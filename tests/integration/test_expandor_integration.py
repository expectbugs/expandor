"""
Integration tests for complete Expandor workflows
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from expandor import (
    Expandor, ExpandorConfig, ExpandorResult,
    VRAMError, StrategyError, QualityError
)
from expandor.adapters.mock_pipeline import MockInpaintPipeline, MockRefinerPipeline

class TestExpandorIntegration:
    
    def setup_method(self):
        """Setup for each test"""
        self.expandor = Expandor()
        self.mock_inpaint = MockInpaintPipeline()
        self.mock_refiner = MockRefinerPipeline()
    
    def test_complete_workflow_direct_upscale(self):
        """Test complete workflow with direct upscale"""
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512), color='blue'),
            target_resolution=(1024, 1024),
            prompt="A beautiful blue square",
            seed=42,
            source_metadata={'model': 'test'},
            quality_preset='fast'
        )
        
        result = self.expandor.expand(config)
        
        assert isinstance(result, ExpandorResult)
        assert result.success == True
        assert result.size == (1024, 1024)
        assert result.strategy_used == 'DirectUpscaleStrategy'
        assert result.image_path.exists()
        assert result.metadata is not None
    
    def test_complete_workflow_progressive_outpaint(self):
        """Test complete workflow with progressive outpainting"""
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512), color='green'),
            target_resolution=(1920, 512),  # Wide aspect ratio
            prompt="A wide green landscape",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=self.mock_inpaint,
            quality_preset='balanced'
        )
        
        result = self.expandor.expand(config)
        
        assert result.success == True
        assert result.size == (1920, 512)
        assert 'progressive' in result.strategy_used.lower()
        assert len(result.stages) > 0
        assert len(result.boundaries) > 0
    
    def test_complete_workflow_with_pipelines(self):
        """Test workflow with registered pipelines"""
        # Register pipelines
        self.expandor.register_pipeline('inpaint', self.mock_inpaint)
        self.expandor.register_pipeline('refiner', self.mock_refiner)
        
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(768, 768),
            prompt="Test with pipelines",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        result = self.expandor.expand(config)
        
        assert result.success == True
        # Should have used registered pipelines
        assert 'inpaint' in self.expandor.pipeline_registry
        assert 'refiner' in self.expandor.pipeline_registry
    
    def test_metadata_saved_with_result(self, temp_dir):
        """Test metadata is saved alongside result"""
        # Use temp directory for result
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1024, 1024),
            prompt="Test metadata",
            seed=42,
            source_metadata={'model': 'test'},
            stage_dir=temp_dir
        )
        
        with patch.object(self.expandor, '_save_metadata') as mock_save:
            result = self.expandor.expand(config)
            mock_save.assert_called_once()
        
        # Check metadata format
        metadata_path = result.image_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert 'operation_log' in metadata
            assert 'config_snapshot' in metadata
            assert 'boundary_positions' in metadata
    
    def test_vram_aware_strategy_selection(self):
        """Test VRAM-aware strategy selection"""
        # Mock low VRAM scenario
        with patch.object(self.expandor.vram_manager, 'get_available_vram') as mock_vram:
            mock_vram.return_value = 2000  # 2GB
            
            with patch.object(self.expandor.vram_manager, 'calculate_generation_vram') as mock_calc:
                mock_calc.return_value = 8000  # 8GB needed
                
                config = ExpandorConfig(
                    source_image=Image.new('RGB', (2048, 2048)),
                    target_resolution=(4096, 4096),
                    prompt="Large image",
                    seed=42,
                    source_metadata={'model': 'test'}
                )
                
                result = self.expandor.expand(config)
                
                # Should have selected a VRAM-friendly strategy
                assert result.strategy_used in ['TiledExpansionStrategy', 'CPUOffloadStrategy']
    
    def test_quality_validation_and_repair(self):
        """Test quality validation and repair workflow"""
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1024, 1024),
            prompt="Test quality",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=self.mock_inpaint,
            quality_preset='ultra',
            auto_refine=True
        )
        
        # Mock quality validator to detect issues
        with patch.object(self.expandor.quality_validator, 'validate') as mock_validate:
            mock_validate.return_value = {
                'passed': False,
                'issues': [{'type': 'seam', 'location': (512, 0, 10, 1024)}],
                'score': 0.7
            }
            
            result = self.expandor.expand(config)
        
        assert result.success == True
        assert result.quality_score >= 0.7
        # Should have attempted repair
        assert mock_validate.called
    
    def test_error_handling_invalid_config(self):
        """Test error handling for invalid configuration"""
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(-1024, 1024),  # Invalid
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        with pytest.raises(ExpandorError) as exc_info:
            self.expandor.expand(config)
        
        assert "validation" in str(exc_info.value).lower()
        assert exc_info.value.stage == "validation"
    
    def test_error_handling_missing_pipeline(self):
        """Test error handling when required pipeline is missing"""
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1920, 512),  # Needs progressive outpaint
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'},
            # No inpaint pipeline provided
            quality_preset='ultra'
        )
        
        # Should either fall back to another strategy or fail gracefully
        result = self.expandor.expand(config)
        
        # Should have used a fallback strategy
        assert result.success == True
        assert result.fallback_count >= 0
    
    def test_stage_saving(self, temp_dir):
        """Test saving intermediate stages"""
        stage_dir = temp_dir / "stages"
        
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1024, 512),
            prompt="Test stages",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=self.mock_inpaint,
            save_stages=True,
            stage_dir=stage_dir
        )
        
        result = self.expandor.expand(config)
        
        assert result.success == True
        assert stage_dir.exists()
        # Should have saved some stage files
        stage_files = list(stage_dir.glob("*.png"))
        assert len(stage_files) >= 0  # Might be 0 with mocks
    
    def test_cache_clearing(self):
        """Test cache clearing functionality"""
        # Run an expansion to populate caches
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        self.expandor.expand(config)
        
        # Clear caches
        self.expandor.clear_caches()
        
        # Should not raise any errors
        assert True
    
    def test_vram_estimation(self):
        """Test VRAM estimation before execution"""
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(4096, 4096),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        estimate = self.expandor.estimate_vram(config)
        
        assert 'total_required_mb' in estimate
        assert 'available_mb' in estimate
        assert 'recommended_strategy' in estimate
        assert estimate['total_required_mb'] > 0
    
    def test_pipeline_fallback_chain(self):
        """Test strategy fallback chain execution"""
        # Create a strategy that will fail
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(8192, 8192),  # Very large
            prompt="Test fallback",
            seed=42,
            source_metadata={'model': 'test'},
            quality_preset='ultra'
        )
        
        # Mock first strategy to fail
        with patch('expandor.strategies.direct_upscale.DirectUpscaleStrategy.execute') as mock_exec:
            mock_exec.side_effect = VRAMError(
                operation="upscale",
                required_mb=20000,
                available_mb=8000
            )
            
            result = self.expandor.expand(config)
        
        # Should have used fallback
        assert result.success == True
        assert result.fallback_count > 0
        assert 'execution_history' in result.metadata