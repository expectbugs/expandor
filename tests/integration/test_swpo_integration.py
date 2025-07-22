"""
SWPO strategy integration tests
"""

import pytest
from PIL import Image

from expandor import Expandor
from expandor.core.exceptions import ExpandorError, StrategyError

from .base_integration_test import BaseIntegrationTest


class TestSWPOIntegration(BaseIntegrationTest):
    """Integration tests for SWPO strategy"""
    
    def test_swpo_extreme_horizontal(self, expandor, test_image_1080p,
                                   mock_inpaint_pipeline, temp_dir):
        """Test SWPO handles extreme horizontal expansion"""
        config = self.create_config(
            source_image=test_image_1080p,
            target_resolution=(7680, 1080),  # 4x width
            inpaint_pipeline=mock_inpaint_pipeline,
            window_size=400,
            overlap_ratio=0.8,
            force_strategy='swpo',  # Force SWPO
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (7680, 1080))
        assert result.strategy_used == 'swpo'
        
        # Check SWPO specific metadata
        assert 'total_windows' in result.metadata
        assert result.metadata['total_windows'] > 1
        
        # Check window parameters
        assert 'window_parameters' in result.metadata
        assert result.metadata['window_parameters']['window_size'] == 400
        assert result.metadata['window_parameters']['overlap_ratio'] == 0.8
    
    def test_swpo_extreme_vertical(self, expandor, test_image_small,
                                  mock_inpaint_pipeline, temp_dir):
        """Test SWPO handles extreme vertical expansion"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(512, 2048),  # 4x height
            inpaint_pipeline=mock_inpaint_pipeline,
            window_size=256,
            overlap_ratio=0.75,
            force_strategy='swpo',
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (512, 2048))
        
        # Verify boundaries tracked
        assert 'boundaries' in result.metadata or 'boundary_positions' in result.metadata
    
    def test_swpo_bidirectional_expansion(self, expandor, test_image_small,
                                        mock_inpaint_pipeline, temp_dir):
        """Test SWPO handles expansion in both directions"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 1024),  # 2x in both dimensions
            inpaint_pipeline=mock_inpaint_pipeline,
            window_size=300,
            overlap_ratio=0.8,
            force_strategy='swpo',
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (1024, 1024))
        self.check_no_artifacts(result, tolerance=0.15)
    
    def test_swpo_without_pipeline_fails(self, expandor, test_image_small, temp_dir):
        """Test SWPO fails properly without inpaint pipeline"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 512),
            force_strategy='swpo',
            # No inpaint pipeline
            stage_dir=temp_dir / "stages"
        )
        
        with pytest.raises((ExpandorError, StrategyError)) as exc_info:
            expandor.expand(config)
        
        assert "inpaint" in str(exc_info.value).lower()
    
    def test_swpo_stage_tracking(self, expandor, test_image_small,
                                mock_inpaint_pipeline, temp_dir):
        """Test SWPO tracks all window stages"""
        stage_dir = temp_dir / "stages"
        
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1536, 512),  # 3x width
            inpaint_pipeline=mock_inpaint_pipeline,
            window_size=256,
            overlap_ratio=0.5,
            force_strategy='swpo',
            save_stages=True,
            stage_dir=stage_dir
        )
        
        result = expandor.expand(config)
        
        # Check window stages saved
        window_stages = list(stage_dir.glob("swpo_window_*.png"))
        assert len(window_stages) > 1
        
        # Check metadata has stage info
        assert 'stages' in result.metadata
        swpo_stages = [s for s in result.metadata['stages'] if 'swpo' in s['name']]
        assert len(swpo_stages) >= len(window_stages)