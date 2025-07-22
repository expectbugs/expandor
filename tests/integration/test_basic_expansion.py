"""
Basic integration test for Expandor
"""

import pytest
from pathlib import Path
from PIL import Image

from expandor import Expandor, ExpandorConfig
from expandor.adapters.mock_pipeline import MockInpaintPipeline

class TestBasicExpansion:
    
    def setup_method(self):
        """Setup for each test"""
        self.expandor = Expandor()
        self.mock_pipeline = MockInpaintPipeline()
        self.test_image_path = Path("tests/fixtures/landscape_1344x768.png")
        
        # Register mock pipeline
        self.expandor.register_pipeline("inpaint", self.mock_pipeline)
    
    def test_simple_expansion(self):
        """Test basic image expansion"""
        # Load test image
        source_image = Image.open(self.test_image_path)
        
        # Create config for 16:9 to 21:9 expansion
        config = ExpandorConfig(
            source_image=source_image,
            target_resolution=(2560, 1080),  # 21:9 aspect
            prompt="A beautiful landscape with mountains",
            seed=42,
            source_metadata={'model': 'SDXL'},
            generation_metadata={},
            inpaint_pipeline=self.mock_pipeline,
            quality_preset='fast',
            save_stages=False
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify results
        assert result.success
        assert result.size == (2560, 1080)
        assert len(result.stages) > 0
        assert result.strategy_used in ['progressive_outpaint', 'direct_upscale']
        assert result.image_path.exists()
        
        # Check quality metrics
        assert result.seams_detected == 0  # Mock should produce no seams
        assert result.vram_peak_mb > 0
        assert result.total_duration_seconds > 0
    
    def test_extreme_aspect_change(self):
        """Test extreme aspect ratio change"""
        source_image = Image.open(self.test_image_path)
        
        # 16:9 to 32:9 (super ultrawide)
        config = ExpandorConfig(
            source_image=source_image,
            target_resolution=(5760, 1080),
            prompt="An expansive landscape panorama",
            seed=123,
            source_metadata={'model': 'SDXL'},
            inpaint_pipeline=self.mock_pipeline,
            quality_preset='balanced'
        )
        
        result = self.expandor.expand(config)
        
        assert result.success
        assert result.size[0] / result.size[1] > 5.0  # Very wide
        assert 'swpo' in result.strategy_used.lower() or 'progressive' in result.strategy_used.lower()