"""
Edge case integration tests
"""

import pytest
from PIL import Image

from expandor import Expandor
from expandor.core.exceptions import ExpandorError, VRAMError

from .base_integration_test import BaseIntegrationTest


class TestEdgeCases(BaseIntegrationTest):
    """Test edge cases and error conditions"""
    
    def test_same_size_input_output(self, expandor, test_image_small,
                                   mock_img2img_pipeline, temp_dir):
        """Test when input and output are same size"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(512, 512),  # Same as input
            img2img_pipeline=mock_img2img_pipeline,
            stage_dir=temp_dir / "stages"
        )
        
        # Should handle gracefully
        result = expandor.expand(config)
        self.validate_result(result, (512, 512))
        
        # Might use direct strategy or skip processing
        assert result.strategy_used in ['direct_upscale', 'passthrough']
    
    def test_tiny_image_expansion(self, expandor, mock_inpaint_pipeline, temp_dir):
        """Test expanding very small images"""
        tiny_img = self.create_test_image(64, 64)
        
        config = self.create_config(
            source_image=tiny_img,
            target_resolution=(512, 512),  # 8x expansion
            inpaint_pipeline=mock_inpaint_pipeline,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        self.validate_result(result, (512, 512))
    
    def test_unusual_aspect_ratios(self, expandor, mock_inpaint_pipeline, temp_dir):
        """Test unusual aspect ratios"""
        # Ultra-wide
        img = self.create_test_image(1920, 400)
        
        config = self.create_config(
            source_image=img,
            target_resolution=(3840, 400),  # Keep ultra-wide
            inpaint_pipeline=mock_inpaint_pipeline,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        self.validate_result(result, (3840, 400))
    
    def test_invalid_target_resolution(self, expandor, test_image_small, temp_dir):
        """Test invalid target resolutions fail loud"""
        # Zero dimension
        with pytest.raises(ExpandorError) as exc_info:
            config = self.create_config(
                source_image=test_image_small,
                target_resolution=(0, 512),
                stage_dir=temp_dir / "stages"
            )
            expandor.expand(config)
        
        assert "Invalid target resolution" in str(exc_info.value)
        
        # Negative dimension
        with pytest.raises(ExpandorError) as exc_info:
            config = self.create_config(
                source_image=test_image_small,
                target_resolution=(512, -512),
                stage_dir=temp_dir / "stages"
            )
            expandor.expand(config)
        
        assert "Invalid target resolution" in str(exc_info.value)
    
    def test_corrupted_source_image(self, expandor, temp_dir):
        """Test handling of corrupted source images"""
        # Create invalid image path
        bad_path = temp_dir / "nonexistent.png"
        
        with pytest.raises(ExpandorError) as exc_info:
            config = self.create_config(
                source_image=bad_path,
                target_resolution=(512, 512),
                stage_dir=temp_dir / "stages"
            )
            expandor.expand(config)
        
        assert "not found" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
    
    def test_memory_pressure_handling(self, expandor, test_image_4k,
                                    mock_inpaint_pipeline, temp_dir, monkeypatch):
        """Test handling of memory pressure"""
        # Mock very low memory
        def mock_get_available_vram():
            return 512.0  # Only 512MB
        
        monkeypatch.setattr(
            expandor.vram_manager,
            'get_available_vram',
            mock_get_available_vram
        )
        
        config = self.create_config(
            source_image=test_image_4k,
            target_resolution=(7680, 4320),  # 8K target
            inpaint_pipeline=mock_inpaint_pipeline,
            allow_cpu_offload=True,
            stage_dir=temp_dir / "stages"
        )
        
        # Should use CPU offload or fail loud
        try:
            result = expandor.expand(config)
            assert result.strategy_used in ["cpu_offload", "tiled_expansion"]
        except VRAMError as e:
            # Also acceptable - fail loud
            assert "VRAM" in str(e)
    
    def test_dimension_rounding(self, expandor, test_image_small,
                               mock_inpaint_pipeline, temp_dir):
        """Test that dimensions are properly rounded"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1023, 767),  # Not multiples of 8
            inpaint_pipeline=mock_inpaint_pipeline,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        # Should be rounded to multiples of 8
        assert result.size[0] % 8 == 0
        assert result.size[1] % 8 == 0
        
        # Should be close to requested size
        assert abs(result.size[0] - 1023) < 8
        assert abs(result.size[1] - 767) < 8
    
    def test_force_strategy_override(self, expandor, test_image_small,
                                   mock_inpaint_pipeline, temp_dir):
        """Test force_strategy overrides automatic selection"""
        # Force SWPO even for simple expansion
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(768, 768),  # Simple 1.5x
            inpaint_pipeline=mock_inpaint_pipeline,
            force_strategy='swpo',
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        assert result.strategy_used == 'swpo'