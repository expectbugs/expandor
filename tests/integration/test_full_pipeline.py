"""
Full pipeline integration tests
Tests complete expansion workflows with all strategies.
"""

from pathlib import Path

import pytest
from PIL import ImageDraw

from expandor import Expandor
from expandor.adapters.mock_pipeline import MockImg2ImgPipeline, MockInpaintPipeline
from expandor.core.config import ExpandorConfig
from expandor.core.exceptions import ExpandorError, VRAMError

from .base_integration_test import BaseIntegrationTest


class TestFullPipelineIntegration(BaseIntegrationTest):
    """Test complete expansion pipelines"""

    def test_simple_upscale(
        self, expandor, test_image_small, mock_img2img_pipeline, temp_dir
    ):
        """Test simple 2x upscale"""
        # Register the mock pipeline to make it available
        expandor.register_pipeline("img2img", mock_img2img_pipeline)

        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 1024),  # 2x upscale
            img2img_pipeline=mock_img2img_pipeline,
            stage_dir=temp_dir / "stages",
        )

        result = expandor.expand(config)

        self.validate_result(result, (1024, 1024))
        # Could use various strategies depending on available tools
        # Strategy names can be either the registry key or the class name
        valid_strategies = [
            "direct",
            "direct_upscale",
            "DirectUpscaleStrategy",
            "hybrid_adaptive",
            "HybridAdaptiveStrategy",
            "progressive",
            "progressive_outpaint",
            "ProgressiveOutpaintStrategy",
            "tiled",
            "TiledExpansionStrategy",
        ]
        assert (
            result.strategy_used in valid_strategies
        ), f"Unexpected strategy: {result.strategy_used}"
        self.check_no_artifacts(result)

    def test_progressive_expansion(
        self, expandor, test_image_1080p, mock_inpaint_pipeline, temp_dir
    ):
        """Test progressive outpainting for aspect ratio change"""
        config = self.create_config(
            source_image=test_image_1080p,
            target_resolution=(2560, 1440),  # 16:9 to 16:9 with expansion
            inpaint_pipeline=mock_inpaint_pipeline,
            quality_preset="high",
            stage_dir=temp_dir / "stages",
        )

        result = expandor.expand(config)

        self.validate_result(result, (2560, 1440))
        assert result.strategy_used in [
            "progressive",
            "progressive_outpaint",
            "ProgressiveOutpaintStrategy",
            "hybrid_adaptive",
            "HybridAdaptiveStrategy",
        ]

        # Check boundaries were tracked
        assert (
            "boundary_positions" in result.metadata or "boundaries" in result.metadata
        )

    def test_extreme_aspect_ratio_change(
        self, expandor, test_image_1080p, mock_inpaint_pipeline, temp_dir
    ):
        """Test extreme aspect ratio change (16:9 to 32:9)"""
        config = self.create_config(
            source_image=test_image_1080p,
            target_resolution=(3840, 1080),  # Double width
            inpaint_pipeline=mock_inpaint_pipeline,
            quality_preset="balanced",
            window_size=300,
            overlap_ratio=0.8,
            stage_dir=temp_dir / "stages",
        )

        result = expandor.expand(config)

        self.validate_result(result, (3840, 1080))
        # Should use SWPO or progressive for extreme ratio
        assert result.strategy_used in [
            "swpo",
            "SWPOStrategy",
            "progressive",
            "progressive_outpaint",
            "ProgressiveOutpaintStrategy",
            "hybrid_adaptive",
            "HybridAdaptiveStrategy",
        ]

    def test_vram_constrained_expansion(
        self, expandor, test_image_4k, mock_inpaint_pipeline, temp_dir, monkeypatch
    ):
        """Test expansion with limited VRAM"""

        # Mock limited VRAM
        def mock_get_available_vram():
            return 2048.0  # 2GB only

        monkeypatch.setattr(
            expandor.vram_manager, "get_available_vram", mock_get_available_vram
        )

        config = self.create_config(
            source_image=test_image_4k,
            target_resolution=(7680, 4320),  # 8K target
            inpaint_pipeline=mock_inpaint_pipeline,
            allow_tiled=True,
            allow_cpu_offload=True,
            stage_dir=temp_dir / "stages",
        )

        result = expandor.expand(config)

        self.validate_result(result, (7680, 4320))
        # Should use tiled or CPU offload strategy
        assert result.strategy_used in [
            "tiled",
            "tiled_expansion",
            "TiledExpansionStrategy",
            "cpu_offload",
            "CPUOffloadStrategy",
            "hybrid_adaptive",
            "HybridAdaptiveStrategy",
        ]

    def test_no_pipeline_fails_loud(self, expandor, test_image_small, temp_dir):
        """Test that expansion fails loud without pipelines"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 768),  # Aspect change requires pipeline
            # No pipelines provided
            stage_dir=temp_dir / "stages",
        )

        with pytest.raises(ExpandorError) as exc_info:
            expandor.expand(config)

        # Should have clear error message
        assert "pipeline" in str(exc_info.value).lower()

    def test_quality_enforcement(self, expandor, mock_inpaint_pipeline, temp_dir):
        """Test quality enforcement with auto-refinement"""
        # Create image with intentional seam
        img = self.create_test_image(512, 512, pattern="solid")
        draw = ImageDraw.Draw(img)
        # Add visible seam
        draw.line([(256, 0), (256, 512)], fill=(255, 0, 0), width=2)

        config = self.create_config(
            source_image=img,
            target_resolution=(1024, 512),  # Horizontal expansion
            inpaint_pipeline=mock_inpaint_pipeline,
            quality_preset="ultra",
            auto_refine=True,
            stage_dir=temp_dir / "stages",
        )

        result = expandor.expand(config)

        self.validate_result(result, (1024, 512))

        # Should have quality validation in metadata
        assert (
            "quality_refined" in result.metadata
            or "final_quality_score" in result.metadata
        )

    def test_stage_saving(
        self, expandor, test_image_small, mock_inpaint_pipeline, temp_dir
    ):
        """Test that stages are saved correctly"""
        stage_dir = temp_dir / "stages"

        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(768, 768),  # Requires processing
            inpaint_pipeline=mock_inpaint_pipeline,
            save_stages=True,
            stage_dir=stage_dir,
        )

        result = expandor.expand(config)

        self.validate_result(result, (768, 768))

        # Check stages were saved
        assert stage_dir.exists()
        stage_files = list(stage_dir.glob("*.png"))
        assert len(stage_files) > 0, "No stage files saved"

        # Verify stage images can be loaded
        for stage_file in stage_files:
            img = Image.open(stage_file)
            assert img.size[0] > 0 and img.size[1] > 0

    def test_metadata_completeness(
        self, expandor, test_image_small, mock_inpaint_pipeline, temp_dir
    ):
        """Test that metadata is complete"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 768),
            inpaint_pipeline=mock_inpaint_pipeline,
            stage_dir=temp_dir / "stages",
        )

        result = expandor.expand(config)

        # Check required metadata fields
        assert "strategy_used" in result.metadata
        assert "duration_seconds" in result.metadata
        assert "stages" in result.metadata

        # Check stage information
        if result.metadata["stages"]:
            stage = result.metadata["stages"][0]
            assert "name" in stage
            assert "duration" in stage
            assert "method" in stage
