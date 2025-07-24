"""
CPU Offload strategy integration tests
"""

import pytest
from PIL import Image

from expandor import Expandor
from expandor.core.exceptions import ExpandorError, StrategyError

from .base_integration_test import BaseIntegrationTest


class TestCPUOffloadIntegration(BaseIntegrationTest):
    """Integration tests for CPU Offload strategy"""

    def test_cpu_offload_basic(
        self, expandor, test_image_small, mock_inpaint_pipeline, temp_dir
    ):
        """Test basic CPU offload operation"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 1024),
            inpaint_pipeline=mock_inpaint_pipeline,
            strategy_override="cpu_offload",
            allow_cpu_offload=True,
            stage_dir=temp_dir / "stages",
        )

        result = expandor.expand(config)

        self.validate_result(result, (1024, 1024))
        assert result.strategy_used == "cpu_offload"

        # Check CPU offload specific metadata
        assert "tile_size" in result.metadata
        assert result.metadata["tile_size"] >= 384  # Minimum tile size

    def test_cpu_offload_not_allowed_fails(self, expandor, test_image_small, temp_dir):
        """Test CPU offload fails when not allowed"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 1024),
            strategy_override="cpu_offload",
            allow_cpu_offload=False,  # Not allowed
            stage_dir=temp_dir / "stages",
        )

        with pytest.raises((ExpandorError, StrategyError)) as exc_info:
            expandor.expand(config)

        assert "not allowed" in str(exc_info.value).lower()

    def test_cpu_offload_extreme_memory_constraint(
        self, expandor, test_image_4k, mock_img2img_pipeline, temp_dir, monkeypatch
    ):
        """Test CPU offload with extreme memory constraints"""

        # Mock very low memory
        def mock_get_available_vram():
            return 256.0  # Only 256MB

        monkeypatch.setattr(
            expandor.vram_manager, "get_available_vram", mock_get_available_vram
        )

        config = self.create_config(
            source_image=test_image_4k,
            target_resolution=(5760, 3240),  # 1.5x expansion
            img2img_pipeline=mock_img2img_pipeline,
            strategy_override="cpu_offload",
            allow_cpu_offload=True,
            stage_dir=temp_dir / "stages",
        )

        result = expandor.expand(config)

        self.validate_result(result, (5760, 3240))

        # Should use very small tiles
        assert result.metadata["tile_size"] <= 512

    def test_cpu_offload_stage_tracking(
        self, expandor, test_image_small, mock_inpaint_pipeline, temp_dir
    ):
        """Test CPU offload tracks processing stages"""
        stage_dir = temp_dir / "stages"

        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 768),  # Aspect change
            inpaint_pipeline=mock_inpaint_pipeline,
            strategy_override="cpu_offload",
            allow_cpu_offload=True,
            save_stages=True,
            stage_dir=stage_dir,
        )

        result = expandor.expand(config)

        # Check stages saved
        cpu_stages = list(stage_dir.glob("cpu_offload_stage_*.png"))
        assert len(cpu_stages) > 0

        # Check metadata
        assert "total_stages" in result.metadata
        assert result.metadata["total_stages"] >= len(cpu_stages)
