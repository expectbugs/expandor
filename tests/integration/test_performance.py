"""
Performance integration tests
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from expandor import Expandor
from expandor.core.config import ExpandorConfig

from .base_integration_test import BaseIntegrationTest


class TestPerformance(BaseIntegrationTest):
    """Test performance characteristics"""

    def test_expansion_performance(
        self, expandor, test_image_small, mock_inpaint_pipeline, temp_dir
    ):
        """Test expansion completes in reasonable time"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 768),
            inpaint_pipeline=mock_inpaint_pipeline,
            stage_dir=temp_dir / "stages",
        )

        start_time = time.time()
        result = expandor.expand(config)
        duration = time.time() - start_time

        # Should complete in reasonable time (adjust based on mock latency)
        assert duration < 10.0  # 10 seconds max for test

        # Check duration is tracked
        assert result.total_duration_seconds > 0
        assert abs(result.total_duration_seconds - duration) < 1.0

    def test_memory_usage_tracking(
        self, expandor, test_image_1080p, mock_inpaint_pipeline, temp_dir
    ):
        """Test memory usage is tracked"""
        config = self.create_config(
            source_image=test_image_1080p,
            target_resolution=(3840, 2160),  # 4K
            inpaint_pipeline=mock_inpaint_pipeline,
            stage_dir=temp_dir / "stages",
        )

        result = expandor.expand(config)

        # Check memory tracking in metadata
        if "peak_vram_mb" in result.metadata:
            assert result.metadata["peak_vram_mb"] > 0
            print(f"Peak VRAM usage: {result.metadata['peak_vram_mb']:.1f} MB")

    def test_concurrent_expansions(
        self, expandor, test_image_small, mock_img2img_pipeline, temp_dir
    ):
        """Test concurrent expansions don't interfere"""

        def expand_image(index):
            config = self.create_config(
                source_image=test_image_small,
                target_resolution=(768, 768),
                img2img_pipeline=mock_img2img_pipeline,
                seed=42 + index,  # Different seeds
                stage_dir=temp_dir / f"stages_{index}",
            )
            return expandor.expand(config)

        # Run multiple expansions concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(expand_image, i) for i in range(3)]
            results = [f.result() for f in futures]

        # All should succeed
        for result in results:
            self.validate_result(result, (768, 768))

    def test_stage_cleanup(
        self, expandor, test_image_small, mock_inpaint_pipeline, temp_dir
    ):
        """Test temporary files are cleaned up"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 768),
            inpaint_pipeline=mock_inpaint_pipeline,
            save_stages=False,  # Don't save stages
            stage_dir=temp_dir / "stages",
        )

        # Track temp files before
        temp_files_before = (
            list(expandor._temp_base.glob("*"))
            if hasattr(expandor, "_temp_base")
            else []
        )

        result = expandor.expand(config)

        # Track temp files after
        temp_files_after = (
            list(expandor._temp_base.glob("*"))
            if hasattr(expandor, "_temp_base")
            else []
        )

        # Should not accumulate temp files
        assert (
            len(temp_files_after) <= len(temp_files_before) + 1
        )  # Allow for result image
