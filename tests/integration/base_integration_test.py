"""
Base class for integration tests
Provides common fixtures and utilities for testing Expandor.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
from PIL import Image, ImageDraw

from expandor import Expandor
from expandor.adapters.mock_pipeline import MockImg2ImgPipeline, MockInpaintPipeline
from expandor.adapters.mock_pipeline_adapter import MockPipelineAdapter
from expandor.core.config import ExpandorConfig
from expandor.core.exceptions import ExpandorError, VRAMError
from expandor.core.result import ExpandorResult


class BaseIntegrationTest:
    """
    Base class for Expandor integration tests.

    Provides common fixtures and validation methods following
    the FAIL LOUD philosophy.
    """

    @pytest.fixture(autouse=True)
    def setup_logging(self, caplog):
        """Setup logging for tests"""
        caplog.set_level(logging.INFO)
        self.caplog = caplog

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp(prefix="expandor_test_")
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_adapter(self):
        """Create mock pipeline adapter for testing"""
        return MockPipelineAdapter(device="cpu", dtype="fp32")

    @pytest.fixture
    def expandor(self, mock_adapter, temp_dir):
        """Create Expandor instance for testing"""
        # New API: Expandor requires adapter
        expandor = Expandor(mock_adapter)
        # Override temp directory for testing
        expandor._temp_base = temp_dir
        return expandor

    @pytest.fixture
    def mock_inpaint_pipeline(self):
        """Create mock inpaint pipeline"""
        return MockInpaintPipeline()

    @pytest.fixture
    def mock_img2img_pipeline(self):
        """Create mock img2img pipeline"""
        return MockImg2ImgPipeline()

    @pytest.fixture
    def test_image_small(self):
        """Create small test image (512x512)"""
        return self.create_test_image(512, 512)

    @pytest.fixture
    def test_image_1080p(self):
        """Create 1080p test image"""
        return self.create_test_image(1920, 1080)

    @pytest.fixture
    def test_image_4k(self):
        """Create 4K test image"""
        return self.create_test_image(3840, 2160)

    def create_test_image(
        self, width: int, height: int, pattern: str = "gradient"
    ) -> Image.Image:
        """
        Create test image with specified pattern.

        Args:
            width: Image width
            height: Image height
            pattern: Pattern type ('gradient', 'checkerboard', 'solid')

        Returns:
            Test image
        """
        img = Image.new("RGB", (width, height))

        if pattern == "gradient":
            # Create gradient pattern for seam detection
            pixels = img.load()
            for x in range(width):
                for y in range(height):
                    r = int((x / width) * 255)
                    g = int((y / height) * 255)
                    b = 128
                    pixels[x, y] = (r, g, b)

        elif pattern == "checkerboard":
            # Create checkerboard pattern
            draw = ImageDraw.Draw(img)
            square_size = 64
            for x in range(0, width, square_size):
                for y in range(0, height, square_size):
                    if (x // square_size + y // square_size) % 2 == 0:
                        draw.rectangle(
                            [x, y, x + square_size, y + square_size],
                            fill=(255, 255, 255),
                        )
                    else:
                        draw.rectangle(
                            [x, y, x + square_size, y + square_size], fill=(0, 0, 0)
                        )

        elif pattern == "solid":
            # Solid color
            img.paste((100, 150, 200), [0, 0, width, height])

        return img

    def create_config(
        self, source_image: Image.Image, target_resolution: Tuple[int, int], **kwargs
    ) -> ExpandorConfig:
        """
        Create ExpandorConfig with defaults for testing.

        Args:
            source_image: Source image
            target_resolution: Target resolution
            **kwargs: Additional config parameters

        Returns:
            ExpandorConfig instance
        """
        defaults = {
            "prompt": "Extend the scene naturally with perfect quality",
            "seed": 42,
            "source_metadata": {"model": "test", "original_prompt": "test scene"},
            "quality_preset": "high",
            "save_stages": True,
            "stage_dir": kwargs.get("temp_dir", Path("temp/stages")),
        }

        # Override with provided kwargs
        defaults.update(kwargs)

        return ExpandorConfig(
            source_image=source_image, target_resolution=target_resolution, **defaults
        )

    def validate_result(self, result: ExpandorResult, expected_size: Tuple[int, int]):
        """
        Validate expansion result - FAIL LOUD on issues.

        Args:
            result: ExpandorResult to validate
            expected_size: Expected (width, height)

        Raises:
            AssertionError: On any validation failure
        """
        # Check result structure
        assert hasattr(result, "image_path"), "Result missing image_path"
        assert hasattr(result, "size"), "Result missing size"
        assert hasattr(result, "metadata"), "Result missing metadata"

        # Verify image exists and can be loaded
        assert (
            result.image_path.exists()
        ), f"Result image not found at {result.image_path}"

        # Verify image can be loaded
        img = Image.open(result.image_path)
        assert (
            img.size == expected_size
        ), f"Size mismatch: {img.size} != {expected_size}"

        # Verify metadata
        assert (
            result.size == expected_size
        ), f"Result size mismatch: {result.size} != {expected_size}"
        assert result.total_duration_seconds > 0, "No duration recorded"
        assert result.strategy_used is not None, "No strategy recorded"

        # Log validation success
        print(
            f"✓ Result validated: {result.size}, {result.strategy_used} strategy, "
            f"{result.total_duration_seconds:.2f}s"
        )

    def check_no_artifacts(self, result: ExpandorResult, tolerance: float = 0.1):
        """
        Verify no significant artifacts detected.

        Args:
            result: ExpandorResult to check
            tolerance: Quality tolerance (0-1)

        Raises:
            AssertionError: If artifacts exceed tolerance
        """
        # Check quality metrics in metadata
        metadata = result.metadata

        if "final_quality_score" in metadata:
            assert metadata["final_quality_score"] > (
                1.0 - tolerance
            ), f"Quality score {metadata['final_quality_score']} below threshold"

        if "seams_detected" in metadata:
            assert (
                metadata["seams_detected"] == 0
            ), f"Detected {metadata['seams_detected']} seams!"

        # Check artifact metrics
        if "artifacts_fixed" in metadata:
            print(f"ℹ️  Fixed {metadata['artifacts_fixed']} artifacts during processing")

    def save_debug_info(self, result: ExpandorResult, test_name: str, temp_dir: Path):
        """Save debug information for failed tests."""
        debug_dir = temp_dir / "debug" / test_name
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Save result image
        if result.image_path and result.image_path.exists():
            shutil.copy(result.image_path, debug_dir / "result.png")

        # Save metadata
        if hasattr(result, "metadata"):
            import json

            with open(debug_dir / "metadata.json", "w") as f:
                json.dump(result.metadata, f, indent=2, default=str)

        # Save stages if available
        if hasattr(result, "stage_results") and result.stage_results:
            with open(debug_dir / "stages.txt", "w") as f:
                for stage in result.stage_results:
                    f.write(f"{stage}\n")

        print(f"Debug info saved to: {debug_dir}")
