"""
Unit tests for enhanced artifact detection
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from expandor.core.exceptions import QualityError
from expandor.processors.artifact_detector_enhanced import (
    ArtifactSeverity,
    EnhancedArtifactDetector,
)


class TestEnhancedArtifactDetector:
    """Test enhanced artifact detection"""

    @pytest.fixture
    def detector(self):
        return EnhancedArtifactDetector()

    @pytest.fixture
    def test_image_with_seam(self):
        """Create test image with visible seam"""
        img = Image.new("RGB", (512, 512), color=(100, 100, 100))
        draw = ImageDraw.Draw(img)

        # Left half different color
        draw.rectangle([0, 0, 256, 512], fill=(50, 50, 50))
        # Right half different color
        draw.rectangle([256, 0, 512, 512], fill=(150, 150, 150))

        return img

    def test_detects_color_discontinuity(self, detector, test_image_with_seam):
        """Test detection of color discontinuity at boundary"""
        boundaries = [{"position": 256, "direction": "vertical", "step": 0}]

        result = detector.detect_artifacts_comprehensive(
            test_image_with_seam, boundaries, "ultra"
        )

        assert result.has_artifacts
        assert result.severity in [ArtifactSeverity.HIGH, ArtifactSeverity.CRITICAL]
        assert result.detection_scores["color"] > 0.5

    def test_clean_image_passes(self, detector):
        """Test clean image has no artifacts"""
        # Create solid color image with slight noise to avoid perfect uniformity
        img = Image.new("RGB", (512, 512))
        pixels = img.load()

        # Use numpy for slight random noise
        np.random.seed(42)
        for x in range(512):
            for y in range(512):
                # Solid color with tiny noise
                base_color = 128
                noise = int(np.random.normal(0, 2))  # Very small noise
                color = max(0, min(255, base_color + noise))
                pixels[x, y] = (color, color, color)

        result = detector.detect_artifacts_comprehensive(img, [], "ultra")

        # With ultra-strict settings, even clean images might have low severity
        # This aligns with the FAIL LOUD philosophy
        assert result.severity in [ArtifactSeverity.NONE, ArtifactSeverity.LOW]

        # If artifacts are detected, they should be minimal
        if result.has_artifacts:
            assert result.artifact_percentage < 1.0  # Less than 1% artifacts
            assert result.severity == ArtifactSeverity.LOW

    def test_recommendations_generated(self, detector, test_image_with_seam):
        """Test that recommendations are generated"""
        boundaries = [{"position": 256, "direction": "vertical", "step": 0}]

        result = detector.detect_artifacts_comprehensive(
            test_image_with_seam, boundaries, "balanced"
        )

        assert len(result.recommendations) > 0
        assert any("discontinuit" in r.lower() for r in result.recommendations)

    def test_critical_fails_loud(self, detector):
        """Test critical artifacts raise QualityError"""
        # Create image with extreme artifacts
        img = Image.new("RGB", (512, 512))
        draw = ImageDraw.Draw(img)

        # Create checkerboard pattern (extreme artifacts)
        for x in range(0, 512, 32):
            for y in range(0, 512, 32):
                if (x // 32 + y // 32) % 2 == 0:
                    draw.rectangle([x, y, x + 32, y + 32], fill=(255, 255, 255))
                else:
                    draw.rectangle([x, y, x + 32, y + 32], fill=(0, 0, 0))

        boundaries = [
            {"position": i * 32, "direction": "vertical", "step": i} for i in range(16)
        ]

        with pytest.raises(QualityError) as exc_info:
            detector.detect_artifacts_comprehensive(img, boundaries, "ultra")

        assert "Critical" in str(exc_info.value)
