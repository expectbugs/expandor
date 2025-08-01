"""Unit tests for Hybrid Adaptive Strategy"""

from unittest.mock import Mock, patch

import pytest
from PIL import Image

from expandor.core.config import ExpandorConfig
from expandor.strategies.experimental.hybrid_adaptive import HybridAdaptiveStrategy


class TestHybridAdaptiveStrategy:

    @pytest.fixture
    def strategy(self):
        return HybridAdaptiveStrategy()

    def test_simple_upscale_plan(self, strategy):
        """Test planning for simple upscale"""
        config = ExpandorConfig(
            source_image=Image.new("RGB", (512, 512)),
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={},
        )

        # Mock Real-ESRGAN availability
        with patch.object(strategy.strategies["direct"], "realesrgan_path", "/mock/path"):
            plan = strategy._analyze_expansion(config)

        assert len(plan.steps) == 1
        # Accept either direct or progressive for simple upscale
        assert plan.steps[0]["strategy"] in ["direct", "progressive"]
        # Rationale varies by strategy chosen
        assert "upscale" in plan.rationale.lower()

    def test_extreme_ratio_plan(self, strategy):
        """Test planning for extreme aspect ratio"""
        config = ExpandorConfig(
            source_image=Image.new("RGB", (512, 512)),
            target_resolution=(4096, 512),  # 8:1 ratio
            prompt="Test",
            seed=42,
            source_metadata={},
        )

        # Mock inpaint pipeline to enable SWPO
        strategy.inpaint_pipeline = Mock()
        plan = strategy._analyze_expansion(config)

        assert plan.steps[0]["strategy"] == "swpo"
        assert "Extreme" in plan.rationale
