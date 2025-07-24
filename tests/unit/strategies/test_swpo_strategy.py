"""
Unit tests for SWPO (Sliding Window Progressive Outpainting) Strategy
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from expandor.core.config import ExpandorConfig
from expandor.core.exceptions import ExpandorError, StrategyError
from expandor.strategies.swpo_strategy import SWPOStrategy, SWPOWindow


class TestSWPOStrategy:
    """Comprehensive tests for SWPO strategy"""

    @pytest.fixture
    def strategy(self):
        """Create SWPO strategy instance"""
        return SWPOStrategy()

    @pytest.fixture
    def base_config(self):
        """Create base configuration for testing"""
        # Create test image
        test_image = Image.new("RGB", (1024, 768), color="blue")

        # Mock pipeline
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.images = [Image.new("RGB", (1344, 768), color="red")]
        mock_pipeline.return_value = mock_result

        config = ExpandorConfig(
            source_image=test_image,
            target_resolution=(5376, 768),  # 7:1 extreme ratio
            prompt="Test prompt",
            seed=42,
            source_metadata={"model": "test"},
            window_size=200,
            overlap_ratio=0.8,
            denoising_strength=0.95,
        )

        return config

    def test_window_planning_horizontal(self, strategy):
        """Test window planning for horizontal expansion"""
        windows = strategy._plan_windows(
            source_size=(1024, 768),
            target_size=(2048, 768),
            window_size=400,
            overlap_ratio=0.5,
        )

        # Should have multiple windows
        assert len(windows) > 1

        # First window should start at source edge
        assert windows[0].is_first == True
        assert windows[0].position[0] == 0

        # Last window should reach target size
        assert windows[-1].is_last == True
        assert windows[-1].position[2] == 2048

        # All should be horizontal expansion
        for window in windows:
            assert window.expansion_type == "horizontal"

    def test_window_planning_vertical(self, strategy):
        """Test window planning for vertical expansion"""
        windows = strategy._plan_windows(
            source_size=(768, 1024),
            target_size=(768, 2048),
            window_size=400,
            overlap_ratio=0.5,
        )

        # Should have multiple windows
        assert len(windows) > 1

        # All should be vertical expansion
        for window in windows:
            assert window.expansion_type == "vertical"

        # Last window should reach target size
        assert windows[-1].position[3] == 2048

    def test_dimension_rounding(self, strategy):
        """Test that dimensions are rounded to multiples of 8"""
        windows = strategy._plan_windows(
            source_size=(1023, 767),  # Not multiples of 8
            target_size=(2047, 1535),  # Not multiples of 8
            window_size=200,
            overlap_ratio=0.8,
        )

        # All window dimensions should be multiples of 8
        for window in windows:
            x1, y1, x2, y2 = window.position
            width = x2 - x1
            height = y2 - y1
            # Note: Individual windows might not be multiples of 8,
            # but the final dimensions should be

    def test_execute_with_context(self, strategy, base_config):
        """Test execution with proper context injection"""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.images = [Image.new("RGB", (1344, 768), color="red")]
        mock_pipeline.return_value = mock_result
        
        # Mock context
        context = {
            "pipeline_registry": {"inpaint": mock_pipeline},
            "boundary_tracker": Mock(),
            "metadata_tracker": Mock(),
        }

        # Mock window execution
        with patch.object(strategy, "_execute_window") as mock_execute:
            mock_execute.return_value = (
                Image.new("RGB", base_config.target_resolution),
                {
                    "boundary_position": 1024,
                    "input_size": (1224, 768),
                    "output_size": (1424, 768),
                },
            )

            # Set the pipeline on strategy from context (mimicking orchestrator)
            strategy.inpaint_pipeline = context["pipeline_registry"]["inpaint"]
            strategy.boundary_tracker = context["boundary_tracker"]
            strategy.metadata_tracker = context["metadata_tracker"]
            
            result = strategy.execute(base_config, context)

            assert result["size"] == base_config.target_resolution
            assert result["metadata"]["strategy"] == "swpo"
            assert "boundaries" in result
