"""Unit tests for CPU Offload Strategy"""

from unittest.mock import Mock, patch

import pytest
from PIL import Image

from expandor.core.config import ExpandorConfig
from expandor.core.exceptions import StrategyError
from expandor.strategies.cpu_offload import CPUOffloadStrategy


class TestCPUOffloadStrategy:

    @pytest.fixture
    def strategy(self):
        return CPUOffloadStrategy()

    def test_requires_cpu_offload_allowed(self, strategy):
        """Test that strategy fails if CPU offload not allowed"""
        config = ExpandorConfig(
            source_image=Image.new("RGB", (512, 512)),
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={},
            use_cpu_offload=False,
        )

        with pytest.raises(StrategyError) as exc_info:
            strategy.execute(config)

        assert "not allowed" in str(exc_info.value)

    def test_tile_size_calculation(self, strategy):
        """Test optimal tile size calculation"""
        # Test with various VRAM amounts
        tile_512 = strategy._calculate_optimal_tile_size(512)
        assert 384 <= tile_512 <= 768  # Within bounds
        assert tile_512 % 64 == 0  # Multiple of 64

        tile_2048 = strategy._calculate_optimal_tile_size(2048)
        assert tile_2048 <= 768  # Maximum
        assert tile_2048 % 64 == 0  # Multiple of 64

        tile_1024 = strategy._calculate_optimal_tile_size(1024)
        assert 384 <= tile_1024 <= 768  # Within bounds
        assert tile_1024 % 64 == 0  # Multiple of 64
