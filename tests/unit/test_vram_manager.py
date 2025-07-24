"""
Test VRAM Manager functionality
"""

import pytest
import torch

from expandor.core.vram_manager import VRAMManager


class TestVRAMManager:

    def setup_method(self):
        """Setup for each test"""
        self.vram_manager = VRAMManager()

    def test_calculate_generation_vram(self):
        """Test VRAM calculation for different resolutions"""
        # Test 1080p
        result = self.vram_manager.calculate_generation_vram(1920, 1080)
        assert isinstance(result, float)
        assert result > 0

        # Test 4K
        result_4k = self.vram_manager.calculate_generation_vram(3840, 2160)
        assert result_4k > result

        # Test with batch size
        result_batch2 = self.vram_manager.calculate_generation_vram(
            1920, 1080, batch_size=2
        )
        assert result_batch2 == result * 2

    def test_calculate_generation_vram_with_dtype(self):
        """Test VRAM calculation with different dtypes"""
        # Test float16 (default)
        result_f16 = self.vram_manager.calculate_generation_vram(
            1920, 1080, dtype="float16"
        )

        # Test float32
        result_f32 = self.vram_manager.calculate_generation_vram(
            1920, 1080, dtype="float32"
        )

        # float32 should use 2x memory of float16
        assert result_f32 > result_f16

        # Test bfloat16
        result_bf16 = self.vram_manager.calculate_generation_vram(
            1920, 1080, dtype="bfloat16"
        )

        # bfloat16 should be same as float16
        assert result_bf16 == result_f16

    def test_determine_strategy(self):
        """Test strategy determination"""
        # Small image should use full strategy
        strategy = self.vram_manager.determine_strategy(1024, 1024)
        assert strategy["strategy"] in ["full", "tiled", "cpu_offload"]

        # Huge image might need tiling or CPU offload
        strategy_huge = self.vram_manager.determine_strategy(8192, 8192)
        assert strategy_huge["vram_required_mb"] > strategy["vram_required_mb"]

    def test_get_available_vram(self):
        """Test VRAM availability check"""
        vram = self.vram_manager.get_available_vram()

        if torch.cuda.is_available():
            assert vram is not None
            assert vram > 0
        else:
            # CPU-only system
            assert vram is None

    def test_peak_usage_tracking(self):
        """Test peak VRAM usage tracking"""
        # Initial peak should be 0
        assert self.vram_manager.get_peak_usage() == 0.0

        # Track some usage
        self.vram_manager.track_peak_usage(1000.0)
        assert self.vram_manager.get_peak_usage() == 1000.0

        # Track lower usage - peak should not change
        self.vram_manager.track_peak_usage(500.0)
        assert self.vram_manager.get_peak_usage() == 1000.0

        # Track higher usage - peak should update
        self.vram_manager.track_peak_usage(1500.0)
        assert self.vram_manager.get_peak_usage() == 1500.0

    def test_clear_cache(self):
        """Test cache clearing"""
        # Should not raise any errors
        self.vram_manager.clear_cache()

        # Test with CUDA if available
        if torch.cuda.is_available():
            # Create some tensor to use memory
            tensor = torch.randn(1000, 1000).cuda()
            self.vram_manager.clear_cache()
            # Cache should be cleared without error
