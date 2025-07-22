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
    
    def test_determine_strategy(self):
        """Test strategy determination"""
        # Small image should use full strategy
        strategy = self.vram_manager.determine_strategy(1024, 1024)
        assert strategy['strategy'] in ['full', 'tiled', 'cpu_offload']
        
        # Huge image might need tiling or CPU offload
        strategy_huge = self.vram_manager.determine_strategy(8192, 8192)
        assert strategy_huge['vram_required_mb'] > strategy['vram_required_mb']
    
    def test_get_available_vram(self):
        """Test VRAM availability check"""
        vram = self.vram_manager.get_available_vram()
        
        if torch.cuda.is_available():
            assert vram is not None
            assert vram > 0
        else:
            # CPU-only system
            assert vram is None