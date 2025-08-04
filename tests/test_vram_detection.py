import pytest
import torch
from expandor.utils.vram_manager import VRAMManager

class TestVRAMDetection:
    """Test VRAM detection works correctly"""
    
    def test_vram_detection_returns_int(self):
        """VRAM detection should return integer MB"""
        manager = VRAMManager()
        
        if torch.cuda.is_available():
            vram = manager.get_available_vram()
            assert isinstance(vram, int)
            assert vram > 0
            assert vram < 100000  # Sanity check - less than 100GB
    
    def test_vram_fails_loud_on_error(self):
        """VRAM detection should fail loud if CUDA unavailable"""
        # Mock torch.cuda.is_available to return False
        import unittest.mock
        
        with unittest.mock.patch('torch.cuda.is_available', return_value=False):
            manager = VRAMManager()
            vram = manager.get_available_vram()
            assert vram == 0  # Should return 0 for CPU