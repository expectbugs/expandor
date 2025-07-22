"""
VRAM Manager for Dynamic Resource Management
Adapted from ai-wallpaper project (https://github.com/user/ai-wallpaper)
Original: ai_wallpaper/core/vram_calculator.py
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
import logging

class VRAMManager:
    """Calculate VRAM requirements and determine expansion strategy"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # SDXL base requirements (measured empirically)
        # Copy from lines 17-20 of vram_calculator.py
        self.MODEL_OVERHEAD_MB = 6144  # 6GB for SDXL refiner
        self.ATTENTION_MULTIPLIER = 4   # Attention needs ~4x image memory
        self.SAFETY_BUFFER = 0.2        # 20% safety margin
        
    def calculate_generation_vram(self, 
                                 width: int, 
                                 height: int,
                                 batch_size: int = 1,
                                 model_type: str = "sdxl") -> float:
        """
        Calculate ACCURATE VRAM requirements for refinement.
        Copy implementation from lines 22-76 of vram_calculator.py
        """
        pixels = width * height
        
        # Bytes per pixel - always use float16 for SDXL models
        # TODO: Add model-specific dtype mapping if needed for other models
        bytes_per_pixel = 2  # float16 = 2 bytes per pixel
        
        # Image tensor memory (BCHW format)
        # 1 batch × 4 channels (latent) × H × W
        latent_h = height // 8  # VAE downscales by 8
        latent_w = width // 8
        latent_pixels = latent_h * latent_w
        
        # Memory calculations (in MB)
        latent_memory_mb = (latent_pixels * 4 * bytes_per_pixel) / (1024 * 1024)
        
        # Attention memory scales with sequence length
        attention_memory_mb = (latent_pixels * self.ATTENTION_MULTIPLIER * bytes_per_pixel) / (1024 * 1024)
        
        # Activations and gradients
        activation_memory_mb = latent_memory_mb * 2  # Conservative estimate
        
        # Total image-related memory
        image_memory_mb = latent_memory_mb + attention_memory_mb + activation_memory_mb
        
        # Add model overhead
        total_vram_mb = self.MODEL_OVERHEAD_MB + image_memory_mb
        
        # Add safety buffer
        total_with_buffer_mb = total_vram_mb * (1 + self.SAFETY_BUFFER)
        
        # Return total VRAM with safety buffer
        return total_with_buffer_mb * batch_size
    
    def get_available_vram(self) -> Optional[float]:
        """Get available VRAM in MB - from lines 78-92"""
        if not torch.cuda.is_available():
            return None
            
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_mb = free_bytes / (1024 * 1024)
            total_mb = total_bytes / (1024 * 1024)
            
            self.logger.debug(f"VRAM: {free_mb:.0f}MB free / {total_mb:.0f}MB total")
            return free_mb
        except Exception as e:
            self.logger.warning(f"Could not get VRAM info: {e}")
            return None
    
    def determine_strategy(self, width: int, height: int) -> Dict[str, Any]:
        """
        Determine the best expansion strategy based on VRAM.
        Adapted from determine_refinement_strategy (lines 94-end)
        """
        # Calculate required VRAM
        required_mb = self.calculate_generation_vram(width, height)
        
        # Get available VRAM
        available_mb = self.get_available_vram()
        
        if available_mb is None:
            # CPU only - must use CPU offload
            return {
                'strategy': 'cpu_offload',
                'vram_available_mb': 0,
                'vram_required_mb': required_mb,
                'details': {
                    'warning': 'No GPU available - using CPU offload (very slow)'
                }
            }
        
        # Full strategy possible?
        if required_mb <= available_mb:
            return {
                'strategy': 'full',
                'vram_available_mb': available_mb,
                'vram_required_mb': required_mb,
                'details': {
                    'message': 'Full processing possible'
                }
            }
        
        # Calculate tile size for tiled strategy
        # Start with 1024x1024 and reduce until it fits
        tile_size = 1024
        overlap = 256
        
        while tile_size >= 512:
            tile_required_mb = self.calculate_generation_vram(tile_size, tile_size)
            if tile_required_mb <= available_mb * 0.8:
                return {
                    'strategy': 'tiled',
                    'vram_available_mb': available_mb,
                    'vram_required_mb': required_mb,
                    'details': {
                        'tile_size': tile_size,
                        'overlap': overlap,
                        'message': f'Using {tile_size}x{tile_size} tiles with {overlap}px overlap'
                    }
                }
            tile_size -= 256
            overlap = max(128, tile_size // 4)
        
        # Last resort - CPU offload
        return {
            'strategy': 'cpu_offload',
            'vram_available_mb': available_mb,
            'vram_required_mb': required_mb,
            'details': {
                'warning': 'Insufficient VRAM even for tiling - using CPU offload'
            }
        }
    
    def estimate_requirement(self, config) -> Dict[str, float]:
        """Estimate VRAM for ExpandorConfig"""
        target_w, target_h = config.target_resolution
        return self.calculate_generation_vram(target_w, target_h)