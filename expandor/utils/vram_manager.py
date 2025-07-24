"""
VRAM Manager for Expandor
Simple wrapper around GPUMemoryManager for Phase 4 compatibility
"""

import logging
from typing import Any, Dict, Optional

import torch

from .memory_utils import GPUMemoryManager, MemoryStats

logger = logging.getLogger(__name__)


class VRAMManager:
    """
    Manages VRAM usage and optimization.
    Wrapper around GPUMemoryManager for Phase 4 API compatibility.
    """

    def __init__(self):
        self._gpu_manager = GPUMemoryManager()
        self.logger = logger

    def get_available_vram(self) -> float:
        """Get available VRAM in MB"""
        if not torch.cuda.is_available():
            return float("inf")  # No VRAM limit for CPU

        stats = self._gpu_manager.get_memory_stats()
        return stats.gpu_free_mb

    def get_total_vram(self) -> float:
        """Get total VRAM in MB"""
        if not torch.cuda.is_available():
            return 0.0

        stats = self._gpu_manager.get_memory_stats()
        return stats.gpu_total_mb

    def get_used_vram(self) -> float:
        """Get used VRAM in MB"""
        if not torch.cuda.is_available():
            return 0.0

        stats = self._gpu_manager.get_memory_stats()
        return stats.gpu_used_mb

    def estimate_operation_vram(
        self, operation: str, width: int, height: int, model_type: str = "sdxl"
    ) -> float:
        """
        Estimate VRAM for operation.

        Args:
            operation: Type of operation (generate, inpaint, img2img)
            width: Image width
            height: Image height
            model_type: Model type (sdxl, sd15, flux)

        Returns:
            Estimated VRAM usage in MB
        """
        # Base model VRAM requirements
        base_vram = {
            "sd15": 4000,
            "sd2": 6000,
            "sdxl": 10000,
            "sd3": 16000,
            "flux": 24000,
        }.get(model_type, 8000)

        # Calculate pixel-based overhead
        pixels = width * height
        pixel_overhead = (pixels / (1024 * 1024)) * 4  # ~4MB per megapixel

        # Operation multipliers
        operation_multipliers = {
            "generate": 1.0,
            "img2img": 1.2,
            "inpaint": 1.5,
            "upscale": 0.8,
        }
        multiplier = operation_multipliers.get(operation, 1.0)

        # Calculate total
        total_vram = (base_vram + pixel_overhead) * multiplier

        # Add safety margin
        return total_vram * 1.2

    def get_memory_efficient_settings(self, target_vram: float) -> Dict[str, Any]:
        """Get settings for target VRAM limit"""
        settings = {}

        if target_vram < 4000:  # Less than 4GB
            settings.update(
                {
                    "enable_attention_slicing": True,
                    "enable_vae_slicing": True,
                    "enable_cpu_offload": True,
                    "sequential_cpu_offload": True,
                    "batch_size": 1,
                    "tile_size": 512,
                }
            )
        elif target_vram < 8000:  # Less than 8GB
            settings.update(
                {
                    "enable_attention_slicing": True,
                    "enable_vae_slicing": True,
                    "enable_cpu_offload": False,
                    "batch_size": 1,
                    "tile_size": 768,
                }
            )
        else:  # 8GB or more
            settings.update(
                {
                    "enable_attention_slicing": "auto",
                    "enable_vae_slicing": False,
                    "enable_cpu_offload": False,
                    "batch_size": 2,
                    "tile_size": 1024,
                }
            )

        return settings

    def clear_cache(self):
        """Clear CUDA cache"""
        self._gpu_manager.clear_gpu_cache()

    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics"""
        stats = self._gpu_manager.get_memory_stats()
        return {
            "vram_used": stats.gpu_used_mb,
            "vram_total": stats.gpu_total_mb,
            "vram_free": stats.gpu_free_mb,
            "vram_percentage": (
                (stats.gpu_used_mb / stats.gpu_total_mb * 100)
                if stats.gpu_total_mb > 0
                else 0
            ),
            "ram_used": stats.cpu_used_mb,
            "ram_total": stats.cpu_total_mb,
            "ram_free": stats.cpu_free_mb,
        }

    def check_vram_availability(self, required_mb: float) -> bool:
        """Check if enough VRAM is available"""
        available = self.get_available_vram()
        return available >= required_mb * 1.1  # 10% safety margin
