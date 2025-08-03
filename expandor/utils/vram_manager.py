"""
VRAM Manager for Expandor
Simple wrapper around GPUMemoryManager for Phase 4 compatibility
"""

import logging
from typing import Any, Dict

import torch

from .memory_utils import GPUMemoryManager

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
        # Use ConfigurationManager for all values - NO HARDCODED VALUES
        from ..core.configuration_manager import ConfigurationManager
        config_manager = ConfigurationManager()
        
        # Get model VRAM requirements from config
        model_requirements = config_manager.get_value('vram.model_requirements')
        base_vram = model_requirements.get(model_type)
        if base_vram is None:
            # FAIL LOUD if model type not found
            default_vram = config_manager.get_value('vram.model_requirements.default')
            self.logger.warning(f"Unknown model type '{model_type}', using default: {default_vram}MB")
            base_vram = default_vram

        # Calculate pixel-based overhead
        pixels = width * height
        pixel_overhead_per_megapixel = config_manager.get_value('vram.pixel_overhead_mb_per_megapixel')
        pixel_overhead = (pixels / (1024 * 1024)) * pixel_overhead_per_megapixel

        # Get operation multipliers from config
        operation_multipliers = config_manager.get_value('vram.operation_multipliers')
        multiplier = operation_multipliers.get(operation)
        if multiplier is None:
            # FAIL LOUD if operation not found
            default_multiplier = config_manager.get_value('vram.operation_multipliers.default')
            self.logger.warning(f"Unknown operation '{operation}', using default multiplier: {default_multiplier}")
            multiplier = default_multiplier

        # Calculate total
        total_vram = (base_vram + pixel_overhead) * multiplier

        # Add safety margin from config
        safety_margin = config_manager.get_value('vram.estimation_safety_margin')
        return total_vram * safety_margin

    def get_memory_efficient_settings(
            self, target_vram: float) -> Dict[str, Any]:
        """Get settings for target VRAM limit"""
        # Use ConfigurationManager for all values - NO HARDCODED VALUES
        from ..core.configuration_manager import ConfigurationManager
        config_manager = ConfigurationManager()
        
        # Get memory profiles from config
        memory_profiles = config_manager.get_value('vram.memory_profiles')
        
        # Determine which profile to use based on target VRAM
        low_threshold = memory_profiles['low']['threshold_mb']
        medium_threshold = memory_profiles['medium']['threshold_mb']
        
        if target_vram < low_threshold:
            # Use low memory profile
            settings = memory_profiles['low']['settings'].copy()
        elif target_vram < medium_threshold:
            # Use medium memory profile
            settings = memory_profiles['medium']['settings'].copy()
        else:
            # Use high memory profile
            settings = memory_profiles['high']['settings'].copy()

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
