"""
VRAM Manager for Dynamic Resource Management
Adapted from ai-wallpaper project (https://github.com/user/ai-wallpaper)
Original: ai_wallpaper/core/vram_calculator.py
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch

from .exceptions import VRAMError


class VRAMManager:
    """Calculate VRAM requirements and determine expansion strategy"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize ConfigurationManager
        from .configuration_manager import ConfigurationManager
        self.config_manager = ConfigurationManager()

        # Load VRAM configuration
        try:
            # Model overhead from config based on model type
            # Will be set dynamically in calculate_generation_vram based on model_type
            self.model_overheads = self.config_manager.get_value('vram.model_requirements')
            
            # Attention multiplier is an architectural constant (Q, K, V, attention scores)
            # This represents the ~4x memory needed for attention mechanisms
            self.ATTENTION_MULTIPLIER = self.config_manager.get_value('vram.attention_multiplier')
            
            # Safety factor from config (note: config has safety_factor, not buffer)
            # If safety_factor is 0.9, then buffer is 1 - 0.9 = 0.1 (10%)
            safety_factor = self.config_manager.get_value('vram.safety_factor')
            self.SAFETY_BUFFER = 1.0 - safety_factor  # Convert factor to buffer
            
        except ValueError as e:
            # FAIL LOUD on missing configuration
            raise ValueError(
                f"Failed to load VRAM configuration!\n{str(e)}\n"
                f"Please ensure vram configuration section exists in master_defaults.yaml"
            )

        # Track peak usage
        self.peak_usage_mb = 0.0

        # Reservation tracking
        self.reservations = {}  # component_name -> MB reserved
        self.total_reserved_mb = 0.0

    def calculate_generation_vram(
        self,
        width: int,
        height: int,
        batch_size: int = 1,
        model_type: str = "sdxl",
        dtype: str = "float16",
    ) -> float:
        """
        Calculate ACCURATE VRAM requirements for refinement.
        Copy implementation from lines 22-76 of vram_calculator.py
        """
        # pixels = width * height  # Unused calculation

        # Dtype mapping for different precision modes
        dtype_map = {"float16": 2, "float32": 4, "bfloat16": 2}
        # FAIL LOUD if dtype not recognized
        if dtype not in dtype_map:
            raise ValueError(
                f"Unknown dtype for VRAM calculation: {dtype}\n"
                f"Supported dtypes: {list(dtype_map.keys())}\n"
                f"Please specify a valid dtype for the model."
            )
        bytes_per_pixel = dtype_map[dtype]

        # Image tensor memory (BCHW format)
        # 1 batch × 4 channels (latent) × H × W
        latent_h = height // 8  # VAE downscales by 8
        latent_w = width // 8
        latent_pixels = latent_h * latent_w

        # Memory calculations (in MB)
        latent_memory_mb = (latent_pixels * 4 *
                            bytes_per_pixel) / (1024 * 1024)

        # Attention memory scales with sequence length
        attention_memory_mb = (
            latent_pixels * self.ATTENTION_MULTIPLIER * bytes_per_pixel
        ) / (1024 * 1024)

        # Activations and gradients
        activation_memory_mb = latent_memory_mb * 2  # Conservative estimate

        # Total image-related memory
        image_memory_mb = latent_memory_mb + attention_memory_mb + activation_memory_mb

        # Get model overhead from config based on model type
        # Use default if specific model not found
        if model_type not in self.model_overheads:
            # Use configured default, not hardcoded value
            if 'default' not in self.model_overheads:
                raise ValueError(
                    f"Model type '{model_type}' not found in VRAM requirements and no default configured!\n"
                    f"Available models: {list(self.model_overheads.keys())}\n"
                    f"Please add '{model_type}' or 'default' to vram.model_requirements in config."
                )
            model_overhead_mb = self.model_overheads['default']
        else:
            model_overhead_mb = self.model_overheads[model_type]
        total_vram_mb = model_overhead_mb + image_memory_mb

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

            self.logger.debug(
                f"VRAM: {free_mb:.0f}MB free / {total_mb:.0f}MB total")
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
                "strategy": "cpu_offload",
                "vram_available_mb": 0,
                "vram_required_mb": required_mb,
                "details": {
                    "warning": "No GPU available - using CPU offload (very slow)"},
            }

        # Full strategy possible?
        if required_mb <= available_mb:
            return {
                "strategy": "full",
                "vram_available_mb": available_mb,
                "vram_required_mb": required_mb,
                "details": {"message": "Full processing possible"},
            }

        # Calculate tile size for tiled strategy
        # Get initial tile size and overlap from config
        try:
            tiled_config = self.config_manager.get_value('strategies.tiled_expansion')
            tile_size = tiled_config['default_tile_size']
            overlap = tiled_config['overlap']
        except (KeyError, ValueError):
            # FAIL LOUD if config missing
            raise ValueError(
                "Tiled expansion configuration not found!\n"
                "Required: strategies.tiled_expansion.default_tile_size and overlap"
            )

        min_tile_size = tiled_config['min_tile_size']
        vram_safety_ratio = tiled_config['vram_safety_ratio']
        tile_size_step = self.config_manager.get_value('vram.tile_size_step')
        min_overlap = self.config_manager.get_value('vram.min_overlap')
        
        while tile_size >= min_tile_size:
            tile_required_mb = self.calculate_generation_vram(
                tile_size, tile_size)
            if tile_required_mb <= available_mb * vram_safety_ratio:
                return {
                    "strategy": "tiled",
                    "vram_available_mb": available_mb,
                    "vram_required_mb": required_mb,
                    "details": {
                        "tile_size": tile_size,
                        "overlap": overlap,
                        "message": f"Using {tile_size}x{tile_size} tiles with {overlap}px overlap",
                    },
                }
            tile_size -= tile_size_step
            overlap = max(min_overlap, tile_size // 4)

        # Last resort - CPU offload
        return {
            "strategy": "cpu_offload",
            "vram_available_mb": available_mb,
            "vram_required_mb": required_mb,
            "details": {
                "warning": "Insufficient VRAM even for tiling - using CPU offload"},
        }

    def estimate_requirement(self, config) -> Dict[str, float]:
        """Estimate VRAM for ExpandorConfig"""
        target_w, target_h = config.target_resolution
        vram_mb = self.calculate_generation_vram(target_w, target_h)
        return {
            "required_mb": vram_mb,
            "peak_mb": vram_mb * (1 + self.SAFETY_BUFFER),  # Use configured safety buffer
        }

    def track_peak_usage(self, current_mb: float) -> None:
        """Track peak VRAM usage during operations"""
        self.peak_usage_mb = max(self.peak_usage_mb, current_mb)

    def clear_cache(self) -> None:
        """Clear CUDA cache to free up VRAM"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_peak_usage(self) -> float:
        """Get peak VRAM usage recorded"""
        return self.peak_usage_mb

    def estimate_pipeline_memory(
        self, pipeline_type: str = "sdxl", include_vae: bool = True
    ) -> float:
        """
        Estimate memory required for a pipeline.

        Args:
            pipeline_type: Type of pipeline ('sdxl', 'sd15', 'flux')
            include_vae: Include VAE memory requirements

        Returns:
            Estimated memory in MB
        """
        # Get base memory from configured model requirements
        pipeline_key = pipeline_type.lower()
        if pipeline_key not in self.model_overheads:
            # Use configured default, not hardcoded value
            if 'default' not in self.model_overheads:
                raise ValueError(
                    f"Pipeline type '{pipeline_type}' not found in VRAM requirements and no default configured!\n"
                    f"Available types: {list(self.model_overheads.keys())}\n"
                    f"Please add '{pipeline_key}' or 'default' to vram.model_requirements in config."
                )
            base_memory = self.model_overheads['default']
        else:
            base_memory = self.model_overheads[pipeline_key]

        # VAE memory is typically included in the model requirements
        # but if we need to add it separately, it would be ~10% of model size
        # This is kept simple since most model requirements include VAE
        if include_vae:
            # VAE is typically already included in model requirements
            # No additional memory needed
            pass

        return base_memory

    def get_safe_tile_size(
        self,
        available_mb: Optional[float] = None,
        model_type: str = "sdxl",
        safety_factor: float = 0.8,
    ) -> int:
        """
        Calculate safe tile size based on available VRAM.

        Args:
            available_mb: Available VRAM (auto-detect if None)
            model_type: Model type for constraints
            safety_factor: Safety factor (0-1)

        Returns:
            Safe tile size in pixels
        """
        if available_mb is None:
            available_mb = self.get_available_vram()
            if available_mb is None:
                # No GPU, return minimum tile size from config
                return self.config_manager.get_value('vram.min_dimension_size')

        # Apply safety factor
        safe_vram = available_mb * safety_factor

        # Model-specific memory requirements (MB per megapixel)
        mb_per_mp = self.config_manager.get_value("vram.model_memory_per_megapixel")
        
        # Get memory requirement for model type - FAIL LOUD if not found
        model_key = model_type.lower()
        if model_key not in mb_per_mp:
            # Use 'unknown' if available, otherwise fail
            if 'unknown' not in mb_per_mp:
                raise ValueError(
                    f"Model type '{model_type}' not found in VRAM memory per megapixel config!\n"
                    f"Available models: {list(mb_per_mp.keys())}\n"
                    f"Please add '{model_key}' or 'unknown' to vram.model_memory_per_megapixel in config."
                )
            mb_per_megapixel = mb_per_mp['unknown']
        else:
            mb_per_megapixel = mb_per_mp[model_key]
        
        mb_per_pixel = mb_per_megapixel / 1_000_000

        # Calculate maximum pixels
        max_pixels = int(safe_vram / mb_per_pixel)

        # Convert to square tile size
        tile_size = int(max_pixels**0.5)

        # Apply constraints - load from config
        # Use ConfigurationManager for all config access
        from ..core.configuration_manager import ConfigurationManager
        config_manager = ConfigurationManager()
        
        # Get dimension constraints from configuration
        min_size = config_manager.get_value('vram.min_dimension_size')
        max_size = config_manager.get_value('vram.max_dimension_size')

        # Round to multiple of 64 for better compatibility
        tile_size = max(min_size, min(tile_size, max_size))
        tile_size = (tile_size // 64) * 64

        self.logger.debug(
            f"Calculated safe tile size: {tile_size} for {
                safe_vram:.0f}MB VRAM"
        )

        return tile_size

    def reserve(self, component: str, mb_required: float) -> bool:
        """
        Reserve VRAM for a component

        Args:
            component: Name of component requesting VRAM
            mb_required: Megabytes required

        Returns:
            True if reservation successful, False otherwise

        Raises:
            VRAMError: If insufficient VRAM available
        """
        available = self.get_available_vram()
        if available is None:
            available = 0  # CPU mode

        already_reserved = self.total_reserved_mb
        free_mb = available - already_reserved

        if mb_required > free_mb:
            raise VRAMError(
                f"Insufficient VRAM for {component}: "
                f"need {mb_required}MB, have {free_mb}MB free "
                f"({available}MB total - {already_reserved}MB reserved)"
            )

        # Make reservation
        self.reservations[component] = mb_required
        self.total_reserved_mb += mb_required
        self.logger.info(
            f"Reserved {mb_required}MB for {component} "
            f"({self.total_reserved_mb}MB total reserved)"
        )
        return True

    def release(self, component: str):
        """Release VRAM reservation for component"""
        if component in self.reservations:
            mb_released = self.reservations[component]
            del self.reservations[component]
            self.total_reserved_mb -= mb_released
            self.logger.info(
                f"Released {mb_released}MB from {component} "
                f"({self.total_reserved_mb}MB still reserved)"
            )

    def get_free_vram(self) -> float:
        """Get free VRAM after reservations"""
        available = self.get_available_vram()
        if available is None:
            return 0.0
        return max(0, available - self.total_reserved_mb)

    @contextmanager
    def vram_allocation(self, component: str, mb_required: float):
        """
        Context manager for VRAM allocation

        Usage:
            with vram_manager.vram_allocation("pipeline", 4000):
                # Use pipeline with guaranteed VRAM
                pass
            # VRAM automatically released
        """
        try:
            self.reserve(component, mb_required)
            yield
        finally:
            self.release(component)
