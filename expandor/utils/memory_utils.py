"""
Memory management utilities for Expandor
Handles GPU memory optimization, CPU offloading, and memory tracking.
"""

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch

from ..core.configuration_manager import ConfigurationManager

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory statistics snapshot"""

    gpu_used_mb: float
    gpu_total_mb: float
    gpu_free_mb: float
    cpu_used_mb: float
    cpu_total_mb: float
    cpu_free_mb: float
    timestamp: float


class GPUMemoryManager:
    """
    Manages GPU memory for optimal performance.

    Features:
    - Memory tracking and profiling
    - Automatic cache clearing
    - Memory pressure detection
    - CPU offload management
    """

    def __init__(self):
        self.stats_history: List[MemoryStats] = []
        self.offloaded_tensors: Dict[str, Tuple[torch.Tensor, str]] = {}
        self.peak_gpu_usage = 0.0
        self.enable_profiling = False
        self.config_manager = ConfigurationManager()

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.mem_get_info()
            bytes_to_mb = self.config_manager.get_value('constants.memory.bytes_per_mb')
            gpu_free = gpu_stats[0] / bytes_to_mb  # Convert to MB
            gpu_total = gpu_stats[1] / bytes_to_mb
            gpu_used = gpu_total - gpu_free
        else:
            gpu_free = gpu_total = gpu_used = 0.0

        # CPU memory
        cpu_info = psutil.virtual_memory()
        bytes_to_mb = self.config_manager.get_value('constants.memory.bytes_per_mb')
        cpu_total = cpu_info.total / bytes_to_mb
        cpu_free = cpu_info.available / bytes_to_mb
        cpu_used = cpu_info.used / bytes_to_mb

        stats = MemoryStats(
            gpu_used_mb=gpu_used,
            gpu_total_mb=gpu_total,
            gpu_free_mb=gpu_free,
            cpu_used_mb=cpu_used,
            cpu_total_mb=cpu_total,
            cpu_free_mb=cpu_free,
            timestamp=time.time(),
        )

        # Track peak usage
        if gpu_used > self.peak_gpu_usage:
            self.peak_gpu_usage = gpu_used

        # Store history if profiling
        if self.enable_profiling:
            self.stats_history.append(stats)

        return stats

    def clear_cache(self, aggressive: Optional[bool] = None):
        """
        Clear GPU cache to free memory.

        Args:
            aggressive: If True, also runs garbage collection (None = use config default from 'constants.memory.default_aggressive_clear')
        """
        if aggressive is None:
            aggressive = self.config_manager.get_value('constants.memory.default_aggressive_clear')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if aggressive:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def estimate_tensor_memory(
        self, shape: Tuple[int, ...], dtype: Optional[torch.dtype] = None
    ) -> float:
        """
        Estimate memory required for a tensor.

        Args:
            shape: Tensor shape
            dtype: Data type (None = use config default torch.float32)

        Returns:
            Memory in MB
        """
        if dtype is None:
            dtype_str = self.config_manager.get_value('constants.memory.default_dtype')
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
                'int32': torch.int32,
                'int64': torch.int64,
                'uint8': torch.uint8,
                'bool': torch.bool
            }
            if dtype_str not in dtype_map:
                raise ValueError(
                    f"Unknown dtype string: {dtype_str}\n"
                    f"Valid options: {list(dtype_map.keys())}"
                )
            dtype = dtype_map[dtype_str]
        # Calculate number of elements
        num_elements = 1
        for dim in shape:
            num_elements *= dim

        # Bytes per element
        dtype_sizes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int32: 4,
            torch.int64: 8,
            torch.uint8: 1,
            torch.bool: 1,
        }

        # FAIL LOUD if dtype not recognized
        if dtype not in dtype_sizes:
            raise ValueError(
                f"Unknown dtype: {dtype}\n"
                f"Supported dtypes: {list(dtype_sizes.keys())}\n"
                f"Please add the dtype to the dtype_sizes mapping."
            )
        bytes_per_element = dtype_sizes[dtype]
        total_bytes = num_elements * bytes_per_element

        # Convert to MB
        bytes_to_mb = self.config_manager.get_value('constants.memory.bytes_per_mb')
        return total_bytes / bytes_to_mb

    def has_sufficient_memory(
        self, required_mb: float, safety_factor: Optional[float] = None
    ) -> bool:
        """
        Check if sufficient GPU memory is available.

        Args:
            required_mb: Required memory in MB
            safety_factor: Safety multiplier (None = use config default from 'constants.memory.default_safety_factor')

        Returns:
            True if sufficient memory available
        """
        if safety_factor is None:
            safety_factor = self.config_manager.get_value('constants.memory.default_safety_factor')
        stats = self.get_memory_stats()
        required_with_safety = required_mb * safety_factor
        return stats.gpu_free_mb >= required_with_safety

    @contextmanager
    def memory_efficient_scope(
        self, name: Optional[str] = None, clear_on_exit: Optional[bool] = None
    ):
        """
        Context manager for memory-efficient operations.

        Args:
            name: Scope name for logging (None = use config default from 'constants.memory.default_scope_name')
            clear_on_exit: Clear cache on exit (None = use config default from 'constants.memory.default_clear_on_exit')
        """
        if name is None:
            name = self.config_manager.get_value('constants.memory.default_scope_name')
        if clear_on_exit is None:
            clear_on_exit = self.config_manager.get_value('constants.memory.default_clear_on_exit')
        start_stats = self.get_memory_stats()
        logger.debug(
            f"Entering {name} - GPU free: {start_stats.gpu_free_mb:.1f}MB")

        try:
            yield self
        finally:
            if clear_on_exit:
                self.clear_cache()

            end_stats = self.get_memory_stats()
            used = start_stats.gpu_free_mb - end_stats.gpu_free_mb
            logger.debug(f"Exiting {name} - Used: {used:.1f}MB")

    def get_optimal_batch_size(
        self,
        base_memory_mb: float,
        batch_memory_mb: float,
        max_batch_size: Optional[int] = None,
        safety_factor: Optional[float] = None,
    ) -> int:
        """
        Calculate optimal batch size based on available memory.

        Args:
            base_memory_mb: Base memory requirement
            batch_memory_mb: Memory per batch item
            max_batch_size: Maximum allowed batch size (None = use config default from 'constants.memory.default_max_batch_size')
            safety_factor: Safety factor (0-1) (None = use config default from 'constants.memory.default_batch_safety_factor')

        Returns:
            Optimal batch size
        """
        if max_batch_size is None:
            max_batch_size = self.config_manager.get_value('constants.memory.default_max_batch_size')
        if safety_factor is None:
            safety_factor = self.config_manager.get_value('constants.memory.default_batch_safety_factor')
        stats = self.get_memory_stats()
        available = stats.gpu_free_mb * safety_factor

        if available <= base_memory_mb:
            return 1

        batch_budget = available - base_memory_mb
        optimal_size = int(batch_budget / batch_memory_mb)

        return max(1, min(optimal_size, max_batch_size))


# Global memory manager instance
gpu_memory_manager = GPUMemoryManager()


def offload_to_cpu(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """
    Offload tensor to CPU memory.

    Args:
        tensor: Tensor to offload
        name: Identifier for the tensor

    Returns:
        CPU tensor
    """
    if tensor.is_cuda:
        cpu_tensor = tensor.cpu()
        gpu_memory_manager.offloaded_tensors[name] = (
            cpu_tensor, str(tensor.device))
        logger.debug(
            f"Offloaded {name} to CPU ({
                tensor.element_size() *
                tensor.nelement() /
                gpu_memory_manager.config_manager.get_value('constants.memory.bytes_per_mb'):.1f}MB)"
        )
        return cpu_tensor
    return tensor


def load_to_gpu(
    tensor: torch.Tensor, name: str, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Load tensor back to GPU.

    Args:
        tensor: CPU tensor
        name: Identifier
        device: Target device (uses original if not specified)

    Returns:
        GPU tensor
    """
    if not tensor.is_cuda:
        if name in gpu_memory_manager.offloaded_tensors:
            _, original_device = gpu_memory_manager.offloaded_tensors[name]
            target_device = device or torch.device(original_device)
        else:
            target_device = device or torch.device("cuda")

        gpu_tensor = tensor.to(target_device)
        logger.debug(
            f"Loaded {name} to GPU ({
                tensor.element_size() *
                tensor.nelement() /
                gpu_memory_manager.config_manager.get_value('constants.memory.bytes_per_mb'):.1f}MB)"
        )

        # Remove from offloaded list
        if name in gpu_memory_manager.offloaded_tensors:
            del gpu_memory_manager.offloaded_tensors[name]

        return gpu_tensor
    return tensor


def estimate_model_memory(model: Any, include_gradients: Optional[bool] = None) -> float:
    """
    Estimate memory required for a model.

    Args:
        model: PyTorch model
        include_gradients: Include gradient storage (None = use config default from 'constants.memory.default_include_gradients')

    Returns:
        Estimated memory in MB
    """
    # Get config manager
    config_manager = ConfigurationManager()
    if include_gradients is None:
        include_gradients = config_manager.get_value('constants.memory.default_include_gradients')
    total_params = 0
    total_bytes = 0

    for param in model.parameters():
        total_params += param.nelement()
        bytes_per_element = param.element_size()
        param_bytes = param.nelement() * bytes_per_element

        # Add gradient storage if needed
        if include_gradients and param.requires_grad:
            # Load gradient multiplier from config
            try:
                from pathlib import Path

                from .config_loader import ConfigLoader
                config_dir = Path(__file__).parent.parent / "config"
                loader = ConfigLoader(config_dir)
                proc_config = loader.load_config_file("processing_params.yaml")
                gradient_multiplier = proc_config.get(
                    'memory_params', {}).get(
                    'gradient_memory_multiplier', 2)
            except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
                raise ValueError(
                    f"Failed to load memory parameters configuration: {e}")
            param_bytes *= gradient_multiplier

        total_bytes += param_bytes

    # Add buffer for activations (rough estimate)
    # Load activation multipliers from config
    try:
        from pathlib import Path

        from .config_loader import ConfigLoader
        config_dir = Path(__file__).parent.parent / "config"
        loader = ConfigLoader(config_dir)
        proc_config = loader.load_config_file("processing_params.yaml")
        mem_params = proc_config.get('memory_params', {})
        activation_multiplier = (
            mem_params.get(
                'activation_multiplier_with_grad' if include_gradients else 'activation_multiplier_no_grad',
                None))
        if activation_multiplier is None:
            key = 'activation_multiplier_with_grad' if include_gradients else 'activation_multiplier_no_grad'
            raise ValueError(
                f"{key} not found in memory_params configuration!\n"
                "This value must be explicitly set in processing_params.yaml"
            )
    except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
        raise ValueError(
            f"Failed to load memory parameters configuration: {e}")
    total_bytes *= activation_multiplier

    bytes_to_mb = config_manager.get_value('constants.memory.bytes_per_mb')
    total_mb = total_bytes / bytes_to_mb
    logger.debug(
        f"Model memory estimate: {
            total_mb:.1f}MB ({
            total_params:,        } parameters)"
    )

    return total_mb


class MemoryEfficientTiling:
    """Helper for memory-efficient tiled processing."""

    @staticmethod
    def calculate_optimal_tile_size(
        image_size: Tuple[int, int],
        available_memory_mb: float,
        bytes_per_pixel: Optional[float] = None,
        overlap: Optional[int] = None,
        min_tile: Optional[int] = None,
        max_tile: Optional[int] = None,
    ) -> int:
        """
        Calculate optimal tile size for available memory.

        Args:
            image_size: (width, height) of image
            available_memory_mb: Available GPU memory
            bytes_per_pixel: Memory per pixel (None = use config default from 'constants.memory.default_bytes_per_pixel')
            overlap: Tile overlap in pixels (None = use config default from 'constants.memory.default_tile_overlap')
            min_tile: Minimum tile size (None = use config default from 'constants.memory.default_min_tile_size')
            max_tile: Maximum tile size (None = use config default from 'constants.memory.default_max_tile_size')

        Returns:
            Optimal tile size
        """
        # Get config manager
        config_manager = ConfigurationManager()
        if bytes_per_pixel is None:
            bytes_per_pixel = config_manager.get_value('constants.memory.default_bytes_per_pixel')
        if overlap is None:
            overlap = config_manager.get_value('constants.memory.default_tile_overlap')
        if min_tile is None:
            min_tile = config_manager.get_value('constants.memory.default_min_tile_size')
        if max_tile is None:
            max_tile = config_manager.get_value('constants.memory.default_max_tile_size')
        # Binary search for optimal tile size
        left, right = min_tile, max_tile
        optimal = min_tile

        while left <= right:
            mid = (left + right) // 2

            # Calculate memory for this tile size
            tile_pixels = (mid + overlap * 2) ** 2
            bytes_to_mb = config_manager.get_value('constants.memory.bytes_per_mb')
            tile_memory_mb = (tile_pixels * bytes_per_pixel) / bytes_to_mb

            # Add overhead for processing (2x for input/output)
            overhead_multiplier = config_manager.get_value('constants.memory.processing_overhead_multiplier')
            total_memory_mb = tile_memory_mb * overhead_multiplier

            if total_memory_mb <= available_memory_mb:
                optimal = mid
                left = mid + 1
            else:
                right = mid - 1

        # Ensure tile size is multiple of alignment (for model compatibility)
        alignment = config_manager.get_value('constants.dimensions.alignment_multiple')
        optimal = (optimal // alignment) * alignment

        return max(min_tile, min(optimal, max_tile))


def profile_memory_usage(func):
    """
    Decorator to profile memory usage of a function.

    Usage:
        @profile_memory_usage
        def my_function():
            ...
    """

    def wrapper(*args, **kwargs):
        # Start profiling
        gpu_memory_manager.enable_profiling = True
        start_stats = gpu_memory_manager.get_memory_stats()

        try:
            # Run function
            result = func(*args, **kwargs)

            # End profiling
            end_stats = gpu_memory_manager.get_memory_stats()

            # Report
            gpu_used = end_stats.gpu_used_mb - start_stats.gpu_used_mb
            cpu_used = end_stats.cpu_used_mb - start_stats.cpu_used_mb

            logger.info(
                f"{func.__name__} memory usage: "
                f"GPU: {gpu_used:.1f}MB, CPU: {cpu_used:.1f}MB, "
                f"Peak GPU: {gpu_memory_manager.peak_gpu_usage:.1f}MB"
            )

            return result

        finally:
            gpu_memory_manager.enable_profiling = False

    return wrapper
