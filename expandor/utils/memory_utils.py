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

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.mem_get_info()
            gpu_free = gpu_stats[0] / (1024**2)  # Convert to MB
            gpu_total = gpu_stats[1] / (1024**2)
            gpu_used = gpu_total - gpu_free
        else:
            gpu_free = gpu_total = gpu_used = 0.0

        # CPU memory
        cpu_info = psutil.virtual_memory()
        cpu_total = cpu_info.total / (1024**2)
        cpu_free = cpu_info.available / (1024**2)
        cpu_used = cpu_info.used / (1024**2)

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

    def clear_cache(self, aggressive: bool = False):
        """
        Clear GPU cache to free memory.

        Args:
            aggressive: If True, also runs garbage collection
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if aggressive:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def estimate_tensor_memory(
        self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32
    ) -> float:
        """
        Estimate memory required for a tensor.

        Args:
            shape: Tensor shape
            dtype: Data type

        Returns:
            Memory in MB
        """
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
        return total_bytes / (1024**2)

    def has_sufficient_memory(
        self, required_mb: float, safety_factor: float = 1.2
    ) -> bool:
        """
        Check if sufficient GPU memory is available.

        Args:
            required_mb: Required memory in MB
            safety_factor: Safety multiplier

        Returns:
            True if sufficient memory available
        """
        stats = self.get_memory_stats()
        required_with_safety = required_mb * safety_factor
        return stats.gpu_free_mb >= required_with_safety

    @contextmanager
    def memory_efficient_scope(
        self, name: str = "operation", clear_on_exit: bool = True
    ):
        """
        Context manager for memory-efficient operations.

        Args:
            name: Scope name for logging
            clear_on_exit: Clear cache on exit
        """
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
        max_batch_size: int = 16,
        safety_factor: float = 0.8,
    ) -> int:
        """
        Calculate optimal batch size based on available memory.

        Args:
            base_memory_mb: Base memory requirement
            batch_memory_mb: Memory per batch item
            max_batch_size: Maximum allowed batch size
            safety_factor: Safety factor (0-1)

        Returns:
            Optimal batch size
        """
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
                1024**2:.1f}MB)"
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
                1024**2:.1f}MB)"
        )

        # Remove from offloaded list
        if name in gpu_memory_manager.offloaded_tensors:
            del gpu_memory_manager.offloaded_tensors[name]

        return gpu_tensor
    return tensor


def estimate_model_memory(model: Any, include_gradients: bool = True) -> float:
    """
    Estimate memory required for a model.

    Args:
        model: PyTorch model
        include_gradients: Include gradient storage

    Returns:
        Estimated memory in MB
    """
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
                'activation_multiplier_with_grad',
                4) if include_gradients else mem_params.get(
                'activation_multiplier_no_grad',
                2))
    except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
        raise ValueError(
            f"Failed to load memory parameters configuration: {e}")
    total_bytes *= activation_multiplier

    total_mb = total_bytes / (1024**2)
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
        bytes_per_pixel: float = 12,
        overlap: int = 64,
        min_tile: int = 256,
        max_tile: int = 2048,
    ) -> int:
        """
        Calculate optimal tile size for available memory.

        Args:
            image_size: (width, height) of image
            available_memory_mb: Available GPU memory
            bytes_per_pixel: Memory per pixel (channels * dtype_size * overhead)
            overlap: Tile overlap in pixels
            min_tile: Minimum tile size
            max_tile: Maximum tile size

        Returns:
            Optimal tile size
        """
        # Binary search for optimal tile size
        left, right = min_tile, max_tile
        optimal = min_tile

        while left <= right:
            mid = (left + right) // 2

            # Calculate memory for this tile size
            tile_pixels = (mid + overlap * 2) ** 2
            tile_memory_mb = (tile_pixels * bytes_per_pixel) / (1024**2)

            # Add overhead for processing (2x for input/output)
            total_memory_mb = tile_memory_mb * 2

            if total_memory_mb <= available_memory_mb:
                optimal = mid
                left = mid + 1
            else:
                right = mid - 1

        # Ensure tile size is multiple of 8 (for model compatibility)
        optimal = (optimal // 8) * 8

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
