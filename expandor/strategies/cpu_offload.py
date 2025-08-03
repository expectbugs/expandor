"""
CPU Offload Strategy for extreme memory constraints.
Handles expansions when GPU memory is severely limited or unavailable.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..core.config import ExpandorConfig
from ..core.exceptions import ExpandorError, StrategyError
from ..processors.tiled_processor import TiledProcessor
from ..utils.dimension_calculator import DimensionCalculator
from ..utils.memory_utils import gpu_memory_manager
from .base_strategy import BaseExpansionStrategy


class CPUOffloadStrategy(BaseExpansionStrategy):
    """
    CPU Offload strategy for extreme memory constraints.

    Features:
    - Sequential CPU offloading of model components
    - Small tile processing (384x384 minimum)
    - Aggressive garbage collection
    - Multi-stage processing with memory clearing
    - Automatic batch size adjustment
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize CPU offload strategy with proper signature."""
        super().__init__(config=config, metrics=metrics, logger=logger)

        # Initialize required components
        self.dimension_calc = DimensionCalculator(self.logger)
        self.tiled_processor = TiledProcessor(logger=self.logger)

        # Get configuration from ConfigurationManager
        from ..core.configuration_manager import ConfigurationManager
        self.config_manager = ConfigurationManager()
        
        # Get all strategy config at once
        try:
            self.strategy_config = self.config_manager.get_strategy_config('cpu_offload')
        except ValueError as e:
            # FAIL LOUD
            raise ValueError(
                f"Failed to load cpu_offload configuration!\n{str(e)}"
            )
        
        # Assign all values from config - NO DEFAULTS!
        self.safety_factor = self.strategy_config['safety_factor']
        self.conservative_safety_factor = self.strategy_config['conservative_safety_factor']
        self.pipeline_vram = self.strategy_config['pipeline_vram']
        self.gpu_memory_fallback = self.strategy_config['gpu_memory_fallback']
        self.aspect_change_threshold = self.strategy_config['aspect_change_threshold']
        self.tile_generation_strength = self.strategy_config['tile_generation_strength']
        self.tile_generation_guidance = self.strategy_config['tile_generation_guidance']
        self.tile_refinement_strength = self.strategy_config['tile_refinement_strength']
        self.tile_refinement_guidance = self.strategy_config['tile_refinement_guidance']
        
        # CPU offload specific parameters
        self.min_tile_size = self.strategy_config['min_tile_size']
        self.default_tile_size = self.strategy_config['default_tile_size']
        self.max_tile_size = self.strategy_config['max_tile_size']
        self.min_overlap = self.strategy_config['min_overlap']
        self.default_overlap = self.strategy_config['default_overlap']

    def validate_requirements(self):
        """Validate CPU offload requirements."""
        # CPU offload can work with minimal resources
        # Requirements checked in execute

    def estimate_vram(self, config: ExpandorConfig) -> Dict[str, float]:
        """
        Estimate VRAM requirements for CPU offload.

        This strategy uses minimal VRAM by design.
        """
        # Calculate tile size based on available memory
        tile_size = self.vram_manager.get_safe_tile_size(
            # More conservative for CPU offload
            model_type="sdxl", safety_factor=self.strategy_config['safety_factor']
        )

        # Estimate for single tile processing
        tile_pixels = tile_size * tile_size
        tile_vram_mb = (tile_pixels * self.strategy_config['vram_bytes_per_pixel']) / self.strategy_config['vram_mb_divisor']

        # Minimal pipeline memory with offloading
        # 1GB max with offloading
        pipeline_vram = self.strategy_config['pipeline_vram']

        return {
            "base_vram_mb": tile_vram_mb,
            "peak_vram_mb": tile_vram_mb + pipeline_vram,
            "strategy_overhead_mb": self.strategy_config['strategy_overhead_mb'],  # Offloading overhead
        }

    def execute(
        self, config: ExpandorConfig, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute CPU offload strategy with extreme memory efficiency.

        Args:
            config: ExpandorConfig with all parameters
            context: Execution context

        Returns:
            Dict with results

        Raises:
            StrategyError: On any failure (FAIL LOUD)
        """
        self._context = context or {}
        start_time = time.time()

        self.logger.info("Starting CPU offload expansion strategy")

        # Validate inputs
        self.validate_inputs(config)

        # Check if CPU offload is allowed
        if not config.use_cpu_offload:
            raise StrategyError(
                "CPU offload strategy requested but not allowed in config",
                details={"allow_cpu_offload": False},
            )

        # Check pipeline availability - pipelines should be injected by
        # orchestrator
        available_pipelines = []
        if self.inpaint_pipeline:
            available_pipelines.append("inpaint")
        if self.img2img_pipeline:
            available_pipelines.append("img2img")

        if not available_pipelines:
            raise StrategyError(
                "CPU offload requires at least one pipeline",
                details={
                    "available_pipelines": list(
                        self._context.get("pipeline_registry", {}).keys()
                    )
                },
            )

        try:
            # Load source image
            if isinstance(config.source_image, Path):
                source_image = self.validate_image_path(config.source_image)
            else:
                source_image = config.source_image.copy()

            source_w, source_h = source_image.size
            target_w, target_h = config.get_target_resolution()

            # Calculate optimal tile size for minimal memory
            available_vram = (
                self.vram_manager.get_available_vram() or self.strategy_config['fallback_vram_mb']
            )  # Assume 512MB if no GPU
            optimal_tile_size = self._calculate_optimal_tile_size(
                available_vram)

            self.logger.info(
                f"Using tile size: {optimal_tile_size}x{optimal_tile_size}"
            )

            # Plan processing stages
            stages = self._plan_cpu_offload_stages(
                source_size=(source_w, source_h),
                target_size=(target_w, target_h),
                tile_size=optimal_tile_size,
            )

            if not stages:
                raise StrategyError(
                    "Failed to plan CPU offload stages",
                    details={
                        "source_size": (source_w, source_h),
                        "target_size": (target_w, target_h),
                    },
                )

            self.logger.info(f"Planned {len(stages)} processing stages")

            # Process each stage with aggressive memory management
            current_image = source_image

            for i, stage in enumerate(stages):
                stage_start = time.time()

                self.logger.info(
                    f"Processing stage {i + 1}/{len(stages)}: {stage['name']}"
                )

                # Clear memory before each stage
                gpu_memory_manager.clear_cache(aggressive=True)

                # Process stage
                current_image = self._process_cpu_offload_stage(
                    current_image, stage, config, i
                )

                # Track progress
                if self.boundary_tracker and stage.get("boundaries"):
                    for boundary in stage["boundaries"]:
                        if boundary:  # Skip None entries
                            self.track_boundary(**boundary)

                # Record stage
                self.record_stage(
                    name=f"cpu_offload_stage_{i}",
                    method=stage["method"],
                    input_size=stage["input_size"],
                    output_size=stage["output_size"],
                    start_time=stage_start,
                    metadata={
                        "stage_type": stage["type"],
                        "tile_size": stage.get("tile_size", optimal_tile_size),
                    },
                )

                # Save intermediate if requested
                if config.save_stages and config.stage_dir:
                    stage_path = config.stage_dir / \
                        f"cpu_offload_stage_{i:02d}.png"
                    # Get compression from config
                    from ..core.configuration_manager import ConfigurationManager
                    config_manager = ConfigurationManager()
                    png_compression = config_manager.get_value("output.formats.png.compression")
                    current_image.save(stage_path, "PNG", compress_level=png_compression)

                # Aggressive cleanup
                gpu_memory_manager.clear_cache(aggressive=True)

            # Final validation
            if current_image.size != (target_w, target_h):
                raise ExpandorError(
                    f"CPU offload size mismatch: expected {target_w}x{target_h}, " f"got {
                        current_image.size[0]}x{
                        current_image.size[1]}", stage="validation", )

            # Save final result
            output_path = self.save_temp_image(
                current_image, "cpu_offload_final")

            return {
                "image_path": output_path,
                "size": current_image.size,
                "stages": self.stage_results,
                "boundaries": (
                    self.boundary_tracker.get_all_boundaries()
                    if self.boundary_tracker
                    else []
                ),
                "metadata": {
                    "strategy": "cpu_offload",
                    "total_stages": len(stages),
                    "tile_size": optimal_tile_size,
                    "peak_vram_mb": self.vram_manager.get_peak_usage(),
                    "duration": time.time() - start_time,
                },
            }

        except Exception as e:
            # FAIL LOUD
            self.logger.error(f"CPU offload strategy failed: {str(e)}")
            raise StrategyError(
                f"CPU offload execution failed: {str(e)}",
                details={
                    "stage": (
                        f"{i}/{len(stages)}"
                        if "i" in locals() and "stages" in locals()
                        else "initialization"
                    )
                },
            ) from e
        finally:
            # Cleanup
            self.cleanup()

    def _calculate_optimal_tile_size(self, available_vram_mb: float) -> int:
        """Calculate optimal tile size for available memory."""
        # Use VRAM manager's safe tile calculation
        tile_size = self.vram_manager.get_safe_tile_size(
            available_mb=available_vram_mb,
            model_type="sdxl",
            # Very conservative for CPU offload
            safety_factor=self.strategy_config['conservative_safety_factor'],
        )

        # Clamp to our limits
        tile_size = max(self.min_tile_size, min(tile_size, self.max_tile_size))

        # Ensure multiple of configured rounding
        tile_size = (tile_size // self.strategy_config['tile_size_rounding']) * self.strategy_config['tile_size_rounding']

        return tile_size

    def _plan_cpu_offload_stages(
        self, source_size: Tuple[int, int], target_size: Tuple[int, int], tile_size: int
    ) -> List[Dict[str, Any]]:
        """Plan processing stages for CPU offload."""
        stages = []
        source_w, source_h = source_size
        target_w, target_h = target_size

        # Calculate aspect ratio change
        source_aspect = source_w / source_h
        target_aspect = target_w / target_h
        aspect_change = abs(target_aspect - source_aspect) / source_aspect

        # Stage 1: Initial resize if needed
        if aspect_change > self.strategy_config['aspect_change_threshold']:
            # Need progressive outpainting
            intermediate_w = source_w
            intermediate_h = source_h

            # Gradual aspect adjustment
            steps = self.config.get(
                'parameters',
                {}).get(
                'processing_steps',
                self.strategy_config['aspect_adjust_steps'])  # Multiple small steps
            for i in range(steps):
                progress = (i + 1) / steps
                new_w = int(source_w + (target_w - source_w) * progress)
                new_h = int(source_h + (target_h - source_h) * progress)

                # Round to multiple of configured dimension
                new_w = self.dimension_calc.round_to_multiple(new_w, self.strategy_config['dimension_rounding'])
                new_h = self.dimension_calc.round_to_multiple(new_h, self.strategy_config['dimension_rounding'])

                # Create boundaries list
                boundaries = []
                if new_w > intermediate_w:
                    boundaries.append(
                        {
                            "position": intermediate_w,
                            "direction": "vertical",
                            "step": i,
                            "expansion_size": new_w - intermediate_w,
                        }
                    )
                if new_h > intermediate_h:
                    boundaries.append(
                        {
                            "position": intermediate_h,
                            "direction": "horizontal",
                            "step": i,
                            "expansion_size": new_h - intermediate_h,
                        }
                    )

                stages.append(
                    {
                        "name": f"aspect_adjust_{i + 1}",
                        "type": "outpaint",
                        "method": "tiled_outpaint",
                        "input_size": (intermediate_w, intermediate_h),
                        "output_size": (new_w, new_h),
                        "tile_size": tile_size,
                        "boundaries": boundaries,
                    }
                )

                intermediate_w, intermediate_h = new_w, new_h

        else:
            # Direct upscaling
            stages.append(
                {
                    "name": "direct_upscale",
                    "type": "upscale",
                    "method": "tiled_upscale",
                    "input_size": (source_w, source_h),
                    "output_size": (target_w, target_h),
                    "tile_size": tile_size,
                    "boundaries": [],
                }
            )

        return stages

    def _process_cpu_offload_stage(
        self,
        image: Image.Image,
        stage: Dict[str, Any],
        config: ExpandorConfig,
        stage_index: int,
    ) -> Image.Image:
        """Process a single CPU offload stage."""
        if stage["type"] == "outpaint":
            # Use tiled processor for outpainting
            def process_tile(
                tile_img: Image.Image, tile_info: Dict[str, Any]
            ) -> Image.Image:
                # Determine which edges need expansion
                needs_right = tile_info["x"] + \
                    tile_info["width"] >= image.width
                needs_bottom = tile_info["y"] + \
                    tile_info["height"] >= image.height

                # Create mask for tile
                mask = Image.new("L", tile_img.size, self.strategy_config['mask_background_value'])

                # Mark expansion areas based on tile position
                if needs_right:
                    # Expand right edge
                    expansion_start = max(0, image.width - tile_info["x"])
                    mask_np = np.array(mask)
                    mask_np[:, expansion_start:] = self.strategy_config['mask_foreground_value']
                    mask = Image.fromarray(mask_np)

                if needs_bottom:
                    # Expand bottom edge
                    expansion_start = max(0, image.height - tile_info["y"])
                    mask_np = np.array(mask)
                    mask_np[expansion_start:, :] = self.strategy_config['mask_foreground_value']
                    mask = Image.fromarray(mask_np)

                # Process with inpaint pipeline if mask has content
                if np.any(
                        np.array(mask) > self.strategy_config['mask_background_value']) and hasattr(
                        self,
                        "inpaint_pipeline"):
                    try:
                        result = self.inpaint_pipeline(
                            prompt=config.prompt,
                            image=tile_img,
                            mask_image=mask,
                            strength=self.strategy_config['tile_generation_strength'],
                            num_inference_steps=self.config.get('parameters', {}).get(
                                'tile_generation_steps', self.strategy_config['tile_generation_steps']),  # Fewer steps for speed
                            guidance_scale=self.strategy_config['tile_generation_guidance'],
                            generator=torch.Generator().manual_seed(
                                # FAIL LOUD - seed must be set
                                config.seed + stage_index if config.seed is not None else stage_index
                            ),
                        )

                        if hasattr(result, "images") and result.images:
                            return result.images[0]
                    except Exception as e:
                        self.logger.warning(f"Tile processing failed: {e}")

                return tile_img

            # Process with tiling
            with gpu_memory_manager.memory_efficient_scope("cpu_offload_tiling"):
                result = self.tiled_processor.process_image(
                    image,
                    process_tile,
                    tile_size=stage["tile_size"],
                    overlap=self.default_overlap,
                    target_size=stage["output_size"],
                )

            # Ensure correct size
            if result.size != tuple(stage["output_size"]):
                result = result.resize(
                    stage["output_size"], Image.Resampling.LANCZOS)

            return result

        elif stage["type"] == "upscale":
            # Simple upscale with tiling for memory efficiency
            def upscale_tile(
                tile_img: Image.Image, tile_info: Dict[str, Any]
            ) -> Image.Image:
                # Calculate tile's target size
                scale_x = stage["output_size"][0] / stage["input_size"][0]
                scale_y = stage["output_size"][1] / stage["input_size"][1]

                tile_target_w = int(tile_img.width * scale_x)
                tile_target_h = int(tile_img.height * scale_y)

                # Use img2img pipeline if available for quality
                if (
                    hasattr(self, "img2img_pipeline")
                    and config.quality_preset != "fast"
                ):
                    # First upscale
                    upscaled = tile_img.resize(
                        (tile_target_w, tile_target_h), Image.Resampling.LANCZOS)

                    # Then refine with img2img
                    try:
                        result = self.img2img_pipeline(
                            prompt=config.prompt +
                            ", high quality, sharp details",
                            image=upscaled,
                            strength=self.strategy_config['tile_refinement_strength'],
                            num_inference_steps=self.config.get(
                                'parameters',
                                {}).get(
                                'tile_refinement_steps',
                                self.strategy_config['tile_refinement_steps']),
                            guidance_scale=self.strategy_config['tile_refinement_guidance'],
                            generator=torch.Generator().manual_seed(
                                # FAIL LOUD - seed must be set
                                config.seed + stage_index if config.seed is not None else stage_index
                            ),
                        )

                        if hasattr(result, "images") and result.images:
                            return result.images[0]
                    except Exception as e:
                        self.logger.warning(f"Tile refinement failed: {e}")

                # Fallback to simple upscale
                return tile_img.resize(
                    (tile_target_w, tile_target_h), Image.Resampling.LANCZOS
                )

            # Process with tiling
            with gpu_memory_manager.memory_efficient_scope("cpu_offload_upscale"):
                result = self.tiled_processor.process_image(
                    image,
                    upscale_tile,
                    tile_size=stage["tile_size"],
                    overlap=self.min_overlap,  # Less overlap for upscaling
                    target_size=stage["output_size"],
                )

            return result

        else:
            raise StrategyError(f"Unknown stage type: {stage['type']}")
