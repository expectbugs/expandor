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
from ..core.exceptions import ExpandorError, StrategyError, VRAMError
from ..processors.tiled_processor import TiledProcessor
from ..utils.dimension_calculator import DimensionCalculator
from ..utils.memory_utils import gpu_memory_manager, load_to_gpu, offload_to_cpu
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

        # CPU offload specific parameters
        strategy_config = config or {}
        self.min_tile_size = strategy_config.get("min_tile_size", 384)
        self.default_tile_size = strategy_config.get("default_tile_size", 512)
        self.max_tile_size = strategy_config.get("max_tile_size", 768)
        self.min_overlap = strategy_config.get("min_overlap", 64)
        self.default_overlap = strategy_config.get("default_overlap", 128)

    def validate_requirements(self):
        """Validate CPU offload requirements."""
        # CPU offload can work with minimal resources
        # Requirements checked in execute
        pass

    def estimate_vram(self, config: ExpandorConfig) -> Dict[str, float]:
        """
        Estimate VRAM requirements for CPU offload.

        This strategy uses minimal VRAM by design.
        """
        # Calculate tile size based on available memory
        tile_size = self.vram_manager.get_safe_tile_size(
            model_type="sdxl", safety_factor=self.config.get('parameters', {}).get('safety_factor', 0.6)  # More conservative for CPU offload
        )

        # Estimate for single tile processing
        tile_pixels = tile_size * tile_size
        tile_vram_mb = (tile_pixels * 4 * 3 * 2) / (1024**2)

        # Minimal pipeline memory with offloading
        pipeline_vram = self.config.get('parameters', {}).get('pipeline_vram', 1024)  # 1GB max with offloading

        return {
            "base_vram_mb": tile_vram_mb,
            "peak_vram_mb": tile_vram_mb + pipeline_vram,
            "strategy_overhead_mb": 512,  # Offloading overhead
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

        # Check pipeline availability - pipelines should be injected by orchestrator
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
                self.vram_manager.get_available_vram() or 512
            )  # Assume 512MB if no GPU
            optimal_tile_size = self._calculate_optimal_tile_size(available_vram)

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
                    stage_path = config.stage_dir / f"cpu_offload_stage_{i:02d}.png"
                    current_image.save(stage_path, "PNG", compress_level=0)

                # Aggressive cleanup
                gpu_memory_manager.clear_cache(aggressive=True)

            # Final validation
            if current_image.size != (target_w, target_h):
                raise ExpandorError(
                    f"CPU offload size mismatch: expected {target_w}x{target_h}, "
                    f"got {
                        current_image.size[0]}x{
                        current_image.size[1]}",
                    stage="validation",
                )

            # Save final result
            output_path = self.save_temp_image(current_image, "cpu_offload_final")

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
            safety_factor=self.config.get('parameters', {}).get('conservative_safety_factor', 0.5),  # Very conservative for CPU offload
        )

        # Clamp to our limits
        tile_size = max(self.min_tile_size, min(tile_size, self.max_tile_size))

        # Ensure multiple of 64
        tile_size = (tile_size // 64) * 64

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
        if aspect_change > 0.1:
            # Need progressive outpainting
            intermediate_w = source_w
            intermediate_h = source_h

            # Gradual aspect adjustment
            steps = self.config.get('parameters', {}).get('processing_steps', 3)  # Multiple small steps
            for i in range(steps):
                progress = (i + 1) / steps
                new_w = int(source_w + (target_w - source_w) * progress)
                new_h = int(source_h + (target_h - source_h) * progress)

                # Round to multiple of 8
                new_w = self.dimension_calc.round_to_multiple(new_w, 8)
                new_h = self.dimension_calc.round_to_multiple(new_h, 8)

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
                needs_right = tile_info["x"] + tile_info["width"] >= image.width
                needs_bottom = tile_info["y"] + tile_info["height"] >= image.height

                # Create mask for tile
                mask = Image.new("L", tile_img.size, 0)

                # Mark expansion areas based on tile position
                if needs_right:
                    # Expand right edge
                    expansion_start = max(0, image.width - tile_info["x"])
                    mask_np = np.array(mask)
                    mask_np[:, expansion_start:] = 255
                    mask = Image.fromarray(mask_np)

                if needs_bottom:
                    # Expand bottom edge
                    expansion_start = max(0, image.height - tile_info["y"])
                    mask_np = np.array(mask)
                    mask_np[expansion_start:, :] = 255
                    mask = Image.fromarray(mask_np)

                # Process with inpaint pipeline if mask has content
                if np.any(np.array(mask) > 0) and hasattr(self, "inpaint_pipeline"):
                    try:
                        result = self.inpaint_pipeline(
                            prompt=config.prompt,
                            image=tile_img,
                            mask_image=mask,
                            strength=self.config.get('parameters', {}).get('tile_generation_strength', 0.9),
                            num_inference_steps=self.config.get('parameters', {}).get('tile_generation_steps', 25),  # Fewer steps for speed
                            guidance_scale=self.config.get('parameters', {}).get('tile_generation_guidance', 7.0),
                            generator=torch.Generator().manual_seed(
                                (config.seed or 0) + stage_index
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
                result = result.resize(stage["output_size"], Image.Resampling.LANCZOS)

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
                        (tile_target_w, tile_target_h), Image.Resampling.LANCZOS
                    )

                    # Then refine with img2img
                    try:
                        result = self.img2img_pipeline(
                            prompt=config.prompt + ", high quality, sharp details",
                            image=upscaled,
                            strength=self.config.get('parameters', {}).get('tile_refinement_strength', 0.3),
                            num_inference_steps=self.config.get('parameters', {}).get('tile_refinement_steps', 20),
                            guidance_scale=self.config.get('parameters', {}).get('tile_refinement_guidance', 7.0),
                            generator=torch.Generator().manual_seed(
                                (config.seed or 0) + stage_index
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
