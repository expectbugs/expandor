"""
Sliding Window Progressive Outpainting (SWPO) Strategy
Implements progressive expansion using overlapping windows for seamless results.
Based on ai-wallpaper's AspectAdjuster._sliding_window_adjust implementation.
"""

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter

from ..core.config import ExpandorConfig
from ..core.exceptions import ExpandorError, StrategyError, VRAMError
from ..utils.dimension_calculator import DimensionCalculator
from ..utils.image_utils import (blend_images, create_gradient_mask,
                                 extract_edge_colors)
from ..utils.memory_utils import gpu_memory_manager
from .base_strategy import BaseExpansionStrategy


@dataclass
class SWPOWindow:
    """Represents a single window in SWPO processing"""

    index: int
    position: Tuple[int, int, int, int]  # x1, y1, x2, y2
    expansion_type: str  # 'horizontal' or 'vertical'
    expansion_size: int
    overlap_size: int
    is_first: bool
    is_last: bool


class SWPOStrategy(BaseExpansionStrategy):
    """
    Sliding Window Progressive Outpainting strategy for extreme aspect ratios.

    Key features:
    - Overlapping windows maintain context throughout expansion
    - Configurable window size and overlap ratio
    - Automatic VRAM management with cache clearing
    - Optional final unification pass
    - Zero tolerance for visible seams
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize SWPO strategy with proper BaseExpansionStrategy signature."""
        super().__init__(config=config, metrics=metrics, logger=logger)

        # Initialize dimension calculator
        self.dimension_calc = DimensionCalculator(self.logger)

        # Get configuration from ConfigurationManager
        from ..core.configuration_manager import ConfigurationManager
        self.config_manager = ConfigurationManager()
        
        # Get all strategy config at once
        try:
            self.strategy_params = self.config_manager.get_strategy_config('swpo')
        except ValueError as e:
            # FAIL LOUD
            raise ValueError(
                f"Failed to load swpo configuration!\n{str(e)}"
            )

        # FAIL LOUD if required params missing
        required = [
            'default_overlap_ratio',
            'default_denoising_strength',
            'unification_strength',
            'window_blur_radius',
            'edge_blur_ratio']
        missing = [p for p in required if p not in self.strategy_params]
        if missing:
            raise ValueError(
                f"SWPOStrategy missing required parameters: {missing}\n"
                f"Please check config/strategy_parameters.yaml"
            )

        # Strategy-specific configuration - ALL from config, NO DEFAULTS!
        self.default_window_size = self.strategy_params['default_window_size']
        self.default_overlap_ratio = self.strategy_params['default_overlap_ratio']
        self.default_denoising_strength = self.strategy_params['default_denoising_strength']
        self.default_edge_blur_width = self.strategy_params['default_edge_blur_width']
        self.clear_cache_every_n_windows = self.strategy_params['clear_cache_every_n_windows']

    def validate_requirements(self):
        """
        Validate SWPO requirements - FAIL LOUD if not met.

        Raises:
            StrategyError: If requirements not satisfied
        """
        # Requirements will be checked in execute method

    def estimate_vram(self, config: ExpandorConfig) -> Dict[str, float]:
        """
        Estimate VRAM requirements for SWPO execution.

        Args:
            config: ExpandorConfig instance

        Returns:
            Dictionary with VRAM estimates
        """
        # Get base estimate from parent
        base_estimate = super().estimate_vram(config)

        # SWPO processes windows sequentially
        window_size = config.window_size or self.default_window_size
        overlap_ratio = config.overlap_ratio or self.default_overlap_ratio

        # Calculate effective window area
        overlap_pixels = int(window_size * overlap_ratio)
        effective_area = window_size * window_size

        # Estimate VRAM for single window - use configured values
        # Formula: pixels * channels * precision * batch * safety_factor
        from ..core.configuration_manager import ConfigurationManager
        config_manager = ConfigurationManager()
        
        channels = config_manager.get_value("vram.image_channels")
        bytes_per_pixel = config_manager.get_value("vram.bytes_per_pixel")
        batch_size = config_manager.get_value("vram.default_batch_size")
        safety_factor = config_manager.get_value("vram.estimation_safety_factor")
        
        window_vram_mb = (effective_area * channels * bytes_per_pixel * batch_size * safety_factor) / (1024**2)

        # Add pipeline overhead
        pipeline_vram = self.vram_manager.estimate_pipeline_memory(
            pipeline_type="sdxl", include_vae=True
        )

        swpo_vram = window_vram_mb + pipeline_vram

        return {
            "base_vram_mb": base_estimate["base_vram_mb"],
            "peak_vram_mb": swpo_vram,
            "strategy_overhead_mb": window_vram_mb,
        }

    def execute(
        self, config: ExpandorConfig, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute SWPO expansion with comprehensive error handling.

        Process:
        1. Plan window strategy
        2. Execute each window with overlap
        3. Track boundaries for seam detection
        4. Optional final unification pass
        5. Validate results

        Args:
            config: ExpandorConfig with all parameters
            context: Execution context with injected components

        Returns:
            Dict with image_path, size, stages, boundaries, metadata

        Raises:
            StrategyError: On any failure (FAIL LOUD)
        """
        self._context = context or {}
        start_time = time.time()

        self.logger.info("Starting SWPO expansion strategy")

        # Validate inputs LOUDLY
        self.validate_inputs(config)

        # Check pipeline requirement LOUDLY
        if not self.inpaint_pipeline:
            raise StrategyError(
                "SWPO requires inpaint_pipeline - none provided",
                details={
                    "available_pipelines": list(
                        self._context.get("pipeline_registry", {}).keys()
                    )
                },
            )

        # Store pipeline reference
        # Pipelines should already be injected by orchestrator

        with gpu_memory_manager.memory_efficient_scope("swpo_execution"):
            try:
                # Load source image
                if isinstance(config.source_image, Path):
                    current_image = self.validate_image_path(
                        config.source_image)
                else:
                    current_image = config.source_image.copy()

                source_w, source_h = current_image.size
                target_w, target_h = config.get_target_resolution()

                # Ensure dimensions are multiples of 8
                target_w = self.dimension_calc.round_to_multiple(target_w, 8)
                target_h = self.dimension_calc.round_to_multiple(target_h, 8)

                # Get window configuration from config
                # FAIL LOUD - window size and overlap ratio should be set by config system
                if config.window_size is None:
                    raise ValueError(
                        "window_size not set in configuration. "
                        "This should have been set by ExpandorConfig defaults."
                    )
                if config.overlap_ratio is None:
                    raise ValueError(
                        "overlap_ratio not set in configuration. "
                        "This should have been set by ExpandorConfig defaults."
                    )
                window_size = config.window_size
                overlap_ratio = config.overlap_ratio

                # Plan SWPO windows
                windows = self._plan_windows(
                    source_size=(source_w, source_h),
                    target_size=(target_w, target_h),
                    window_size=window_size,
                    overlap_ratio=overlap_ratio,
                )

                if not windows:
                    raise StrategyError(
                        "Failed to plan SWPO windows",
                        details={
                            "source_size": (source_w, source_h),
                            "target_size": (target_w, target_h),
                        },
                    )

                self.logger.info(f"Planned {len(windows)} SWPO windows")

                # Process each window
                for i, window in enumerate(windows):
                    window_start = time.time()

                    # Check VRAM before processing
                    required_vram = self.estimate_vram(config)["peak_vram_mb"]
                    if not self.check_vram(required_vram):
                        raise VRAMError(
                            operation=f"swpo_window_{i}",
                            required_mb=required_vram,
                            available_mb=self.vram_manager.get_available_vram() or 0,
                        )

                    self.logger.info(
                        f"Processing window {i + 1}/{len(windows)}")

                    # Process window
                    current_image, window_result = self._execute_window(
                        current_image, window, config, i
                    )

                    # Track boundary
                    if self.boundary_tracker:
                        boundary_position = window_result["boundary_position"]
                        self.track_boundary(
                            position=boundary_position,
                            direction=window.expansion_type,
                            step=i,
                            expansion_size=window.expansion_size,
                        )

                    # Record stage
                    self.record_stage(
                        name=f"swpo_window_{i}",
                        method="sliding_window_outpaint",
                        input_size=window_result["input_size"],
                        output_size=window_result["output_size"],
                        start_time=window_start,
                        metadata={
                            "window_index": i,
                            "expansion_type": window.expansion_type,
                            "expansion_size": window.expansion_size,
                            "overlap_size": window.overlap_size,
                            "window_bounds": window.position,
                        },
                    )

                    # Clear cache periodically
                    if (i + 1) % self.clear_cache_every_n_windows == 0:
                        self.logger.debug(
                            f"Clearing cache after window {
                                i + 1}"
                        )
                        gpu_memory_manager.clear_cache(aggressive=True)

                    # Save intermediate if requested
                    if config.save_stages and config.stage_dir:
                        stage_path = config.stage_dir / \
                            f"swpo_window_{i:03d}.png"
                        # Get compression from config
                        from ..core.configuration_manager import ConfigurationManager
                        config_manager = ConfigurationManager()
                        png_compression = config_manager.get_value("output.formats.png.compression")
                        current_image.save(stage_path, "PNG", compress_level=png_compression)

                # Optional final unification pass
                if (
                    getattr(config, "final_unification_pass", True)
                    and self.img2img_pipeline
                ):
                    self.logger.info("Executing final unification pass")
                    unify_start = time.time()

                    current_image = self._unification_pass(
                        current_image, config)

                    self.record_stage(
                        name="swpo_unification",
                        method="unification_refinement",
                        input_size=(
                            target_w,
                            target_h),
                        output_size=(
                            target_w,
                            target_h),
                        start_time=unify_start,
                        metadata={
                            "strength": self.strategy_params['unification_strength'],
                            "purpose": "seamless_blending"},
                    )

                # Final validation
                if current_image.size != (target_w, target_h):
                    raise ExpandorError(
                        f"SWPO size mismatch: expected {target_w}x{target_h}, "
                        f"got {current_image.size[0]}x{current_image.size[1]}",
                        stage="validation",
                    )

                # Save final result
                output_path = self.save_temp_image(current_image, "swpo_final")

                # Return proper dict
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
                        "strategy": "swpo",
                        "total_windows": len(windows),
                        "window_parameters": {
                            "window_size": window_size,
                            "overlap_ratio": overlap_ratio,
                            "denoising_strength": getattr(
                                config,
                                "denoising_strength",
                                self.default_denoising_strength,
                            ),
                        },
                        "duration": time.time() - start_time,
                        "vram_peak_mb": (
                            self.vram_manager.get_peak_usage()
                            if self.vram_manager
                            else None
                        ),
                    },
                }

            except Exception as e:
                # FAIL LOUD - no silent failures
                self.logger.error(f"SWPO strategy failed: {str(e)}")
                raise StrategyError(
                    f"SWPO execution failed: {str(e)}",
                    details={
                        "stage": (
                            "window_processing"
                            if "window" in locals()
                            else "initialization"
                        ),
                        "progress": (
                            f"{i}/{len(windows)}"
                            if "i" in locals() and "windows" in locals()
                            else "0/0"
                        ),
                    },
                ) from e
            finally:
                # Always cleanup
                self.cleanup()

    def _plan_windows(
        self,
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
        window_size: int,
        overlap_ratio: float,
    ) -> List[SWPOWindow]:
        """
        Plan sliding windows for progressive expansion.

        Critical requirements:
        - All dimensions must be multiples of 8
        - Overlap must be sufficient to maintain context
        - Windows must cover entire expansion area
        - Last window must reach exact target dimensions

        Args:
            source_size: (width, height) of source
            target_size: (width, height) of target
            window_size: Size of each window
            overlap_ratio: Overlap between windows (0-1)

        Returns:
            List of SWPOWindow objects
        """
        source_w, source_h = source_size
        target_w, target_h = target_size

        windows = []

        # Determine expansion direction and size
        width_expansion = target_w - source_w
        height_expansion = target_h - source_h

        if width_expansion <= 0 and height_expansion <= 0:
            # No expansion needed
            return windows

        # Calculate overlap size
        overlap_size = int(window_size * overlap_ratio)
        effective_step = window_size - overlap_size

        # Ensure step is multiple of dimension_multiple
        effective_step = self.dimension_calc.round_to_multiple(
            effective_step, self.strategy_params['dimension_multiple'])

        # Recalculate overlap based on adjusted step
        overlap_size = window_size - effective_step

        current_w = source_w
        current_h = source_h
        window_index = 0

        # Plan horizontal expansion windows
        if width_expansion > 0:
            steps_needed = math.ceil(width_expansion / effective_step)

            for i in range(steps_needed):
                # Calculate this window's expansion
                if i == steps_needed - 1:
                    # Last window - expand to exact target
                    next_w = target_w
                else:
                    next_w = min(current_w + window_size, target_w)
                    next_w = self.dimension_calc.round_to_multiple(
                        next_w, self.strategy_params['dimension_multiple'])

                expansion = next_w - current_w

                # Calculate window bounds
                if i == 0:
                    # First window starts at source edge
                    x1 = 0
                else:
                    # Subsequent windows overlap
                    x1 = current_w - overlap_size

                window = SWPOWindow(
                    index=window_index,
                    position=(x1, 0, next_w, current_h),
                    expansion_type="horizontal",
                    expansion_size=expansion,
                    overlap_size=overlap_size if i > 0 else 0,
                    is_first=(i == 0),
                    is_last=(i == steps_needed - 1),
                )

                windows.append(window)
                current_w = next_w
                window_index += 1

        # Plan vertical expansion windows (after horizontal is complete)
        if height_expansion > 0:
            steps_needed = math.ceil(height_expansion / effective_step)

            for i in range(steps_needed):
                # Calculate this window's expansion
                if i == steps_needed - 1:
                    # Last window - expand to exact target
                    next_h = target_h
                else:
                    next_h = min(current_h + window_size, target_h)
                    next_h = self.dimension_calc.round_to_multiple(
                        next_h, self.strategy_params['dimension_multiple'])

                expansion = next_h - current_h

                # Calculate window bounds
                if i == 0:
                    # First window starts at source edge
                    y1 = 0
                else:
                    # Subsequent windows overlap
                    y1 = current_h - overlap_size

                window = SWPOWindow(
                    index=window_index,
                    position=(0, y1, current_w, next_h),
                    expansion_type="vertical",
                    expansion_size=expansion,
                    overlap_size=overlap_size if i > 0 else 0,
                    is_first=(i == 0),
                    is_last=(i == steps_needed - 1),
                )

                windows.append(window)
                current_h = next_h
                window_index += 1

        return windows

    def _execute_window(
        self,
        current_image: Image.Image,
        window: SWPOWindow,
        config: ExpandorConfig,
        window_index: int,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Execute a single SWPO window with proper masking and blending.

        Critical steps:
        1. Create canvas at window size
        2. Position current image with overlap
        3. Create gradient mask for smooth blending
        4. Analyze edges for color consistency
        5. Execute inpainting
        6. Extract and blend result
        7. Track new boundaries

        Returns:
            Tuple of (updated_image, result_metadata)
        """
        x1, y1, x2, y2 = window.position
        window_w = x2 - x1
        window_h = y2 - y1

        # Create canvas at window size
        canvas = Image.new("RGB", (window_w, window_h))
        # Start with black (preserve)
        mask = Image.new("L", (window_w, window_h), 0)

        # Position current image on canvas
        if window.expansion_type == "horizontal":
            # Expanding horizontally
            paste_x = 0 if window.is_first else window.overlap_size
            paste_y = 0

            # Crop relevant portion of current image
            crop_x1 = x1 if window.is_first else x1 + window.overlap_size
            crop_region = current_image.crop(
                (crop_x1, 0, current_image.width, current_image.height)
            )
            canvas.paste(crop_region, (paste_x, paste_y))

            # Mark new area for generation (right side)
            mask_x = (
                current_image.width - x1
                if window.is_first
                else crop_region.width + paste_x
            )
            mask_w = window_w - mask_x
            mask.paste(255, (mask_x, 0, mask_x + mask_w, window_h))

            # Record boundary position
            boundary_position = current_image.width

        else:  # vertical
            # Expanding vertically
            paste_x = 0
            paste_y = 0 if window.is_first else window.overlap_size

            # Crop relevant portion of current image
            crop_y1 = y1 if window.is_first else y1 + window.overlap_size
            crop_region = current_image.crop(
                (0, crop_y1, current_image.width, current_image.height)
            )
            canvas.paste(crop_region, (paste_x, paste_y))

            # Mark new area for generation (bottom)
            mask_y = (
                current_image.height - y1
                if window.is_first
                else crop_region.height + paste_y
            )
            mask_h = window_h - mask_y
            mask.paste(255, (0, mask_y, window_w, mask_y + mask_h))

            # Record boundary position
            boundary_position = current_image.height

        # Create gradient mask for smooth blending
        if not window.is_first and window.overlap_size > 0:
            gradient_size = min(
                window.overlap_size // self.strategy_params['gradient_size_divisor'], 
                self.strategy_params['gradient_size_max']
            )

            if window.expansion_type == "horizontal":
                # Horizontal gradient at mask edge
                gradient = create_gradient_mask(
                    width=gradient_size,
                    height=window_h,
                    direction="left",
                    blur_radius=gradient_size // self.strategy_params['gradient_blur_divisor'],
                )
                # Apply gradient to transition zone
                mask_np = np.array(mask)
                gradient_np = np.array(gradient)
                transition_x = mask_x
                if transition_x + gradient_size <= window_w:
                    mask_np[:, transition_x: transition_x + gradient_size] = (
                        gradient_np
                    )
                mask = Image.fromarray(mask_np)

            else:  # vertical
                # Vertical gradient at mask edge
                gradient = create_gradient_mask(
                    width=window_w,
                    height=gradient_size,
                    direction="top",
                    blur_radius=gradient_size // self.strategy_params['gradient_blur_divisor'],
                )
                # Apply gradient to transition zone
                mask_np = np.array(mask)
                gradient_np = np.array(gradient)
                transition_y = mask_y
                if transition_y + gradient_size <= window_h:
                    mask_np[transition_y:transition_y + gradient_size, :] = (
                        gradient_np
                    )
                mask = Image.fromarray(mask_np)

        # CRITICAL: Apply differential diffusion concept
        # Use varying mask opacity for better blending
        mask = self._apply_differential_mask(mask, window.expansion_type)

        # Apply edge blur for seamless transition
        # CRITICAL: Calculate blur based on expansion size (40% minimum)
        expansion_size = window.expansion_size
        optimal_blur = max(
            int(expansion_size * self.strategy_params['blur_radius_min_percentage']),
            self.default_edge_blur_width
        )
        edge_blur = getattr(config, "edge_blur_radius", optimal_blur)
        mask = mask.filter(ImageFilter.GaussianBlur(edge_blur))

        # Edge analysis for color continuity
        edge_colors = self._analyze_edge_colors(
            canvas, mask, window.expansion_type)

        # Pre-fill with edge colors (helps with coherence)
        if edge_colors:
            canvas = self._apply_edge_fill(
                canvas,
                mask,
                edge_colors,
                blur_radius=self.strategy_params['window_blur_radius'])

        # Add noise to masked areas (improves generation)
        canvas = self._add_noise_to_mask(
            canvas, mask, strength=self.strategy_params['mask_noise_strength']
        )

        # Analyze source image context for better prompting
        source_context = self._analyze_source_context(canvas, mask)

        # Enhance prompt with context and direction
        enhanced_prompt = self._enhance_window_prompt(
            config.prompt, source_context, window
        )

        # Execute inpainting
        try:
            result = self.inpaint_pipeline(
                prompt=enhanced_prompt,
                image=canvas,
                mask_image=mask,
                height=window_h,
                width=window_w,
                # Adaptive strength based on window properties
                strength=self._calculate_window_strength(
                    window, getattr(
                        config, "denoising_strength", self.default_denoising_strength)
                ),
                num_inference_steps=self._calculate_steps(
                    window.expansion_size),
                guidance_scale=self._calculate_guidance_scale(
                    window.expansion_size),
                generator=torch.Generator().manual_seed(config.seed + window_index),
            )

            if not hasattr(result, "images") or not result.images:
                raise StrategyError(
                    f"Pipeline returned no images for window {window_index}",
                    details={"window": window.__dict__},
                )

            generated = result.images[0]

        except Exception as e:
            raise StrategyError(
                f"Inpainting failed for window {window_index}: {str(e)}",
                details={"window": window.__dict__},
            ) from e

        # Update current image with generated content
        if window.expansion_type == "horizontal":
            # Expand canvas if needed
            if generated.width > current_image.width:
                new_canvas = Image.new("RGB", (x2, current_image.height))
                new_canvas.paste(current_image, (0, 0))
                current_image = new_canvas

            # Paste generated content
            if window.is_first:
                current_image.paste(generated, (x1, y1))
            else:
                # Extract non-overlap portion
                extract_x = window.overlap_size
                extract_region = generated.crop(
                    (extract_x, 0, generated.width, generated.height)
                )
                paste_x = current_image.width - window.expansion_size
                current_image.paste(extract_region, (paste_x, 0))

        else:  # vertical
            # Expand canvas if needed
            if generated.height > current_image.height:
                new_canvas = Image.new("RGB", (current_image.width, y2))
                new_canvas.paste(current_image, (0, 0))
                current_image = new_canvas

            # Paste generated content
            if window.is_first:
                current_image.paste(generated, (x1, y1))
            else:
                # Extract non-overlap portion
                extract_y = window.overlap_size
                extract_region = generated.crop(
                    (0, extract_y, generated.width, generated.height)
                )
                paste_y = current_image.height - window.expansion_size
                current_image.paste(extract_region, (0, paste_y))

        # Prepare result metadata
        result_metadata = {
            "input_size": (canvas.width, canvas.height),
            "output_size": current_image.size,
            "window_size": (window_w, window_h),
            "boundary_position": boundary_position,
            "expansion_type": window.expansion_type,
            "expansion_size": window.expansion_size,
        }

        return current_image, result_metadata

    def _unification_pass(
        self, image: Image.Image, config: ExpandorConfig
    ) -> Image.Image:
        """
        Optional final pass to unify the entire image.
        Uses very low denoising strength to preserve content while smoothing transitions.
        """
        if not self.img2img_pipeline:
            self.logger.warning(
                "No img2img pipeline available for unification pass")
            return image

        strength = getattr(
            config,
            "unification_strength",
            self.strategy_params['unification_strength'])

        try:
            result = self.img2img_pipeline(
                prompt=config.prompt
                + ", seamless, unified composition, perfect quality",
                image=image,
                strength=strength,
                num_inference_steps=self.strategy_params.get(
                    'unification_steps', config.num_inference_steps),
                guidance_scale=self.strategy_params.get(
                    'unification_guidance', config.guidance_scale),
                generator=torch.Generator().manual_seed(
                    config.seed + self.strategy_params['unification_seed_offset']
                ),
            )

            if hasattr(result, "images") and result.images:
                return result.images[0]
            else:
                self.logger.warning(
                    "Unification pass returned no image, using original"
                )
                return image

        except Exception as e:
            self.logger.warning(
                f"Unification pass failed: {e}, using original")
            return image

    def _analyze_edge_colors(
        self, image: Image.Image, mask: Image.Image, expansion_type: str
    ) -> Optional[Dict[str, np.ndarray]]:
        """Analyze edge colors for pre-filling expansion areas"""
        img_array = np.array(image)
        mask_array = np.array(mask)

        edge_colors = {}

        # Find edge of existing content
        if expansion_type == "horizontal":
            # Find rightmost non-masked column
            for x in range(img_array.shape[1] - 1, -1, -1):
                if np.mean(mask_array[:, x]) < self.strategy_params['mask_binary_threshold']:  # Found edge
                    edge_colors["primary"] = extract_edge_colors(
                        image, "right", width=self.strategy_params['edge_width']
                    )
                    break
        else:  # vertical
            # Find bottom-most non-masked row
            for y in range(img_array.shape[0] - 1, -1, -1):
                if np.mean(mask_array[y, :]) < self.strategy_params['mask_binary_threshold']:  # Found edge
                    edge_colors["primary"] = extract_edge_colors(
                        image, "bottom", width=self.strategy_params['edge_width']
                    )
                    break

        return edge_colors if edge_colors else None

    def _apply_edge_fill(
        self,
        image: Image.Image,
        mask: Image.Image,
        edge_colors: Dict[str, np.ndarray],
        blur_radius: int,
    ) -> Image.Image:
        """Apply edge color fill to masked areas for better coherence"""
        img_array = np.array(image)
        mask_array = np.array(mask) / self.strategy_params['mask_conversion_divisor']

        if "primary" in edge_colors:
            # Get average color from edge
            edge_color_mean = np.mean(edge_colors["primary"], axis=(0, 1))

            # Apply to masked areas with gradient
            for c in range(3):
                img_array[:, :, c] = (
                    img_array[:, :, c] * (1 - mask_array)
                    + edge_color_mean[c] * mask_array
                )

        # Convert back and apply blur for smooth transition
        result = Image.fromarray(img_array.astype(np.uint8))
        if blur_radius > 0:
            # Create blurred version
            blurred = result.filter(ImageFilter.GaussianBlur(blur_radius))
            # Blend based on mask
            result = blend_images(result, blurred, mask)

        return result

    def _add_noise_to_mask(
        self, image: Image.Image, mask: Image.Image, strength: float = 0.02
    ) -> Image.Image:
        """Add subtle noise to masked areas to improve inpainting"""
        img_array = np.array(image)
        mask_array = np.array(mask) / self.strategy_params['mask_conversion_divisor']

        # Generate noise
        noise = np.random.normal(
            self.strategy_params['noise_min_value'], 
            strength * self.strategy_params['noise_max_value'], 
            img_array.shape
        )

        # Apply noise only to masked areas
        for c in range(3):
            img_array[:, :, c] = img_array[:, :, c] + \
                (noise[:, :, c] * mask_array)

        # Clip values
        img_array = np.clip(
            img_array, 
            self.strategy_params['noise_min_value'], 
            self.strategy_params['noise_max_value']
        ).astype(np.uint8)

        return Image.fromarray(img_array)

    def _calculate_steps(self, expansion_size: int) -> int:
        """Calculate inference steps based on expansion size"""
        # More steps for larger expansions
        if expansion_size < self.strategy_params['step_thresholds']['small']:
            return self.strategy_params['step_values']['small']
        elif expansion_size < self.strategy_params['step_thresholds']['medium']:
            return self.strategy_params['step_values']['medium']
        elif expansion_size < self.strategy_params['step_thresholds']['large']:
            return self.strategy_params['step_values']['large']
        else:
            return self.strategy_params['step_values']['xlarge']

    def _calculate_guidance_scale(self, expansion_size: int) -> float:
        """Calculate guidance scale based on expansion size"""
        # Moderate guidance for better blending
        if expansion_size < self.strategy_params['guidance_thresholds']['small']:
            return self.strategy_params['guidance_values']['small']
        elif expansion_size < self.strategy_params['guidance_thresholds']['medium']:
            return self.strategy_params['guidance_values']['medium']
        elif expansion_size < self.strategy_params['guidance_thresholds']['large']:
            return self.strategy_params['guidance_values']['large']
        else:
            return self.strategy_params['guidance_values']['xlarge']

    def _calculate_window_strength(
            self,
            window: SWPOWindow,
            base_strength: float) -> float:
        """Calculate adaptive strength for each window

        CRITICAL: Balance generation with preservation
        First windows need more generation, later windows need more preservation
        """
        # Start with base
        strength = base_strength

        # Adjust based on window position
        if window.is_first:
            # First window needs more generation but not too much
            strength = min(
                strength * self.strategy_params['first_window_strength_multiplier'], 
                self.strategy_params['first_window_strength_max']
            )
        elif window.is_last:
            # Last window needs more preservation
            strength = max(
                strength * self.strategy_params['last_window_strength_multiplier'], 
                self.strategy_params['last_window_strength_min']
            )
        else:
            # Middle windows balance both
            strength = min(strength, self.strategy_params['middle_window_strength_max'])

        # Further adjust based on overlap
        if window.overlap_size > 0:
            # More overlap = need more preservation
            overlap_ratio = window.overlap_size / window.expansion_size
            if overlap_ratio > self.strategy_params['overlap_ratio_threshold']:
                strength *= self.strategy_params['overlap_strength_multiplier']

        return strength

    def _analyze_source_context(
            self, image: Image.Image, mask: Image.Image) -> Dict[str, Any]:
        """Analyze the source content near expansion area for context"""
        img_array = np.array(image)
        mask_array = np.array(mask)

        # Find edge region (where mask transitions)
        edge_mask = np.zeros_like(mask_array)
        kernel_size = self.strategy_params['kernel_size']

        # Dilate and subtract to find edge
        from scipy import ndimage
        dilated = ndimage.binary_dilation(
            mask_array > 128, iterations=kernel_size // 2)
        eroded = ndimage.binary_erosion(
            mask_array < 128, iterations=kernel_size // 2)
        edge_mask = dilated & eroded

        # Sample colors from edge region
        edge_pixels = img_array[edge_mask]

        if len(edge_pixels) > 0:
            edge_mean = np.mean(edge_pixels, axis=0)
            edge_std = np.std(edge_pixels, axis=0)

            # Analyze texture complexity
            gray_edge = np.mean(edge_pixels, axis=1)
            texture_variance = np.var(gray_edge)

            return {
                "edge_color_mean": edge_mean.tolist(),
                "edge_color_std": edge_std.tolist(),
                "texture_complexity": float(texture_variance),
                "has_high_frequency": texture_variance > self.strategy_params['texture_variance_high_threshold'],
                "is_smooth": texture_variance < self.strategy_params['texture_variance_smooth_threshold']
            }

        return {"edge_color_mean": self.strategy_params['default_edge_color'], "texture_complexity": 0}

    def _enhance_window_prompt(
        self, base_prompt: str, context: Dict[str, Any], window: SWPOWindow
    ) -> str:
        """Enhance prompt based on window position and image context"""
        enhanced = base_prompt

        # Add texture descriptors
        if context.get("has_high_frequency", False):
            enhanced += ", detailed texture, high frequency details preserved"
        elif context.get("is_smooth", False):
            enhanced += ", smooth gradients, soft transitions"

        # Add direction-specific guidance
        if window.expansion_type == "horizontal":
            if window.is_first:
                enhanced += ", seamless left/right expansion, natural horizontal continuation"
            else:
                enhanced += ", continuing horizontal flow, maintaining perspective"
        else:  # vertical
            if window.is_first:
                enhanced += ", seamless top/bottom expansion, natural vertical continuation"
            else:
                enhanced += ", continuing vertical elements, consistent lighting"

        # Add window-specific guidance
        if not window.is_first:
            enhanced += ", perfectly matching existing content, no visible seams"

        # Always end with quality emphasis
        enhanced += ", ultra high quality, photorealistic blending"

        return enhanced

    def _apply_differential_mask(
            self,
            mask: Image.Image,
            expansion_type: str) -> Image.Image:
        """Apply differential diffusion concept with varying mask opacity

        Based on latest research showing that varying mask opacity
        creates better blends than binary masks
        """
        mask_np = np.array(mask, dtype=np.float32) / self.strategy_params['mask_conversion_divisor']

        # Create differential zones
        # Zone 1: Full generation (white - 255)
        # Zone 2: High blend (light gray - 200)
        # Zone 3: Medium blend (gray - 150)
        # Zone 4: Light blend (dark gray - 100)
        # Zone 5: Preserve (black - 0)

        # Find transition regions
        h, w = mask_np.shape

        # Apply multi-level opacity based on distance from edge
        for y in range(h):
            for x in range(w):
                if mask_np[y, x] > self.strategy_params['mask_generate_threshold']:  # Full generation area
                    # Find distance to nearest black pixel
                    dist_to_edge = self._min_distance_to_black(
                        mask_np, x, y, max_dist=self.strategy_params['differential_mask_max_dist'])

                    if dist_to_edge < self.strategy_params['differential_mask_distance_thresholds']['near']:
                        mask_np[y, x] = self.strategy_params['differential_mask_opacity_values']['high_blend']
                    elif dist_to_edge < self.strategy_params['differential_mask_distance_thresholds']['medium']:
                        mask_np[y, x] = self.strategy_params['differential_mask_opacity_values']['very_high_blend']
                    # else keep at full_generation value

        # Convert back to image
        return Image.fromarray((mask_np * 255).astype(np.uint8))

    def _min_distance_to_black(
            self,
            mask: np.ndarray,
            x: int,
            y: int,
            max_dist: int) -> int:
        """Find minimum distance to a black (preserved) pixel"""
        h, w = mask.shape

        for d in range(1, max_dist):
            # Check square perimeter at distance d
            for dx in range(-d, d + 1):
                for dy in [-d, d]:  # Top and bottom edges
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] < self.strategy_params['mask_preserve_threshold']:
                        return d

            for dy in range(-d + 1,
                            d):  # Left and right edges (excluding corners)
                for dx in [-d, d]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] < self.strategy_params['mask_preserve_threshold']:
                        return d

        return max_dist
