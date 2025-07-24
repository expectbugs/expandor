"""
Progressive Outpainting Strategy
Adapted from ai-wallpaper AspectAdjuster
Original: ai_wallpaper/processing/aspect_adjuster.py
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..core.config import ExpandorConfig
from ..core.vram_manager import VRAMManager
from ..utils.dimension_calculator import DimensionCalculator
from .base_strategy import BaseExpansionStrategy


class ProgressiveOutpaintStrategy(BaseExpansionStrategy):
    """Progressive aspect ratio adjustment with zero quality compromise"""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(config=config, metrics=metrics, logger=logger)
        # Note: logger is already set by parent class
        self.dimension_calc = DimensionCalculator(self.logger)
        self.model_metadata = {}  # Initialize for boundary tracking

        # Configuration from AspectAdjuster __init__ (lines 47-79)
        # NOTE: Original loads from ConfigManager, we're hardcoding defaults
        # Progressive outpainting config
        self.prog_enabled = True
        self.max_supported = 8.0

        # Outpaint settings from lines 59-68
        self.outpaint_strength = 0.95  # High enough to generate content
        self.min_strength = 0.20  # Minimum strength for final passes
        self.max_strength = 0.95  # Maximum strength
        self.outpaint_prompt_suffix = (
            ", seamless expansion, extended scenery, natural continuation"
        )
        self.base_mask_blur = 32
        self.base_steps = 60

        # Expansion ratios from lines 75-79 - REDUCED for maximum context
        self.first_step_ratio = 1.4  # Was 2.0
        self.middle_step_ratio = 1.25  # Was 1.5
        self.final_step_ratio = 1.15  # Was 1.3

    def _analyze_edge_colors(
        self, image: Image.Image, edge: str, sample_width: int = 50
    ) -> Dict:
        """
        Analyze colors at image edge for better continuation.
        Copy from lines 81-123 of aspect_adjuster.py
        """
        if image.mode != "RGB":
            raise ValueError(f"Image must be RGB, got {image.mode}")

        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # Validate dimensions
        if h < sample_width or w < sample_width:
            raise ValueError(
                f"Image too small ({w}x{h}) for {sample_width}px edge sampling"
            )

        if edge == "left":
            sample = img_array[:, :sample_width]
        elif edge == "right":
            sample = img_array[:, -sample_width:]
        elif edge == "top":
            sample = img_array[:sample_width, :]
        elif edge == "bottom":
            sample = img_array[-sample_width:, :]
        else:
            raise ValueError(f"Invalid edge: {edge}")

        # Calculate dominant colors
        pixels = sample.reshape(-1, 3)
        mean_color = np.mean(pixels, axis=0)
        median_color = np.median(pixels, axis=0)

        # Calculate color variance
        color_std = np.std(pixels, axis=0)

        return {
            "mean_rgb": mean_color.tolist(),
            "median_rgb": median_color.tolist(),
            "color_variance": float(np.mean(color_std)),
            "is_uniform": float(np.mean(color_std)) < 20,
            "sample_size": pixels.shape[0],
        }

    def execute(
        self, config: ExpandorConfig, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute progressive outpainting strategy"""
        self._context = context or {}

        # IMPORTANT: This method needs access to inpaint_pipeline
        # Pipeline should be injected by orchestrator
        if not hasattr(self, "inpaint_pipeline") or not self.inpaint_pipeline:
            raise RuntimeError(
                "No inpainting pipeline available for progressive outpainting"
            )

        # Calculate steps
        if isinstance(config.source_image, Path):
            current_image = Image.open(config.source_image)
        else:
            current_image = config.source_image

        current_size = current_image.size
        target_w, target_h = config.get_target_resolution()
        target_aspect = target_w / target_h

        # Calculate progressive steps
        steps = self.dimension_calc.calculate_progressive_strategy(
            current_size, target_aspect
        )

        if not steps:
            # No expansion needed
            return {
                "image_path": (
                    config.source_image
                    if isinstance(config.source_image, Path)
                    else None
                ),
                "size": current_size,
                "stages": [],
            }

        # Execute progressive adjustment
        # Save initial image if needed
        if isinstance(config.source_image, Image.Image):
            temp_dir = Path("temp/progressive")
            temp_dir.mkdir(parents=True, exist_ok=True)
            current_path = temp_dir / f"initial_{current_size[0]}x{current_size[1]}.png"
            config.source_image.save(current_path, "PNG", compress_level=0)
        else:
            current_path = config.source_image

        stages = []
        boundaries = []

        for i, step in enumerate(steps):
            step_num = i + 1
            self.logger.info(
                f"Progressive Step {step_num}/{len(steps)}: {step['description']}"
            )

            # Execute outpaint step
            current_path = self._execute_outpaint_step(
                current_path, config.prompt, step
            )

            # Track boundaries for seam detection
            boundary_position = (
                step["current_size"][0]
                if step["direction"] == "horizontal"
                else step["current_size"][1]
            )
            self.track_boundary(
                position=boundary_position,
                direction=step["direction"],
                step=step_num,
                expansion_size=(
                    step["target_size"][0] - step["current_size"][0]
                    if step["direction"] == "horizontal"
                    else step["target_size"][1] - step["current_size"][1]
                ),
                source_size=step["current_size"],
                target_size=step["target_size"],
                method="progressive_outpaint",
            )

            # Also keep local list for return value
            boundaries.append(
                {
                    "step": step_num,
                    "position": boundary_position,
                    "direction": step["direction"],
                }
            )

            # Update current state
            current_image = Image.open(current_path)

            stages.append(
                {
                    "name": f"progressive_step_{step_num}",
                    "input_size": step["current_size"],
                    "output_size": step["target_size"],
                    "method": "progressive_outpaint",
                }
            )

        return {
            "image_path": current_path,
            "size": current_image.size,
            "stages": stages,
            "boundaries": boundaries,
        }

    def _execute_outpaint_step(
        self, image_path: Path, prompt: str, step_info: Dict
    ) -> Path:
        """
        Execute a single outpaint step.
        Adapted from _execute_outpaint_step lines 524-698
        """
        # Validate input
        if image_path is None:
            raise ValueError("Image path cannot be None")
        if not isinstance(image_path, Path):
            image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        current_w, current_h = image.size
        target_w, target_h = step_info["target_size"]

        # Round to model constraints (SDXL uses 8x multiples)
        target_w = self.dimension_calc.round_to_multiple(target_w, 8)
        target_h = self.dimension_calc.round_to_multiple(target_h, 8)

        # Create canvas and mask
        canvas = Image.new("RGB", (target_w, target_h), color="black")
        mask = Image.new("L", (target_w, target_h), color="white")

        # Calculate padding
        pad_left = (target_w - current_w) // 2
        pad_top = (target_h - current_h) // 2

        # Place image
        canvas.paste(image, (pad_left, pad_top))

        # Create mask (black = keep, white = generate)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(
            [pad_left, pad_top, pad_left + current_w - 1, pad_top + current_h - 1],
            fill="black",
        )

        # Apply adaptive mask blur
        mask_blur = self._get_adaptive_blur(step_info)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))

        # Pre-fill with edge colors (lines 574-625)
        canvas = self._prefill_canvas_with_edge_colors(
            canvas, image, pad_left, pad_top, current_w, current_h
        )

        # Enhance prompt
        enhanced_prompt = prompt + self.outpaint_prompt_suffix

        # Get adaptive parameters
        num_steps = self._get_adaptive_steps(step_info)
        guidance = self._get_adaptive_guidance(step_info)

        # Execute pipeline
        result = self.inpaint_pipeline(
            prompt=enhanced_prompt,
            image=canvas,
            mask_image=mask,
            strength=self.outpaint_strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            width=target_w,
            height=target_h,
        ).images[0]

        # Save result
        temp_path = self._save_temp_result(
            result, current_w, current_h, target_w, target_h
        )

        return temp_path

    def _prefill_canvas_with_edge_colors(
        self, canvas, source_image, pad_left, pad_top, current_w, current_h
    ):
        """
        Pre-fill empty areas with edge-extended colors.
        Adapted from lines 574-625 of aspect_adjuster.py
        """
        canvas_array = np.array(canvas)

        # Analyze edges
        left_colors = self._analyze_edge_colors(source_image, "left")
        right_colors = self._analyze_edge_colors(source_image, "right")
        top_colors = self._analyze_edge_colors(source_image, "top")
        bottom_colors = self._analyze_edge_colors(source_image, "bottom")

        # Fill empty areas with gradient from nearest edge
        for y in range(canvas_array.shape[0]):
            for x in range(canvas_array.shape[1]):
                # Skip if pixel already has content
                if not np.all(canvas_array[y, x] == 0):
                    continue

                # Calculate distance to original image
                dist_left = max(0, pad_left - x)
                dist_right = max(0, x - (pad_left + current_w - 1))
                dist_top = max(0, pad_top - y)
                dist_bottom = max(0, y - (pad_top + current_h - 1))

                # Determine which edge is closest and fill accordingly
                if dist_left > 0 and dist_left >= max(
                    dist_right, dist_top, dist_bottom
                ):
                    base_color = np.array(left_colors["median_rgb"])
                    variation = np.random.normal(
                        0, left_colors["color_variance"] * 0.2, 3
                    )
                elif dist_right > 0 and dist_right >= max(
                    dist_left, dist_top, dist_bottom
                ):
                    base_color = np.array(right_colors["median_rgb"])
                    variation = np.random.normal(
                        0, right_colors["color_variance"] * 0.2, 3
                    )
                elif dist_top > 0 and dist_top >= max(
                    dist_left, dist_right, dist_bottom
                ):
                    base_color = np.array(top_colors["median_rgb"])
                    variation = np.random.normal(
                        0, top_colors["color_variance"] * 0.2, 3
                    )
                else:
                    base_color = np.array(bottom_colors["median_rgb"])
                    variation = np.random.normal(
                        0, bottom_colors["color_variance"] * 0.2, 3
                    )

                canvas_array[y, x] = np.clip(base_color + variation, 0, 255).astype(
                    np.uint8
                )

        return Image.fromarray(canvas_array)

    def _get_adaptive_blur(self, step_info: Dict) -> int:
        """Calculate adaptive mask blur based on expansion size"""
        expansion_ratio = step_info.get("expansion_ratio", 1.5)
        # Larger blur for larger expansions
        if expansion_ratio > 1.8:
            return int(self.base_mask_blur * 1.5)
        elif expansion_ratio > 1.5:
            return int(self.base_mask_blur * 1.2)
        else:
            return self.base_mask_blur

    def _get_adaptive_steps(self, step_info: Dict) -> int:
        """Calculate adaptive inference steps"""
        step_type = step_info.get("step_type", "progressive")
        if step_type == "initial":
            return int(self.base_steps * 1.2)  # More steps for first expansion
        elif step_type == "final":
            return int(self.base_steps * 0.8)  # Fewer steps for final touch
        else:
            return self.base_steps

    def _get_adaptive_guidance(self, step_info: Dict) -> float:
        """Calculate adaptive guidance scale"""
        # Lower guidance for better blending
        return 7.5

    def _save_temp_result(self, image, old_w, old_h, new_w, new_h):
        """Save intermediate result"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"prog_{old_w}x{old_h}_to_{new_w}x{new_h}_{timestamp}.png"

        # Create temp directory
        temp_dir = Path("temp/progressive")
        temp_dir.mkdir(parents=True, exist_ok=True)

        save_path = temp_dir / filename
        image.save(save_path, "PNG", compress_level=0)

        return save_path
