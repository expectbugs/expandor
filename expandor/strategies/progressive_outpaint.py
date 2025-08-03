"""
Progressive Outpainting Strategy
Adapted from ai-wallpaper AspectAdjuster
Original: ai_wallpaper/processing/aspect_adjuster.py
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..core.config import ExpandorConfig
from ..utils.dimension_calculator import DimensionCalculator
from ..utils.path_resolver import PathResolver
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
        self.dimension_calc = DimensionCalculator(self.logger)
        self.model_metadata = {}
        
        # Get configuration from ConfigurationManager
        from ..core.configuration_manager import ConfigurationManager
        self.config_manager = ConfigurationManager()
        
        # Get all strategy config at once
        try:
            self.strategy_config = self.config_manager.get_strategy_config('progressive_outpaint')
        except ValueError as e:
            # FAIL LOUD
            raise ValueError(
                f"Failed to load progressive_outpaint configuration!\n{str(e)}"
            )
        
        # Assign all values from config - NO DEFAULTS!
        self.prog_enabled = True  # This can stay hardcoded as it defines the strategy
        self.max_supported = self.strategy_config['max_supported_ratio']
        self.outpaint_strength = self.strategy_config['base_strength']
        self.min_strength = self.strategy_config['min_strength']
        self.max_strength = self.strategy_config['max_strength']
        self.first_step_ratio = self.strategy_config['first_step_ratio']
        self.middle_step_ratio = self.strategy_config['middle_step_ratio']
        self.final_step_ratio = self.strategy_config['final_step_ratio']
        self.outpaint_prompt_suffix = self.strategy_config['outpaint_prompt_suffix']
        self.base_mask_blur = self.strategy_config['base_mask_blur']
        self.base_steps = self.strategy_config['base_steps']
        
        # Get the other parameters
        self.seam_repair_multiplier = self.strategy_config['seam_repair_multiplier']
        self.blur_radius_ratio = self.strategy_config['blur_radius_ratio']
        self.mask_blur_ratio = self.strategy_config['mask_blur_ratio']
        self.edge_preservation_ratio = self.strategy_config['edge_preservation_ratio']

    def _analyze_edge_colors(
        self, image: Image.Image, edge: str, sample_width: Optional[int] = None
    ) -> Dict:
        """
        Analyze colors at image edge for better continuation.
        Copy from lines 81-123 of aspect_adjuster.py
        """
        rgb_mode = self.config_manager.get_value("processing.image_mode", "RGB")
        if image.mode != rgb_mode:
            raise ValueError(f"Image must be {rgb_mode}, got {image.mode}")

        # Get sample_width from config if not provided
        if sample_width is None:
            sample_width = self.config_manager.get_value('strategies.progressive_outpaint.edge_sample_width')

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
            "is_uniform": float(np.mean(color_std)) < self.strategy_config['color_variance_uniform_threshold'],
            "sample_size": pixels.shape[0],
        }

    def execute(
        self, config: ExpandorConfig, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute progressive outpainting strategy"""
        self._context = context or {}

        # Apply config values - FAIL LOUD if not set
        if not hasattr(config, 'denoising_strength') or config.denoising_strength is None:
            raise ValueError(
                "denoising_strength not set in config! This is a required value.\n"
                "Please ensure it's set in your configuration or command line."
            )
        self.outpaint_strength = config.denoising_strength
        
        if not hasattr(config, 'num_inference_steps') or config.num_inference_steps is None:
            raise ValueError(
                "num_inference_steps not set in config! This is a required value.\n"
                "Please ensure it's set in your configuration or command line."
            )
        self.base_steps = config.num_inference_steps
        
        if not hasattr(config, 'guidance_scale') or config.guidance_scale is None:
            raise ValueError(
                "guidance_scale not set in config! This is a required value.\n"
                "Please ensure it's set in your configuration or command line."
            )
        self.guidance_scale = config.guidance_scale

        # Store config for use in helper methods
        self._current_config = config

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
            path_resolver = PathResolver(self.logger)
            base_temp_dir = self.config_manager.get_path("paths.temp_dir")
            temp_dir = base_temp_dir / "progressive"
            temp_dir = path_resolver.resolve_path(temp_dir, create=True, path_type="directory")
            current_path = temp_dir / \
                f"initial_{current_size[0]}x{current_size[1]}.png"
            png_compression = self.config_manager.get_value("output.formats.png.compression")
            config.source_image.save(current_path, "PNG", compress_level=png_compression)
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
        rounding = self.config_manager.get_value("processing.dimension_rounding")
        target_w = self.dimension_calc.round_to_multiple(target_w, rounding)
        target_h = self.dimension_calc.round_to_multiple(target_h, rounding)

        # Create canvas and mask
        rgb_mode = self.config_manager.get_value("processing.image_mode", "RGB")
        mask_mode = self.config_manager.get_value("processing.mask_mode", "L")
        canvas_bg = self.config_manager.get_value("processing.canvas_background_color", "black")
        mask_bg = self.config_manager.get_value("processing.mask_background_color", "white")
        canvas = Image.new(rgb_mode, (target_w, target_h), color=canvas_bg)
        mask = Image.new(mask_mode, (target_w, target_h), color=mask_bg)

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

        # Analyze image for context-aware prompt enhancement
        image_context = self._analyze_image_context(image)

        # Enhance prompt with image-specific context
        enhanced_prompt = self._enhance_prompt_with_context(
            prompt, image_context, step_info
        )

        # Get adaptive parameters
        num_steps = self._get_adaptive_steps(step_info)
        guidance = self._get_adaptive_guidance(step_info)

        # Adaptive strength based on step type
        strength = self._get_adaptive_strength(step_info)

        # Execute pipeline
        result = self.inpaint_pipeline(
            prompt=enhanced_prompt,
            image=canvas,
            mask_image=mask,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            width=target_w,
            height=target_h,
        ).images[0]

        # CRITICAL: Two-pass approach for seamless blending
        # Second pass to refine seams with lower denoising
        # Default to True if not specified in step_info
        enable_seam_fix = step_info.get("enable_seam_fix") if "enable_seam_fix" in step_info else True
        if enable_seam_fix:
            result = self._refine_seams(
                result, canvas, mask, enhanced_prompt,
                current_w, current_h, pad_left, pad_top
            )

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
                        0,
                        left_colors["color_variance"] *
                        self.edge_preservation_ratio *
                        2,
                        3)
                elif dist_right > 0 and dist_right >= max(
                    dist_left, dist_top, dist_bottom
                ):
                    base_color = np.array(right_colors["median_rgb"])
                    variation = np.random.normal(
                        0,
                        right_colors["color_variance"] *
                        self.edge_preservation_ratio *
                        2,
                        3)
                elif dist_top > 0 and dist_top >= max(
                    dist_left, dist_right, dist_bottom
                ):
                    base_color = np.array(top_colors["median_rgb"])
                    variation = np.random.normal(
                        0,
                        top_colors["color_variance"] *
                        self.edge_preservation_ratio *
                        2,
                        3)
                else:
                    base_color = np.array(bottom_colors["median_rgb"])
                    variation = np.random.normal(
                        0,
                        bottom_colors["color_variance"] *
                        self.edge_preservation_ratio *
                        2,
                        3)

                canvas_array[y, x] = np.clip(
                    base_color + variation, 
                    self.strategy_config['fill_color'] - self.strategy_config['fill_color'],  # 0
                    self.strategy_config['fill_color']  # 255
                ).astype(np.uint8)

        return Image.fromarray(canvas_array)

    def _get_adaptive_blur(self, step_info: Dict) -> int:
        """Calculate adaptive mask blur based on expansion size

        CRITICAL: Must be 40% of new content dimension minimum
        as per CLAUDE.md guidance for seamless blending
        """
        current_w, current_h = step_info["current_size"]
        target_w, target_h = step_info["target_size"]

        # Calculate expansion dimensions
        width_expansion = target_w - current_w
        height_expansion = target_h - current_h

        # Get the larger expansion dimension
        max_expansion = max(width_expansion, height_expansion)

        # CRITICAL: 40% of new content dimension minimum
        optimal_blur = int(
            max_expansion *
            self.blur_radius_ratio)

        # But ensure minimum blur for small expansions
        return max(optimal_blur, self.base_mask_blur * self.strategy_config['mask_blur_multiplier'])

    def _get_adaptive_steps(self, step_info: Dict) -> int:
        """Calculate adaptive inference steps"""
        # Use config value if available
        if hasattr(
                self,
                '_current_config') and hasattr(
                self._current_config,
                'num_inference_steps'):
            base_steps = self._current_config.num_inference_steps
        else:
            base_steps = self.base_steps

        step_type = step_info.get("step_type", "progressive")
        if step_type == "initial":
            return int(base_steps * self.strategy_config['first_step_multiplier'])  # More steps for first expansion
        elif step_type == "final":
            return int(base_steps * self.strategy_config['final_step_multiplier'])  # Fewer steps for final touch
        else:
            return base_steps

    def _get_adaptive_guidance(self, step_info: Dict) -> float:
        """Calculate adaptive guidance scale"""
        # Use config value if available
        if hasattr(
                self,
                '_current_config') and hasattr(
                self._current_config,
                'guidance_scale'):
            base_guidance = self._current_config.guidance_scale
        else:
            base_guidance = self.guidance_scale  # Already validated in execute()

        # Lower guidance for better blending
        return base_guidance

    def _get_adaptive_strength(self, step_info: Dict) -> float:
        """Calculate adaptive denoising strength based on step

        CRITICAL: Must balance generation with preservation
        Too high = disconnected content
        Too low = no new content
        """
        step_type = step_info.get("step_type", "progressive")
        # Expansion ratio must be provided in step_info
        if "expansion_ratio" not in step_info:
            raise ValueError(
                "expansion_ratio not found in step_info!\n"
                "This is a required value for adaptive strength calculation."
            )
        expansion_ratio = step_info["expansion_ratio"]

        # Use config value if available, otherwise use instance default
        if hasattr(
                self,
                '_current_config') and hasattr(
                self._current_config,
                'denoising_strength'):
            strength = self._current_config.denoising_strength
        else:
            strength = self.outpaint_strength

        # Adjust based on step type
        if step_type == "initial":
            # First step needs more generation
            strength = min(strength * self.strategy_config['high_complexity_multiplier'], self.max_strength)
        elif step_type == "final":
            # Final steps need more preservation
            strength = max(strength * self.strategy_config['low_complexity_multiplier'], self.min_strength)

        # Further adjust based on expansion size
        if expansion_ratio > self.strategy_config['large_expansion_ratio_threshold']:
            # Large expansions need careful balance
            strength = min(strength, self.strategy_config['large_expansion_strength_limit'])

        return strength

    def _save_temp_result(self, image, old_w, old_h, new_w, new_h):
        """Save intermediate result"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"prog_{old_w}x{old_h}_to_{new_w}x{new_h}_{timestamp}.png"

        # Create temp directory using PathResolver
        path_resolver = PathResolver(self.logger)
        base_temp_dir = self.config_manager.get_path("paths.temp_dir")
        temp_dir = base_temp_dir / "progressive"
        temp_dir = path_resolver.resolve_path(temp_dir, create=True, path_type="directory")

        save_path = temp_dir / filename
        png_compression = self.config_manager.get_value("output.formats.png.compression")
        image.save(save_path, "PNG", compress_level=png_compression)

        return save_path

    def _analyze_image_context(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image to understand its content and style for better prompting"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # Analyze color distribution
        mean_color = np.mean(img_array, axis=(0, 1))
        std_color = np.std(img_array, axis=(0, 1))

        # Detect if image is grayscale-ish
        color_variance = np.std(mean_color)
        is_monochrome = color_variance < self.strategy_config['monochrome_threshold']

        # Get RGB normalization value from config
        norm_divisor = self.config_manager.get_value("processing.rgb_max_value")
        
        # Analyze brightness
        brightness = np.mean(img_array) / norm_divisor
        is_dark = brightness < self.strategy_config['brightness_dark_threshold']
        is_bright = brightness > self.strategy_config['brightness_bright_threshold']

        # Analyze contrast
        contrast = np.std(img_array) / norm_divisor
        is_high_contrast = contrast > self.strategy_config['contrast_high_threshold']

        # Edge complexity (simple metric)
        edges = np.gradient(np.mean(img_array, axis=2))
        edge_density = np.mean(np.abs(edges[0]) + np.abs(edges[1]))
        is_detailed = edge_density > self.strategy_config['edge_density_detailed_threshold']

        return {
            "mean_color": mean_color.tolist(),
            "is_monochrome": is_monochrome,
            "is_dark": is_dark,
            "is_bright": is_bright,
            "is_high_contrast": is_high_contrast,
            "is_detailed": is_detailed,
            "brightness": float(brightness),
            "contrast": float(contrast),
            "edge_density": float(edge_density)
        }

    def _enhance_prompt_with_context(
        self, base_prompt: str, image_context: Dict[str, Any], step_info: Dict
    ) -> str:
        """Create context-aware prompt based on image analysis"""
        # Start with base prompt
        enhanced = base_prompt

        # Add style descriptors based on image analysis
        style_additions = []

        if image_context["is_monochrome"]:
            style_additions.append("monochrome")
            style_additions.append("black and white")

        if image_context["is_dark"]:
            style_additions.append("dark atmosphere")
            style_additions.append("low key lighting")
        elif image_context["is_bright"]:
            style_additions.append("bright lighting")
            style_additions.append("high key")

        if image_context["is_high_contrast"]:
            style_additions.append("high contrast")
            style_additions.append("dramatic lighting")

        if image_context["is_detailed"]:
            style_additions.append("highly detailed")
            style_additions.append("intricate")

        # Add style descriptors if any
        if style_additions:
            enhanced += ", " + ", ".join(style_additions)

        # Add expansion-specific suffixes
        direction = step_info.get("direction", "both")
        if direction == "horizontal":
            enhanced += ", seamless horizontal continuation, extending scenery naturally to the sides"
        elif direction == "vertical":
            enhanced += ", seamless vertical expansion, natural sky and ground extension"
        else:
            enhanced += self.outpaint_prompt_suffix

        # Add quality descriptors
        enhanced += ", maintaining consistent style and lighting, perfect seamless blend"

        return enhanced

    def _refine_seams(
        self,
        initial_result: Image.Image,
        original_canvas: Image.Image,
        original_mask: Image.Image,
        prompt: str,
        current_w: int,
        current_h: int,
        pad_left: int,
        pad_top: int
    ) -> Image.Image:
        """Second pass to refine seams with targeted inpainting"""
        # Create a mask that focuses on the transition area
        seam_mask = Image.new("L", initial_result.size, 0)
        mask_draw = ImageDraw.Draw(seam_mask)

        # Define seam region (40% of expansion area as per CLAUDE.md)
        seam_width = int(max(pad_left, pad_top) *
                         self.seam_repair_multiplier)
        if seam_width < self.strategy_config['min_seam_width']:
            seam_width = self.strategy_config['min_seam_width']  # Minimum seam width

        # Create seam mask based on padding direction
        if pad_left > 0:  # Left padding
            mask_draw.rectangle([pad_left -
                                 seam_width //
                                 2, 0, pad_left +
                                 current_w +
                                 seam_width //
                                 2, initial_result.height], fill=self.strategy_config['fill_color'])
        if pad_top > 0:  # Top padding
            mask_draw.rectangle(
                [0, pad_top - seam_width // 2,
                 initial_result.width, pad_top + current_h + seam_width // 2],
                fill=self.strategy_config['fill_color']
            )

        # Right side seam if expanded right
        right_pad = initial_result.width - (pad_left + current_w)
        if right_pad > 0:
            mask_draw.rectangle([pad_left +
                                 current_w -
                                 seam_width //
                                 2, 0, pad_left +
                                 current_w +
                                 seam_width //
                                 2, initial_result.height], fill=self.strategy_config['fill_color'])

        # Bottom seam if expanded down
        bottom_pad = initial_result.height - (pad_top + current_h)
        if bottom_pad > 0:
            mask_draw.rectangle(
                [0, pad_top + current_h - seam_width // 2,
                 initial_result.width, pad_top + current_h + seam_width // 2],
                fill=self.strategy_config['fill_color']
            )

        # Apply heavy blur to seam mask for gradual transition
        seam_mask = seam_mask.filter(ImageFilter.GaussianBlur(seam_width // self.strategy_config['seam_blur_divisor']))

        # Refine with very low denoising to blend seams
        refined = self.inpaint_pipeline(
            prompt=prompt + ", perfect seamless blend, no visible edges",
            image=initial_result,
            mask_image=seam_mask,
            strength=min(
                self.config_manager.get_value('strategies.progressive_outpaint.seam_repair_max_strength'),
                self.outpaint_strength *
                self.seam_repair_multiplier),
            # Use configured seam repair values
            num_inference_steps=self.config_manager.get_value('strategies.progressive_outpaint.seam_repair_steps'),
            guidance_scale=self.config_manager.get_value('strategies.progressive_outpaint.seam_repair_guidance'),
            # Lower guidance for better blending
            width=initial_result.width,
            height=initial_result.height,
        ).images[0]

        return refined
