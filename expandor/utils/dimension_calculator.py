"""
Dimension Calculator - Resolution and Aspect Ratio Management
Adapted from ai-wallpaper project
Original: ai_wallpaper/core/resolution_manager.py
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..core.configuration_manager import ConfigurationManager


@dataclass
class ResolutionConfig:
    """Configuration for a specific resolution (matches ai-wallpaper)"""

    width: int
    height: int
    aspect_ratio: float
    total_pixels: int
    name: Optional[str] = None

    @classmethod
    def from_tuple(cls,
                   resolution: Tuple[int,
                                     int],
                   name: Optional[str] = None):
        width, height = resolution
        return cls(
            width=width,
            height=height,
            aspect_ratio=width / height,
            total_pixels=width * height,
            name=name,
        )


class DimensionCalculator:
    """Manages dimension calculations and strategies"""

    # Copy from lines 36-47 of resolution_manager.py
    def _get_presets(self) -> Dict[str, Tuple[int, int]]:
        """Get resolution presets from config"""
        return {
            "1080p": (self.config_manager.get_value('constants.common_dimensions.fhd'),
                      self.config_manager.get_value('constants.common_dimensions.standard_heights')[3]),  # 1920x1080
            "1440p": (self.config_manager.get_value('constants.common_dimensions.qhd'),
                      self.config_manager.get_value('constants.common_dimensions.standard_heights')[4]),  # 2560x1440
            "4K": (self.config_manager.get_value('constants.common_dimensions.4k'),
                   self.config_manager.get_value('constants.common_dimensions.standard_heights')[5]),  # 3840x2160
            "5K": (self.config_manager.get_value('constants.common_dimensions.standard_widths')[6],
                   self.config_manager.get_value('constants.common_dimensions.standard_heights')[6] // 2),  # 5120x2880
            "8K": (self.config_manager.get_value('constants.common_dimensions.8k'),
                   self.config_manager.get_value('constants.common_dimensions.standard_heights')[6]),  # 7680x4320
            "ultrawide_1440p": (3440, self.config_manager.get_value('constants.common_dimensions.standard_heights')[4]),  # 3440x1440
            "ultrawide_4K": (self.config_manager.get_value('constants.common_dimensions.standard_widths')[6],
                              self.config_manager.get_value('constants.common_dimensions.standard_heights')[5]),  # 5120x2160
            "super_ultrawide": (5760, self.config_manager.get_value('constants.common_dimensions.standard_heights')[3]),  # 5760x1080
            "portrait_4K": (self.config_manager.get_value('constants.common_dimensions.standard_heights')[5],
                            self.config_manager.get_value('constants.common_dimensions.4k')),  # 2160x3840
            "square_4K": (2880, 2880),
        }
    
    @property
    def PRESETS(self) -> Dict[str, Tuple[int, int]]:
        return self._get_presets()

    # Copy from lines 50-59 of resolution_manager.py
    def _get_sdxl_optimal_dimensions(self) -> List[Tuple[int, int]]:
        """Get SDXL optimal dimensions from config"""
        sdxl_base = self.config_manager.get_value('constants.common_dimensions.sdxl_base')
        sd_medium = self.config_manager.get_value('constants.common_dimensions.sd_medium')
        sdxl_high = self.config_manager.get_value('constants.common_dimensions.sdxl_high')
        return [
            (sdxl_base, sdxl_base),  # 1:1 (1024x1024)
            (1152, 896),  # 4:3.11
            (1216, 832),  # 3:2.05
            (1344, sd_medium),  # 16:9.14 (1344x768)
            (sdxl_high, 640),  # 2.4:1 (1536x640)
            (sd_medium, 1344),  # 9:16 (portrait) (768x1344)
            (896, 1152),  # 3:4 (portrait)
            (640, sdxl_high),  # 1:2.4 (tall portrait) (640x1536)
        ]
    
    @property
    def SDXL_OPTIMAL_DIMENSIONS(self) -> List[Tuple[int, int]]:
        return self._get_sdxl_optimal_dimensions()

    # Copy from lines 61-66
    def _get_flux_constraints(self) -> Dict[str, int]:
        """Get FLUX constraints from config"""
        alignment = self.config_manager.get_value('constants.dimensions.alignment_multiple')
        sdxl_base = self.config_manager.get_value('constants.common_dimensions.sdxl_base')
        return {
            "divisible_by": alignment * 2,  # 16 = 8 * 2
            "max_dimension": self.config_manager.get_value('constants.common_dimensions.tile_sizes')[4],  # 2048
            "optimal_pixels": sdxl_base * sdxl_base,  # 1024 * 1024
        }
    
    @property
    def FLUX_CONSTRAINTS(self) -> Dict[str, int]:
        return self._get_flux_constraints()

    def __init__(self, logger: Optional[logging.Logger] = None):
        # In ai-wallpaper: self.logger = get_logger(self.__class__.__name__)
        # For standalone, we adapt to use standard logging
        self.logger = logger or logging.getLogger(__name__)
        self.config_manager = ConfigurationManager()

    def round_to_multiple(
            self,
            value: int,
            multiple: Optional[int] = None) -> int:
        """Round value to nearest multiple"""
        if multiple is None:
            # Load from config
            try:
                from pathlib import Path

                from .config_loader import ConfigLoader
                config_dir = Path(__file__).parent.parent / "config"
                loader = ConfigLoader(config_dir)
                proc_config = loader.load_config_file("processing_params.yaml")
                multiple = proc_config.get(
                    'dimension_calculation', {}).get(
                    'dimension_multiple', 8)
            except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
                raise ValueError(
                    f"Failed to load dimension calculation configuration: {e}")
        return ((value + multiple // 2) // multiple) * multiple

    def get_optimal_generation_size(
        self, target_resolution: Tuple[int, int], model_type: str
    ) -> Tuple[int, int]:
        """
        Copy implementation from lines 70-93
        """
        target_config = ResolutionConfig.from_tuple(target_resolution)

        if model_type == "sdxl":
            return self._get_sdxl_optimal_size(target_config)
        elif model_type == "flux":
            return self._get_flux_optimal_size(target_config)
        elif model_type in ["dalle3", "gpt_image_1"]:
            # These models use fixed 1024x1024
            return (1024, 1024)
        else:
            # Default: use SDXL logic
            return self._get_sdxl_optimal_size(target_config)

    def _get_sdxl_optimal_size(
            self, target: ResolutionConfig) -> Tuple[int, int]:
        """Copy from lines 95-118"""
        # Find closest aspect ratio match
        best_match = None
        best_diff = float("inf")

        for dims in self.SDXL_OPTIMAL_DIMENSIONS:
            width, height = dims
            aspect = width / height
            diff = abs(aspect - target.aspect_ratio)

            if diff < best_diff:
                best_diff = diff
                best_match = dims

        if self.logger:
            self.logger.info(
                f"Target: {target.width}x{target.height} (aspect {target.aspect_ratio:.2f}) -> "
                f"Using SDXL trained size: {best_match[0]}x{best_match[1]}"
            )

        return best_match

    def _get_flux_optimal_size(
            self, target: ResolutionConfig) -> Tuple[int, int]:
        """Copy from lines 120-142"""
        # FLUX works best around 1MP
        scale = math.sqrt(
            self.FLUX_CONSTRAINTS["optimal_pixels"] /
            target.total_pixels)

        # Calculate dimensions
        width = int(target.width * scale)
        height = int(target.height * scale)

        # Ensure divisible by alignment multiple
        divisible_by = self.FLUX_CONSTRAINTS["divisible_by"]
        width = (width // divisible_by) * divisible_by
        height = (height // divisible_by) * divisible_by

        # Ensure within max dimension
        max_dim = self.FLUX_CONSTRAINTS["max_dimension"]
        if width > max_dim or height > max_dim:
            scale = max_dim / max(width, height)
            width = int(width * scale)
            height = int(height * scale)
            divisible_by = self.FLUX_CONSTRAINTS["divisible_by"]
            width = (width // divisible_by) * divisible_by
            height = (height // divisible_by) * divisible_by

        return (width, height)

    def calculate_progressive_strategy(
        self,
        current_size: Tuple[int, int],
        target_aspect: float,
        max_expansion_per_step: Optional[float] = None,
    ) -> List[Dict]:
        """
        Calculate progressive outpainting steps.
        Copy core logic from lines 223-405 of resolution_manager.py
        """
        if max_expansion_per_step is None:
            max_expansion_per_step = self.config_manager.get_value('constants.expansion.max_expansion_per_step')
        
        # Input validation
        if not current_size or len(current_size) != 2:
            raise ValueError(f"Invalid current_size: {current_size}")

        current_w, current_h = current_size

        if current_w <= 0 or current_h <= 0:
            raise ValueError(f"Invalid dimensions: {current_w}x{current_h}")

        if target_aspect <= 0:
            raise ValueError(f"Invalid target_aspect: {target_aspect}")

        current_aspect = current_w / current_h

        # If aspect change is minimal, return empty strategy
        aspect_threshold = self.config_manager.get_value('constants.expansion.aspect_change_threshold')
        if abs(current_aspect - target_aspect) < aspect_threshold:
            return []

        # Check if expansion is too extreme
        aspect_change_ratio = max(
            target_aspect / current_aspect, current_aspect / target_aspect
        )
        max_ratio = self.config_manager.get_value('constants.expansion.max_aspect_ratio_change')
        if aspect_change_ratio > max_ratio:
            raise ValueError(
                f"Aspect ratio change {
                    aspect_change_ratio:.1f}x exceeds maximum supported ratio of {max_ratio}x")

        steps = []

        # Determine expansion direction
        if target_aspect > current_aspect:
            # Expanding width
            target_w = int(current_h * target_aspect)
            target_h = current_h
            direction = "horizontal"

            # Calculate total expansion needed
            total_expansion = target_w / current_w

            # Progressive expansion logic
            temp_w = current_w
            temp_h = current_h

            # First step: Can be larger
            initial_factor = self.config_manager.get_value('constants.expansion.initial_expansion_factor')
            if total_expansion >= initial_factor:
                next_w = min(int(temp_w * initial_factor), target_w)
                steps.append(
                    {
                        "method": "outpaint",
                        "current_size": (
                            temp_w,
                            temp_h),
                        "target_size": (
                            next_w,
                            temp_h),
                        "expansion_ratio": next_w /
                        temp_w,
                        "direction": direction,
                        "step_type": "initial",
                        "description": f"Initial {initial_factor}x expansion: {temp_w}x{temp_h} → {next_w}x{temp_h}",
                    })
                temp_w = next_w

            # Middle steps
            step_num = 2
            middle_factor = self.config_manager.get_value('constants.expansion.middle_expansion_factor')
            completion = self.config_manager.get_value('constants.expansion.completion_threshold')
            while temp_w < target_w * completion:
                if temp_w * middle_factor <= target_w:
                    next_w = int(temp_w * middle_factor)
                else:
                    next_w = target_w

                steps.append(
                    {
                        "method": "outpaint",
                        "current_size": (
                            temp_w,
                            temp_h),
                        "target_size": (
                            next_w,
                            temp_h),
                        "expansion_ratio": next_w /
                        temp_w,
                        "direction": direction,
                        "step_type": "progressive",
                        "description": f"Step {step_num}: {temp_w}x{temp_h} → {next_w}x{temp_h}",
                    })
                temp_w = next_w
                step_num += 1

                if step_num > 10:
                    raise RuntimeError(
                        f"Too many expansion steps ({step_num})")

            # Final adjustment if needed
            if temp_w < target_w:
                steps.append(
                    {
                        "method": "outpaint",
                        "current_size": (
                            temp_w,
                            temp_h),
                        "target_size": (
                            target_w,
                            temp_h),
                        "expansion_ratio": target_w /
                        temp_w,
                        "direction": direction,
                        "step_type": "final",
                        "description": f"Final adjustment: {temp_w}x{temp_h} → {target_w}x{temp_h}",
                    })

        else:
            # Expanding height (similar logic)
            target_w = current_w
            target_h = int(current_w / target_aspect)
            direction = "vertical"

            # Similar progressive logic for vertical expansion
            temp_w = current_w
            temp_h = current_h

            # Calculate total expansion needed
            total_expansion = target_h / current_h

            # First step: Can be larger (2.0x)
            if total_expansion >= 2.0:
                next_h = min(int(temp_h * 2.0), target_h)
                steps.append(
                    {
                        "method": "outpaint",
                        "current_size": (
                            temp_w,
                            temp_h),
                        "target_size": (
                            temp_w,
                            next_h),
                        "expansion_ratio": next_h /
                        temp_h,
                        "direction": direction,
                        "step_type": "initial",
                        "description": f"Initial 2x expansion: {temp_w}x{temp_h} → {temp_w}x{next_h}",
                    })
                temp_h = next_h

            # Middle steps
            step_num = 2
            while temp_h < target_h * completion:
                if temp_h * middle_factor <= target_h:
                    next_h = int(temp_h * middle_factor)
                else:
                    next_h = target_h

                steps.append(
                    {
                        "method": "outpaint",
                        "current_size": (
                            temp_w,
                            temp_h),
                        "target_size": (
                            temp_w,
                            next_h),
                        "expansion_ratio": next_h /
                        temp_h,
                        "direction": direction,
                        "step_type": "progressive",
                        "description": f"Step {step_num}: {temp_w}x{temp_h} → {temp_w}x{next_h}",
                    })
                temp_h = next_h
                step_num += 1

                if step_num > 10:
                    raise RuntimeError(
                        f"Too many expansion steps ({step_num})")

            # Final adjustment if needed
            if temp_h < target_h:
                steps.append(
                    {
                        "method": "outpaint",
                        "current_size": (
                            temp_w,
                            temp_h),
                        "target_size": (
                            temp_w,
                            target_h),
                        "expansion_ratio": target_h /
                        temp_h,
                        "direction": direction,
                        "step_type": "final",
                        "description": f"Final adjustment: {temp_w}x{temp_h} → {temp_w}x{target_h}",
                    })

        return steps

    def calculate_sliding_window_strategy(
        self,
        current_size: Tuple[int, int],
        target_size: Tuple[int, int],
        window_size: int = 200,
        overlap_ratio: float = 0.8,
    ) -> List[Dict]:
        """
        Calculate sliding window (SWPO) strategy.
        Copy from calculate_sliding_window_strategy starting at line 415
        """
        # Input validation
        current_w, current_h = current_size
        target_w, target_h = target_size

        if window_size <= 0:
            raise ValueError(f"Invalid window_size: {window_size}")

        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(f"Invalid overlap_ratio: {overlap_ratio}")

        steps = []

        # Calculate step size (window minus overlap)
        step_size = int(window_size * (1.0 - overlap_ratio))

        # Determine if we need horizontal, vertical, or both expansions
        need_horizontal = target_w > current_w
        need_vertical = target_h > current_h

        if need_horizontal:
            # Calculate horizontal sliding windows
            temp_w = current_w
            temp_h = target_h if need_vertical else current_h
            window_num = 1

            while temp_w < target_w:
                # Calculate next window position
                next_w = min(temp_w + window_size, target_w)

                # Ensure we reach exactly target_w on last step
                if target_w - next_w < step_size:
                    next_w = target_w
                else:
                    # Round to multiple of 8 for SDXL compatibility
                    next_w = self.round_to_multiple(next_w, 8)

                steps.append(
                    {
                        "method": "sliding_window",
                        "current_size": (
                            temp_w,
                            temp_h),
                        "target_size": (
                            next_w,
                            temp_h),
                        "window_size": next_w -
                        temp_w,
                        "overlap_size": (
                            window_size -
                            step_size if window_num > 1 else 0),
                        "direction": "horizontal",
                        "window_number": window_num,
                        "description": f"H-Window {window_num}: {temp_w}x{temp_h} → {next_w}x{temp_h}",
                    })

                # Calculate actual step taken
                if window_num == 1:
                    actual_step = next_w - temp_w
                else:
                    actual_step = step_size
                temp_w = temp_w + actual_step
                window_num += 1

        # Similar logic for vertical windows if needed
        if need_vertical:
            # Calculate vertical sliding windows
            temp_w = target_w  # Use final width from horizontal expansion
            temp_h = (
                current_h if not need_horizontal else target_h
            )  # Use target height if horizontal was done
            window_num = 1

            while temp_h < target_h:
                # Calculate next window position
                next_h = min(temp_h + window_size, target_h)

                # Ensure we reach exactly target_h on last step
                if target_h - next_h < step_size:
                    next_h = target_h
                else:
                    # Round to multiple of 8 for SDXL compatibility
                    next_h = self.round_to_multiple(next_h, 8)

                steps.append(
                    {
                        "method": "sliding_window",
                        "current_size": (
                            temp_w,
                            temp_h),
                        "target_size": (
                            temp_w,
                            next_h),
                        "window_size": next_h -
                        temp_h,
                        "overlap_size": (
                            window_size -
                            step_size if window_num > 1 else 0),
                        "direction": "vertical",
                        "window_number": window_num,
                        "description": f"V-Window {window_num}: {temp_w}x{temp_h} → {temp_w}x{next_h}",
                    })

                # Calculate actual step taken
                if window_num == 1:
                    actual_step = next_h - temp_h
                else:
                    actual_step = step_size
                temp_h = temp_h + actual_step
                window_num += 1

        return steps
