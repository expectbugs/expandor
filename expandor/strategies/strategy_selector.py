"""
Strategy selection logic for Expandor
"""

import logging
from typing import Dict, Optional, Tuple

from . import STRATEGY_REGISTRY, get_strategy_class
from .base_strategy import BaseExpansionStrategy
from ..core.configuration_manager import ConfigurationManager

logger = logging.getLogger(__name__)


class StrategySelector:
    """Selects the appropriate expansion strategy based on requirements"""

    def __init__(self):
        self.logger = logger
        self._strategy_cache = {}
        
        # Load all strategy selection thresholds from config
        self.config_manager = ConfigurationManager()
        
        # Load thresholds
        self.aspect_ratio_extreme_threshold = self.config_manager.get_value(
            'image_processing.strategy_selection.aspect_ratio_extreme_threshold')
        self.expansion_extreme_threshold = self.config_manager.get_value(
            'image_processing.strategy_selection.expansion_extreme_threshold')
        self.vram_high_threshold = self.config_manager.get_value(
            'image_processing.strategy_selection.vram_high_threshold')
        self.vram_medium_threshold = self.config_manager.get_value(
            'image_processing.strategy_selection.vram_medium_threshold')
        self.vram_low_threshold = self.config_manager.get_value(
            'image_processing.strategy_selection.vram_low_threshold')
        self.vram_very_low_threshold = self.config_manager.get_value(
            'image_processing.strategy_selection.vram_very_low_threshold')
        self.expansion_small_threshold = self.config_manager.get_value(
            'image_processing.strategy_selection.expansion_small_threshold')
        self.expansion_moderate_threshold = self.config_manager.get_value(
            'image_processing.strategy_selection.expansion_moderate_threshold')

    def select_strategy(
        self,
        current_size: Tuple[int, int],
        target_size: Tuple[int, int],
        available_vram: float,
        user_preference: Optional[str] = None,
        quality_priority: bool = True,
    ) -> Tuple[str, str]:
        """
        Select the best strategy for the given expansion.

        Args:
            current_size: Current image dimensions (width, height)
            target_size: Target image dimensions (width, height)
            available_vram: Available VRAM in MB
            user_preference: User's preferred strategy or 'auto'
            quality_priority: Prioritize quality over speed

        Returns:
            Tuple of (strategy_name, reason_string)
        """
        # Calculate expansion metrics
        expansion_factor = self._calculate_expansion_factor(
            current_size, target_size)
        aspect_ratio_change = self._calculate_aspect_ratio_change(
            current_size, target_size
        )

        self.logger.info(
            f"Selecting strategy for {current_size} → {target_size} "
            f"(expansion: {
                expansion_factor:.2f}x, aspect change: {
                aspect_ratio_change:.2f})"
        )

        # User preference takes precedence if valid
        if user_preference and user_preference != "auto":
            if user_preference in STRATEGY_REGISTRY:
                reason = f"User requested: {user_preference}"
                self.logger.info(f"Using user preference: {user_preference}")
                return user_preference, reason
            else:
                self.logger.warning(
                    f"Invalid user preference: {user_preference}, using auto selection")

        # Auto selection based on requirements
        strategy_name, reason = self._auto_select_strategy(
            current_size,
            target_size,
            expansion_factor,
            aspect_ratio_change,
            available_vram,
            quality_priority,
        )

        self.logger.info(f"Selected strategy: {strategy_name} ({reason})")
        return strategy_name, reason

    def _auto_select_strategy(
        self,
        current_size: Tuple[int, int],
        target_size: Tuple[int, int],
        expansion_factor: float,
        aspect_ratio_change: float,
        available_vram: float,
        quality_priority: bool,
    ) -> Tuple[str, str]:
        """Auto-select the best strategy"""

        # Check for extreme aspect ratio changes
        if aspect_ratio_change > self.aspect_ratio_extreme_threshold and expansion_factor > self.expansion_extreme_threshold:
            if available_vram > self.vram_high_threshold:
                return "swpo", "Extreme aspect ratio change with sufficient VRAM"
            else:
                return (
                    "tiled_expansion",
                    "Extreme aspect ratio change with limited VRAM",
                )

        # Check for low VRAM scenarios
        if available_vram < self.vram_low_threshold:
            if available_vram < self.vram_very_low_threshold:
                return "cpu_offload", "Very low VRAM requires CPU offload"
            return "tiled_expansion", "Low VRAM requires tiled processing"

        # Check expansion factor
        if expansion_factor <= self.expansion_small_threshold:
            return "direct_upscale", "Small expansion factor allows direct upscaling"
        elif expansion_factor <= self.expansion_moderate_threshold:
            if quality_priority:
                return (
                    "progressive_outpaint",
                    "Moderate expansion with quality priority",
                )
            else:
                return "direct_upscale", "Moderate expansion with speed priority"
        else:
            # Large expansion
            if available_vram > self.vram_medium_threshold:
                return "progressive_outpaint", "Large expansion with sufficient VRAM"
            else:
                return (
                    "tiled_expansion",
                    "Large expansion requires memory-efficient processing",
                )

    def _calculate_expansion_factor(
        self, current_size: Tuple[int, int], target_size: Tuple[int, int]
    ) -> float:
        """Calculate the expansion factor"""
        current_pixels = current_size[0] * current_size[1]
        target_pixels = target_size[0] * target_size[1]
        return (target_pixels / current_pixels) ** 0.5

    def _calculate_aspect_ratio_change(
        self, current_size: Tuple[int, int], target_size: Tuple[int, int]
    ) -> float:
        """Calculate the aspect ratio change"""
        current_ratio = current_size[0] / current_size[1]
        target_ratio = target_size[0] / target_size[1]
        return abs(target_ratio - current_ratio)

    def get_strategy_instance(
        self, strategy_name: str, pipeline_adapter, config
    ) -> BaseExpansionStrategy:
        """Get or create a strategy instance"""

        # Check cache
        cache_key = f"{strategy_name}_{id(pipeline_adapter)}"
        if cache_key in self._strategy_cache:
            return self._strategy_cache[cache_key]

        # Create new instance
        try:
            strategy_class = get_strategy_class(strategy_name)
            strategy_instance = strategy_class(pipeline_adapter, config)

            # Cache it
            self._strategy_cache[cache_key] = strategy_instance

            return strategy_instance

        except Exception as e:
            self.logger.error(
                f"Failed to create strategy {strategy_name}: {e}")
            # Fall back to direct upscale
            if strategy_name != "direct_upscale":
                self.logger.warning("Falling back to direct_upscale strategy")
                return self.get_strategy_instance(
                    "direct_upscale", pipeline_adapter, config
                )
            raise

    def clear_cache(self):
        """Clear the strategy instance cache"""
        self._strategy_cache.clear()

    def register_strategy(
            self,
            name: str,
            module_path: str,
            priority: int = 50):
        """Register a custom strategy"""
        STRATEGY_REGISTRY[name] = module_path
        self.logger.info(f"Registered custom strategy: {name}")

    def list_strategies(self) -> Dict[str, str]:
        """List all available strategies"""
        return STRATEGY_REGISTRY.copy()
