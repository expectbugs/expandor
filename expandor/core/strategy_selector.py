"""
Strategy selection with VRAM-aware multi-factor decision making
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from PIL import Image

from ..strategies import get_strategy_class
from ..strategies.base_strategy import BaseExpansionStrategy
from .exceptions import StrategyError, VRAMError
from .vram_manager import VRAMManager


@dataclass
class SelectionMetrics:
    """Metrics used for strategy selection"""

    source_size: tuple
    target_size: tuple
    area_ratio: float
    aspect_change: float
    absolute_width_change: int
    absolute_height_change: int
    has_inpaint: bool
    has_refiner: bool
    has_img2img: bool
    is_extreme_ratio: bool
    model_type: str
    vram_available_mb: float
    vram_required_mb: float
    quality_preset: str


class StrategySelector:
    """Intelligent strategy selection with multi-factor decision making"""

    def __init__(
        self,
        config: Dict[str, Any],
        vram_manager: VRAMManager,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize strategy selector

        Args:
            config: Loaded configuration from YAML files
            vram_manager: VRAM manager instance
            logger: Logger instance
        """
        self.config = config
        self.vram_manager = vram_manager
        self.logger = logger or logging.getLogger(__name__)
        self.last_selection_reason = ""

        # Load strategy configurations using ConfigurationManager
        from .configuration_manager import ConfigurationManager
        self.config_manager = ConfigurationManager()
        
        self.strategy_config = self.config_manager.get_value("strategies")
        self.quality_presets = self.config_manager.get_value("quality_presets")
        self.vram_strategies = self.config_manager.get_value("vram")
        self.strategy_parameters = self.config_manager.get_value("strategies")

        # Cache for initialized strategies
        self._strategy_cache = {}

    def _load_strategy_class(
            self,
            strategy_name: str) -> Type[BaseExpansionStrategy]:
        """Load strategy class using the strategies module registry"""
        return get_strategy_class(strategy_name)

    def select_strategy(self, config) -> Tuple[str, str, SelectionMetrics]:
        """
        Select optimal strategy based on multiple factors

        Args:
            config: ExpandorConfig instance

        Returns:
            Tuple of (strategy_name, reason, metrics)
        """
        # Calculate decision metrics
        metrics = self._calculate_metrics(config)

        # Log metrics for debugging
        self.logger.debug(f"Selection metrics: {metrics.__dict__}")

        # Check if user specified a strategy
        user_strategy = None

        # First check for internal_strategy (maps user-friendly names)
        if hasattr(config, "internal_strategy"):
            internal = config.internal_strategy
            if internal != "auto":
                user_strategy = internal
        # Fall back to other checks
        elif hasattr(config, "effective_strategy"):
            user_strategy = config.effective_strategy
        elif hasattr(config, "strategy_override") and config.strategy_override:
            user_strategy = config.strategy_override
        elif hasattr(config, "strategy") and config.strategy != "auto":
            user_strategy = config.strategy

        if user_strategy:
            # Map user-friendly names to internal names
            strategy_mapping = {
                "direct": "direct_upscale",
                "progressive": "progressive_outpaint",
                "tiled": "tiled_expansion",
                "swpo": "swpo",
                "cpu_offload": "cpu_offload",
                "hybrid": "hybrid_adaptive",
            }

            # Use mapped name if available, otherwise use as-is
            mapped_strategy = strategy_mapping.get(
                user_strategy, user_strategy)

            self.logger.info(
                f"Using user-specified strategy: {user_strategy} (mapped to {mapped_strategy})")
            return mapped_strategy, "User specified", metrics

        # VRAM-based strategy selection
        strategy_name = self._select_by_vram(metrics, config)

        # If VRAM selection didn't force a specific strategy, use smart
        # selection
        if not strategy_name:
            strategy_name = self._select_by_metrics(metrics, config)

        self.logger.info(
            f"Selected {strategy_name} strategy (reason: {
                self.last_selection_reason})"
        )

        return strategy_name, self.last_selection_reason, metrics

    def select(self, config, dry_run: Optional[bool] = None) -> BaseExpansionStrategy:
        """
        Select and instantiate optimal strategy

        Args:
            config: ExpandorConfig instance
            dry_run: If True, only calculate without caching (None = use config default from 'constants.cli.default_dry_run')

        Returns:
            Selected strategy instance
        """
        if dry_run is None:
            dry_run = self.config_manager.get_value('constants.cli.default_dry_run')
        strategy_name, reason, metrics = self.select_strategy(config)

        # Get or create strategy instance
        strategy = self._get_strategy(strategy_name, metrics, dry_run)

        return strategy

    def get_selection_reason(self) -> str:
        """Get human-readable reason for last strategy selection"""
        return self.last_selection_reason

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        from ..strategies import STRATEGY_REGISTRY

        return list(STRATEGY_REGISTRY.keys())

    def clear_cache(self):
        """Clear strategy instance cache"""
        self._strategy_cache.clear()

    def _calculate_metrics(self, config) -> SelectionMetrics:
        """Calculate all metrics needed for strategy selection"""
        # Get source dimensions
        if isinstance(config.source_image, Path):
            with Image.open(config.source_image) as img:
                source_w, source_h = img.size
        else:
            source_w, source_h = config.source_image.size

        target_w, target_h = config.target_resolution

        # Calculate ratios
        source_aspect = source_w / source_h
        target_aspect = target_w / target_h
        area_ratio = (target_w * target_h) / (source_w * source_h)
        aspect_change = max(
            target_aspect / source_aspect, source_aspect / target_aspect
        )

        # Check available pipelines - with new adapter pattern, assume capabilities
        # In the future, this should check the adapter's capabilities
        has_inpaint = True  # Most adapters support inpainting
        has_refiner = False  # Refiner is optional
        has_img2img = True  # Most adapters support img2img

        # Check if extreme aspect ratio change
        # FAIL LOUD - get required configuration value
        extreme_threshold = self.config_manager.get_value(
            "strategies.progressive_outpaint.aspect_ratio_thresholds.extreme"
        )
        is_extreme = aspect_change > extreme_threshold

        # Calculate VRAM requirements
        vram_available = self.vram_manager.get_available_vram()
        if vram_available is None:
            raise RuntimeError(
                "FATAL: Failed to detect available VRAM!\n"
                "Cannot select appropriate strategy without VRAM information.\n"
                "Solutions:\n"
                "1. Specify --vram-limit explicitly\n"
                "2. Force a specific strategy with --strategy\n"
                "3. Check GPU drivers and CUDA availability"
            )
        if vram_available <= 0:
            self.logger.warning(f"No VRAM available ({vram_available}MB), will use CPU strategies")
        vram_required = self.vram_manager.calculate_generation_vram(
            target_w, target_h)

        # Get model type from metadata
        model_type = "unknown"
        if hasattr(config, "source_metadata") and config.source_metadata:
            # source_metadata is runtime data, so .get() with default is appropriate here
            model_type = config.source_metadata.get("model", "unknown")

        return SelectionMetrics(
            source_size=(source_w, source_h),
            target_size=(target_w, target_h),
            area_ratio=area_ratio,
            aspect_change=aspect_change,
            absolute_width_change=abs(target_w - source_w),
            absolute_height_change=abs(target_h - source_h),
            has_inpaint=has_inpaint,
            has_refiner=has_refiner,
            has_img2img=has_img2img,
            is_extreme_ratio=is_extreme,
            model_type=model_type,
            vram_available_mb=vram_available,
            vram_required_mb=vram_required,
            quality_preset=config.quality_preset,
        )

    def _select_by_vram(
            self,
            metrics: SelectionMetrics,
            config) -> Optional[str]:
        """
        Select strategy based on VRAM constraints

        Returns strategy name if VRAM forces a specific choice, None otherwise
        """
        # FAIL LOUD - get required VRAM thresholds
        vram_thresholds = self.config_manager.get_value("vram.thresholds")

        # No GPU available - must use CPU offload
        if metrics.vram_available_mb == 0:
            self.last_selection_reason = "No GPU available - CPU only mode"
            if getattr(config, "allow_cpu_offload", True):
                return "cpu_offload"
            else:
                raise VRAMError(
                    operation="strategy_selection",
                    required_mb=metrics.vram_required_mb,
                    available_mb=0,
                    message="No GPU available and CPU offload is disabled",
                )

        # Apply safety factor
        # FAIL LOUD - get required safety factor
        safety_factor = self.config_manager.get_value("vram.safety_factor")
        safe_vram = metrics.vram_available_mb * safety_factor

        # Check if we need VRAM-limited strategies
        if metrics.vram_required_mb > safe_vram:
            self.logger.warning(
                f"VRAM constrained: need {metrics.vram_required_mb:.0f}MB, "
                f"have {safe_vram:.0f}MB safe"
            )

            # Try tiled processing first
            # FAIL LOUD - get required threshold
            tiled_threshold = self.config_manager.get_value("vram.thresholds.tiled_processing")
            if getattr(
                config, "allow_tiled", True
            ) and safe_vram >= tiled_threshold:
                self.last_selection_reason = f"Insufficient VRAM for full processing ({
                    metrics.vram_required_mb:.0f}MB > {
                    safe_vram:.0f}MB)"
                return "tiled_expansion"

            # Fall back to CPU offload
            elif getattr(config, "allow_cpu_offload", True):
                self.last_selection_reason = (
                    "Insufficient VRAM even for tiled processing"
                )
                return "cpu_offload"

            else:
                # No VRAM-friendly strategies allowed
                raise VRAMError(
                    operation="strategy_selection",
                    required_mb=metrics.vram_required_mb,
                    available_mb=metrics.vram_available_mb,
                    message=f"Insufficient VRAM and all fallback strategies disabled. " f"Need {
                        metrics.vram_required_mb:.0f}MB, have {
                        safe_vram:.0f}MB",
                )

        # VRAM is sufficient - let metrics decide
        return None

    def _select_by_metrics(self, metrics: SelectionMetrics, config) -> str:
        """
        Select strategy based on image metrics and available pipelines

        This implements the smart multi-factor decision logic
        """
        # Check for extreme aspect ratio change requiring SWPO
        if metrics.is_extreme_ratio and metrics.has_inpaint:
            # Check if SWPO is enabled
            if "swpo" not in self.strategy_config:
                raise ValueError(
                    "FATAL: swpo configuration not found in strategies!\n"
                    "This is required for extreme aspect ratio handling."
                )
            swpo_config = self.strategy_config["swpo"]
            
            # Check if enabled - default to True if not specified
            if "enabled" not in swpo_config:
                swpo_enabled = True
            else:
                swpo_enabled = swpo_config["enabled"]
            
            if swpo_enabled:
                self.last_selection_reason = f"Extreme aspect ratio change ({
                    metrics.aspect_change:.1f}x)"
                return "swpo"

        # Get strategy selection thresholds from config
        try:
            selection_config = self.config_manager.get_value("image_processing.strategy_selection")
            aspect_threshold = selection_config["simple_upscale_threshold"]
            moderate_expansion = selection_config["expansion_moderate_threshold"]
            # For aspect change tolerance, use a smaller value (this is a tolerance, not a threshold)
            aspect_tolerance = 1.1  # This is a tolerance value, kept as algorithmic constant
            # For massive upscale, calculate from moderate expansion
            massive_expansion = moderate_expansion * 4  # 16 if moderate is 4
        except (KeyError, ValueError) as e:
            # FAIL LOUD - configuration must exist
            raise ValueError(
                f"Strategy selection configuration not found!\n{str(e)}\n"
                f"Required: image_processing.strategy_selection with thresholds"
            )

        # Check for significant aspect change needing progressive outpainting
        if metrics.aspect_change > aspect_threshold and metrics.has_inpaint:
            if "progressive_outpainting" not in self.strategy_config:
                raise ValueError(
                    "FATAL: progressive_outpainting configuration not found in strategies!\n"
                    "This is required for aspect ratio change handling."
                )
            prog_config = self.strategy_config["progressive_outpainting"]
            
            # Check if enabled - default to True if not specified
            if "enabled" not in prog_config:
                prog_enabled = True
            else:
                prog_enabled = prog_config["enabled"]
            
            if prog_enabled:
                self.last_selection_reason = f"Significant aspect ratio change ({
                    metrics.aspect_change:.1f}x)"
                return "progressive_outpaint"

        # Simple upscaling case - no aspect change, reasonable size increase
        if metrics.area_ratio < moderate_expansion and metrics.aspect_change < aspect_tolerance:
            self.last_selection_reason = (
                f"Simple upscale ({metrics.area_ratio:.1f}x area increase)"
            )
            return "direct_upscale"

        # Massive upscaling might benefit from tiling even with sufficient VRAM
        if metrics.area_ratio > massive_expansion and metrics.quality_preset == "ultra":
            self.last_selection_reason = (
                f"Massive upscale ({metrics.area_ratio:.1f}x) with ultra quality"
            )
            return "tiled_expansion"

        # Default to intelligent hybrid approach
        self.last_selection_reason = (
            "Complex expansion - using adaptive hybrid strategy"
        )
        return "hybrid_adaptive"

    def _get_strategy(
        self, strategy_name: str, metrics: SelectionMetrics, dry_run: bool
    ) -> BaseExpansionStrategy:
        """
        Get or create strategy instance

        Args:
            strategy_name: Name of strategy to create
            metrics: Selection metrics
            dry_run: If True, create new instance without caching

        Returns:
            Strategy instance

        Raises:
            StrategyError: If strategy not found or initialization fails
        """
        # Check cache if not dry run
        if not dry_run and strategy_name in self._strategy_cache:
            return self._strategy_cache[strategy_name]

        # Create new instance
        try:
            strategy_class = self._load_strategy_class(strategy_name)

            # Get strategy-specific config from strategies.yaml
            if strategy_name not in self.strategy_config:
                raise ValueError(
                    f"FATAL: Strategy '{strategy_name}' not found in strategies configuration!\n"
                    f"Available strategies: {list(self.strategy_config.keys())}\n"
                    f"Please add configuration for {strategy_name} in master_defaults.yaml"
                )
            strategy_config = self.strategy_config[strategy_name]
            
            # Merge with parameters from strategy_parameters.yaml if available
            if hasattr(self, 'strategy_parameters') and strategy_name in self.strategy_parameters:
                strategy_params = self.strategy_parameters[strategy_name]
                # Merge parameters into the config's parameters section
                if 'parameters' not in strategy_config:
                    strategy_config['parameters'] = {}
                strategy_config['parameters'].update(strategy_params)

            # Create instance with config and metrics
            strategy = strategy_class(
                config=strategy_config, metrics=metrics, logger=self.logger
            )

            # Cache if not dry run
            if not dry_run:
                self._strategy_cache[strategy_name] = strategy

            return strategy

        except Exception as e:
            raise StrategyError(
                f"Failed to create {strategy_name} strategy: {
                    str(e)}"
            )
