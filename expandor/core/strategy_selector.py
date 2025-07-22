"""
Strategy selection with VRAM-aware multi-factor decision making
"""

import logging
import importlib
from typing import Dict, List, Optional, Any, Type, Tuple
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from .vram_manager import VRAMManager
from .exceptions import VRAMError, StrategyError
from ..strategies.base_strategy import BaseExpansionStrategy
from ..strategies import get_strategy_class

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
    
    def __init__(self, 
                 config: Dict[str, Any],
                 vram_manager: VRAMManager,
                 logger: Optional[logging.Logger] = None):
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
        
        # Load strategy configurations
        self.strategy_config = config.get('strategies', {})
        self.quality_presets = config.get('quality_presets', {})
        self.vram_strategies = config.get('vram_strategies', {})
        
        # Cache for initialized strategies
        self._strategy_cache = {}
    
    def _load_strategy_class(self, strategy_name: str) -> Type[BaseExpansionStrategy]:
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
        
        # Check if user specified a strategy override
        if hasattr(config, 'strategy_override') and config.strategy_override:
            self.logger.info(f"Using user-specified strategy: {config.strategy_override}")
            return config.strategy_override, "User override", metrics
        
        # VRAM-based strategy selection
        strategy_name = self._select_by_vram(metrics, config)
        
        # If VRAM selection didn't force a specific strategy, use smart selection
        if not strategy_name:
            strategy_name = self._select_by_metrics(metrics, config)
        
        self.logger.info(
            f"Selected {strategy_name} strategy (reason: {self.last_selection_reason})"
        )
        
        return strategy_name, self.last_selection_reason, metrics
    
    def select(self, config, dry_run: bool = False) -> BaseExpansionStrategy:
        """
        Select and instantiate optimal strategy
        
        Args:
            config: ExpandorConfig instance
            dry_run: If True, only calculate without caching
            
        Returns:
            Selected strategy instance
        """
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
        aspect_change = max(target_aspect/source_aspect, source_aspect/target_aspect)
        
        # Check available pipelines
        has_inpaint = hasattr(config, 'inpaint_pipeline') and config.inpaint_pipeline is not None
        has_refiner = hasattr(config, 'refiner_pipeline') and config.refiner_pipeline is not None
        has_img2img = hasattr(config, 'img2img_pipeline') and config.img2img_pipeline is not None
        
        # Check if extreme aspect ratio change
        is_extreme = aspect_change > self.strategy_config.get(
            'progressive_outpainting', {}
        ).get('aspect_ratio_thresholds', {}).get('extreme', 4.0)
        
        # Calculate VRAM requirements
        vram_available = self.vram_manager.get_available_vram() or 0
        vram_required = self.vram_manager.calculate_generation_vram(target_w, target_h)
        
        # Get model type from metadata
        model_type = 'unknown'
        if hasattr(config, 'source_metadata'):
            model_type = config.source_metadata.get('model', 'unknown')
        
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
            quality_preset=config.quality_preset
        )
    
    def _select_by_vram(self, metrics: SelectionMetrics, config) -> Optional[str]:
        """
        Select strategy based on VRAM constraints
        
        Returns strategy name if VRAM forces a specific choice, None otherwise
        """
        vram_thresholds = self.vram_strategies.get('thresholds', {
            'tiled_processing': 4000,
            'minimum': 2000
        })
        
        # No GPU available - must use CPU offload
        if metrics.vram_available_mb == 0:
            self.last_selection_reason = "No GPU available - CPU only mode"
            if getattr(config, 'allow_cpu_offload', True):
                return 'cpu_offload'
            else:
                raise VRAMError(
                    operation="strategy_selection",
                    required_mb=metrics.vram_required_mb,
                    available_mb=0,
                    message="No GPU available and CPU offload is disabled"
                )
        
        # Apply safety factor
        safety_factor = self.vram_strategies.get('safety_factor', 0.8)
        safe_vram = metrics.vram_available_mb * safety_factor
        
        # Check if we need VRAM-limited strategies
        if metrics.vram_required_mb > safe_vram:
            self.logger.warning(
                f"VRAM constrained: need {metrics.vram_required_mb:.0f}MB, "
                f"have {safe_vram:.0f}MB safe"
            )
            
            # Try tiled processing first
            if getattr(config, 'allow_tiled', True) and safe_vram >= vram_thresholds.get('tiled_processing', 4000):
                self.last_selection_reason = f"Insufficient VRAM for full processing ({metrics.vram_required_mb:.0f}MB > {safe_vram:.0f}MB)"
                return 'tiled_expansion'
            
            # Fall back to CPU offload
            elif getattr(config, 'allow_cpu_offload', True):
                self.last_selection_reason = "Insufficient VRAM even for tiled processing"
                return 'cpu_offload'
            
            else:
                # No VRAM-friendly strategies allowed
                raise VRAMError(
                    operation="strategy_selection",
                    required_mb=metrics.vram_required_mb,
                    available_mb=metrics.vram_available_mb,
                    message=f"Insufficient VRAM and all fallback strategies disabled. "
                           f"Need {metrics.vram_required_mb:.0f}MB, have {safe_vram:.0f}MB"
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
            swpo_config = self.strategy_config.get('swpo', {})
            if swpo_config.get('enabled', True):
                self.last_selection_reason = f"Extreme aspect ratio change ({metrics.aspect_change:.1f}x)"
                return 'swpo'
        
        # Check for significant aspect change needing progressive outpainting  
        if metrics.aspect_change > 1.5 and metrics.has_inpaint:
            prog_config = self.strategy_config.get('progressive_outpainting', {})
            if prog_config.get('enabled', True):
                self.last_selection_reason = f"Significant aspect ratio change ({metrics.aspect_change:.1f}x)"
                return 'progressive_outpaint'
        
        # Simple upscaling case - no aspect change, reasonable size increase
        if metrics.area_ratio < 4 and metrics.aspect_change < 1.1:
            self.last_selection_reason = f"Simple upscale ({metrics.area_ratio:.1f}x area increase)"
            return 'direct_upscale'
        
        # Massive upscaling might benefit from tiling even with sufficient VRAM
        if metrics.area_ratio > 16 and metrics.quality_preset == 'ultra':
            self.last_selection_reason = f"Massive upscale ({metrics.area_ratio:.1f}x) with ultra quality"
            return 'tiled_expansion'
        
        # Default to intelligent hybrid approach
        self.last_selection_reason = "Complex expansion - using adaptive hybrid strategy"
        return 'hybrid_adaptive'
    
    def _get_strategy(self, strategy_name: str, metrics: SelectionMetrics, dry_run: bool) -> BaseExpansionStrategy:
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
            
            # Get strategy-specific config
            strategy_config = self.strategy_config.get(strategy_name, {})
            
            # Create instance with config and metrics
            strategy = strategy_class(
                config=strategy_config,
                metrics=metrics,
                logger=self.logger
            )
            
            # Cache if not dry run
            if not dry_run:
                self._strategy_cache[strategy_name] = strategy
            
            return strategy
            
        except Exception as e:
            raise StrategyError(
                f"Failed to create {strategy_name} strategy: {str(e)}"
            )