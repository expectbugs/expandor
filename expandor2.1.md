# Expandor Phase 2 Step 1: Implement Base Architecture

## Overview

This step implements the core foundation of Expandor: the strategy selector with VRAM awareness, pipeline orchestrator with fallback chains, metadata tracking system, and boundary tracking for seam detection. Each component must be bulletproof with no silent failures.

## Step 2.1.1: Create Core Expandor Class

### Location: `expandor/core/expandor.py`

Replace the placeholder implementation with the full core class:

```python
"""
Main Expandor class - Universal image expansion orchestrator
"""

import logging
import time
import json
import pkg_resources
from typing import Optional, Dict, Any, List
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from .config import ExpandorConfig
from .exceptions import ExpandorError, VRAMError, StrategyError, QualityError
from .strategy_selector import StrategySelector
from .pipeline_orchestrator import PipelineOrchestrator
from .metadata_tracker import MetadataTracker
from .boundary_tracker import BoundaryTracker
from .vram_manager import VRAMManager
from .result import ExpandorResult, StageResult
from ..utils.config_loader import ConfigLoader
from ..utils.logging_utils import setup_logger

class Expandor:
    """Universal image expansion and adaptation system"""
    
    def __init__(self, config_path: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize Expandor with optional config override
        
        Args:
            config_path: Optional path to custom config directory
            logger: Optional logger instance (creates one if not provided)
        """
        # Setup logging first - CRITICAL for fail-loud philosophy
        self.logger = logger or setup_logger("expandor", level=logging.INFO)
        self.logger.info("Initializing Expandor Universal Image Resolution Adaptation System")
        
        # Load configuration
        try:
            self.config_loader = ConfigLoader(config_path or self._get_default_config_path())
            self.config = self.config_loader.load_all_configs()
            self.logger.info(f"Loaded configuration from: {self.config_loader.config_dir}")
        except Exception as e:
            raise ExpandorError(
                f"Failed to load configuration: {str(e)}\n"
                f"Ensure config files exist and are valid YAML",
                stage="initialization"
            )
        
        # Initialize core components
        try:
            # VRAM Manager - Must be first to check hardware
            self.vram_manager = VRAMManager(self.logger)
            available_vram = self.vram_manager.get_available_vram()
            if available_vram:
                self.logger.info(f"VRAM available: {available_vram:.0f}MB")
            else:
                self.logger.warning("No CUDA device detected - CPU mode only")
            
            # Strategy Selector with config and VRAM awareness
            self.strategy_selector = StrategySelector(
                config=self.config,
                vram_manager=self.vram_manager,
                logger=self.logger
            )
            
            # Pipeline Orchestrator for executing strategies
            self.orchestrator = PipelineOrchestrator(
                config=self.config,
                logger=self.logger
            )
            
            # Metadata Tracker for operation history
            self.metadata_tracker = MetadataTracker(self.logger)
            
            # Boundary Tracker for seam detection
            self.boundary_tracker = BoundaryTracker(self.logger)
            
            # Pipeline registry for external models
            self.pipeline_registry = {}
            
            # Cache for loaded strategies
            self._strategy_cache = {}
            
        except Exception as e:
            raise ExpandorError(
                f"Failed to initialize core components: {str(e)}",
                stage="initialization"
            )
        
        self.logger.info("Expandor initialization complete")
    
    def expand(self, config: ExpandorConfig) -> ExpandorResult:
        """
        Main expansion method with comprehensive error handling
        
        This is the primary entry point for all image expansion operations.
        It orchestrates the entire pipeline from input validation through
        quality verification.
        
        Args:
            config: Expansion configuration with all parameters
            
        Returns:
            ExpandorResult with image path, metadata, and quality metrics
            
        Raises:
            ExpandorError: On any unrecoverable error (fail loud)
        """
        # Start timing immediately
        operation_start = time.time()
        
        # Initialize tracking
        self.metadata_tracker.start_operation(config)
        self.boundary_tracker.reset()
        
        # Validate inputs - fail fast on bad config
        try:
            self._validate_config(config)
        except Exception as e:
            raise ExpandorError(
                f"Configuration validation failed: {str(e)}",
                stage="validation",
                partial_result={"config": str(config)}
            )
        
        # Log operation start
        self.logger.info(
            f"Starting expansion: {config.source_image} -> {config.target_resolution} "
            f"[{config.quality_preset} quality]"
        )
        
        try:
            # Pre-execution setup
            self._prepare_workspace(config)
            
            # Register any provided pipelines
            if config.inpaint_pipeline:
                self.register_pipeline("inpaint", config.inpaint_pipeline)
            if config.refiner_pipeline:
                self.register_pipeline("refiner", config.refiner_pipeline)
            if config.img2img_pipeline:
                self.register_pipeline("img2img", config.img2img_pipeline)
            
            # Select strategy with VRAM awareness
            self.logger.info("Selecting optimal expansion strategy...")
            strategy = self.strategy_selector.select(config)
            self.metadata_tracker.record_event("strategy_selected", {
                "strategy": strategy.__class__.__name__,
                "reason": self.strategy_selector.get_selection_reason()
            })
            
            # Execute with fallback chain
            self.logger.info(f"Executing {strategy.__class__.__name__}...")
            result = self.orchestrator.execute(
                strategy=strategy,
                config=config,
                metadata_tracker=self.metadata_tracker,
                boundary_tracker=self.boundary_tracker
            )
            
            # Post-execution validation and repair
            self.logger.info("Validating quality and checking for artifacts...")
            result = self._validate_and_repair(result, config)
            
            # Calculate final metrics
            result.total_duration_seconds = time.time() - operation_start
            result.vram_peak_mb = self.vram_manager.get_peak_usage()
            
            # Update metadata with complete operation info
            result.metadata.update({
                "expandor_version": "0.1.0",
                "config_snapshot": config.__dict__,
                "operation_log": self.metadata_tracker.get_operation_log(),
                "boundary_positions": self.boundary_tracker.get_all_boundaries()
            })
            
            # Save metadata alongside image
            self._save_metadata(result)
            
            # Log success
            self.logger.info(
                f"Expansion complete: {result.size} in {result.total_duration_seconds:.1f}s "
                f"[{result.strategy_used}, {result.seams_detected} seams, "
                f"{result.quality_score:.2f} quality]"
            )
            
            return result
            
        except VRAMError as e:
            # VRAM errors are special - log GPU state
            self._log_vram_error(e)
            raise
            
        except Exception as e:
            # Get partial result for debugging
            partial_result = self.metadata_tracker.get_partial_result()
            
            # Fail loud with comprehensive error info
            raise ExpandorError(
                f"Expansion failed at {self.metadata_tracker.current_stage}: {str(e)}",
                stage=self.metadata_tracker.current_stage,
                partial_result=partial_result
            )
        
        finally:
            # Cleanup temporary files if not saving stages
            if not config.save_stages:
                self._cleanup_temp_files()
    
    def register_pipeline(self, name: str, pipeline: Any):
        """
        Register a pipeline for strategies to use
        
        Args:
            name: Pipeline identifier (e.g., 'inpaint', 'refiner', 'img2img')
            pipeline: Pipeline instance (diffusers pipeline or mock)
                     Expected interface for pipelines:
                     - __call__ method that accepts:
                       - prompt (str)
                       - image (PIL.Image)
                       - For inpaint: mask_image (PIL.Image)
                       - Optional: strength, guidance_scale, num_inference_steps
                     - Returns object with .images[0] containing result
        """
        self.pipeline_registry[name] = pipeline
        self.logger.debug(f"Registered pipeline: {name}")
        
        # Make pipeline available to orchestrator
        self.orchestrator.register_pipeline(name, pipeline)
    
    def estimate_vram(self, config: ExpandorConfig) -> Dict[str, float]:
        """
        Estimate VRAM requirements before execution
        
        Args:
            config: Expansion configuration
            
        Returns:
            Dictionary with VRAM estimates for each stage
        """
        # Get basic requirement
        base_estimate = self.vram_manager.estimate_requirement(config)
        
        # Get strategy-specific requirements
        strategy = self.strategy_selector.select(config, dry_run=True)
        strategy_estimate = strategy.estimate_vram(config)
        
        return {
            "total_required_mb": max(
                base_estimate["total_with_buffer_mb"],
                strategy_estimate.get("peak_vram_mb", 0)
            ),
            "available_mb": self.vram_manager.get_available_vram() or 0,
            "base_estimate": base_estimate,
            "strategy_estimate": strategy_estimate,
            "recommended_strategy": strategy.__class__.__name__
        }
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available expansion strategies"""
        return self.strategy_selector.get_available_strategies()
    
    def _validate_config(self, config: ExpandorConfig):
        """
        Validate configuration thoroughly
        
        Checks all parameters for validity and compatibility
        """
        # Basic parameter validation
        if not config.prompt or len(config.prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        
        if config.seed < 0:
            raise ValueError(f"Invalid seed: {config.seed} (must be non-negative)")
        
        # Resolution validation
        target_w, target_h = config.target_resolution
        if target_w <= 0 or target_h <= 0:
            raise ValueError(f"Invalid target resolution: {target_w}x{target_h}")
        
        if target_w > 65536 or target_h > 65536:
            raise ValueError(f"Target resolution too large: {target_w}x{target_h} (max 65536)")
        
        # Source image validation
        if isinstance(config.source_image, Path):
            if not config.source_image.exists():
                raise FileNotFoundError(f"Source image not found: {config.source_image}")
            
            # Try to open and validate
            try:
                with Image.open(config.source_image) as img:
                    if img.size[0] <= 0 or img.size[1] <= 0:
                        raise ValueError(f"Invalid source image dimensions: {img.size}")
            except Exception as e:
                raise ValueError(f"Cannot read source image: {str(e)}")
        
        elif isinstance(config.source_image, Image.Image):
            if config.source_image.size[0] <= 0 or config.source_image.size[1] <= 0:
                raise ValueError(f"Invalid source image dimensions: {config.source_image.size}")
        else:
            raise TypeError(
                f"source_image must be Path or PIL.Image, got {type(config.source_image)}"
            )
        
        # Quality preset validation
        available_presets = self.config.get("quality_presets", {}).keys()
        if config.quality_preset not in available_presets:
            raise ValueError(
                f"Invalid quality preset: {config.quality_preset}. "
                f"Available: {', '.join(available_presets)}"
            )
        
        # VRAM limit validation
        if config.vram_limit_mb is not None and config.vram_limit_mb <= 0:
            raise ValueError(f"Invalid VRAM limit: {config.vram_limit_mb}MB")
        
        # Parameter range validation
        if not 0.0 <= config.overlap_ratio < 1.0:
            raise ValueError(f"Invalid overlap_ratio: {config.overlap_ratio} (must be 0.0-0.99)")
        
        if not 0.0 <= config.min_strength <= 1.0:
            raise ValueError(f"Invalid min_strength: {config.min_strength} (must be 0.0-1.0)")
        
        if not 0.0 <= config.max_strength <= 1.0:
            raise ValueError(f"Invalid max_strength: {config.max_strength} (must be 0.0-1.0)")
        
        if config.min_strength > config.max_strength:
            raise ValueError(
                f"min_strength ({config.min_strength}) cannot be greater than "
                f"max_strength ({config.max_strength})"
            )
    
    def _prepare_workspace(self, config: ExpandorConfig):
        """Prepare workspace for operation"""
        # Create temp directory
        self.temp_dir = Path("temp") / f"expandor_{int(time.time())}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set stage directory if saving stages
        if config.save_stages:
            if config.stage_dir:
                self.stage_dir = config.stage_dir
            else:
                self.stage_dir = self.temp_dir / "stages"
            self.stage_dir.mkdir(parents=True, exist_ok=True)
            
            # Update config to include stage dir
            config.stage_dir = self.stage_dir
    
    def _validate_and_repair(self, result: ExpandorResult, config: ExpandorConfig) -> ExpandorResult:
        """
        Validate quality and repair artifacts if needed
        
        This is where Expandor's zero-tolerance for artifacts is enforced
        """
        if config.artifact_detection_level == "disabled":
            self.logger.info("Artifact detection disabled by config")
            return result
        
        # Run quality validation
        from ..processors.quality_validator import QualityValidator
        validator = QualityValidator(
            config=self.config,
            logger=self.logger
        )
        
        validation_result = validator.validate(
            image_path=result.image_path,
            metadata=result.metadata,
            detection_level=config.artifact_detection_level
        )
        
        result.seams_detected = validation_result["seam_count"]
        result.quality_score = validation_result["quality_score"]
        
        # If artifacts found and repair enabled
        if validation_result["issues_found"] and config.quality_preset != "fast":
            self.logger.warning(
                f"Found {validation_result['seam_count']} seams, "
                f"attempting repair..."
            )
            
            # Get repair strategy
            repair_config = self.config["quality_presets"][config.quality_preset]
            max_attempts = repair_config.get("seam_repair_attempts", 3)
            
            # Attempt repairs
            from ..processors.seam_repair import SeamRepairProcessor
            repairer = SeamRepairProcessor(
                pipelines=self.pipeline_registry,
                logger=self.logger
            )
            
            for attempt in range(max_attempts):
                self.logger.info(f"Repair attempt {attempt + 1}/{max_attempts}")
                
                repair_result = repairer.repair_seams(
                    image_path=result.image_path,
                    artifact_mask=validation_result["mask"],
                    prompt=config.prompt,
                    metadata=result.metadata
                )
                
                # Re-validate
                validation_result = validator.validate(
                    image_path=repair_result["image_path"],
                    metadata=repair_result["metadata"],
                    detection_level=config.artifact_detection_level
                )
                
                result.artifacts_fixed += repair_result["seams_repaired"]
                result.image_path = repair_result["image_path"]
                result.seams_detected = validation_result["seam_count"]
                
                if not validation_result["issues_found"]:
                    self.logger.info("All artifacts successfully repaired!")
                    break
            else:
                self.logger.warning(
                    f"Could not repair all artifacts after {max_attempts} attempts. "
                    f"Remaining seams: {validation_result['seam_count']}"
                )
        
        return result
    
    def _save_metadata(self, result: ExpandorResult):
        """Save metadata JSON alongside result image"""
        metadata_path = result.image_path.with_suffix('.json')
        
        # Convert result to serializable dict
        metadata = {
            "image_path": str(result.image_path),
            "size": result.size,
            "success": result.success,
            "stages": [stage.to_dict() for stage in result.stages],
            "boundaries": result.boundaries,
            "seams_detected": result.seams_detected,
            "artifacts_fixed": result.artifacts_fixed,
            "refinement_passes": result.refinement_passes,
            "quality_score": result.quality_score,
            "vram_peak_mb": result.vram_peak_mb,
            "total_duration_seconds": result.total_duration_seconds,
            "strategy_used": result.strategy_used,
            "fallback_count": result.fallback_count,
            "metadata": result.metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.debug(f"Saved metadata to: {metadata_path}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.debug(f"Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                self.logger.warning(f"Could not clean up temp files: {e}")
    
    def _log_vram_error(self, error: VRAMError):
        """Log detailed VRAM error information"""
        self.logger.error("=" * 60)
        self.logger.error("VRAM ERROR - Insufficient GPU Memory")
        self.logger.error("=" * 60)
        
        # Get current VRAM state
        vram_state = self.vram_manager.get_detailed_vram_state()
        
        self.logger.error(f"Required VRAM: {error.required_mb:.0f}MB")
        self.logger.error(f"Available VRAM: {vram_state['free_mb']:.0f}MB")
        self.logger.error(f"Total VRAM: {vram_state['total_mb']:.0f}MB")
        self.logger.error(f"Used VRAM: {vram_state['used_mb']:.0f}MB")
        
        if vram_state.get('processes'):
            self.logger.error("\nProcesses using VRAM:")
            for proc in vram_state['processes']:
                self.logger.error(f"  - {proc['name']}: {proc['memory_mb']:.0f}MB")
        
        self.logger.error("\nSuggestions:")
        self.logger.error("  1. Close other GPU applications")
        self.logger.error("  2. Use a lower quality preset")
        self.logger.error("  3. Enable tiled processing")
        self.logger.error("  4. Reduce target resolution")
        self.logger.error("=" * 60)
    
    def _get_default_config_path(self) -> Path:
        """Get default config directory path"""
        # Check common locations
        module_dir = Path(__file__).parent.parent
        config_dir = module_dir / "config"
        
        if config_dir.exists():
            return config_dir
        
        # Fallback to package data
        import pkg_resources
        try:
            config_dir = Path(pkg_resources.resource_filename('expandor', 'config'))
            if config_dir.exists():
                return config_dir
        except:
            pass
        
        raise ExpandorError(
            "Cannot find configuration directory. "
            "Please specify config_path or ensure package is properly installed."
        )
    
    def clear_caches(self):
        """
        Clear all internal caches to free memory
        
        Useful for long-running processes or after processing large images
        """
        # Clear strategy cache in selector
        if hasattr(self, 'strategy_selector'):
            self.strategy_selector.clear_cache()
        
        # Clear any cached strategies
        if hasattr(self, '_strategy_cache'):
            self._strategy_cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Cleared all caches and freed memory")
```

### Critical Implementation Notes:

1. **Fail-Loud Philosophy**: Every error is explicit with detailed context
2. **VRAM Awareness**: Checks GPU state before and during operations
3. **Metadata Tracking**: Records every decision and operation
4. **Quality Validation**: Zero tolerance for artifacts
5. **Pipeline Registry**: Flexible integration with any image generation model
6. **Comprehensive Logging**: Every significant action is logged

## Step 2.1.2: Create Strategy Selector

### Location: `expandor/core/strategy_selector.py`

```python
"""
Intelligent strategy selection with VRAM awareness
"""

import logging
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from .config import ExpandorConfig
from .vram_manager import VRAMManager
from .exceptions import VRAMError, StrategyError
# Strategies will be lazy loaded to avoid circular imports

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
    
    # Strategy registry - maps strategy names to import paths
    STRATEGY_REGISTRY = {
        'progressive_outpaint': 'expandor.strategies.progressive_outpaint.ProgressiveOutpaintStrategy',
        'swpo': 'expandor.strategies.swpo_strategy.SWPOStrategy',
        'direct_upscale': 'expandor.strategies.direct_upscale.DirectUpscaleStrategy',
        'tiled_expansion': 'expandor.strategies.tiled_expansion.TiledExpansionStrategy',
        'cpu_offload': 'expandor.strategies.cpu_offload.CPUOffloadStrategy',
        'hybrid_adaptive': 'expandor.strategies.hybrid_adaptive.HybridAdaptiveStrategy'
    }
    
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
        # Cache for loaded strategy classes
        self._strategy_class_cache = {}
    
    def _load_strategy_class(self, strategy_name: str):
        """Lazy load strategy class to avoid circular imports"""
        if strategy_name not in self._strategy_class_cache:
            if strategy_name not in self.STRATEGY_REGISTRY:
                raise StrategyError(f"Unknown strategy: {strategy_name}")
            
            # Get the import path
            import_path = self.STRATEGY_REGISTRY[strategy_name]
            module_path, class_name = import_path.rsplit('.', 1)
            
            # Dynamic import
            import importlib
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, class_name)
            
            self._strategy_class_cache[strategy_name] = strategy_class
        
        return self._strategy_class_cache[strategy_name]
    
    def select(self, config: ExpandorConfig, dry_run: bool = False) -> BaseExpansionStrategy:
        """
        Select optimal strategy based on multiple factors
        
        Args:
            config: Expansion configuration
            dry_run: If True, only calculate without initializing strategy
            
        Returns:
            Selected strategy instance
            
        Raises:
            StrategyError: If no suitable strategy can be found
        """
        # Calculate decision metrics
        metrics = self._calculate_metrics(config)
        
        # Log metrics for debugging
        self.logger.debug(f"Selection metrics: {metrics.__dict__}")
        
        # Check if user specified a strategy override
        if config.strategy_override:
            self.logger.info(f"Using user-specified strategy: {config.strategy_override}")
            return self._get_strategy(config.strategy_override, metrics, dry_run)
        
        # VRAM-based strategy selection
        strategy_name = self._select_by_vram(metrics, config)
        
        # If VRAM selection didn't force a specific strategy, use smart selection
        if not strategy_name:
            strategy_name = self._select_by_metrics(metrics, config)
        
        # Get or create strategy instance
        strategy = self._get_strategy(strategy_name, metrics, dry_run)
        
        self.logger.info(
            f"Selected {strategy_name} strategy (reason: {self.last_selection_reason})"
        )
        
        return strategy
    
    def get_selection_reason(self) -> str:
        """Get human-readable reason for last strategy selection"""
        return self.last_selection_reason
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self.STRATEGY_REGISTRY.keys())
    
    def _calculate_metrics(self, config: ExpandorConfig) -> SelectionMetrics:
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
        has_inpaint = config.inpaint_pipeline is not None
        has_refiner = config.refiner_pipeline is not None
        has_img2img = config.img2img_pipeline is not None
        
        # Check if extreme aspect ratio change
        is_extreme = aspect_change > self.strategy_config.get(
            'progressive_outpainting', {}
        ).get('aspect_ratio_thresholds', {}).get('extreme', 4.0)
        
        # Calculate VRAM requirements
        vram_available = self.vram_manager.get_available_vram() or 0
        vram_calc = self.vram_manager.estimate_requirement({
            'target_resolution': (target_w, target_h),
            'source_metadata': config.source_metadata
        })
        vram_required = vram_calc['total_with_buffer_mb']
        
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
            model_type=config.source_metadata.get('model', 'unknown').lower(),
            vram_available_mb=vram_available,
            vram_required_mb=vram_required,
            quality_preset=config.quality_preset
        )
    
    def _select_by_vram(self, metrics: SelectionMetrics, config: ExpandorConfig) -> Optional[str]:
        """
        Select strategy based on VRAM constraints
        
        Returns strategy name if VRAM forces a specific choice, None otherwise
        """
        vram_thresholds = self.vram_strategies.get('thresholds', {})
        
        # No GPU available - must use CPU offload
        if metrics.vram_available_mb == 0:
            self.last_selection_reason = "No GPU available - CPU only mode"
            if config.allow_cpu_offload:
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
            if config.allow_tiled and safe_vram >= vram_thresholds.get('tiled_processing', 4000):
                self.last_selection_reason = f"Insufficient VRAM for full processing ({metrics.vram_required_mb:.0f}MB > {safe_vram:.0f}MB)"
                return 'tiled_expansion'
            
            # Fall back to CPU offload
            elif config.allow_cpu_offload:
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
    
    def _select_by_metrics(self, metrics: SelectionMetrics, config: ExpandorConfig) -> str:
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
        if metrics.area_ratio > 16 and config.quality_preset == 'ultra':
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
        # Validate strategy name
        if strategy_name not in self.STRATEGY_REGISTRY:
            raise StrategyError(
                f"Unknown strategy: {strategy_name}. "
                f"Available: {list(self.STRATEGY_REGISTRY.keys())}"
            )
        
        # Check cache if not dry run
        if not dry_run and strategy_name in self._strategy_cache:
            return self._strategy_cache[strategy_name]
        
        # Create new instance
        try:
            strategy_class = self._load_strategy_class(strategy_name)
            
            # Get strategy-specific config
            strategy_config = self.strategy_config.get(strategy_name, {})
            
            # Create instance with config
            strategy = strategy_class(
                config=strategy_config,
                logger=self.logger
            )
            
            # Set metrics separately if the strategy supports it
            if hasattr(strategy, 'set_metrics'):
                strategy.set_metrics(metrics)
            
            # Cache if not dry run
            if not dry_run:
                self._strategy_cache[strategy_name] = strategy
            
            return strategy
            
        except Exception as e:
            raise StrategyError(
                f"Failed to initialize {strategy_name} strategy: {str(e)}"
            )

    def clear_cache(self):
        """Clear strategy cache to free memory"""
        self._strategy_cache.clear()
        self.logger.debug("Cleared strategy cache")
```

### Critical Implementation Notes:

1. **Multi-Factor Decision Making**: Considers VRAM, aspect ratio, area change, and available pipelines
2. **VRAM Priority**: VRAM constraints override all other factors
3. **Fallback Chain**: Always has a path to success (even if slow)
4. **Strategy Caching**: Reuses initialized strategies for efficiency
5. **Clear Selection Reasoning**: Tracks why each strategy was chosen

## Step 2.1.3: Create Pipeline Orchestrator

### Location: `expandor/core/pipeline_orchestrator.py`

```python
"""
Pipeline orchestrator with fallback chains
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, List
from pathlib import Path

from .config import ExpandorConfig
from .metadata_tracker import MetadataTracker
from .boundary_tracker import BoundaryTracker
from .result import ExpandorResult, StageResult
from .exceptions import ExpandorError, StrategyError, VRAMError
from ..strategies.base_strategy import BaseExpansionStrategy

class PipelineOrchestrator:
    """
    Executes expansion strategies with fallback support
    
    This class is responsible for:
    1. Executing strategies with proper error handling
    2. Managing fallback chains when strategies fail
    3. Coordinating between strategy, metadata, and boundary tracking
    4. Ensuring atomic operations (all-or-nothing)
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize orchestrator
        
        Args:
            config: Global configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.pipeline_registry = {}
        
        # Load fallback configuration
        self.fallback_config = config.get('vram_strategies', {}).get('fallback_chain', {})
        
        # Track execution state
        self.current_strategy = None
        self.execution_history = []
    
    def execute(self,
                strategy: BaseExpansionStrategy,
                config: ExpandorConfig,
                metadata_tracker: MetadataTracker,
                boundary_tracker: BoundaryTracker) -> ExpandorResult:
        """
        Execute strategy with comprehensive error handling and fallbacks
        
        Args:
            strategy: Primary strategy to execute
            config: Expansion configuration
            metadata_tracker: Metadata tracking instance
            boundary_tracker: Boundary tracking instance
            
        Returns:
            ExpandorResult with processed image and metadata
            
        Raises:
            ExpandorError: If all strategies fail
        """
        # Track start
        execution_start = time.time()
        self.execution_history.clear()
        
        # Prepare strategy with pipelines and trackers
        self._prepare_strategy(strategy, boundary_tracker)
        
        # Build fallback chain
        fallback_chain = self._build_fallback_chain(strategy, config)
        
        # Execute with fallbacks
        last_error = None
        fallback_count = 0
        
        for attempt, current_strategy in enumerate(fallback_chain):
            try:
                self.current_strategy = current_strategy
                strategy_name = current_strategy.__class__.__name__
                
                self.logger.info(
                    f"Executing {strategy_name} "
                    f"{'(primary)' if attempt == 0 else f'(fallback {attempt})'}"
                )
                
                # Record execution attempt
                metadata_tracker.record_event("strategy_execution_start", {
                    "strategy": strategy_name,
                    "attempt": attempt + 1,
                    "is_fallback": attempt > 0
                })
                
                # Execute strategy
                result = self._execute_single_strategy(
                    current_strategy,
                    config,
                    metadata_tracker,
                    boundary_tracker
                )
                
                # Success! Update result metadata
                result.strategy_used = strategy_name
                result.fallback_count = fallback_count
                result.metadata["execution_history"] = self.execution_history
                
                # Record success
                metadata_tracker.record_event("strategy_execution_success", {
                    "strategy": strategy_name,
                    "duration": time.time() - execution_start
                })
                
                return result
                
            except Exception as e:
                # Record failure
                error_info = {
                    "strategy": strategy_name,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
                self.execution_history.append({
                    "strategy": strategy_name,
                    "status": "failed",
                    "error": str(e),
                    "attempt": attempt + 1
                })
                
                metadata_tracker.record_event("strategy_execution_failed", error_info)
                
                # Log based on error type
                if isinstance(e, VRAMError):
                    self.logger.warning(f"{strategy_name} failed due to VRAM: {e}")
                else:
                    self.logger.error(f"{strategy_name} failed: {e}")
                
                last_error = e
                fallback_count += 1
                
                # Check if we should continue with fallbacks
                if attempt == len(fallback_chain) - 1:
                    # This was the last strategy
                    break
                
                if not self._should_continue_fallback(e, config):
                    break
                
                # Prepare for next attempt
                self.logger.info("Attempting fallback strategy...")
                time.sleep(0.5)  # Brief pause before retry
        
        # All strategies failed - fail loud
        self._handle_complete_failure(last_error, self.execution_history)
    
    def register_pipeline(self, name: str, pipeline: Any):
        """Register a pipeline for strategies to use"""
        self.pipeline_registry[name] = pipeline
    
    def _prepare_strategy(self, strategy: BaseExpansionStrategy, boundary_tracker: BoundaryTracker):
        """
        Prepare strategy with necessary components
        
        Injects pipelines and trackers into strategy
        """
        # Inject pipelines
        for name, pipeline in self.pipeline_registry.items():
            if hasattr(strategy, f"{name}_pipeline"):
                setattr(strategy, f"{name}_pipeline", pipeline)
            elif hasattr(strategy, f"set_{name}_pipeline"):
                getattr(strategy, f"set_{name}_pipeline")(pipeline)
        
        # Inject boundary tracker
        strategy.boundary_tracker = boundary_tracker
        
        # Validate strategy has required pipelines
        strategy.validate_requirements()
    
    def _execute_single_strategy(self,
                                strategy: BaseExpansionStrategy,
                                config: ExpandorConfig,
                                metadata_tracker: MetadataTracker,
                                boundary_tracker: BoundaryTracker) -> ExpandorResult:
        """
        Execute a single strategy with proper tracking
        
        This method ensures all tracking is properly updated during execution
        """
        # Create execution context
        # This context is passed to strategies to provide access to shared resources
        context = {
            "config": config,                          # Original ExpandorConfig
            "metadata_tracker": metadata_tracker,      # For recording events/metrics
            "boundary_tracker": boundary_tracker,      # For tracking expansion boundaries
            "pipeline_registry": self.pipeline_registry,  # Available AI pipelines
            "stage_callback": lambda stage: self._record_stage(stage, metadata_tracker),  # Stage recording function
            "save_stages": config.save_stages,         # Whether to save intermediate results
            "stage_dir": config.stage_dir             # Directory for saving stages
        }
        
        # Execute strategy
        try:
            # Update metadata tracker stage
            metadata_tracker.current_stage = f"executing_{strategy.__class__.__name__}"
            
            # Execute with context
            raw_result = strategy.execute(config, context)
            
            # Convert to ExpandorResult if needed
            if isinstance(raw_result, ExpandorResult):
                return raw_result
            else:
                return self._convert_to_result(raw_result, strategy, config)
                
        except Exception as e:
            # Ensure strategy cleanup on error
            if hasattr(strategy, 'cleanup'):
                try:
                    strategy.cleanup()
                except:
                    pass  # Don't let cleanup errors mask the real error
            raise
    
    def _build_fallback_chain(self, 
                             primary_strategy: BaseExpansionStrategy,
                             config: ExpandorConfig) -> List[BaseExpansionStrategy]:
        """
        Build chain of strategies to try in order
        
        Returns list with primary strategy first, then fallbacks
        """
        chain = [primary_strategy]
        
        # Check if fallbacks are disabled
        if config.quality_preset == 'fast':
            return chain  # No fallbacks in fast mode
        
        # Get strategy name
        primary_name = primary_strategy.__class__.__name__.lower()
        
        # Add configured fallbacks
        # Map fallback config to strategy classes
        fallback_map = {
            'tiled_large': ('tiled_expansion', {'tile_size': 1024}),
            'tiled_medium': ('tiled_expansion', {'tile_size': 768}),
            'tiled_small': ('tiled_expansion', {'tile_size': 512}),
            'cpu_offload': ('cpu_offload', {})
        }
        
        # Build fallback chain based on config
        for priority in sorted(self.fallback_config.keys()):
            fallback_name = self.fallback_config[priority]
            
            if fallback_name in fallback_map:
                strategy_name, extra_config = fallback_map[fallback_name]
                
                # Skip if same as primary
                if strategy_name == primary_name:
                    continue
                
                # Skip if disabled
                if strategy_name == 'tiled_expansion' and not config.allow_tiled:
                    continue
                if strategy_name == 'cpu_offload' and not config.allow_cpu_offload:
                    continue
                
                # Create fallback strategy
                try:
                    # Load strategy class using lazy loading
                    strategy_class = self._load_strategy_class(strategy_name)
                    
                    # Create instance with extra config
                    fallback = strategy_class(
                        config={**self.config.get('strategies', {}).get(strategy_name, {}), 
                               **extra_config},
                        logger=self.logger
                    )
                    
                    chain.append(fallback)
                    
                except Exception as e:
                    self.logger.warning(f"Could not create fallback {strategy_name}: {e}")
        
        return chain
    
    def _should_continue_fallback(self, error: Exception, config: ExpandorConfig) -> bool:
        """
        Determine if we should try fallback after an error
        
        Some errors are unrecoverable and fallbacks won't help
        """
        # Don't retry on configuration errors
        if isinstance(error, (ValueError, TypeError)):
            return False
        
        # Don't retry on file errors
        if isinstance(error, (FileNotFoundError, PermissionError)):
            return False
        
        # Always retry VRAM errors if fallbacks available
        if isinstance(error, VRAMError):
            return True
        
        # Retry other errors based on quality preset
        if config.quality_preset in ['ultra', 'high']:
            return True  # Try hard to succeed
        
        return False
    
    def _record_stage(self, stage: StageResult, metadata_tracker: MetadataTracker):
        """Record a stage completion"""
        metadata_tracker.record_event("stage_complete", {
            "name": stage.name,
            "method": stage.method,
            "input_size": stage.input_size,
            "output_size": stage.output_size,
            "duration": stage.duration_seconds,
            "vram_used": stage.vram_used_mb,
            "artifacts_detected": stage.artifacts_detected
        })
    
    def _convert_to_result(self, 
                          raw_result: Dict[str, Any],
                          strategy: BaseExpansionStrategy,
                          config: ExpandorConfig) -> ExpandorResult:
        """Convert strategy output to ExpandorResult"""
        # Extract required fields
        image_path = raw_result.get('image_path')
        if not image_path:
            raise StrategyError(f"{strategy.__class__.__name__} did not return image_path")
        
        # Create result
        result = ExpandorResult(
            image_path=Path(image_path),
            size=raw_result.get('size', (0, 0)),
            success=True,
            stages=raw_result.get('stages', []),
            boundaries=raw_result.get('boundaries', []),
            seams_detected=0,  # Will be set by quality validation
            artifacts_fixed=0,
            refinement_passes=raw_result.get('refinement_passes', 0),
            quality_score=1.0,  # Will be updated by validation
            vram_peak_mb=0,  # Will be set by main expandor
            total_duration_seconds=0,  # Will be set by main expandor
            strategy_used=strategy.__class__.__name__,
            fallback_count=0,
            metadata=raw_result.get('metadata', {})
        )
        
        return result
    
    def _handle_complete_failure(self, last_error: Exception, history: List[Dict]):
        """Handle case where all strategies failed"""
        # Build comprehensive error message
        error_lines = [
            "All expansion strategies failed!",
            f"Attempted {len(history)} strategies:",
            ""
        ]
        
        for item in history:
            error_lines.append(
                f"  - {item['strategy']} (attempt {item['attempt']}): {item['error']}"
            )
        
        error_lines.extend([
            "",
            "This indicates a critical issue that prevents image expansion.",
            "Possible causes:",
            "  1. Insufficient system resources (RAM/VRAM)",
            "  2. Corrupted input image",  
            "  3. Invalid configuration",
            "  4. Missing required pipelines",
            "",
            "Last error details:"
        ])
        
        error_message = "\n".join(error_lines)
        
        # Raise comprehensive error
        raise ExpandorError(
            error_message,
            stage="pipeline_execution",
            partial_result={"execution_history": history}
        ) from last_error
```

### Critical Implementation Notes:

1. **Atomic Operations**: Either complete success or complete failure
2. **Comprehensive Fallbacks**: Automatic retry with degraded strategies
3. **Detailed History**: Track every execution attempt
4. **Strategy Preparation**: Inject all required components
5. **Smart Failure Handling**: Different handling for different error types

## Step 2.1.4: Create Metadata Tracker

### Location: `expandor/core/metadata_tracker.py`

```python
"""
Metadata tracking system for operation history
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

class MetadataTracker:
    """
    Tracks all metadata during expansion operations
    
    Provides:
    - Operation event logging
    - Stage tracking
    - Performance metrics
    - Debug information collection
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize metadata tracker"""
        self.logger = logger or logging.getLogger(__name__)
        self.reset()
    
    def reset(self):
        """Reset tracker for new operation"""
        self.operation_id = f"op_{int(time.time() * 1000)}"
        self.start_time = time.time()
        self.current_stage = "initialization"
        self.events = []
        self.metrics = {}
        self.config_snapshot = None
        self.stage_timings = {}
        self.stage_start = None
    
    def start_operation(self, config):
        """
        Start tracking a new operation
        
        Args:
            config: ExpandorConfig instance
        """
        self.reset()
        
        # Snapshot configuration
        self.config_snapshot = {
            "source_image": str(config.source_image) if isinstance(config.source_image, Path) else "<PIL.Image>",
            "target_resolution": config.target_resolution,
            "quality_preset": config.quality_preset,
            "prompt": config.prompt[:100] + "..." if len(config.prompt) > 100 else config.prompt,
            "seed": config.seed,
            "model_type": config.source_metadata.get('model', 'unknown'),
            "has_pipelines": {
                "inpaint": config.inpaint_pipeline is not None,
                "refiner": config.refiner_pipeline is not None,
                "img2img": config.img2img_pipeline is not None
            }
        }
        
        # Record start event
        self.record_event("operation_start", {
            "operation_id": self.operation_id,
            "config": self.config_snapshot
        })
        
        self.logger.debug(f"Started tracking operation: {self.operation_id}")
    
    def enter_stage(self, stage_name: str):
        """
        Mark entry into a new stage
        
        Args:
            stage_name: Name of the stage
        """
        # End previous stage timing
        if self.stage_start is not None:
            elapsed = time.time() - self.stage_start
            self.stage_timings[self.current_stage] = elapsed
        
        # Start new stage
        self.current_stage = stage_name
        self.stage_start = time.time()
        
        self.record_event("stage_enter", {
            "stage": stage_name,
            "timestamp": datetime.now().isoformat()
        })
    
    def exit_stage(self, success: bool = True, error: Optional[str] = None):
        """
        Mark exit from current stage
        
        Args:
            success: Whether stage completed successfully
            error: Error message if failed
        """
        if self.stage_start is not None:
            elapsed = time.time() - self.stage_start
            self.stage_timings[self.current_stage] = elapsed
            self.stage_start = None
        
        self.record_event("stage_exit", {
            "stage": self.current_stage,
            "success": success,
            "duration": self.stage_timings.get(self.current_stage, 0),
            "error": error
        })
    
    def record_event(self, event_type: str, data: Dict[str, Any]):
        """
        Record an event during operation
        
        Args:
            event_type: Type of event (e.g., "strategy_selected", "vram_check")
            data: Event data
        """
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "relative_time": time.time() - self.start_time,
            "stage": self.current_stage,
            "data": data
        }
        
        self.events.append(event)
        
        # Log significant events
        if event_type in ["strategy_selected", "stage_complete", "error"]:
            self.logger.debug(f"Event: {event_type} - {data}")
    
    def record_metric(self, metric_name: str, value: Any):
        """
        Record a performance metric
        
        Args:
            metric_name: Name of metric
            value: Metric value
        """
        self.metrics[metric_name] = value
    
    def add_stage_metadata(self, stage_name: str, metadata: Dict[str, Any]):
        """
        Add metadata for a specific stage
        
        Args:
            stage_name: Name of stage
            metadata: Stage-specific metadata
        """
        self.record_event("stage_metadata", {
            "stage": stage_name,
            "metadata": metadata
        })
    
    def get_operation_log(self) -> Dict[str, Any]:
        """
        Get complete operation log
        
        Returns:
            Dictionary with all operation data
        """
        return {
            "operation_id": self.operation_id,
            "duration": time.time() - self.start_time,
            "config": self.config_snapshot,
            "events": self.events,
            "metrics": self.metrics,
            "stage_timings": self.stage_timings,
            "event_summary": self._summarize_events()
        }
    
    def get_partial_result(self) -> Dict[str, Any]:
        """
        Get partial result for error reporting
        
        Returns:
            Dictionary with data collected so far
        """
        return {
            "operation_id": self.operation_id,
            "duration_before_error": time.time() - self.start_time,
            "last_stage": self.current_stage,
            "events_count": len(self.events),
            "last_events": self.events[-5:] if self.events else [],
            "metrics": self.metrics,
            "stage_timings": self.stage_timings
        }
    
    def save_to_file(self, filepath: Path):
        """
        Save operation log to JSON file
        
        Args:
            filepath: Path to save log
        """
        log_data = self.get_operation_log()
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        self.logger.debug(f"Saved operation log to: {filepath}")
    
    def _summarize_events(self) -> Dict[str, Any]:
        """Create summary of events by type"""
        summary = {}
        
        for event in self.events:
            event_type = event['type']
            if event_type not in summary:
                summary[event_type] = {
                    'count': 0,
                    'first_at': event['relative_time'],
                    'last_at': event['relative_time']
                }
            
            summary[event_type]['count'] += 1
            summary[event_type]['last_at'] = event['relative_time']
        
        return summary
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary
        
        Returns:
            Dictionary with performance metrics
        """
        total_time = time.time() - self.start_time
        
        # Calculate stage percentages
        stage_percentages = {}
        for stage, duration in self.stage_timings.items():
            stage_percentages[stage] = (duration / total_time) * 100
        
        return {
            "total_duration": total_time,
            "stages": self.stage_timings,
            "stage_percentages": stage_percentages,
            "metrics": self.metrics,
            "events_per_second": len(self.events) / total_time if total_time > 0 else 0
        }
```

### Critical Implementation Notes:

1. **Comprehensive Tracking**: Every significant action is recorded
2. **Stage Timing**: Precise timing for performance analysis
3. **Event History**: Complete audit trail for debugging
4. **Partial Results**: Can provide debug info even on failure
5. **Performance Metrics**: Built-in performance analysis

## Step 2.1.5: Create Boundary Tracker

### Location: `expandor/core/boundary_tracker.py`

```python
"""
Boundary tracking for seam detection
"""

import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

@dataclass
class BoundaryInfo:
    """Information about an expansion boundary"""
    position: int  # Pixel position of boundary
    direction: str  # 'horizontal' or 'vertical'
    step: int  # Which expansion step created this
    expansion_size: int  # How many pixels were added
    source_size: Tuple[int, int]  # Size before expansion
    target_size: Tuple[int, int]  # Size after expansion
    method: str  # Method used (e.g., 'outpaint', 'sliding_window')
    metadata: Dict = field(default_factory=dict)  # Additional info

class BoundaryTracker:
    """
    Tracks expansion boundaries for artifact detection
    
    This is critical for detecting seams where new content meets old.
    Every expansion creates potential seam locations that must be tracked.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize boundary tracker"""
        self.logger = logger or logging.getLogger(__name__)
        self.reset()
    
    def reset(self):
        """Reset tracker for new operation"""
        self.boundaries: List[BoundaryInfo] = []
        self.horizontal_positions: Set[int] = set()
        self.vertical_positions: Set[int] = set()
        self.boundary_map: Dict[str, List[BoundaryInfo]] = {
            'horizontal': [],
            'vertical': []
        }
    
    def add_boundary(self,
                    position: int,
                    direction: str,
                    step: int,
                    expansion_size: int,
                    source_size: Tuple[int, int],
                    target_size: Tuple[int, int],
                    method: str = 'unknown',
                    metadata: Optional[Dict] = None):
        """
        Add a boundary to track
        
        Args:
            position: Pixel position where new content meets old
            direction: 'horizontal' or 'vertical'
            step: Expansion step number
            expansion_size: Pixels added in this expansion
            source_size: Image size before expansion
            target_size: Image size after expansion
            method: Method used for expansion
            metadata: Additional tracking data
        """
        if direction not in ['horizontal', 'vertical']:
            raise ValueError(f"Invalid direction: {direction}")
        
        boundary = BoundaryInfo(
            position=position,
            direction=direction,
            step=step,
            expansion_size=expansion_size,
            source_size=source_size,
            target_size=target_size,
            method=method,
            metadata=metadata or {}
        )
        
        self.boundaries.append(boundary)
        self.boundary_map[direction].append(boundary)
        
        # Track unique positions
        if direction == 'horizontal':
            self.horizontal_positions.add(position)
        else:
            self.vertical_positions.add(position)
        
        self.logger.debug(
            f"Added {direction} boundary at {position} "
            f"(step {step}, {expansion_size}px expansion)"
        )
    
    def add_progressive_boundaries(self,
                                  current_size: Tuple[int, int],
                                  target_size: Tuple[int, int],
                                  step: int,
                                  method: str = 'progressive'):
        """
        Add boundaries for a progressive expansion
        
        This is the most common case - expanding from center outward
        """
        current_w, current_h = current_size
        target_w, target_h = target_size
        
        # Horizontal expansion
        if target_w > current_w:
            pad_left = (target_w - current_w) // 2
            pad_right = pad_left + current_w
            
            # Left boundary
            self.add_boundary(
                position=pad_left,
                direction='horizontal',
                step=step,
                expansion_size=(target_w - current_w) // 2,
                source_size=current_size,
                target_size=target_size,
                method=method,
                metadata={'side': 'left', 'pad_amount': pad_left}
            )
            
            # Right boundary  
            self.add_boundary(
                position=pad_right,
                direction='horizontal',
                step=step,
                expansion_size=(target_w - current_w) // 2,
                source_size=current_size,
                target_size=target_size,
                method=method,
                metadata={'side': 'right', 'pad_amount': target_w - pad_right}
            )
        
        # Vertical expansion
        if target_h > current_h:
            pad_top = (target_h - current_h) // 2
            pad_bottom = pad_top + current_h
            
            # Top boundary
            self.add_boundary(
                position=pad_top,
                direction='vertical',
                step=step,
                expansion_size=(target_h - current_h) // 2,
                source_size=current_size,
                target_size=target_size,
                method=method,
                metadata={'side': 'top', 'pad_amount': pad_top}
            )
            
            # Bottom boundary
            self.add_boundary(
                position=pad_bottom,
                direction='vertical',
                step=step,
                expansion_size=(target_h - current_h) // 2,
                source_size=current_size,
                target_size=target_size,
                method=method,
                metadata={'side': 'bottom', 'pad_amount': target_h - pad_bottom}
            )
    
    def add_sliding_window_boundaries(self,
                                     window_positions: List[Tuple[int, int]],
                                     direction: str,
                                     step: int):
        """
        Add boundaries for sliding window expansion
        
        Args:
            window_positions: List of (start, end) positions for each window
            direction: 'horizontal' or 'vertical'
            step: Step number
        """
        for i, (start, end) in enumerate(window_positions):
            # Each window creates a boundary at its edge
            if i > 0:  # Not the first window
                # Boundary where this window starts (overlaps with previous)
                self.add_boundary(
                    position=start,
                    direction=direction,
                    step=step,
                    expansion_size=end - start,
                    source_size=(0, 0),  # Not applicable for SWPO
                    target_size=(0, 0),
                    method='sliding_window',
                    metadata={
                        'window_index': i,
                        'window_start': start,
                        'window_end': end,
                        'is_overlap': True
                    }
                )
    
    def add_tile_boundaries(self,
                           tile_grid: List[Tuple[int, int, int, int]],
                           step: int):
        """
        Add boundaries for tiled processing
        
        Args:
            tile_grid: List of (x1, y1, x2, y2) tile coordinates
            step: Step number
        """
        # Extract unique boundaries
        x_boundaries = set()
        y_boundaries = set()
        
        for x1, y1, x2, y2 in tile_grid:
            if x1 > 0:
                x_boundaries.add(x1)
            if x2 < float('inf'):  # Not the last tile
                x_boundaries.add(x2)
            if y1 > 0:
                y_boundaries.add(y1)
            if y2 < float('inf'):
                y_boundaries.add(y2)
        
        # Add vertical boundaries
        for x in sorted(x_boundaries):
            self.add_boundary(
                position=x,
                direction='horizontal',
                step=step,
                expansion_size=0,  # Tiles don't expand
                source_size=(0, 0),
                target_size=(0, 0),
                method='tiled',
                metadata={'boundary_type': 'tile_edge'}
            )
        
        # Add horizontal boundaries
        for y in sorted(y_boundaries):
            self.add_boundary(
                position=y,
                direction='vertical',
                step=step,
                expansion_size=0,
                source_size=(0, 0),
                target_size=(0, 0),
                method='tiled',
                metadata={'boundary_type': 'tile_edge'}
            )
    
    def get_all_boundaries(self) -> List[Dict]:
        """
        Get all boundaries as serializable list
        
        Returns:
            List of boundary dictionaries
        """
        return [
            {
                'position': b.position,
                'direction': b.direction,
                'step': b.step,
                'expansion_size': b.expansion_size,
                'source_size': b.source_size,
                'target_size': b.target_size,
                'method': b.method,
                'metadata': b.metadata
            }
            for b in self.boundaries
        ]
    
    def get_boundaries_for_detection(self) -> Dict[str, List[int]]:
        """
        Get boundaries formatted for artifact detection
        
        Returns:
            Dictionary with 'horizontal' and 'vertical' position lists
        """
        return {
            'horizontal': sorted(list(self.horizontal_positions)),
            'vertical': sorted(list(self.vertical_positions))
        }
    
    def get_boundary_regions(self, margin: int = 10) -> List[Tuple[int, int, int, int]]:
        """
        Get regions around boundaries for focused processing
        
        Args:
            margin: Pixels on each side of boundary
            
        Returns:
            List of (x1, y1, x2, y2) regions
        """
        regions = []
        
        # Get image dimensions from last boundary
        if self.boundaries:
            last_boundary = self.boundaries[-1]
            width, height = last_boundary.target_size
        else:
            return regions
        
        # Horizontal boundaries create vertical strips
        for x in self.horizontal_positions:
            x1 = max(0, x - margin)
            x2 = min(width, x + margin)
            regions.append((x1, 0, x2, height))
        
        # Vertical boundaries create horizontal strips
        for y in self.vertical_positions:
            y1 = max(0, y - margin)
            y2 = min(height, y + margin)
            regions.append((0, y1, width, y2))
        
        return regions
    
    def get_critical_boundaries(self) -> List[BoundaryInfo]:
        """
        Get boundaries most likely to have visible seams
        
        Returns boundaries from progressive/SWPO operations
        """
        critical = []
        
        for boundary in self.boundaries:
            # Progressive and sliding window boundaries are most critical
            if boundary.method in ['progressive', 'sliding_window']:
                critical.append(boundary)
            # Large expansions are also critical
            elif boundary.expansion_size > 200:
                critical.append(boundary)
        
        return critical
    
    def summarize(self) -> Dict[str, Any]:
        """
        Get summary of tracked boundaries
        
        Returns:
            Summary dictionary
        """
        return {
            'total_boundaries': len(self.boundaries),
            'horizontal_count': len(self.horizontal_positions),
            'vertical_count': len(self.vertical_positions),
            'methods_used': list(set(b.method for b in self.boundaries)),
            'expansion_steps': len(set(b.step for b in self.boundaries)),
            'critical_boundaries': len(self.get_critical_boundaries()),
            'largest_expansion': max((b.expansion_size for b in self.boundaries), default=0)
        }
```

### Critical Implementation Notes:

1. **Precise Tracking**: Every pixel where new meets old is tracked
2. **Multiple Methods**: Handles progressive, SWPO, and tiled boundaries
3. **Critical Detection**: Identifies boundaries most likely to have seams
4. **Region Generation**: Creates focused areas for artifact detection
5. **Serializable Output**: All data can be saved with image metadata

## Step 2.1.6: Create VRAM Manager

### Location: `expandor/core/vram_manager.py`

```python
"""
VRAM Manager for Dynamic Resource Management
Expandor's standalone VRAM management (inspired by ai-wallpaper's VRAMCalculator)
"""

import torch
from typing import Dict, Any, Optional
import logging

class VRAMManager:
    """Manage VRAM requirements and determine processing strategy"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # SDXL base requirements (measured empirically)
        self.MODEL_OVERHEAD_MB = 6144  # 6GB for SDXL refiner
        self.ATTENTION_MULTIPLIER = 4   # Attention needs ~4x image memory
        self.SAFETY_BUFFER = 0.2        # 20% safety margin
        
        # Track peak usage
        self._peak_usage_mb = 0
    
    def calculate_generation_vram(self, 
                                 width: int, 
                                 height: int,
                                 batch_size: int = 1,
                                 model_type: str = "sdxl") -> float:
        """
        Calculate ACCURATE VRAM requirements for refinement.
        
        Args:
            width: Image width
            height: Image height  
            dtype: Data type (float16 or float32)
            
        Returns:
            Dict with detailed VRAM breakdown
        """
        pixels = width * height
        
        # Bytes per pixel based on dtype
        bytes_per_pixel = 2 if dtype == torch.float16 else 4
        
        # Image tensor memory (BCHW format)
        # 1 batch × 4 channels (latent) × H × W
        latent_h = height // 8  # VAE downscales by 8
        latent_w = width // 8
        latent_pixels = latent_h * latent_w
        
        # Memory calculations (in MB)
        latent_memory_mb = (latent_pixels * 4 * bytes_per_pixel) / (1024 * 1024)
        
        # Attention memory scales with sequence length
        # For SDXL: roughly latent_pixels * multiplier
        attention_memory_mb = (latent_pixels * self.ATTENTION_MULTIPLIER * bytes_per_pixel) / (1024 * 1024)
        
        # Activations and gradients
        activation_memory_mb = latent_memory_mb * 2  # Conservative estimate
        
        # Total image-related memory
        image_memory_mb = latent_memory_mb + attention_memory_mb + activation_memory_mb
        
        # Add model overhead
        total_vram_mb = self.MODEL_OVERHEAD_MB + image_memory_mb
        
        # Add safety buffer
        total_with_buffer_mb = total_vram_mb * (1 + self.SAFETY_BUFFER)
        
        # Return total VRAM with safety buffer
        return total_with_buffer_mb * batch_size
    
    def get_available_vram(self) -> Optional[float]:
        """Get available VRAM in MB"""
        if not torch.cuda.is_available():
            return None
            
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_mb = free_bytes / (1024 * 1024)
            total_mb = total_bytes / (1024 * 1024)
            
            self.logger.debug(f"VRAM: {free_mb:.0f}MB free / {total_mb:.0f}MB total")
            return free_mb
        except Exception as e:
            self.logger.error(f"Failed to get VRAM info: {e}")
            return None
    
    def estimate_requirement(self, config) -> Dict[str, float]:
        """
        Estimate VRAM requirement from ExpandorConfig
        
        Args:
            config: ExpandorConfig instance
            
        Returns:
            Dict with VRAM estimates
        """
        width, height = config.target_resolution
        return self.calculate_generation_vram(width, height)
    
    def get_current_usage(self) -> float:
        """Get current VRAM usage in MB"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_mb = (total_bytes - free_bytes) / (1024 * 1024)
            
            # Update peak if current is higher
            if used_mb > self._peak_usage_mb:
                self._peak_usage_mb = used_mb
            
            return used_mb
        except:
            return 0.0
    
    def get_peak_usage(self) -> float:
        """Get peak VRAM usage recorded"""
        return self._peak_usage_mb
    
    def determine_strategy(self, width: int, height: int) -> Dict[str, Any]:
        """
        Determine the best processing strategy based on VRAM.
        
        Returns:
            Strategy dict with:
            - strategy: 'full', 'tiled', or 'cpu_offload'
            - details: Strategy-specific parameters
        """
        # Calculate requirements
        required_mb = self.calculate_generation_vram(width, height)
        
        # Get available VRAM
        available_mb = self.get_available_vram()
        
        if available_mb is None:
            # No CUDA - use CPU offload
            return {
                'strategy': 'cpu_offload',
                'details': {
                    'reason': 'No CUDA available',
                    'warning': 'Using CPU - will be VERY slow'
                },
                'vram_required_mb': required_mb,
                'vram_available_mb': 0
            }
        
        # Strategy decision
        if required_mb <= available_mb:
            # Full processing possible
            return {
                'strategy': 'full',
                'details': {
                    'message': f"Full processing possible: {required_mb:.0f}MB < {available_mb:.0f}MB"
                },
                'vram_required_mb': required_mb,
                'vram_available_mb': available_mb
            }
        
        # Need tiled processing
        # Calculate optimal tile size
        if vram_info['total_vram_mb'] > 0:
            pixels_per_mb = vram_info['pixels'] / vram_info['total_vram_mb']
        else:
            pixels_per_mb = 1024 * 1024 / 100  # Fallback
        
        max_pixels = int((available_mb - self.MODEL_OVERHEAD_MB) * pixels_per_mb * 0.8)
        
        # Tile size (ensure divisible by 128 for SDXL)
        tile_size = int(max_pixels ** 0.5)
        tile_size = max(512, ((tile_size // 128) * 128))
        
        # Can we do tiled processing?
        tile_required_mb = self.calculate_generation_vram(tile_size, tile_size)
        if tile_required_mb <= available_mb:
            return {
                'strategy': 'tiled',
                'details': {
                    'tile_size': tile_size,
                    'overlap': min(256, tile_size // 4),
                    'message': f"Tiled processing: {tile_size}x{tile_size} tiles"
                },
                'vram_required_mb': required_mb,
                'vram_available_mb': available_mb
            }
        
        # Last resort - CPU offload
        return {
            'strategy': 'cpu_offload',
            'details': {
                'reason': 'Image too large even for tiled processing',
                'tile_size': 512,  # Minimum tile size
                'warning': 'Using aggressive CPU offloading - will be SLOW'
            },
            'vram_required_mb': required_mb,
            'vram_available_mb': available_mb
        }
    
    def get_detailed_vram_state(self) -> Dict[str, Any]:
        """Get detailed VRAM state information"""
        if not torch.cuda.is_available():
            return {
                'available': False,
                'free_mb': 0,
                'total_mb': 0,
                'used_mb': 0,
                'processes': []
            }
        
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_mb = free_bytes / (1024 * 1024)
            total_mb = total_bytes / (1024 * 1024)
            used_mb = total_mb - free_mb
            
            # Update peak usage
            if used_mb > self._peak_usage_mb:
                self._peak_usage_mb = used_mb
            
            # Get process info if possible (requires nvidia-ml-py)
            processes = []
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for p in procs:
                    processes.append({
                        'pid': p.pid,
                        'memory_mb': p.usedGpuMemory / (1024 * 1024),
                        'name': 'Unknown'  # Would need psutil to get name
                    })
            except:
                # nvidia-ml-py not available or error
                pass
            
            return {
                'available': True,
                'free_mb': free_mb,
                'total_mb': total_mb,
                'used_mb': used_mb,
                'processes': processes
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get detailed VRAM state: {e}")
            return {
                'available': False,
                'error': str(e)
            }
```

### Critical Implementation Notes:

1. **Standalone Implementation**: Expandor's own VRAM management, not dependent on ai-wallpaper
2. **API Inspired By**: Similar interface to ai-wallpaper's VRAMCalculator for familiarity
3. **Peak Tracking**: Records maximum VRAM usage
4. **Error Handling**: Graceful degradation on errors
5. **Optional Process Info**: Can show what's using VRAM if pynvml available

## Step 2.1.7: Create Supporting Classes

### Location: `expandor/core/result.py`

```python
"""
Result classes for Expandor operations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

@dataclass
class StageResult:
    """Result from a single processing stage"""
    name: str
    method: str
    input_size: Tuple[int, int]
    output_size: Tuple[int, int]
    duration_seconds: float
    vram_used_mb: float
    artifacts_detected: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'method': self.method,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'duration_seconds': self.duration_seconds,
            'vram_used_mb': self.vram_used_mb,
            'artifacts_detected': self.artifacts_detected,
            'metadata': self.metadata
        }

@dataclass
class ExpandorResult:
    """Comprehensive result from expansion operation"""
    # Core results
    image_path: Path
    size: Tuple[int, int]
    success: bool = True
    
    # Stage tracking
    stages: List[StageResult] = field(default_factory=list)
    boundaries: List[Dict] = field(default_factory=list)
    
    # Quality metrics
    seams_detected: int = 0
    artifacts_fixed: int = 0
    refinement_passes: int = 0
    quality_score: float = 1.0
    
    # Resource usage
    vram_peak_mb: float = 0.0
    total_duration_seconds: float = 0.0
    strategy_used: str = ""
    fallback_count: int = 0
    
    # Full metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Error information (if success=False)
    error: Optional[Exception] = None
    error_stage: Optional[str] = None
    
    @property
    def image(self):
        """Load and return the image (lazy loading)"""
        if hasattr(self, '_image_cache'):
            return self._image_cache
        
        from PIL import Image
        self._image_cache = Image.open(self.image_path)
        return self._image_cache
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """Get summary of all stages"""
        total_vram = sum(s.vram_used_mb for s in self.stages)
        total_time = sum(s.duration_seconds for s in self.stages)
        
        return {
            'stage_count': len(self.stages),
            'total_stage_time': total_time,
            'total_vram_used': total_vram,
            'avg_vram_per_stage': total_vram / len(self.stages) if self.stages else 0,
            'stages': [s.to_dict() for s in self.stages]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'image_path': str(self.image_path),
            'size': self.size,
            'success': self.success,
            'stages': [s.to_dict() for s in self.stages],
            'boundaries': self.boundaries,
            'seams_detected': self.seams_detected,
            'artifacts_fixed': self.artifacts_fixed,
            'refinement_passes': self.refinement_passes,
            'quality_score': self.quality_score,
            'vram_peak_mb': self.vram_peak_mb,
            'total_duration_seconds': self.total_duration_seconds,
            'strategy_used': self.strategy_used,
            'fallback_count': self.fallback_count,
            'metadata': self.metadata,
            'error': str(self.error) if self.error else None,
            'error_stage': self.error_stage
        }
```

### Location: `expandor/utils/logging_utils.py`

```python
"""
Logging utilities for Expandor
"""

import logging
import sys
from pathlib import Path
from typing import Optional

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logger(name: str, 
                 level: int = logging.INFO,
                 log_file: Optional[Path] = None) -> logging.Logger:
    """
    Setup logger with consistent formatting
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with color
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Format with colors for console
    console_format = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Plain format for file
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger
```

### Location: `expandor/utils/config_loader.py`

```python
"""
Configuration loading utilities
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class ConfigLoader:
    """Loads and merges configuration from YAML files"""
    
    def __init__(self, config_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize config loader
        
        Args:
            config_dir: Directory containing config files
            logger: Optional logger
        """
        self.config_dir = Path(config_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        if not self.config_dir.exists():
            raise ValueError(f"Config directory not found: {config_dir}")
    
    def load_all_configs(self) -> Dict[str, Any]:
        """
        Load and merge all configuration files
        
        Returns:
            Merged configuration dictionary
        """
        config = {}
        
        # Define config files to load in order
        config_files = [
            'strategies.yaml',
            'quality_presets.yaml',
            'model_constraints.yaml',
            'vram_strategies.yaml'
        ]
        
        for filename in config_files:
            filepath = self.config_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        file_config = yaml.safe_load(f)
                        if file_config:
                            # Extract the base key (filename without .yaml)
                            key = filename.replace('.yaml', '')
                            config[key] = file_config.get(key, file_config)
                            self.logger.debug(f"Loaded {filename}")
                except Exception as e:
                    self.logger.error(f"Failed to load {filename}: {e}")
                    raise
            else:
                self.logger.warning(f"Config file not found: {filename}")
        
        return config
    
    def load_file(self, filename: str) -> Dict[str, Any]:
        """
        Load a specific config file
        
        Args:
            filename: Name of config file
            
        Returns:
            Configuration dictionary
        """
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    
    def load_config(self, name: str) -> Dict[str, Any]:
        """Load specific config section by name."""
        # First check if it's a specific file
        try:
            return self.load_config_file(f"{name}.yaml")
        except FileNotFoundError:
            # Try loading all configs and extracting section
            all_configs = self.load_all_configs()
            
            # Map common names to config keys
            name_map = {
                'strategies': 'strategies',
                'quality': 'quality_validation',
                'vram': 'vram_strategies',
                'models': 'model_constraints'
            }
            
            config_key = name_map.get(name, name)
            if config_key in all_configs:
                return all_configs[config_key]
            
            # Return defaults if not found
            return self._get_defaults(name)
    
    def _get_defaults(self, name: str) -> Dict[str, Any]:
        """Hardcoded defaults when config missing."""
        defaults = {
            'strategies': {
                'progressive_max_aspect_ratio': 4.0,
                'swpo_window_size': 256,
                'swpo_overlap_ratio': 0.8
            },
            'quality': {
                'artifact_threshold': 20.0,
                'color_threshold': 30.0,
                'gradient_spike_threshold': 50.0
            },
            'vram': {
                'safety_margin': 0.2,
                'clear_cache_threshold': 0.9
            }
        }
        return defaults.get(name, {})
    
    def save_config(self, config: Dict[str, Any], filename: str):
        """
        Save configuration to file
        
        Args:
            config: Configuration dictionary
            filename: Target filename
        """
        filepath = self.config_dir / filename
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        self.logger.debug(f"Saved config to {filename}")
```

## Step 2.1.8: Create Exception Hierarchy

Location: `expandor/core/exceptions.py` (update existing file)

```python
"""
Custom exceptions for Expandor
"""

from typing import Optional, Any, Dict

class ExpandorError(Exception):
    """Base exception for all Expandor errors"""
    def __init__(self, 
                 message: str, 
                 stage: Optional[str] = None, 
                 partial_result: Optional[Any] = None):
        self.message = message
        self.stage = stage
        self.partial_result = partial_result
        super().__init__(message)
    
    def __str__(self):
        parts = [self.message]
        if self.stage:
            parts.append(f" [Stage: {self.stage}]")
        return "".join(parts)

class VRAMError(ExpandorError):
    """VRAM-related errors."""
    def __init__(self, operation: str, required_mb: float, 
                available_mb: float, message: str = ""):
        self.operation = operation
        self.required_mb = required_mb
        self.available_mb = available_mb
        base_msg = f"Insufficient VRAM for {operation}: need {required_mb:.1f}MB, have {available_mb:.1f}MB"
        if message:
            base_msg = f"{base_msg}. {message}"
        super().__init__(base_msg)

class StrategyError(ExpandorError):
    """Strategy selection or execution errors"""
    pass

class QualityError(ExpandorError):
    """Quality validation errors"""
    def __init__(self,
                 message: str,
                 quality_score: float,
                 issues: Dict[str, Any],
                 **kwargs):
        super().__init__(message, **kwargs)
        self.quality_score = quality_score
        self.issues = issues

class PipelineError(ExpandorError):
    """Pipeline execution errors"""
    pass

class ConfigurationError(ExpandorError):
    """Configuration errors"""
    pass
```

## Verification Steps

After implementing all components:

1. **Run unit tests for each component**:
```bash
pytest tests/unit/test_expandor_core.py -v
pytest tests/unit/test_strategy_selector.py -v
pytest tests/unit/test_metadata_tracker.py -v
pytest tests/unit/test_boundary_tracker.py -v
```

2. **Run integration test**:
```bash
pytest tests/integration/test_base_architecture.py -v
```

3. **Verify logging output**:
```python
# Test script
from expandor import Expandor, ExpandorConfig
expandor = Expandor()
print(expandor.get_available_strategies())
```

## Summary

This step implements the complete base architecture for Expandor with:

1. **Core Expandor class** - Main orchestrator with fail-loud philosophy
2. **Strategy Selector** - Intelligent multi-factor strategy selection
3. **Pipeline Orchestrator** - Robust execution with automatic fallbacks
4. **Metadata Tracker** - Comprehensive operation tracking
5. **Boundary Tracker** - Critical seam position tracking
6. **Supporting utilities** - Logging, config loading, exceptions

Every component follows the "quality over all" and "no silent failures" principles. The system is designed to be bulletproof with comprehensive error handling, detailed logging, and automatic fallbacks.

## Key Corrections Made

1. **Import fixes**: Added missing imports (pkg_resources, Path, Image, VRAMError, StrategyError)
2. **VRAMError compatibility**: Updated VRAMError signature to match ai-wallpaper's implementation with operation parameter
3. **Metadata serialization**: Fixed stages serialization to use to_dict() method
4. **Exception handling**: Removed incompatible config parameter from ExpandorError 
5. **Dynamic imports**: Replaced get_strategy_class with explicit imports for fallback strategies

These corrections ensure compatibility with the ai-wallpaper codebase patterns and conventions.