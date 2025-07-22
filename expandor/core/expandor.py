"""
Main Expandor class - Universal image expansion orchestrator
"""

import logging
import time
import json
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
from ..processors.quality_validator import QualityValidator
from ..processors.seam_repair import SeamRepairProcessor

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
            
            # Quality Validator
            self.quality_validator = QualityValidator(self.config, self.logger)
            
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
                config=config
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
                "config_snapshot": self._serialize_config(config),
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
            # Log error details
            self.logger.error(f"Expansion failed: {str(e)}")
            self.metadata_tracker.record_event("expansion_failed", {
                "error": str(e),
                "stage": self.metadata_tracker.current_stage,
                "duration": time.time() - operation_start
            })
            
            # Re-raise with context
            raise ExpandorError(
                f"Expansion failed during {self.metadata_tracker.current_stage}: {str(e)}",
                stage=self.metadata_tracker.current_stage,
                config=config,
                partial_result=self.metadata_tracker.get_partial_result()
            ) from e
            
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
        target_w, target_h = config.target_resolution
        base_vram = self.vram_manager.calculate_generation_vram(target_w, target_h)
        
        # Get strategy-specific requirements
        strategy = self.strategy_selector.select(config, dry_run=True)
        strategy_estimate = strategy.estimate_vram(config)
        
        return {
            "total_required_mb": max(
                base_vram,
                strategy_estimate.get("peak_vram_mb", 0)
            ),
            "available_mb": self.vram_manager.get_available_vram() or 0,
            "base_estimate_mb": base_vram,
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
        available_presets = list(self.config.get("quality_presets", {}).keys())
        if not available_presets:
            available_presets = ["ultra", "high", "balanced", "fast"]
        
        if config.quality_preset not in available_presets:
            raise ValueError(
                f"Invalid quality preset: {config.quality_preset}. "
                f"Available: {', '.join(available_presets)}"
            )
        
        # Pipeline validation
        has_any_pipeline = any([
            config.inpaint_pipeline,
            config.refiner_pipeline,
            config.img2img_pipeline
        ])
        
        if not has_any_pipeline:
            self.logger.warning(
                "No AI pipelines provided. Only direct upscaling will be available."
            )
    
    def _prepare_workspace(self, config: ExpandorConfig):
        """Prepare directories and workspace for execution"""
        # Create temp directories
        temp_dirs = [
            Path("temp"),
            Path("temp/progressive"),
            Path("temp/tiles"),
            Path("temp/upscale")
        ]
        
        for dir_path in temp_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create stage directory if saving stages
        if config.save_stages and config.stage_dir:
            config.stage_dir.mkdir(parents=True, exist_ok=True)
    
    def _validate_and_repair(self, result: ExpandorResult, config: ExpandorConfig) -> ExpandorResult:
        """
        Validate quality and repair artifacts if needed
        
        Args:
            result: Initial expansion result
            config: Expansion configuration
            
        Returns:
            Updated result with quality metrics and repairs
        """
        # Skip validation in fast mode
        if config.quality_preset == 'fast':
            result.quality_score = 1.0
            return result
        
        # Get boundaries for artifact detection
        critical_boundaries = self.boundary_tracker.get_critical_boundaries()
        
        # Validate quality
        validation_result = self.quality_validator.validate(
            image_path=result.image_path,
            boundaries=critical_boundaries,
            quality_preset=config.quality_preset
        )
        
        result.quality_score = validation_result.get('score', 1.0)
        result.seams_detected = len(validation_result.get('issues', []))
        
        # Check if repair needed
        if not validation_result.get('passed', True) and config.auto_refine:
            self.logger.warning(f"Quality issues detected: {validation_result['issues']}")
            
            # Attempt repair if pipelines available
            if config.inpaint_pipeline or config.refiner_pipeline:
                try:
                    repair_processor = SeamRepairProcessor(
                        pipelines=self.pipeline_registry,
                        logger=self.logger
                    )
                    
                    # Create artifact mask from issues
                    artifact_mask = self._create_artifact_mask(
                        result.image_path,
                        validation_result['issues']
                    )
                    
                    # Repair
                    repair_result = repair_processor.repair_seams(
                        image_path=result.image_path,
                        artifact_mask=artifact_mask,
                        prompt=config.prompt,
                        metadata=result.metadata
                    )
                    
                    if repair_result['success']:
                        result.image_path = repair_result['image_path']
                        result.artifacts_fixed = repair_result['repairs_made']
                        result.refinement_passes = repair_result['passes']
                        result.quality_score = repair_result['final_score']
                        
                        self.logger.info(
                            f"Repaired {result.artifacts_fixed} artifacts in "
                            f"{result.refinement_passes} passes"
                        )
                    
                except Exception as e:
                    self.logger.error(f"Artifact repair failed: {e}")
                    # Continue with original result
        
        return result
    
    def _create_artifact_mask(self, image_path: Path, issues: List[Dict]) -> np.ndarray:
        """Create mask for detected artifacts"""
        # This is a simplified implementation
        # In production, would use sophisticated artifact detection
        with Image.open(image_path) as img:
            mask = np.zeros((img.height, img.width), dtype=np.uint8)
            
            # Mark boundary regions
            for issue in issues:
                if 'location' in issue:
                    x, y, w, h = issue['location']
                    mask[y:y+h, x:x+w] = 255
            
            return mask
    
    def _save_metadata(self, result: ExpandorResult):
        """Save metadata JSON alongside result image"""
        metadata_path = result.image_path.with_suffix('.json')
        
        # Convert result to serializable format
        metadata = result.to_dict()
        
        # Add operation log
        metadata['operation_log'] = self.metadata_tracker.get_operation_log()
        
        # Save as JSON
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.debug(f"Saved metadata to: {metadata_path}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created during processing"""
        temp_patterns = [
            "temp/progressive_*.png",
            "temp/tile_*.png",
            "temp/upscaled_*.png",
            "temp/canvas_*.png",
            "temp/mask_*.png"
        ]
        
        import glob
        import os
        
        for pattern in temp_patterns:
            for file_path in glob.glob(pattern):
                try:
                    os.unlink(file_path)
                except:
                    pass
    
    def _log_vram_error(self, error: VRAMError):
        """Log detailed VRAM error information"""
        self.logger.error("=" * 60)
        self.logger.error("VRAM ALLOCATION FAILURE")
        self.logger.error(f"Operation: {error.operation}")
        self.logger.error(f"Required: {error.required_mb:.0f}MB")
        self.logger.error(f"Available: {error.available_mb:.0f}MB")
        self.logger.error("")
        self.logger.error("Suggestions:")
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
        
        # Fallback to current directory
        config_dir = Path("config")
        if config_dir.exists():
            return config_dir
        
        # Create default config directory
        config_dir.mkdir(exist_ok=True)
        self.logger.warning(f"Created config directory at: {config_dir}")
        return config_dir
    
    def _serialize_config(self, config: ExpandorConfig) -> Dict[str, Any]:
        """Serialize config for metadata storage"""
        return {
            "source_image": str(config.source_image) if isinstance(config.source_image, Path) else "<PIL.Image>",
            "target_resolution": config.target_resolution,
            "prompt": config.prompt,
            "seed": config.seed,
            "quality_preset": config.quality_preset,
            "source_metadata": config.source_metadata,
            "save_stages": config.save_stages,
            "stage_dir": str(config.stage_dir) if config.stage_dir else None,
            "auto_refine": config.auto_refine,
            "allow_tiled": config.allow_tiled,
            "allow_cpu_offload": config.allow_cpu_offload,
            "strategy_override": config.strategy_override
        }
    
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