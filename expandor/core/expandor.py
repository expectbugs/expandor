"""
Main Expandor class - Universal image expansion orchestrator
"""

import atexit
import gc
import glob
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .._version import __version__
from ..adapters.base_adapter import BasePipelineAdapter
from ..processors.quality_orchestrator import QualityOrchestrator
from ..processors.quality_validator import QualityValidator
from ..processors.seam_repair import SeamRepairProcessor
from ..utils.config_loader import ConfigLoader
from ..utils.logging_utils import setup_logger
from ..utils.path_resolver import PathResolver
from .boundary_tracker import BoundaryTracker
from .config import ExpandorConfig
from .exceptions import ExpandorError, VRAMError
from .metadata_tracker import MetadataTracker
from .pipeline_orchestrator import PipelineOrchestrator
from .result import ExpandorResult
from .strategy_selector import StrategySelector
from .vram_manager import VRAMManager


class Expandor:
    """Universal image expansion and adaptation system"""

    def __init__(
        self,
        pipeline_adapter: "BasePipelineAdapter",
        config_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Expandor with pipeline adapter

        Args:
            pipeline_adapter: Required pipeline adapter instance
            config_path: Optional config directory path
            logger: Optional logger instance

        Raises:
            TypeError: If pipeline_adapter is not provided
            ExpandorError: If configuration loading fails
        """
        # Validate adapter
        if not pipeline_adapter:
            raise TypeError(
                "pipeline_adapter is required.\n"
                "Example: expandor = Expandor(DiffusersPipelineAdapter())"
            )

        # Setup logging
        self.logger = logger or setup_logger("expandor", level=logging.INFO)
        self.logger.info(
            "Initializing Expandor with %s", type(pipeline_adapter).__name__
        )

        # Set adapter
        self.pipeline_adapter = pipeline_adapter

        # Load configuration
        try:
            config_dir = config_path or self._get_default_config_path()
            self.config_loader = ConfigLoader(config_dir)
            self.config = self.config_loader.load_all_configs()
            self.logger.info("Configuration loaded from: %s", config_dir)
        except Exception as e:
            raise ExpandorError(
                f"Configuration loading failed: {str(e)}\n"
                f"Run 'expandor --setup' to create valid configuration",
                stage="initialization",
            )

        # Initialize all components
        self._initialize_components()

        # Setup cleanup
        self._temp_files: List[Path] = []
        atexit.register(self._cleanup_temp_files)

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
                config=config,
            )

        # Apply quality preset settings to config
        self._apply_quality_preset(config)

        # Log operation start
        self.logger.info(
            f"Starting expansion: {config.source_image} -> {config.get_target_resolution()} "
            f"[{config.quality_preset} quality]"
        )

        try:
            # Pre-execution setup
            self._prepare_workspace(config)

            # Pipelines are now managed by the adapter
            # No need to register from config

            # Select strategy with VRAM awareness
            self.logger.info("Selecting optimal expansion strategy...")
            strategy = self.strategy_selector.select(config)
            self.metadata_tracker.record_event(
                "strategy_selected",
                {
                    "strategy": strategy.__class__.__name__,
                    "reason": self.strategy_selector.get_selection_reason(),
                },
            )

            # Execute with fallback chain
            self.logger.info(f"Executing {strategy.__class__.__name__}...")
            result = self.orchestrator.execute(
                strategy=strategy,
                config=config,
                metadata_tracker=self.metadata_tracker,
                boundary_tracker=self.boundary_tracker,
                temp_dir=self._temp_base,
            )

            # Post-execution validation and repair
            self.logger.info(
                "Validating quality and checking for artifacts...")
            result = self._validate_and_repair(result, config)

            # Calculate final metrics
            result.total_duration_seconds = time.time() - operation_start
            result.vram_peak_mb = self.vram_manager.get_peak_usage()

            # Update metadata with complete operation info
            result.metadata.update(
                {
                    "expandor_version": __version__,
                    "config_snapshot": self._serialize_config(config),
                    "operation_log": self.metadata_tracker.get_operation_log(),
                    "boundary_positions": self.boundary_tracker.get_all_boundaries(),
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
            self.metadata_tracker.record_event(
                "expansion_failed",
                {
                    "error": str(e),
                    "stage": self.metadata_tracker.current_stage,
                    "duration": time.time() - operation_start,
                },
            )

            # Re-raise with context
            raise ExpandorError(
                f"Expansion failed during {
                    self.metadata_tracker.current_stage}: {
                    str(e)}",
                stage=self.metadata_tracker.current_stage,
                config=config,
                partial_result=self.metadata_tracker.get_partial_result(),
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
        target_w, target_h = config.get_target_resolution()
        base_vram = self.vram_manager.calculate_generation_vram(
            target_w, target_h)

        # Get strategy-specific requirements
        strategy = self.strategy_selector.select(config, dry_run=True)
        strategy_estimate = strategy.estimate_vram(config)

        return {
            "total_required_mb": max(
                base_vram, strategy_estimate.get("peak_vram_mb", 0)
            ),
            "available_mb": self.vram_manager.get_available_vram() or 0,
            "base_estimate_mb": base_vram,
            "strategy_estimate": strategy_estimate,
            "recommended_strategy": strategy.__class__.__name__,
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

        if config.seed is not None and config.seed < 0:
            raise ValueError(
                f"Invalid seed: {config.seed} (must be non-negative)"
            )

        # Resolution validation
        target_w, target_h = config.get_target_resolution()
        if target_w <= 0 or target_h <= 0:
            raise ValueError(
                f"Invalid target resolution: {target_w}x{target_h}")

        if target_w > 65536 or target_h > 65536:
            raise ValueError(
                f"Target resolution too large: {target_w}x{target_h} (max 65536)")

        # Source image validation
        if isinstance(config.source_image, Path):
            if not config.source_image.exists():
                raise FileNotFoundError(
                    f"Source image not found: {config.source_image}"
                )

            # Try to open and validate
            try:
                with Image.open(config.source_image) as img:
                    if img.size[0] <= 0 or img.size[1] <= 0:
                        raise ValueError(
                            f"Invalid source image dimensions: {
                                img.size}"
                        )
            except Exception as e:
                raise ValueError(f"Cannot read source image: {str(e)}")

        elif isinstance(config.source_image, Image.Image):
            if config.source_image.size[0] <= 0 or config.source_image.size[1] <= 0:
                raise ValueError(
                    f"Invalid source image dimensions: {
                        config.source_image.size}"
                )
        else:
            raise TypeError(
                f"source_image must be Path or PIL.Image, got {
                    type(
                        config.source_image)}")

        # Quality preset validation
        available_presets = list(self.config.get("quality_presets", {}).keys())
        if not available_presets:
            available_presets = ["ultra", "high", "balanced", "fast"]

        if config.quality_preset not in available_presets:
            raise ValueError(
                f"Invalid quality preset: {config.quality_preset}. "
                f"Available: {', '.join(available_presets)}"
            )

        # Pipeline validation - pipelines now managed by adapter
        # The adapter should provide the necessary pipelines
        if not self.pipeline_adapter:
            self.logger.warning(
                "No pipeline adapter provided. Only basic operations will be available."
            )

    def _prepare_workspace(self, config: ExpandorConfig):
        """Prepare directories and workspace for execution"""
        # Create temp subdirectories using PathResolver
        path_resolver = PathResolver(self.logger)
        temp_subdirs = ["progressive", "tiles", "upscale", "masks", "stages"]

        for subdir in temp_subdirs:
            dir_path = self._temp_base / subdir
            path_resolver.resolve_path(dir_path, create=True, path_type="directory")

        # Create stage directory if saving stages using PathResolver
        if config.save_stages and config.stage_dir:
            path_resolver.resolve_path(config.stage_dir, create=True, path_type="directory")

    @property
    def temp_dir(self) -> Path:
        """Get the temporary directory for this instance"""
        return self._temp_base

    def _validate_and_repair(
        self, result: ExpandorResult, config: ExpandorConfig
    ) -> ExpandorResult:
        """
        Validate quality and repair artifacts if needed

        Args:
            result: Initial expansion result
            config: Expansion configuration

        Returns:
            Updated result with quality metrics and repairs
        """
        # Skip validation in fast mode
        if config.quality_preset == "fast":
            result.quality_score = 1.0
            return result

        # Get boundaries for artifact detection
        critical_boundaries = self.boundary_tracker.get_critical_boundaries()

        # Validate quality
        validation_metadata = {
            "boundaries": critical_boundaries,
            "quality_preset": config.quality_preset,
        }
        validation_result = self.quality_validator.validate(
            image_path=result.image_path,
            metadata=validation_metadata,
            detection_level=config.quality_preset,
        )

        # Extract required validation results - FAIL LOUD if missing
        if "quality_score" not in validation_result:
            raise ExpandorError(
                "Validation result missing required 'quality_score' field",
                stage="validation",
                details={"available_keys": list(validation_result.keys())}
            )
        if "seam_count" not in validation_result:
            raise ExpandorError(
                "Validation result missing required 'seam_count' field",
                stage="validation",
                details={"available_keys": list(validation_result.keys())}
            )
        
        result.quality_score = validation_result["quality_score"]
        result.seams_detected = validation_result["seam_count"]

        # Check if repair needed - auto_refine is now always enabled for
        # quality
        # Check for issues - FAIL LOUD if field missing
        if "issues_found" not in validation_result:
            raise ExpandorError(
                "Validation result missing required 'issues_found' field",
                stage="validation",
                details={"available_keys": list(validation_result.keys())}
            )
        
        if validation_result["issues_found"]:
            self.logger.warning(
                f"Quality issues detected: score={validation_result['quality_score']}"
            )

            # Attempt repair if adapter supports inpainting
            if self.pipeline_adapter and self.pipeline_adapter.supports_inpainting():
                try:
                    # Clear VRAM before artifact repair - critical for high resolution
                    self.logger.info("Clearing VRAM before artifact repair...")
                    import gc
                    import torch
                    
                    # Aggressive memory clearing
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        
                        # Log VRAM status before
                        free_before = torch.cuda.mem_get_info()[0] / (1024**2)
                        total_vram = torch.cuda.mem_get_info()[1] / (1024**2)
                        self.logger.info(f"VRAM before clearing: {free_before:.0f}MB free / {total_vram:.0f}MB total")
                        
                        # If VRAM is critically low, enable CPU offload
                        # Get minimum VRAM requirement from config
                        vram_thresholds = self.config_manager.get_value('vram.thresholds')
                        required_vram = vram_thresholds['tiled_processing']  # Use tiled threshold as minimum for repair
                        
                        if free_before < required_vram:
                            self.logger.warning(f"Low VRAM detected ({free_before:.0f}MB < {required_vram}MB required)")
                            self.logger.info("Enabling CPU offload for memory efficiency...")
                            
                            # Enable CPU offload on adapter if available
                            if hasattr(self.pipeline_adapter, 'enable_cpu_offload'):
                                self.pipeline_adapter.enable_cpu_offload()
                            
                            # Clear again after offload
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        
                        # Final VRAM status
                        free_after = torch.cuda.mem_get_info()[0] / (1024**2)
                        self.logger.info(f"VRAM after optimization: {free_after:.0f}MB free / {total_vram:.0f}MB total")
                    
                    repair_processor = SeamRepairProcessor(
                        pipelines=self.pipeline_registry, 
                        logger=self.logger,
                        config=self.config.get('processing_params', {})
                    )

                    # Create artifact mask from issues
                    # Extract seam locations for artifact mask - FAIL LOUD if missing
                    if "details" not in validation_result:
                        raise ExpandorError(
                            "Validation result missing required 'details' field for artifact repair",
                            stage="repair"
                        )
                    if "seam_locations" not in validation_result["details"]:
                        raise ExpandorError(
                            "Validation details missing required 'seam_locations' field for artifact repair",
                            stage="repair"
                        )
                    
                    artifact_mask = self._create_artifact_mask(
                        result.image_path, validation_result["details"]["seam_locations"]
                    )

                    # Repair with OOM protection
                    try:
                        repair_result = repair_processor.repair_seams(
                            image_path=result.image_path,
                            artifact_mask=artifact_mask,
                            prompt=config.prompt,
                            metadata=result.metadata,
                        )
                    except torch.cuda.OutOfMemoryError as oom_error:
                        self.logger.warning(f"OOM during artifact repair: {oom_error}")
                        self.logger.info("Attempting repair with more aggressive memory management...")
                        
                        # Clear everything again
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        
                        # Try once more with reduced settings if possible
                        try:
                            # Reduce inference steps for lower memory usage
                            reduced_metadata = result.metadata.copy()
                            
                            # Get seam repair config for reduced settings
                            seam_config = self.config_manager.get_processor_config('seam_repair')
                            
                            # Use approximately half of normal values for low memory
                            normal_steps = seam_config.get('artifact_repair_steps', seam_config['seam_repair_steps'])
                            normal_strength = seam_config.get('artifact_repair_strength', seam_config['seam_repair_strength'])
                            
                            reduced_metadata['artifact_repair_steps'] = max(10, normal_steps // 2)  # Half steps, min 10
                            reduced_metadata['artifact_repair_strength'] = max(0.2, normal_strength * 0.6)  # 60% strength, min 0.2
                            
                            repair_result = repair_processor.repair_seams(
                                image_path=result.image_path,
                                artifact_mask=artifact_mask,
                                prompt=config.prompt,
                                metadata=reduced_metadata,
                            )
                            self.logger.info("Successfully repaired with reduced settings")
                        except Exception as e2:
                            self.logger.error(f"Failed even with reduced settings: {e2}")
                            # Create a dummy result to continue
                            repair_result = {
                                "success": False,
                                "image_path": result.image_path,
                                "repairs_made": 0,
                                "passes": 0,
                                "final_score": result.quality_score
                            }

                    if repair_result["success"]:
                        result.image_path = repair_result["image_path"]
                        result.artifacts_fixed = repair_result["repairs_made"]
                        result.refinement_passes = repair_result["passes"]
                        result.quality_score = repair_result["final_score"]

                        self.logger.info(
                            f"Repaired {result.artifacts_fixed} artifacts in "
                            f"{result.refinement_passes} passes"
                        )

                except Exception as e:
                    self.logger.error(f"Artifact repair failed: {e}")
                    # Continue with original result

        return result

    def _create_artifact_mask(
            self,
            image_path: Path,
            issues: List[Dict]) -> np.ndarray:
        """Create mask for detected artifacts"""
        # This is a simplified implementation
        # In production, would use sophisticated artifact detection
        with Image.open(image_path) as img:
            mask = np.zeros((img.height, img.width), dtype=np.uint8)

            # Mark boundary regions
            for issue in issues:
                if "location" in issue:
                    x, y, w, h = issue["location"]
                    mask[y:y + h, x: x + w] = 255

            return mask

    def _save_metadata(self, result: ExpandorResult):
        """Save metadata JSON alongside result image"""
        metadata_path = result.image_path.with_suffix(".json")

        # Convert result to serializable format
        metadata = result.to_dict()

        # Add operation log
        metadata["operation_log"] = self.metadata_tracker.get_operation_log()

        # Save as JSON
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.debug(f"Saved metadata to: {metadata_path}")

    def _cleanup_temp_files(self):
        """Clean up temporary files created during processing"""
        # Clean up files from _temp_files list
        if hasattr(self, "_temp_files"):
            for temp_file in self._temp_files:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                        self.logger.debug(f"Cleaned up temp file: {temp_file}")
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to clean up {temp_file}: {e}")

        # Clean up files in temp subdirectories
        if hasattr(self, "_temp_base") and self._temp_base.exists():
            # Get cleanup patterns from config
            cleanup_patterns = self._config_manager.get_value("processing.temp_cleanup_patterns")
            temp_patterns = [
                self._temp_base / pattern for pattern in cleanup_patterns
            ]

            for pattern in temp_patterns:
                for file_path in glob.glob(str(pattern)):
                    try:
                        Path(file_path).unlink()
                    except Exception as e:
                        self.logger.debug(
                            f"Failed to delete temp file {file_path}: {e}"
                        )

    def _cleanup_temp_base(self):
        """Clean up the entire temp directory on exit"""
        if hasattr(self, "_temp_base") and self._temp_base.exists():
            try:
                shutil.rmtree(self._temp_base)
                self.logger.debug(
                    f"Cleaned up temp directory: {
                        self._temp_base}"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to clean up temp directory {self._temp_base}: {e}"
                )

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

    def _initialize_components(self):
        """Initialize all core components"""
        try:
            # Configuration Manager - Must be first for all components to use
            from .configuration_manager import ConfigurationManager
            self.config_manager = ConfigurationManager()
            
            # VRAM Manager - Must be early to check hardware
            self.vram_manager = VRAMManager(self.logger)
            available_vram = self.vram_manager.get_available_vram()
            if available_vram:
                self.logger.info(f"VRAM available: {available_vram:.0f}MB")
            else:
                self.logger.warning("No CUDA device detected - CPU mode only")

            # Strategy Selector with config and VRAM awareness
            self.strategy_selector = StrategySelector(
                config=self.config, vram_manager=self.vram_manager, logger=self.logger)

            # Pipeline Orchestrator for executing strategies
            self.orchestrator = PipelineOrchestrator(
                config=self.config, logger=self.logger
            )

            # Metadata Tracker for operation history
            self.metadata_tracker = MetadataTracker(self.logger)

            # Boundary Tracker for seam detection
            self.boundary_tracker = BoundaryTracker(self.logger)

            # Quality Validator
            self.quality_validator = QualityValidator(self.config, self.logger)

            # Pipeline registry for external models
            self.pipeline_registry = {}

            # Register pipelines from adapter
            if self.pipeline_adapter:
                # Common pipeline types to check
                pipeline_types = ["inpaint", "img2img", "refiner", "upscale"]
                for pipeline_type in pipeline_types:
                    try:
                        pipeline = self.pipeline_adapter.get_pipeline(
                            pipeline_type)
                        if pipeline:
                            self.pipeline_registry[pipeline_type] = pipeline
                            self.logger.info(
                                f"Registered {pipeline_type} pipeline from adapter")
                        else:
                            self.logger.debug(
                                f"No {pipeline_type} pipeline available from adapter")
                    except Exception as e:
                        # Adapter may not support all pipeline types
                        self.logger.debug(
                            f"Could not get {pipeline_type} pipeline: {e}")

                self.logger.info(
                    f"Total pipelines registered: {len(self.pipeline_registry)}")

                # Share pipeline registry with orchestrator
                self.orchestrator.pipeline_registry = self.pipeline_registry

            # Cache for loaded strategies
            self._strategy_cache = {}

            # Create temp directory
            self._temp_base = Path(tempfile.mkdtemp(prefix="expandor_"))
            self.logger.debug(f"Created temp directory: {self._temp_base}")

            # Register cleanup on exit
            atexit.register(self._cleanup_temp_base)

        except Exception as e:
            raise ExpandorError(
                f"Failed to initialize core components: {str(e)}",
                stage="initialization",
            )

        self.logger.info("Expandor initialization complete")

    def _serialize_config(self, config: ExpandorConfig) -> Dict[str, Any]:
        """Serialize config for metadata storage"""
        return {
            "source_image": (
                str(config.source_image)
                if isinstance(config.source_image, Path)
                else "<PIL.Image>"
            ),
            "target_resolution": config.get_target_resolution(),
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt,
            "seed": config.seed,
            "quality_preset": config.quality_preset,
            "source_metadata": config.source_metadata,
            "save_stages": config.save_stages,
            "stage_dir": str(config.stage_dir) if config.stage_dir else None,
            "strategy": config.strategy,
            "strategy_override": config.strategy_override,
            "use_cpu_offload": config.use_cpu_offload,
            "denoising_strength": config.denoising_strength,
            "guidance_scale": config.guidance_scale,
            "num_inference_steps": config.num_inference_steps,
        }

    def _apply_quality_preset(self, config: ExpandorConfig) -> None:
        """Apply quality preset settings to config"""
        try:
            # Load quality preset
            quality_preset = self.config_loader.load_quality_preset(config.quality_preset)
            
            # Apply expansion settings if present
            if "expansion" in quality_preset:
                expansion_settings = quality_preset["expansion"]
                if "denoising_strength" in expansion_settings:
                    config.denoising_strength = expansion_settings["denoising_strength"]
                    self.logger.debug(
                        f"Applied denoising_strength={config.denoising_strength} from {config.quality_preset} preset"
                    )
            
            # Apply generation settings if present
            if "generation" in quality_preset:
                gen_settings = quality_preset["generation"]
                if "guidance_scale" in gen_settings:
                    config.guidance_scale = gen_settings["guidance_scale"]
                if "num_inference_steps" in gen_settings:
                    config.num_inference_steps = gen_settings["num_inference_steps"]
                    
            self.logger.info(
                f"Applied quality preset '{config.quality_preset}': "
                f"denoising={config.denoising_strength}, "
                f"guidance={config.guidance_scale}, "
                f"steps={config.num_inference_steps}"
            )
            
        except Exception as e:
            self.logger.warning(
                f"Failed to apply quality preset '{config.quality_preset}': {str(e)}. "
                f"Using default values."
            )

    def validate_quality(
        self, result: ExpandorResult, config: ExpandorConfig
    ) -> Dict[str, Any]:
        """
        Validate and potentially refine the expansion result.

        Args:
            result: The expansion result to validate
            config: The configuration used for expansion

        Returns:
            Quality validation results

        Raises:
            QualityError: If quality requirements cannot be met
        """

        # Initialize quality orchestrator with config from ConfigurationManager
        try:
            # Get quality orchestrator config
            orchestrator_config = self.config_manager.get_processor_config('quality_orchestrator')
            
            # Get quality preset specific threshold if available
            if config.quality_preset and config.quality_preset != "custom":
                preset_config = self.config_manager.get_value(f'quality_presets.{config.quality_preset}')
                quality_threshold = preset_config.get('validation', {}).get('quality_threshold', orchestrator_config['quality_threshold'])
            else:
                quality_threshold = orchestrator_config['quality_threshold']
            
            quality_config = {
                "quality_validation": {
                    "quality_threshold": quality_threshold,
                    "max_refinement_passes": orchestrator_config['max_refinement_passes'],
                    "auto_fix_threshold": orchestrator_config['auto_fix_threshold'],
                }}
        except (KeyError, ValueError) as e:
            # FAIL LOUD on missing configuration
            raise ValueError(
                f"Quality orchestrator configuration not found!\n{str(e)}\n"
                f"Required: processors.quality_orchestrator configuration"
            )

        orchestrator = QualityOrchestrator(quality_config, self.logger)

        # Prepare pipeline registry from adapter
        pipeline_registry = self.pipeline_registry

        # Validate and refine
        validation_results = orchestrator.validate_and_refine(
            image_path=result.image_path,
            boundary_tracker=self.boundary_tracker,
            pipeline_registry=pipeline_registry,
            config=config,
        )

        # Update result if refined
        if validation_results["final_image"] != str(result.image_path):
            result.image_path = Path(validation_results["final_image"])
            result.metadata["quality_refined"] = True
            result.metadata["refinement_passes"] = validation_results[
                "refinement_passes"
            ]
            result.metadata["final_quality_score"] = validation_results["quality_score"]

        return validation_results

    def clear_caches(self):
        """
        Clear all internal caches to free memory

        Useful for long-running processes or after processing large images
        """
        # Clear strategy cache in selector
        if hasattr(self, "strategy_selector"):
            self.strategy_selector.clear_cache()

        # Clear any cached strategies
        if hasattr(self, "_strategy_cache"):
            self._strategy_cache.clear()

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self.logger.info("Cleared all caches and freed memory")
