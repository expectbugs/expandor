"""
Diffusers library adapter for Expandor
Provides real pipeline implementation using HuggingFace Diffusers
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from diffusers import (AutoPipelineForImage2Image, AutoPipelineForInpainting,
                       DPMSolverMultistepScheduler,
                       StableDiffusionXLImg2ImgPipeline,
                       StableDiffusionXLInpaintPipeline,
                       StableDiffusionXLPipeline)
from PIL import Image

from ..utils.config_loader import ConfigLoader
from ..utils.logging_utils import setup_logger
from .base_adapter import BasePipelineAdapter
from ..core.configuration_manager import ConfigurationManager


class DiffusersPipelineAdapter(BasePipelineAdapter):
    """
    Adapter for HuggingFace Diffusers pipelines

    Supports:
    - SDXL, SD 1.5, SD 2.x models
    - Image-to-image and inpainting
    - LoRA loading and unloading
    - SDXL refiner models
    - ControlNet for SDXL models (optional feature)

    ControlNet Support (Phase 5):
    - Requires diffusers>=0.24.0 with controlnet extras
    - Currently limited to SDXL models
    - Supports multiple control types (canny, depth, etc.)
    - Use load_controlnet() to load models
    - Use controlnet_* methods for guided generation
    - ALL parameters are REQUIRED - no defaults or auto-detection
    - ALL values from configuration files - no hardcoding
    - Configure via controlnet_config.yaml

    Example:
        # Basic usage
        adapter = DiffusersPipelineAdapter("stabilityai/stable-diffusion-xl-base-1.0")

        # Load ControlNet using model ID from config
        config = adapter._ensure_controlnet_config()
        canny_model_id = config["models"]["sdxl"]["canny"]
        adapter.load_controlnet(canny_model_id, "canny")

        # Or directly with known model ID
        adapter.load_controlnet("diffusers/controlnet-canny-sdxl-1.0", "canny")

        # ALL parameters MUST be provided
        result = adapter.controlnet_img2img(
            image=source,
            control_image=edges,
            prompt="a beautiful landscape",
            negative_prompt="blurry, low quality",  # REQUIRED
            control_type="canny",  # REQUIRED
            controlnet_strength=0.8,  # REQUIRED
            strength=0.75,  # REQUIRED
            num_inference_steps=50,  # REQUIRED
            guidance_scale=7.5  # REQUIRED
        )
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        variant: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_safetensors: Optional[bool] = None,
        enable_xformers: Optional[bool] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        """
        Initialize Diffusers adapter

        Args:
            model_id: HuggingFace model ID
            model_path: Local model path
            device: Device to use
            torch_dtype: Model dtype
            variant: Model variant
            cache_dir: Cache directory for downloads
            use_safetensors: Use safetensors format
            enable_xformers: Enable xformers optimization
            logger: Logger instance
            **kwargs: Additional pipeline arguments
        """
        # Load defaults from configuration - FAIL LOUD if not found
        config_manager = ConfigurationManager()
        
        # Apply configuration defaults for None values
        if device is None:
            device = config_manager.get_value("adapters.diffusers.default_device")
        if torch_dtype is None:
            dtype_str = config_manager.get_value("adapters.diffusers.default_torch_dtype")
            torch_dtype = getattr(torch, dtype_str)
        if variant is None:
            variant = config_manager.get_value("adapters.diffusers.default_variant")
        if use_safetensors is None:
            use_safetensors = config_manager.get_value("adapters.diffusers.default_use_safetensors")
        if enable_xformers is None:
            enable_xformers = config_manager.get_value("adapters.diffusers.default_enable_xformers")
        
        self.model_id = model_id
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.variant = variant
        self.cache_dir = cache_dir
        self.use_safetensors = use_safetensors
        self.enable_xformers = enable_xformers
        self.logger = logger or setup_logger(__name__)
        self.kwargs = kwargs

        # Pipeline instances
        self.inpaint_pipeline = None
        self.img2img_pipeline = None
        self.base_pipeline = None
        self.refiner_pipeline = None

        # LoRA tracking
        self.loaded_loras: Dict[str, float] = {}

        # ControlNet instances (optional feature - lazy initialized)
        self.controlnet_models: Dict[str, Any] = {}
        self.controlnet_pipeline = None  # Single pipeline, swap models dynamically
        self.active_controlnet = None  # Currently active controlnet model

        # Configuration loader - initialized immediately
        config_dir = Path(__file__).parent.parent / "config"
        self.config_loader = ConfigLoader(config_dir, logger=self.logger)

        # Pre-load critical configs for efficiency
        self._vram_config = None  # Loaded when VRAM operations needed
        self._controlnet_config = None  # Loaded when ControlNet used

        # Model type detection
        self.model_type = None
        self.model_config = None
        self.model_type_registry = self._get_model_type_registry()

        # Initialize pipelines
        self._initialize_pipelines()

    def _get_config_value(self, section: str, key: str) -> Any:
        """Get configuration value - FAIL LOUD, no defaults"""
        try:
            # Use ConfigurationManager instead of loading file directly
            from ..core.configuration_manager import ConfigurationManager
            config_manager = ConfigurationManager()
            
            # Build the config path
            config_path = f"processors.{section}.{key}"
            return config_manager.get_value(config_path)
            
        except (ValueError, KeyError) as e:
            # Fail loud - config required
            raise ValueError(
                f"Failed to load {section}.{key} from configuration: {e}\n"
                f"This value must be defined in master_defaults.yaml"
            )

    def _ensure_vram_config(self) -> Dict[str, Any]:
        """Ensure VRAM config is loaded - FAIL LOUD if missing"""
        if self._vram_config is None:
            try:
                self._vram_config = self.config_loader.load_config_file(
                    "vram_strategies.yaml")
            except FileNotFoundError:
                raise FileNotFoundError(
                    "vram_strategies.yaml not found in config directory.\n"
                    "This file is REQUIRED for VRAM estimation.\n"
                    "Create the file or run 'expandor --setup'"
                )
        return self._vram_config

    def _ensure_controlnet_config(self) -> Dict[str, Any]:
        """Ensure ControlNet config is loaded - FAIL LOUD if missing"""
        if self._controlnet_config is None:
            try:
                self._controlnet_config = self.config_loader.load_config_file(
                    "controlnet_config.yaml")
                # Validate the loaded config
                self._validate_controlnet_config(self._controlnet_config)
            except FileNotFoundError:
                raise FileNotFoundError(
                    "controlnet_config.yaml not found.\n"
                    "ControlNet requires configuration to be set up first.\n"
                    "Please run: expandor --setup-controlnet\n"
                    "This will create the necessary configuration files."
                )
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Invalid YAML in controlnet_config.yaml: {
                        str(e)}\n" "Please check the file for syntax errors.\n"
                    "You can regenerate it with: expandor --setup-controlnet --force")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load ControlNet config: {str(e)}\n"
                    "This is unexpected. Please check the config file."
                )

        return self._controlnet_config

    def _validate_controlnet_config(self, config: Dict[str, Any]) -> None:
        """Validate ControlNet configuration structure"""
        required_sections = ["defaults", "extractors", "pipelines", "models"]
        missing_sections = [s for s in required_sections if s not in config]

        if missing_sections:
            raise ValueError(
                f"Invalid controlnet_config.yaml - missing sections: {missing_sections}\n"
                "The config file may be corrupted or outdated.\n"
                "Regenerate with: expandor --setup-controlnet --force"
            )

        # Validate defaults section
        if "defaults" in config:
            required_defaults = [
                "controlnet_strength",
                "strength",
                "num_inference_steps",
                "guidance_scale"]
            defaults = config["defaults"]
            missing_defaults = [
                d for d in required_defaults if d not in defaults]
            if missing_defaults:
                raise ValueError(
                    f"Missing required default values: {missing_defaults}\n"
                    "Please check your controlnet_config.yaml"
                )

    def _get_controlnet_defaults(self) -> Dict[str, Any]:
        """Get default values for ControlNet operations from config"""
        config = self._ensure_controlnet_config()
        if "defaults" not in config:
            raise ValueError(
                "defaults section not found in controlnet_config.yaml\n"
                "This section is REQUIRED for ControlNet operations.\n"
                "The config file should have been auto-created with defaults.\n"
                "If it exists but is missing 'defaults', the file may be corrupted.")
        return config["defaults"]

    def _get_model_type_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Get model type detection registry

        QUALITY OVER ALL: Extensible detection for all model types
        """
        return {
            "sdxl": {
                "patterns": ["stable-diffusion-xl", "sdxl", "xl-base", "xl-refiner"],
                "pipeline_class": "StableDiffusionXLPipeline",
                "inpaint_class": "StableDiffusionXLInpaintPipeline",
                "img2img_class": "StableDiffusionXLImg2ImgPipeline",
                "refiner_class": "StableDiffusionXLImg2ImgPipeline",
                "optimal_resolution": (1024, 1024),
                "resolution_multiple": 8,
                "vae_scale_factor": 8,
            },
            "sd3": {
                "patterns": ["stable-diffusion-3", "sd3", "sd-3"],
                "pipeline_class": "StableDiffusion3Pipeline",
                "inpaint_class": "StableDiffusion3InpaintPipeline",
                "img2img_class": "StableDiffusion3Img2ImgPipeline",
                "refiner_class": None,  # SD3 doesn't use separate refiner
                "optimal_resolution": (1024, 1024),
                "resolution_multiple": 16,
                "vae_scale_factor": 8,
            },
            "flux": {
                "patterns": ["flux", "black-forest-labs/flux"],
                "pipeline_class": "FluxPipeline",
                "inpaint_class": "FluxInpaintPipeline",
                "img2img_class": "FluxImg2ImgPipeline",
                "refiner_class": None,
                "optimal_resolution": (1024, 1024),
                "resolution_multiple": 16,
                "vae_scale_factor": 8,
            },
            "sd2": {
                "patterns": [
                    "stable-diffusion-2",
                    "sd2",
                    "stabilityai/stable-diffusion-2",
                ],
                "pipeline_class": "StableDiffusionPipeline",
                "inpaint_class": "StableDiffusionInpaintPipeline",
                "img2img_class": "StableDiffusionImg2ImgPipeline",
                "refiner_class": None,
                "optimal_resolution": (768, 768),
                "resolution_multiple": 8,
                "vae_scale_factor": 8,
            },
            "sd15": {
                "patterns": [
                    "stable-diffusion-v1",
                    "sd-v1",
                    "runwayml/stable-diffusion",
                ],
                "pipeline_class": "StableDiffusionPipeline",
                "inpaint_class": "StableDiffusionInpaintPipeline",
                "img2img_class": "StableDiffusionImg2ImgPipeline",
                "refiner_class": None,
                "optimal_resolution": (512, 512),
                "resolution_multiple": 8,
                "vae_scale_factor": 8,
            },
        }

    def _detect_model_type(self) -> str:
        """
        Detect model type from model ID or path

        FAIL LOUD: Unknown model types cause errors
        """
        model_identifier = self.model_id or self.model_path or ""
        model_identifier_lower = model_identifier.lower()

        # Check against registry patterns
        for model_type, config in self.model_type_registry.items():
            for pattern in config["patterns"]:
                if pattern in model_identifier_lower:
                    self.logger.info(f"Detected model type: {model_type}")
                    return model_type

        # If we can't detect, try to load model config and inspect
        if self.model_id:
            try:
                from huggingface_hub import model_info

                info = model_info(self.model_id)

                # Check model card and tags
                if info.tags:
                    for tag in info.tags:
                        tag_lower = tag.lower()
                        for model_type, config in self.model_type_registry.items():
                            if any(
                                    pattern in tag_lower for pattern in config["patterns"]):
                                self.logger.info(
                                    f"Detected model type from tags: {model_type}")
                                return model_type

            except Exception as e:
                self.logger.warning(f"Could not fetch model info: {e}")

        # FAIL LOUD - don't guess
        raise ValueError(
            f"Could not detect model type for '{model_identifier}'.\n"
            f"Supported types: {list(self.model_type_registry.keys())}\n"
            f"Please ensure your model ID contains one of these patterns:\n"
            + "\n".join(
                [
                    f"  {t}: {', '.join(c['patterns'])}"
                    for t, c in self.model_type_registry.items()
                ]
            )
        )

    def _initialize_pipelines(self):
        """Initialize the diffusers pipelines"""
        self.logger.info("Initializing Diffusers pipelines...")

        # Detect model type first
        self.model_type = self._detect_model_type()
        self.model_config = self.model_type_registry[self.model_type]

        # Determine model source
        if self.model_path:
            model_source = Path(self.model_path)
            if not model_source.exists():
                raise ValueError(
                    f"Model path does not exist: {
                        self.model_path}"
                )
            model_source = str(model_source)  # Convert to string for diffusers
        elif self.model_id:
            model_source = self.model_id
        else:
            raise ValueError("Either model_id or model_path must be provided")

        # Common arguments
        common_args = {
            "torch_dtype": self.torch_dtype,
            "use_safetensors": self.use_safetensors,
            "cache_dir": self.cache_dir,
            **self.kwargs,
        }

        if self.variant and self.model_id:  # variant only works with model_id
            common_args["variant"] = self.variant

        try:
            # Load base pipeline first to detect model type
            self.logger.info(f"Loading model from: {model_source}")

            # Try SDXL first (most common)
            try:
                # Check if it's a single file or a directory/model_id
                if self.model_path and self.model_path.endswith(('.safetensors', '.ckpt', '.bin')):
                    # Use from_single_file for checkpoint files
                    self.base_pipeline = StableDiffusionXLPipeline.from_single_file(
                        model_source,
                        torch_dtype=self.torch_dtype,
                        use_safetensors=self.use_safetensors,
                        add_watermarker=False,  # Disable watermarker
                        **self.kwargs
                    )
                else:
                    # Use from_pretrained for directories or HuggingFace IDs
                    self.base_pipeline = StableDiffusionXLPipeline.from_pretrained(
                        model_source, **common_args
                    )
                self.model_type = "sdxl"
                self.logger.info("Detected SDXL model")

                # Create specialized pipelines
                self.inpaint_pipeline = StableDiffusionXLInpaintPipeline(
                    vae=self.base_pipeline.vae,
                    text_encoder=self.base_pipeline.text_encoder,
                    text_encoder_2=self.base_pipeline.text_encoder_2,
                    tokenizer=self.base_pipeline.tokenizer,
                    tokenizer_2=self.base_pipeline.tokenizer_2,
                    unet=self.base_pipeline.unet,
                    scheduler=self.base_pipeline.scheduler,
                )

                self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline(
                    vae=self.base_pipeline.vae,
                    text_encoder=self.base_pipeline.text_encoder,
                    text_encoder_2=self.base_pipeline.text_encoder_2,
                    tokenizer=self.base_pipeline.tokenizer,
                    tokenizer_2=self.base_pipeline.tokenizer_2,
                    unet=self.base_pipeline.unet,
                    scheduler=self.base_pipeline.scheduler,
                )

            except Exception as e:
                self.logger.debug(f"Not an SDXL model: {e}")

                # Try auto-detection for other model types
                try:
                    self.inpaint_pipeline = AutoPipelineForInpainting.from_pretrained(
                        model_source, **common_args)
                    self.img2img_pipeline = AutoPipelineForImage2Image.from_pretrained(
                        model_source, **common_args)
                    self.base_pipeline = self.inpaint_pipeline
                    self.model_type = "auto"
                    self.logger.info("Using auto-detected model type")

                except Exception as e:
                    raise ValueError(f"Failed to load model: {e}")

            # Move to device
            self.base_pipeline = self.base_pipeline.to(self.device)
            self.inpaint_pipeline = self.inpaint_pipeline.to(self.device)
            self.img2img_pipeline = self.img2img_pipeline.to(self.device)

            # Enable optimizations
            if self.enable_xformers and self.device == "cuda":
                try:
                    self.base_pipeline.enable_xformers_memory_efficient_attention()
                    self.inpaint_pipeline.enable_xformers_memory_efficient_attention()
                    self.img2img_pipeline.enable_xformers_memory_efficient_attention()
                    self.logger.info(
                        "Enabled xformers memory efficient attention")
                except Exception as e:
                    self.logger.warning(f"Could not enable xformers: {e}")

            # Set scheduler to DPM++ for better quality
            scheduler = DPMSolverMultistepScheduler.from_config(
                self.base_pipeline.scheduler.config
            )
            self.base_pipeline.scheduler = scheduler
            self.inpaint_pipeline.scheduler = scheduler
            self.img2img_pipeline.scheduler = scheduler

            self.logger.info("Pipelines initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize pipelines: {e}")
            raise

    def load_refiner(
            self,
            refiner_id: Optional[str] = None,
            refiner_path: Optional[str] = None):
        """
        Load SDXL refiner model

        Args:
            refiner_id: HuggingFace model ID for refiner
            refiner_path: Local path to refiner
        """
        if self.model_type != "sdxl":
            self.logger.warning("Refiner is only supported for SDXL models")
            return

        refiner_source = refiner_path or refiner_id
        if not refiner_source:
            refiner_source = "stabilityai/stable-diffusion-xl-refiner-1.0"

        try:
            self.logger.info(f"Loading refiner from: {refiner_source}")

            common_args = {
                "torch_dtype": self.torch_dtype,
                "use_safetensors": self.use_safetensors,
                "cache_dir": self.cache_dir,
            }

            if self.variant and refiner_id:
                common_args["variant"] = self.variant

            from diffusers import \
                StableDiffusionXLImg2ImgPipeline as RefinerPipeline

            self.refiner_pipeline = RefinerPipeline.from_pretrained(
                refiner_source, **common_args
            ).to(self.device)

            if self.enable_xformers and self.device == "cuda":
                try:
                    self.refiner_pipeline.enable_xformers_memory_efficient_attention()
                    self.logger.info("Enabled xformers for refiner pipeline")
                except Exception as e:
                    # FAIL LOUD - xformers errors indicate configuration
                    # problems
                    raise RuntimeError(
                        f"Failed to enable xformers for refiner pipeline: {e}\n"
                        f"This usually means xformers is not properly installed.\n"
                        f"Solutions:\n"
                        f"  1. Install xformers: pip install xformers\n"
                        f"  2. Disable xformers in configuration\n"
                        f"  3. Use CPU device instead of CUDA"
                    )

            self.logger.info("Refiner loaded successfully")

        except RuntimeError:
            # Re-raise RuntimeError from xformers
            raise
        except Exception as e:
            # FAIL LOUD - refiner loading failure is critical
            raise RuntimeError(
                f"Failed to load refiner pipeline: {e}\n"
                f"Refiner source: {refiner_source}\n"
                f"This is a critical error - the refiner is required for quality.\n"
                f"Possible solutions:\n"
                f"  1. Check if the model ID is correct\n"
                f"  2. Ensure you have internet connection for model download\n"
                f"  3. Check available disk space for model cache\n"
                f"  4. Disable refiner in configuration if not needed"
            )

    def get_optimal_dimensions(
        self, target_width: int, target_height: int
    ) -> Tuple[int, int]:
        """
        Get optimal dimensions for the model type

        QUALITY OVER ALL: Use model-specific constraints
        """
        if not hasattr(self, "model_config") or not self.model_config:
            # If not initialized yet, use safe defaults
            # Load from config
            try:
                from pathlib import Path

                from ..utils.config_loader import ConfigLoader
                config_dir = Path(__file__).parent.parent / "config"
                loader = ConfigLoader(config_dir)
                proc_config = loader.load_config_file("processing_params.yaml")
                multiple = proc_config.get(
                    'diffusers_adapter', {}).get(
                    'sdxl_dimension_multiple', 8)
            except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
                raise ValueError(
                    f"Failed to load diffusers adapter configuration: {e}")
        else:
            multiple = self.model_config["resolution_multiple"]

        # Round to nearest multiple
        optimal_width = (target_width // multiple) * multiple
        optimal_height = (target_height // multiple) * multiple

        # Ensure minimum size based on model type
        if hasattr(self, "model_config") and self.model_config:
            min_res = min(self.model_config["optimal_resolution"])
            optimal_width = max(optimal_width, min_res)
            optimal_height = max(optimal_height, min_res)
        else:
            # Safe defaults
            optimal_width = max(optimal_width, 512)
            optimal_height = max(optimal_height, 512)

        # Cap at reasonable maximum - load from config
        try:
            from pathlib import Path

            from ..utils.config_loader import ConfigLoader
            config_dir = Path(__file__).parent.parent / "config"
            loader = ConfigLoader(config_dir)
            proc_config = loader.load_config_file("processing_params.yaml")
            max_dimension = proc_config.get(
                'diffusers_adapter', {}).get(
                'max_dimension', 4096)
        except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
            raise ValueError(
                f"Failed to load diffusers adapter configuration: {e}")
        optimal_width = min(optimal_width, max_dimension)
        optimal_height = min(optimal_height, max_dimension)

        self.logger.debug(
            f"Optimal dimensions for {getattr(self, 'model_type', 'unknown')}: "
            f"{target_width}x{target_height} -> {optimal_width}x{optimal_height}"
        )

        return (optimal_width, optimal_height)

    def _create_controlnet_pipeline(self):
        """
        Create a single ControlNet-enabled pipeline

        Uses single pipeline with dynamic model swapping for VRAM efficiency
        FAIL LOUD: Any initialization errors are fatal
        """
        # Import ControlNet pipeline classes
        try:
            from diffusers import StableDiffusionXLControlNetPipeline
        except ImportError as e:
            raise ImportError(
                f"Cannot import ControlNet pipeline classes: {e}\n"
                "ControlNet support requires diffusers>=0.24.0 with controlnet extras.\n"
                "Install with: pip install 'diffusers[controlnet]>=0.24.0'"
            )

        # Pipelines MUST have ControlNet models loaded
        if not self.controlnet_models:
            raise RuntimeError(
                "No ControlNet models loaded but pipeline initialization was called.\n"
                "This is an internal error. Please report this bug.")

        # Only SDXL supports ControlNet currently
        if self.model_type != "sdxl":
            raise RuntimeError(
                f"Cannot create ControlNet pipelines: requires SDXL, got {self.model_type}\n"
                f"This is a bug - ControlNet should not be loaded for non-SDXL models."
            )

        self.logger.info("Creating ControlNet pipeline...")

        try:
            # Ensure base components exist
            if not self.base_pipeline:
                raise RuntimeError(
                    "Cannot create ControlNet pipeline: base pipeline not initialized.\n"
                    "This is an internal error. Please report this issue.")

            # Use first loaded ControlNet as default
            first_type = next(iter(self.controlnet_models.keys()))
            self.active_controlnet = first_type
            controlnet_model = self.controlnet_models[first_type]

            # Create single pipeline that can be used for all operations
            # This is a text2img pipeline that we'll adapt for other operations
            self.controlnet_pipeline = StableDiffusionXLControlNetPipeline(
                vae=self.base_pipeline.vae,
                text_encoder=self.base_pipeline.text_encoder,
                text_encoder_2=self.base_pipeline.text_encoder_2,
                tokenizer=self.base_pipeline.tokenizer,
                tokenizer_2=self.base_pipeline.tokenizer_2,
                unet=self.base_pipeline.unet,
                scheduler=self.base_pipeline.scheduler.from_config(
                    self.base_pipeline.scheduler.config
                ),
                controlnet=controlnet_model,
            ).to(self.device)

            # Apply optimizations
            if self.enable_xformers and self.device == "cuda":
                self.controlnet_pipeline.enable_xformers_memory_efficient_attention()
                self.logger.debug("Enabled xformers for ControlNet pipeline")

            self.logger.info(
                f"Successfully created ControlNet pipeline with {first_type} model")

        except Exception as e:
            # FAIL LOUD with helpful error
            self.controlnet_pipeline = None
            raise RuntimeError(
                f"Failed to create ControlNet pipeline: {str(e)}\n"
                f"Error type: {type(e).__name__}\n"
                f"This may be due to:\n"
                f"  1. Incompatible diffusers version (need >=0.25.0)\n"
                f"  2. Missing ControlNet pipeline classes\n"
                f"  3. GPU memory constraints\n"
                f"To diagnose: pip show diffusers | grep Version"
            ) from e

    def _switch_controlnet(self, control_type: str):
        """
        Switch active ControlNet model in the pipeline

        Args:
            control_type: Type of control to switch to - REQUIRED

        FAIL LOUD: Invalid control type is an error
        """
        if control_type not in self.controlnet_models:
            raise ValueError(
                f"Control type '{control_type}' not loaded.\n"
                f"Available types: {list(self.controlnet_models.keys())}\n"
                f"Load it first with: adapter.load_controlnet(model_id, '{control_type}')"
            )

        if self.controlnet_pipeline is None:
            raise RuntimeError(
                "ControlNet pipeline not initialized.\n"
                "This is an internal error. Please report this bug."
            )

        # Only switch if needed
        if self.active_controlnet != control_type:
            self.logger.debug(
                f"Switching ControlNet from {
                    self.active_controlnet} to {control_type}")
            self.controlnet_pipeline.controlnet = self.controlnet_models[control_type]
            self.active_controlnet = control_type

    def load_lora(
            self,
            lora_path: str,
            weight: Optional[float] = None,
            adapter_name: Optional[str] = None):
        """
        Load LoRA weights

        Args:
            lora_path: Path to LoRA file
            weight: LoRA weight multiplier
            adapter_name: Name for the adapter
        """
        # Get default weight from configuration if not provided
        if weight is None:
            config_manager = ConfigurationManager()
            weight = config_manager.get_value("adapters.diffusers.lora_default_weight")
        
        try:
            lora_path = Path(lora_path)
            if not lora_path.exists():
                raise ValueError(f"LoRA file not found: {lora_path}")

            adapter_name = adapter_name or lora_path.stem

            self.logger.info(f"Loading LoRA: {adapter_name} (weight={weight})")

            # Load LoRA weights into all pipelines
            self.base_pipeline.load_lora_weights(
                str(lora_path), adapter_name=adapter_name
            )

            # Copy to other pipelines
            self.inpaint_pipeline.load_lora_weights(
                str(lora_path), adapter_name=adapter_name
            )
            self.img2img_pipeline.load_lora_weights(
                str(lora_path), adapter_name=adapter_name
            )

            # Track loaded LoRA
            self.loaded_loras[adapter_name] = weight

            # Set LoRA scale
            self._set_lora_scales()

            self.logger.info(f"LoRA loaded successfully: {adapter_name}")

        except Exception as e:
            self.logger.error(f"Failed to load LoRA: {e}")
            raise

    def _set_lora_scales(self):
        """Set scales for all loaded LoRAs"""
        if not self.loaded_loras:
            return

        # Create scale dict
        scales = {name: weight for name, weight in self.loaded_loras.items()}

        # Apply to all pipelines
        for pipeline in [
            self.base_pipeline,
            self.inpaint_pipeline,
            self.img2img_pipeline,
        ]:
            if pipeline and hasattr(pipeline, "set_adapters"):
                pipeline.set_adapters(
                    list(
                        scales.keys()), list(
                        scales.values()))

    def unload_lora(self, adapter_name: Optional[str] = None):
        """
        Unload LoRA weights

        Args:
            adapter_name: Specific LoRA to unload (or all if None)
        """
        if adapter_name:
            if adapter_name in self.loaded_loras:
                del self.loaded_loras[adapter_name]

                # Unload from pipelines
                for pipeline in [
                    self.base_pipeline,
                    self.inpaint_pipeline,
                    self.img2img_pipeline,
                ]:
                    if pipeline and hasattr(pipeline, "delete_adapters"):
                        pipeline.delete_adapters(adapter_name)

                self.logger.info(f"Unloaded LoRA: {adapter_name}")
        else:
            # Unload all
            self.loaded_loras.clear()

            for pipeline in [
                self.base_pipeline,
                self.inpaint_pipeline,
                self.img2img_pipeline,
            ]:
                if pipeline and hasattr(pipeline, "unload_lora_weights"):
                    pipeline.unload_lora_weights()

            self.logger.info("Unloaded all LoRAs")

    def load_pipeline(self, pipeline_type: str, **kwargs) -> Any:
        """
        Load a specific pipeline type

        This is already handled in _initialize_pipelines for DiffusersAdapter
        """
        # Already loaded during initialization
        return self.get_pipeline(pipeline_type)

    def get_pipeline(self, pipeline_type: str) -> Optional[Any]:
        """Get loaded pipeline by type"""
        pipeline_map = {
            "base": self.base_pipeline,
            "inpaint": self.inpaint_pipeline,
            "img2img": self.img2img_pipeline,
            "refiner": self.refiner_pipeline,
        }
        return pipeline_map.get(pipeline_type)

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """Generate image from text"""
        # Get defaults from configuration - FAIL LOUD if not found
        config_manager = ConfigurationManager()
        
        if width is None:
            width = config_manager.get_value("adapters.common.default_width")
        if height is None:
            height = config_manager.get_value("adapters.common.default_height")
        if num_inference_steps is None:
            num_inference_steps = config_manager.get_value("adapters.common.default_num_inference_steps")
        if guidance_scale is None:
            guidance_scale = config_manager.get_value("adapters.common.default_guidance_scale")
        if negative_prompt is None:
            negative_prompt = config_manager.get_value("adapters.common.default_negative_prompt")
        
        # Set seed
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        # Generate
        result = self.base_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs,
        )

        return result.images[0]

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        strength: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """Inpaint masked region"""
        # Get defaults from configuration - FAIL LOUD if not found
        config_manager = ConfigurationManager()
        
        if strength is None:
            strength = config_manager.get_value("adapters.common.default_strength")
        if num_inference_steps is None:
            num_inference_steps = config_manager.get_value("adapters.common.default_num_inference_steps")
        if guidance_scale is None:
            guidance_scale = config_manager.get_value("adapters.common.default_guidance_scale")
        if negative_prompt is None:
            negative_prompt = config_manager.get_value("adapters.common.default_negative_prompt")
        
        # Set seed
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        # Inpaint
        result = self.inpaint_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs,
        )

        return result.images[0]

    def img2img(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        strength: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """Transform image with img2img"""
        # Get defaults from configuration - FAIL LOUD if not found
        config_manager = ConfigurationManager()
        
        if strength is None:
            strength = config_manager.get_value("adapters.common.default_strength")
        if num_inference_steps is None:
            num_inference_steps = config_manager.get_value("adapters.common.default_num_inference_steps")
        if guidance_scale is None:
            guidance_scale = config_manager.get_value("adapters.common.default_guidance_scale")
        if negative_prompt is None:
            negative_prompt = config_manager.get_value("adapters.common.default_negative_prompt")
        
        # Set seed
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        # Transform
        result = self.img2img_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs,
        )

        return result.images[0]

    def refine(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        """Run refinement pipeline"""
        if not self.refiner_pipeline:
            self.logger.warning("No refiner loaded, returning original image")
            return image

        # Use refiner
        result = self.refiner_pipeline(prompt=prompt, image=image, **kwargs)

        return result.images[0]

    def enhance(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        scale_factor: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """
        Enhance/upscale image

        Note: This uses img2img with upscaling.
        For better results, use dedicated upscaling models.
        """
        # Get default scale_factor from configuration if not provided
        if scale_factor is None:
            config_manager = ConfigurationManager()
            scale_factor = config_manager.get_value("adapters.common.default_scale_factor")
        
        # Calculate new dimensions
        new_width = image.width * scale_factor
        new_height = image.height * scale_factor

        # Upscale with PIL first
        upscaled = image.resize(
            (new_width, new_height), Image.Resampling.LANCZOS)

        # Enhance with img2img if prompt provided
        if prompt and self.img2img_pipeline:
            enhanced = self.img2img(
                image=upscaled,
                prompt=prompt,
                # Load enhancement strength from config
                strength=self._get_config_value(
                    'diffusers_adapter',
                    'enhancement_strength',
                    0.3),
                # Low strength to preserve details
                **kwargs,
            )
            return enhanced

        return upscaled

    def unload_pipeline(self, pipeline_type: str):
        """Unload pipeline to free memory"""
        pipeline_map = {
            "base": "base_pipeline",
            "inpaint": "inpaint_pipeline",
            "img2img": "img2img_pipeline",
            "refiner": "refiner_pipeline",
        }

        attr_name = pipeline_map.get(pipeline_type)
        if attr_name and hasattr(self, attr_name):
            pipeline = getattr(self, attr_name)
            if pipeline:
                del pipeline
                setattr(self, attr_name, None)
                self.logger.info(f"Unloaded {pipeline_type} pipeline")

                # Clear CUDA cache
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

    def get_available_pipelines(self) -> List[str]:
        """List available pipeline types"""
        available = []
        if self.base_pipeline:
            available.append("base")
        if self.inpaint_pipeline:
            available.append("inpaint")
        if self.img2img_pipeline:
            available.append("img2img")
        if self.refiner_pipeline:
            available.append("refiner")
        return available

    def supports_inpainting(self) -> bool:
        """Check if adapter supports inpainting"""
        return self.inpaint_pipeline is not None

    def supports_img2img(self) -> bool:
        """Check if adapter supports img2img"""
        return self.img2img_pipeline is not None

    def supports_enhancement(self) -> bool:
        """Check if adapter supports enhancement"""
        return self.img2img_pipeline is not None

    def supports_controlnet(self) -> bool:
        """Check if adapter supports ControlNet"""
        # Model loading supported, generation coming in Phase 5
        return self.model_type == "sdxl"

    def controlnet_inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        control_image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        control_type: str = "canny",
        controlnet_strength: Optional[float] = None,
        strength: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """
        Inpaint with ControlNet guidance

        Uses config-based defaults for all optional parameters
        FAIL LOUD: All dimension and type mismatches are fatal
        """
        # Get defaults from config
        defaults = self._get_controlnet_defaults()
        # Get required defaults from config - FAIL LOUD if missing
        if negative_prompt is None:
            # KeyError if missing
            negative_prompt = defaults["negative_prompt"]
        if controlnet_strength is None:
            # KeyError if missing
            controlnet_strength = defaults["controlnet_strength"]
        if strength is None:
            strength = defaults["strength"]  # KeyError if missing
        if num_inference_steps is None:
            # KeyError if missing
            num_inference_steps = defaults["num_inference_steps"]
        if guidance_scale is None:
            guidance_scale = defaults["guidance_scale"]  # KeyError if missing

        # Validate ControlNet support
        if not self.supports_controlnet():
            raise NotImplementedError(
                f"ControlNet not supported for model type: {self.model_type}\n"
                f"ControlNet currently requires SDXL models.\n"
                f"Use an SDXL model: stabilityai/stable-diffusion-xl-base-1.0"
            )

        # Validate ControlNet is properly initialized - FAIL LOUD on partial
        # setup
        if hasattr(self, 'controlnet_models') and self.controlnet_models:
            # Models loaded but no pipeline
            if not hasattr(
                    self,
                    'controlnet_pipeline') or self.controlnet_pipeline is None:
                # Create pipeline on first use
                self._create_controlnet_pipeline()
        else:
            # No ControlNet loaded at all
            raise RuntimeError(
                "ControlNet is not loaded. Call load_controlnet() first.\n"
                f"Example: adapter.load_controlnet('diffusers/controlnet-{control_type}-sdxl-1.0', '{control_type}')"
            )

        # Switch to the requested control type
        self._switch_controlnet(control_type)

        # For inpainting, we need to create a proper inpaint pipeline
        # Since we only have a text2img pipeline, we'll use it with special
        # handling
        if self.controlnet_pipeline is None:
            raise RuntimeError(
                "ControlNet pipeline not initialized.\n"
                "This is an internal error. Please report this bug."
            )

        # Create proper inpaint pipeline dynamically
        try:
            from diffusers import StableDiffusionXLControlNetInpaintPipeline

            # Create inpaint pipeline from base components
            inpaint_pipeline = StableDiffusionXLControlNetInpaintPipeline(
                vae=self.base_pipeline.vae,
                text_encoder=self.base_pipeline.text_encoder,
                text_encoder_2=self.base_pipeline.text_encoder_2,
                tokenizer=self.base_pipeline.tokenizer,
                tokenizer_2=self.base_pipeline.tokenizer_2,
                unet=self.base_pipeline.unet,
                scheduler=self.base_pipeline.scheduler.from_config(
                    self.base_pipeline.scheduler.config
                ),
                controlnet=self.controlnet_models[control_type],
            ).to(self.device)

            if self.enable_xformers and self.device == "cuda":
                inpaint_pipeline.enable_xformers_memory_efficient_attention()

        except Exception as e:
            raise RuntimeError(
                f"Failed to create ControlNet inpaint pipeline: {str(e)}\n"
                f"This may be due to missing pipeline classes or incompatible versions."
            ) from e

        # Validate dimensions - FAIL LOUD on any mismatch
        if image.size != mask.size:
            raise ValueError(
                f"Image and mask must have same size.\n"
                f"Image: {image.size}, Mask: {mask.size}\n"
                f"Solution: Resize mask to match image size before calling this method:\n"
                f"  mask = mask.resize(image.size, Image.Resampling.LANCZOS)"
            )

        if control_image.size != image.size:
            raise ValueError(
                f"Control image size {
                    control_image.size} doesn't match " f"target image size {
                    image.size}.\n" f"Resize control image to match: control_image.resize({
                    image.size})")

        # Use pre-loaded config
        controlnet_config = self._ensure_controlnet_config()

        if "pipelines" not in controlnet_config:
            raise ValueError(
                "pipelines section not found in controlnet_config.yaml\n"
                "This configuration is REQUIRED for dimension validation.\n"
                "Add: pipelines:\n  dimension_multiple: 8"
            )

        if "dimension_multiple" not in controlnet_config["pipelines"]:
            raise ValueError(
                "dimension_multiple not found in pipelines section of controlnet_config.yaml\n"
                "This value is REQUIRED for SDXL dimension validation.\n"
                "Add: dimension_multiple: 8  # SDXL requires multiples of 8")

        dimension_multiple = controlnet_config["pipelines"]["dimension_multiple"]

        # Ensure dimensions are multiples of requirement
        width, height = image.size
        if width % dimension_multiple != 0 or height % dimension_multiple != 0:
            raise ValueError(
                f"Image dimensions {width}x{height} must be multiples of {dimension_multiple}.\n"
                f"Use: {(width // dimension_multiple) * dimension_multiple}x{(height // dimension_multiple) * dimension_multiple}"
            )

        # Set up generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        # Log parameters
        self.logger.info(
            f"ControlNet inpainting with {control_type} control\n"
            f"  Size: {width}x{height}\n"
            f"  Conditioning scale: {controlnet_strength}\n"
            f"  Strength: {strength}\n"
            f"  Steps: {num_inference_steps}"
        )

        try:
            # Run inference
            result = inpaint_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                control_image=control_image,
                controlnet_strength=controlnet_strength,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                width=width,
                height=height,
            )

            return result.images[0]

        except torch.cuda.OutOfMemoryError as e:
            # FAIL LOUD with VRAM-specific help
            # Get megapixel divisor from config - REQUIRED
            if "calculations" not in controlnet_config:
                raise ValueError(
                    "calculations section not found in controlnet_config.yaml\n"
                    "This section is REQUIRED for error message calculations.\n"
                    "Add: calculations:\n  megapixel_divisor: 1000000")

            if "megapixel_divisor" not in controlnet_config["calculations"]:
                raise ValueError(
                    "megapixel_divisor not found in calculations section of controlnet_config.yaml\n"
                    "This value is REQUIRED for megapixel calculations in error messages.\n"
                    "Add: megapixel_divisor: 1000000  # 1e6")

            mp_divisor = controlnet_config["calculations"]["megapixel_divisor"]

            raise RuntimeError(
                f"Out of GPU memory during ControlNet inpainting.\n"
                f"Current size: {width}x{height} ({width * height / mp_divisor:.1f}MP)\n"
                f"Try reducing image size or enabling CPU offload."
            ) from e
        except Exception as e:
            # FAIL LOUD with general help
            raise RuntimeError(
                f"ControlNet inpainting failed: {str(e)}\n"
                f"Error type: {type(e).__name__}\n"
                f"Debug info:\n"
                f"  Control type: {control_type}\n"
                f"  Image size: {width}x{height}\n"
                f"  Device: {self.device}\n"
                f"Ensure all images are in RGB format and check VRAM usage with nvidia-smi."
            ) from e

    def controlnet_img2img(
        self,
        image: Image.Image,
        control_image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        control_type: str = "canny",
        controlnet_strength: Optional[float] = None,
        strength: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """
        Image-to-image with ControlNet guidance

        Uses config-based defaults for all optional parameters
        FAIL LOUD: Consistent validation with no silent resizing
        """
        # Get defaults from config
        defaults = self._get_controlnet_defaults()
        # Get required defaults from config - FAIL LOUD if missing
        if negative_prompt is None:
            # KeyError if missing
            negative_prompt = defaults["negative_prompt"]
        if controlnet_strength is None:
            # KeyError if missing
            controlnet_strength = defaults["controlnet_strength"]
        if strength is None:
            strength = defaults["strength"]  # KeyError if missing
        if num_inference_steps is None:
            # KeyError if missing
            num_inference_steps = defaults["num_inference_steps"]
        if guidance_scale is None:
            guidance_scale = defaults["guidance_scale"]  # KeyError if missing

        # Validate ControlNet support
        if not self.supports_controlnet():
            raise NotImplementedError(
                f"ControlNet not supported for model type: {self.model_type}\n"
                f"Currently only SDXL models support ControlNet.\n"
                f"Use: stabilityai/stable-diffusion-xl-base-1.0"
            )

        # Validate ControlNet is properly initialized - FAIL LOUD on partial
        # setup
        if hasattr(self, 'controlnet_models') and self.controlnet_models:
            # Models loaded but no pipeline
            if not hasattr(
                    self,
                    'controlnet_pipeline') or self.controlnet_pipeline is None:
                # Create pipeline on first use
                self._create_controlnet_pipeline()
        else:
            # No ControlNet loaded at all
            raise RuntimeError(
                "ControlNet is not loaded. Call load_controlnet() first.\n"
                f"Example: adapter.load_controlnet('diffusers/controlnet-{control_type}-sdxl-1.0', '{control_type}')"
            )

        # Switch to the requested control type
        self._switch_controlnet(control_type)

        # Create proper img2img pipeline dynamically
        try:
            from diffusers import StableDiffusionXLControlNetImg2ImgPipeline

            # Create img2img pipeline from base components
            img2img_pipeline = StableDiffusionXLControlNetImg2ImgPipeline(
                vae=self.base_pipeline.vae,
                text_encoder=self.base_pipeline.text_encoder,
                text_encoder_2=self.base_pipeline.text_encoder_2,
                tokenizer=self.base_pipeline.tokenizer,
                tokenizer_2=self.base_pipeline.tokenizer_2,
                unet=self.base_pipeline.unet,
                scheduler=self.base_pipeline.scheduler.from_config(
                    self.base_pipeline.scheduler.config
                ),
                controlnet=self.controlnet_models[control_type],
            ).to(self.device)

            if self.enable_xformers and self.device == "cuda":
                img2img_pipeline.enable_xformers_memory_efficient_attention()

        except Exception as e:
            raise RuntimeError(
                f"Failed to create ControlNet img2img pipeline: {str(e)}\n"
                f"This may be due to missing pipeline classes or incompatible versions."
            ) from e

        # FAIL LOUD on size mismatch (consistent with inpaint method)
        if control_image.size != image.size:
            raise ValueError(
                f"Control image size {control_image.size} doesn't match "
                f"input image size {image.size}.\n"
                f"Solutions:\n"
                f"  1. Resize control image: control_image.resize({image.size})\n"
                f"  2. Ensure both images are the same size from the start"
            )

        # Use pre-loaded config
        controlnet_config = self._ensure_controlnet_config()

        if "pipelines" not in controlnet_config:
            raise ValueError(
                "pipelines section not found in controlnet_config.yaml\n"
                "This configuration is REQUIRED for dimension validation."
            )

        if "dimension_multiple" not in controlnet_config["pipelines"]:
            raise ValueError(
                "dimension_multiple not found in pipelines section of controlnet_config.yaml\n"
                "This value is REQUIRED for SDXL dimension validation.")

        dimension_multiple = controlnet_config["pipelines"]["dimension_multiple"]

        # FAIL LOUD on non-multiple dimensions
        width, height = image.size
        if width % dimension_multiple != 0 or height % dimension_multiple != 0:
            raise ValueError(
                f"Image dimensions {width}x{height} are not multiples of {dimension_multiple}.\n"
                f"SDXL requires dimensions to be multiples of {dimension_multiple}.\n"
                f"Solution: Resize to {(width // dimension_multiple) * dimension_multiple}x{(height // dimension_multiple) * dimension_multiple}"
            )

        # Generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        self.logger.info(
            f"ControlNet img2img: {control_type} control, "
            f"scale: {controlnet_strength}, strength: {strength}"
        )

        try:
            result = img2img_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                control_image=control_image,
                controlnet_strength=controlnet_strength,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            return result.images[0]

        except Exception as e:
            raise RuntimeError(
                f"ControlNet img2img failed: {str(e)}\n"
                f"Error type: {type(e).__name__}\n"
                f"Size: {width}x{height}, Type: {control_type}\n"
                f"Try lower resolution or reduced conditioning scale."
            ) from e

    def get_controlnet_types(self) -> List[str]:
        """
        Get available ControlNet types

        Returns list of loaded ControlNet model types
        FAIL LOUD: No backwards compatibility
        """
        return list(self.controlnet_models.keys())

    def load_controlnet(
            self,
            controlnet_id: str,
            controlnet_type: str = "canny"):
        """
        Load ControlNet model for guided generation

        Args:
            controlnet_id: Model ID or path (e.g., 'diffusers/controlnet-canny-sdxl-1.0')
            controlnet_type: Type of control (default: 'canny', options: 'canny', 'depth', 'openpose')

        FAIL LOUD: All errors are fatal - no silent failures
        Auto-creates config on first use for user convenience
        """
        # FAIL IMMEDIATELY - no partial infrastructure for unsupported models
        if self.model_type != "sdxl":
            raise NotImplementedError(
                f"ControlNet is only supported for SDXL models.\n"
                f"Your model type: {self.model_type}\n"
                f"Use an SDXL model: stabilityai/stable-diffusion-xl-base-1.0"
            )

        # Access config through method - auto-creates if missing
        # This auto-creates default config if needed
        config = self._ensure_controlnet_config()

        # Try to import ControlNet dependencies
        try:
            from diffusers import ControlNetModel
        except ImportError as e:
            raise ImportError(
                "ControlNet requires diffusers>=0.24.0 with controlnet extras.\n"
                "This is an optional feature that requires additional dependencies.\n"
                "Install with: pip install 'diffusers[controlnet]>=0.24.0'\n"
                f"Original error: {e}"
            )

        try:
            self.logger.info(f"Loading ControlNet model: {controlnet_id}")

            # Load the model
            controlnet = ControlNetModel.from_pretrained(
                controlnet_id,
                torch_dtype=self.torch_dtype,
                cache_dir=self.cache_dir,
                use_safetensors=self.use_safetensors,
                variant=self.variant if self.variant else None,
            )

            # Move to device
            controlnet = controlnet.to(self.device)

            # Store it
            self.controlnet_models[controlnet_type] = controlnet
            self.logger.info(
                f"Successfully loaded {controlnet_type} ControlNet")

            # Note: Pipeline is created lazily on first use for VRAM efficiency
            # This avoids loading unnecessary components until actually needed

            return True

        except Exception as e:
            raise RuntimeError(
                f"Failed to load ControlNet '{controlnet_id}':\n"
                f"  Error: {str(e)}\n"
                f"  Error type: {type(e).__name__}\n"
                f"  Possible solutions:\n"
                f"  1. Check your internet connection\n"
                f"  2. Verify you have access to the model\n"
                f"  3. Ensure sufficient disk space in {self.cache_dir}\n"
                f"  4. Try: huggingface-cli login"
            ) from e

    def generate_with_controlnet(
        self,
        prompt: str,
        control_image: Image.Image,
        negative_prompt: Optional[str] = None,
        control_type: str = "canny",
        width: Optional[int] = None,
        height: Optional[int] = None,
        controlnet_strength: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """
        Text-to-image generation with ControlNet guidance

        Generate new images constrained by control image structure
        Uses config-based defaults for all optional parameters
        """
        # Get defaults from config
        defaults = self._get_controlnet_defaults()
        negative_prompt = negative_prompt if negative_prompt is not None else defaults.get(
            "negative_prompt", "")
        controlnet_strength = controlnet_strength if controlnet_strength is not None else defaults.get(
            "controlnet_strength", 1.0)
        num_inference_steps = num_inference_steps if num_inference_steps is not None else defaults.get(
            "num_inference_steps", 50)
        guidance_scale = guidance_scale if guidance_scale is not None else defaults.get(
            "guidance_scale", 7.5)

        # Default dimensions to control image size if not specified
        if width is None or height is None:
            width, height = control_image.size

        if not self.supports_controlnet():
            raise NotImplementedError(
                "ControlNet requires SDXL models.\n"
                f"Current model type: {self.model_type}"
            )

        # Validate ControlNet is properly initialized - FAIL LOUD on partial
        # setup
        if hasattr(self, 'controlnet_models') and self.controlnet_models:
            # Models loaded but no pipeline
            if not hasattr(
                    self,
                    'controlnet_pipeline') or self.controlnet_pipeline is None:
                # Create pipeline on first use
                self._create_controlnet_pipeline()
        else:
            # No ControlNet loaded
            raise RuntimeError(
                "ControlNet is not loaded. Call load_controlnet() first.\n"
                f"Example: adapter.load_controlnet('diffusers/controlnet-{control_type}-sdxl-1.0', '{control_type}')"
            )

        # Switch to the requested control type
        self._switch_controlnet(control_type)

        # Resize control image if needed
        if control_image.size != (width, height):
            self.logger.info(
                f"Resizing control image from {
                    control_image.size} to ({width}, {height})")
            control_image = control_image.resize(
                (width, height), Image.Resampling.LANCZOS)

        # Use pre-loaded config
        controlnet_config = self._ensure_controlnet_config()
        dimension_multiple = controlnet_config["pipelines"]["dimension_multiple"]

        # Ensure multiple of requirement
        if width % dimension_multiple != 0 or height % dimension_multiple != 0:
            raise ValueError(
                f"Dimensions {width}x{height} must be multiples of {dimension_multiple}.\n"
                f"Use: {(width // dimension_multiple) * dimension_multiple}x{(height // dimension_multiple) * dimension_multiple}"
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        try:
            result = self.controlnet_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,  # StableDiffusionXLControlNetPipeline uses 'image' param
                controlnet_strength=controlnet_strength,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            return result.images[0]
        except Exception as e:
            raise RuntimeError(
                f"ControlNet generation failed: {str(e)}\n"
                f"Error type: {type(e).__name__}\n"
                f"Debug info: size={width}x{height}, type={control_type}"
            ) from e

    def get_available_controlnets(self) -> List[str]:
        """Get list of loaded ControlNet models"""
        if not hasattr(self, "controlnet_models"):
            return []
        return list(self.controlnet_models.keys())

    def estimate_vram(self, operation: str, **kwargs) -> float:
        """
        Estimate VRAM for an operation in MB

        ALL values from configuration - NO HARDCODING
        FAIL LOUD if configuration is incomplete
        """
        # Use pre-loaded config
        # This will fail loud if base config missing
        vram_config = self._ensure_vram_config()

        # Get operation estimates - FAIL LOUD if missing
        if "operation_estimates" not in vram_config:
            raise ValueError(
                "operation_estimates section missing from vram_strategies.yaml\n"
                "This section is required for VRAM estimation.\n"
                "Please run: expandor --setup-controlnet\n"
                "Or manually add the operation_estimates section to the config.")

        operation_estimates = vram_config["operation_estimates"]

        # Get base estimate for model type and operation - FAIL LOUD, no
        # fallbacks
        if self.model_type not in operation_estimates:
            raise ValueError(
                f"Model type '{
                    self.model_type}' not found in operation_estimates.\n" f"Available types: {
                    list(
                        operation_estimates.keys())}\n" f"Add estimates for '{
                    self.model_type}' to vram_strategies.yaml")
        model_estimates = operation_estimates[self.model_type]

        if operation not in model_estimates:
            raise ValueError(
                f"Operation '{operation}' not found for model type '{self.model_type}'.\n"
                f"Available operations: {list(model_estimates.keys())}\n"
                f"Add '{operation}' to vram_strategies.yaml under '{self.model_type}'"
            )
        base = model_estimates[operation]

        # Adjust for resolution if provided
        if "width" in kwargs and "height" in kwargs:
            # Get resolution calculation constants from config
            if "resolution_calculation" not in vram_config:
                raise ValueError(
                    "resolution_calculation section not found in vram_strategies.yaml\n"
                    "This is REQUIRED for resolution-based VRAM scaling.\n"
                    "Add: resolution_calculation:\n  base_pixels: 1048576")
            res_calc = vram_config["resolution_calculation"]

            if "base_pixels" not in res_calc:
                raise ValueError(
                    "base_pixels not found in resolution_calculation section\n"
                    "This value is REQUIRED for VRAM scaling calculations.\n"
                    "Add: base_pixels: 1048576  # 1024*1024"
                )
            base_pixels = res_calc["base_pixels"]

            pixels = kwargs["width"] * kwargs["height"]
            multiplier = pixels / base_pixels
            base = base * multiplier

        # Add LoRA overhead from config
        if "lora_overhead_mb" not in vram_config:
            raise ValueError(
                "lora_overhead_mb not found in vram_strategies.yaml\n"
                "This value is REQUIRED for LoRA VRAM calculations.\n"
                "Add: lora_overhead_mb: 200"
            )
        lora_overhead_mb = vram_config["lora_overhead_mb"]
        lora_overhead = len(self.loaded_loras) * lora_overhead_mb

        # Add ControlNet overhead if applicable
        controlnet_overhead = 0
        if operation.startswith("controlnet_") or self.controlnet_models:
            # Use pre-loaded controlnet config
            if self._controlnet_config is None:
                if operation.startswith("controlnet_"):
                    raise FileNotFoundError(
                        "controlnet_config.yaml not found but ControlNet operation requested.\n"
                        "Create the ControlNet configuration file first.\n"
                        "See documentation for required structure.")
                else:
                    # ControlNet models loaded but no config - FAIL LOUD
                    raise RuntimeError(
                        "ControlNet models are loaded but controlnet_config.yaml not found.\n"
                        "This is an inconsistent state. ControlNet config is REQUIRED when models are loaded.")

            if "vram_overhead" not in self._controlnet_config:
                raise ValueError(
                    "vram_overhead section not found in controlnet_config.yaml\n"
                    "This is REQUIRED for ControlNet VRAM estimation.\n"
                    "Add the vram_overhead section with model_load and operation_active values.")
            vram_overhead = self._controlnet_config["vram_overhead"]

            # Each loaded ControlNet model adds overhead
            if "model_load" not in vram_overhead:
                raise ValueError(
                    "model_load not found in vram_overhead section of controlnet_config.yaml\n"
                    "Add: model_load: 2000  # MB per ControlNet model")
            model_load_overhead = vram_overhead["model_load"]
            controlnet_overhead = len(
                self.controlnet_models) * model_load_overhead

            # Additional overhead for active ControlNet operations
            if operation.startswith("controlnet_"):
                if "operation_active" not in vram_overhead:
                    raise ValueError(
                        "operation_active not found in vram_overhead section of controlnet_config.yaml\n"
                        "Add: operation_active: 1500  # MB for active operations")
                operation_overhead = vram_overhead["operation_active"]
                controlnet_overhead += operation_overhead

        return base + lora_overhead + controlnet_overhead

    def enable_cpu_offload(self):
        """Enable CPU offload for memory efficiency - alias for enable_memory_efficient_mode"""
        self.enable_memory_efficient_mode()
    
    def enable_memory_efficient_mode(self):
        """Enable memory optimizations"""
        try:
            # Enable CPU offload
            if hasattr(self.base_pipeline, "enable_sequential_cpu_offload"):
                self.base_pipeline.enable_sequential_cpu_offload()
                self.logger.info("Enabled CPU offload for base pipeline")

            if hasattr(self.inpaint_pipeline, "enable_sequential_cpu_offload"):
                self.inpaint_pipeline.enable_sequential_cpu_offload()
                self.logger.info("Enabled CPU offload for inpaint pipeline")

            if hasattr(self.img2img_pipeline, "enable_sequential_cpu_offload"):
                self.img2img_pipeline.enable_sequential_cpu_offload()
                self.logger.info("Enabled CPU offload for img2img pipeline")

            if self.refiner_pipeline and hasattr(
                self.refiner_pipeline, "enable_sequential_cpu_offload"
            ):
                self.refiner_pipeline.enable_sequential_cpu_offload()
                self.logger.info("Enabled CPU offload for refiner pipeline")

            # Also enable attention slicing for lower memory usage
            if hasattr(self.base_pipeline, "enable_attention_slicing"):
                self.base_pipeline.enable_attention_slicing()
                self.inpaint_pipeline.enable_attention_slicing()
                self.img2img_pipeline.enable_attention_slicing()
                if self.refiner_pipeline:
                    self.refiner_pipeline.enable_attention_slicing()
                self.logger.info("Enabled attention slicing")

        except Exception as e:
            self.logger.warning(
                f"Could not enable all memory optimizations: {e}")

    def clear_cache(self):
        """Clear any cached data/models"""
        # Delete pipeline components explicitly
        if hasattr(self, 'pipeline') and self.pipeline:
            if hasattr(self.pipeline, 'vae'):
                self.pipeline.vae = None
            if hasattr(self.pipeline, 'unet'):
                self.pipeline.unet = None
            if hasattr(self.pipeline, 'text_encoder'):
                self.pipeline.text_encoder = None
            if hasattr(self.pipeline, 'text_encoder_2'):
                self.pipeline.text_encoder_2 = None

        # Clear CUDA cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.info("Cleared CUDA cache")

        # Clear CPU cache if possible
        import gc

        gc.collect()

        self.logger.info("Cleared memory cache")
