"""
Diffusers library adapter for Expandor
Provides real pipeline implementation using HuggingFace Diffusers
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from PIL import Image

from ..utils.logging_utils import setup_logger
from .base_adapter import BasePipelineAdapter


class DiffusersPipelineAdapter(BasePipelineAdapter):
    """
    Adapter for HuggingFace Diffusers pipelines
    Supports SDXL, SD3, FLUX and other diffusers models
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        variant: Optional[str] = "fp16",
        cache_dir: Optional[str] = None,
        use_safetensors: bool = True,
        enable_xformers: bool = True,
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

        # Model type detection
        self.model_type = None
        self.model_config = None
        self.model_type_registry = self._get_model_type_registry()

        # Initialize pipelines
        self._initialize_pipelines()

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
                                pattern in tag_lower for pattern in config["patterns"]
                            ):
                                self.logger.info(
                                    f"Detected model type from tags: {model_type}"
                                )
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
                        model_source, **common_args
                    )
                    self.img2img_pipeline = AutoPipelineForImage2Image.from_pretrained(
                        model_source, **common_args
                    )
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
                    self.logger.info("Enabled xformers memory efficient attention")
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
        self, refiner_id: Optional[str] = None, refiner_path: Optional[str] = None
    ):
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

            from diffusers import StableDiffusionXLImg2ImgPipeline as RefinerPipeline

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
            multiple = 8
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

        # Cap at reasonable maximum
        max_dimension = 4096  # Increased for modern models
        optimal_width = min(optimal_width, max_dimension)
        optimal_height = min(optimal_height, max_dimension)

        self.logger.debug(
            f"Optimal dimensions for {getattr(self, 'model_type', 'unknown')}: "
            f"{target_width}x{target_height} -> {optimal_width}x{optimal_height}"
        )

        return (optimal_width, optimal_height)

    def load_lora(
        self, lora_path: str, weight: float = 1.0, adapter_name: Optional[str] = None
    ):
        """
        Load LoRA weights

        Args:
            lora_path: Path to LoRA file
            weight: LoRA weight multiplier
            adapter_name: Name for the adapter
        """
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
                pipeline.set_adapters(list(scales.keys()), list(scales.values()))

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
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """Generate image from text"""
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
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """Inpaint masked region"""
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
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """Transform image with img2img"""
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
        scale_factor: int = 2,
        **kwargs,
    ) -> Image.Image:
        """
        Enhance/upscale image

        Note: This uses img2img with upscaling.
        For better results, use dedicated upscaling models.
        """
        # Calculate new dimensions
        new_width = image.width * scale_factor
        new_height = image.height * scale_factor

        # Upscale with PIL first
        upscaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Enhance with img2img if prompt provided
        if prompt and self.img2img_pipeline:
            enhanced = self.img2img(
                image=upscaled,
                prompt=prompt,
                strength=0.3,  # Low strength to preserve details
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
        controlnet_conditioning_scale: float = 1.0,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """
        Inpaint with ControlNet guidance

        FAIL LOUD: Not implemented yet
        """
        raise NotImplementedError(
            "ControlNet support not yet implemented in DiffusersPipelineAdapter.\n"
            "This feature will be added in Phase 5.\n"
            "For now, use standard inpainting without ControlNet."
        )

    def controlnet_img2img(
        self,
        image: Image.Image,
        control_image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        controlnet_conditioning_scale: float = 1.0,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """
        Image-to-image with ControlNet guidance

        FAIL LOUD: Not implemented yet
        """
        raise NotImplementedError(
            "ControlNet support not yet implemented in DiffusersPipelineAdapter.\n"
            "This feature will be added in Phase 5.\n"
            "For now, use standard img2img without ControlNet."
        )

    def get_controlnet_types(self) -> List[str]:
        """
        Get available ControlNet types

        Returns empty list until ControlNet is implemented
        """
        # TODO: Return actual ControlNet types once implemented
        # Will include: ['canny', 'depth', 'openpose', 'mlsd', 'normal', 'seg',
        # 'scribble', 'lineart', 'anime_lineart', 'shuffle', 'inpaint', 'tile']
        return []

    def load_controlnet(self, controlnet_id: str, controlnet_type: str = "canny"):
        """
        Load ControlNet model for guided generation

        Status: Model loading only - generation in Phase 5
        """
        if not self.model_type == "sdxl":
            raise NotImplementedError(
                f"ControlNet is currently only supported for SDXL models.\n"
                f"Your model type: {self.model_type}\n"
                f"Full ControlNet support for all models coming in Phase 5.\n"
                f"For now, use SDXL-based models or wait for the next release."
            )

        try:
            from diffusers import ControlNetModel
        except ImportError:
            raise ImportError(
                "ControlNet requires diffusers>=0.24.0 with controlnet extras.\n"
                "Install with: pip install 'diffusers[controlnet]>=0.24.0'"
            )

        try:
            self.logger.info(f"Loading ControlNet model: {controlnet_id}")

            # Initialize controlnet storage if needed
            if not hasattr(self, "controlnet_models"):
                self.controlnet_models = {}

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
            self.logger.info(f"Successfully loaded {controlnet_type} ControlNet")

            return True

        except Exception as e:
            raise RuntimeError(
                f"Failed to load ControlNet '{controlnet_id}':\n"
                f"  Error: {str(e)}\n"
                f"  Possible solutions:\n"
                f"  1. Check your internet connection\n"
                f"  2. Verify you have access to the model\n"
                f"  3. Ensure sufficient disk space in {self.cache_dir}\n"
                f"  4. Try: huggingface-cli login"
            ) from e

    def generate_with_controlnet(self, **kwargs):
        """ControlNet generation - Phase 5 feature"""
        raise NotImplementedError(
            "ControlNet generation is a Phase 5 feature.\n"
            "Current status: You can load ControlNet models but not use them yet.\n"
            "Workaround: Export your pipeline and use ComfyUI/A1111 directly.\n"
            "Full implementation coming Q1 2024."
        )

    def get_available_controlnets(self) -> List[str]:
        """Get list of loaded ControlNet models"""
        if not hasattr(self, "controlnet_models"):
            return []
        return list(self.controlnet_models.keys())

    def estimate_vram(self, operation: str, **kwargs) -> float:
        """Estimate VRAM for an operation in MB"""
        # Base estimates for different model types
        base_estimates = {
            "sdxl": {
                "generate": 6000,
                "inpaint": 5500,
                "img2img": 5000,
                "refine": 4000,
                "enhance": 3500,
            },
            "sd3": {
                "generate": 8000,
                "inpaint": 7500,
                "img2img": 7000,
                "refine": 5000,
                "enhance": 4500,
            },
            "flux": {
                "generate": 12000,
                "inpaint": 11000,
                "img2img": 10000,
                "refine": 8000,
                "enhance": 7000,
            },
            "sd2": {
                "generate": 4000,
                "inpaint": 3500,
                "img2img": 3000,
                "refine": 2500,
                "enhance": 2000,
            },
            "sd15": {
                "generate": 3000,
                "inpaint": 2500,
                "img2img": 2000,
                "refine": 1800,
                "enhance": 1500,
            },
            "auto": {
                "generate": 5000,
                "inpaint": 4500,
                "img2img": 4000,
                "refine": 3000,
                "enhance": 2500,
            },
        }

        # Get estimates for current model type
        model_estimates = base_estimates.get(self.model_type, base_estimates["auto"])
        base = model_estimates.get(operation, 4000)

        # Adjust for resolution if provided
        if "width" in kwargs and "height" in kwargs:
            pixels = kwargs["width"] * kwargs["height"]
            base_pixels = 1024 * 1024
            multiplier = pixels / base_pixels
            base = base * multiplier

        # Add LoRA overhead
        lora_overhead = len(self.loaded_loras) * 200  # ~200MB per LoRA

        return base + lora_overhead

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
            self.logger.warning(f"Could not enable all memory optimizations: {e}")

    def clear_cache(self):
        """Clear any cached data/models"""
        # Clear CUDA cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.info("Cleared CUDA cache")

        # Clear CPU cache if possible
        import gc

        gc.collect()

        self.logger.info("Cleared memory cache")
