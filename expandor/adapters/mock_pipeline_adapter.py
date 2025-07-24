"""
Mock Pipeline Adapter for Testing
Adapted from ai-wallpaper project
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..adapters.mock_pipeline import (
    MockImg2ImgPipeline,
    MockInpaintPipeline,
    MockRefinerPipeline,
)
from ..utils.logging_utils import setup_logger
from .base_adapter import BasePipelineAdapter


class MockPipelineAdapter(BasePipelineAdapter):
    """
    Mock adapter for testing without real models
    QUALITY OVER ALL: Even mock outputs maintain quality standards
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "fp32",
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        """
        Initialize mock adapter

        Args:
            device: Device to simulate (cpu/cuda/mps)
            dtype: Data type to simulate
            logger: Logger instance
            **kwargs: Additional arguments (for compatibility)
        """
        self.device = device
        self.dtype = dtype
        self.logger = logger or setup_logger(__name__)

        # Initialize mock pipelines
        self.pipelines = {
            "inpaint": MockInpaintPipeline(),
            "img2img": MockImg2ImgPipeline(),
            "refiner": MockRefinerPipeline(),
        }

        # Track loaded LoRAs for testing
        self.loaded_loras = []

        # Simulate VRAM usage
        self.simulated_vram_mb = 0
        self.max_vram_mb = 24576 if device == "cuda" else 0  # Simulate 24GB GPU

        self.logger.info(f"Initialized MockPipelineAdapter on {device} with {dtype}")

    def load_pipeline(self, pipeline_type: str, **kwargs) -> Any:
        """Load a specific pipeline type"""
        if pipeline_type not in self.pipelines:
            # FAIL LOUD
            raise ValueError(
                f"Unknown pipeline type: {pipeline_type}. "
                f"Available types: {list(self.pipelines.keys())}"
            )

        self.logger.info(f"Loaded mock {pipeline_type} pipeline")
        return self.pipelines[pipeline_type]

    def get_pipeline(self, pipeline_type: str) -> Optional[Any]:
        """Get loaded pipeline by type"""
        return self.pipelines.get(pipeline_type)

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
        """Generate image from text prompt"""
        self.logger.info(
            f"Mock generating {width}x{height} image from prompt: {prompt[:50]}..."
        )

        # Simulate VRAM usage
        vram_needed = (width * height * 4) / (1024 * 1024)  # Rough estimate
        if (
            self.device == "cuda"
            and self.simulated_vram_mb + vram_needed > self.max_vram_mb
        ):
            # FAIL LOUD - simulate OOM
            raise RuntimeError(
                f"Mock CUDA out of memory. Tried to allocate {
                    vram_needed:.0f}MB "
                f"(GPU has {
                    self.max_vram_mb -
                    self.simulated_vram_mb:.0f}MB free)"
            )

        # Create a gradient image with some variation based on prompt
        np.random.seed(seed or abs(hash(prompt)) % (2**32))

        # Base gradient
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)

        # Add some noise based on prompt
        noise = np.random.normal(0, 0.05, (height, width))

        # Create RGB channels with different patterns
        r = np.clip((X + noise) * 255, 0, 255).astype(np.uint8)
        g = np.clip((Y + noise) * 255, 0, 255).astype(np.uint8)
        b = np.clip(((X + Y) / 2 + noise) * 255, 0, 255).astype(np.uint8)

        # Add "MOCK" watermark to make it clear this is test output
        # This ensures QUALITY OVER ALL - even mock outputs are clearly labeled
        img_array = np.stack([r, g, b], axis=-1)
        self._add_mock_watermark(img_array)

        return Image.fromarray(img_array, mode="RGB")

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
        """Inpaint masked region of image"""
        self.logger.info(f"Mock inpainting with prompt: {prompt[:50]}...")

        # Use mock pipeline
        pipeline = self.pipelines["inpaint"]
        result = pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=None,  # Mock doesn't use generators
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
        self.logger.info(f"Mock img2img with prompt: {prompt[:50]}...")

        # Use mock pipeline
        pipeline = self.pipelines["img2img"]
        result = pipeline(
            prompt=prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=None,
            **kwargs,
        )

        return result.images[0]

    def refine(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        """Run refinement pipeline"""
        self.logger.info("Mock refining image...")

        # Use mock refiner
        pipeline = self.pipelines["refiner"]
        result = pipeline(prompt=prompt, image=image, **kwargs)

        return result.images[0]

    def enhance(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        scale_factor: int = 2,
        **kwargs,
    ) -> Image.Image:
        """Enhance/upscale image"""
        self.logger.info(f"Mock enhancing image by {scale_factor}x...")

        # Simple bicubic upscale for mock
        new_size = (image.width * scale_factor, image.height * scale_factor)
        enhanced = image.resize(new_size, Image.Resampling.BICUBIC)

        # Add slight sharpening effect
        from PIL import ImageEnhance

        sharpener = ImageEnhance.Sharpness(enhanced)
        enhanced = sharpener.enhance(1.2)

        # Add mock watermark
        img_array = np.array(enhanced)
        self._add_mock_watermark(img_array)

        return Image.fromarray(img_array)

    def get_optimal_dimensions(
        self, target_width: int, target_height: int
    ) -> Tuple[int, int]:
        """Get optimal dimensions for the model"""
        # Round to multiples of 8 for SDXL compatibility
        optimal_width = (target_width // 8) * 8
        optimal_height = (target_height // 8) * 8

        # Ensure minimum size
        optimal_width = max(optimal_width, 512)
        optimal_height = max(optimal_height, 512)

        # Ensure maximum size
        optimal_width = min(optimal_width, 2048)
        optimal_height = min(optimal_height, 2048)

        return (optimal_width, optimal_height)

    def unload_pipeline(self, pipeline_type: str):
        """Unload pipeline to free memory"""
        if pipeline_type in self.pipelines:
            self.logger.info(f"Unloaded mock {pipeline_type} pipeline")
            # Simulate memory release
            self.simulated_vram_mb = max(0, self.simulated_vram_mb - 1000)

    def get_available_pipelines(self) -> List[str]:
        """List available pipeline types"""
        return list(self.pipelines.keys())

    def supports_inpainting(self) -> bool:
        """Check if adapter supports inpainting"""
        return True

    def supports_img2img(self) -> bool:
        """Check if adapter supports img2img"""
        return True

    def supports_enhancement(self) -> bool:
        """Check if adapter supports enhancement"""
        return True

    def supports_controlnet(self) -> bool:
        """Check if adapter supports ControlNet"""
        return False  # Mock doesn't support ControlNet

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
        Mock ControlNet inpainting

        FAIL LOUD: Mock adapter doesn't support ControlNet
        """
        raise NotImplementedError(
            "MockPipelineAdapter does not support ControlNet. "
            "Use DiffusersPipelineAdapter or ComfyUIPipelineAdapter for ControlNet support."
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
        Mock ControlNet img2img

        FAIL LOUD: Mock adapter doesn't support ControlNet
        """
        raise NotImplementedError(
            "MockPipelineAdapter does not support ControlNet. "
            "Use DiffusersPipelineAdapter or ComfyUIPipelineAdapter for ControlNet support."
        )

    def get_controlnet_types(self) -> List[str]:
        """Get available ControlNet types"""
        return []  # No ControlNet support in mock

    def load_controlnet(self, controlnet_type: str, model_id: Optional[str] = None):
        """Load a specific ControlNet model"""
        raise NotImplementedError(
            "MockPipelineAdapter does not support ControlNet. "
            "Use DiffusersPipelineAdapter or ComfyUIPipelineAdapter for ControlNet support."
        )

    def estimate_vram(self, operation: str, **kwargs) -> float:
        """Estimate VRAM for an operation in MB"""
        estimates = {
            "generate": 4000,
            "inpaint": 3500,
            "img2img": 3000,
            "refine": 2000,
            "enhance": 1500,
        }

        base = estimates.get(operation, 2000)

        # Adjust for resolution if provided
        if "width" in kwargs and "height" in kwargs:
            pixels = kwargs["width"] * kwargs["height"]
            base_pixels = 1024 * 1024
            multiplier = pixels / base_pixels
            base = base * multiplier

        return base

    def enable_memory_efficient_mode(self):
        """Enable memory optimizations"""
        self.logger.info("Enabled mock memory efficient mode")
        # Simulate reducing VRAM usage
        self.simulated_vram_mb = self.simulated_vram_mb * 0.7

    def clear_cache(self):
        """Clear any cached data/models"""
        self.logger.info("Cleared mock cache")
        self.simulated_vram_mb = 0

    def load_lora(self, lora_path: str, weight: float, name: str):
        """Load a LoRA model (mock implementation)"""
        # Validate path exists
        if not Path(lora_path).exists():
            # FAIL LOUD
            raise FileNotFoundError(
                f"LoRA file not found: {lora_path}\n"
                f"Please check the path in your configuration."
            )

        self.loaded_loras.append({"path": lora_path, "weight": weight, "name": name})

        self.logger.info(f"Loaded mock LoRA: {name} (weight: {weight})")

    def unload_loras(self):
        """Unload all LoRA models"""
        self.loaded_loras.clear()
        self.logger.info("Unloaded all mock LoRAs")

    def get_loaded_loras(self) -> List[dict]:
        """Get list of loaded LoRAs"""
        return self.loaded_loras.copy()

    def set_scheduler(self, scheduler_name: str):
        """Set the scheduler/sampler (mock implementation)"""
        self.logger.info(f"Set mock scheduler to: {scheduler_name}")

    def get_available_schedulers(self) -> List[str]:
        """Get list of available schedulers"""
        return ["DDIM", "PNDM", "LMSD", "Euler", "EulerA", "DPM++"]

    def get_model_info(self) -> dict:
        """Get information about loaded model"""
        return {
            "name": "mock_model",
            "type": "mock",
            "device": self.device,
            "dtype": self.dtype,
            "loaded_loras": len(self.loaded_loras),
            "vram_usage_mb": self.simulated_vram_mb,
        }

    def supports_tiling(self) -> bool:
        """Check if adapter supports tiled generation"""
        return True

    def enable_tiling(self):
        """Enable tiled VAE decode"""
        self.logger.info("Enabled mock tiling mode")

    def disable_tiling(self):
        """Disable tiled VAE decode"""
        self.logger.info("Disabled mock tiling mode")

    def get_supported_dtypes(self) -> List[str]:
        """Get list of supported data types"""
        return ["fp32", "fp16", "bf16"]

    def get_config_path(self) -> Optional[Path]:
        """Get path to configuration file (if any)"""
        return None  # Mock doesn't use config files

    def validate_config(self, config: dict) -> bool:
        """Validate a configuration dictionary"""
        # Mock accepts any config
        return True

    def get_supported_strategies(self) -> List[str]:
        """Get list of strategies this adapter supports"""
        # Mock adapter supports all basic strategies
        return [
            "direct_upscale",
            "progressive_outpaint",
            "tiled_expansion",
            "swpo",  # Even supports SWPO for testing
        ]

    def _add_mock_watermark(self, img_array: np.ndarray):
        """Add MOCK watermark to image array"""
        # Add text in corner to indicate this is mock output
        # This ensures quality standards - mock outputs are clearly labeled
        h, w = img_array.shape[:2]

        # Create watermark area
        watermark_h, watermark_w = 40, 120
        if h > watermark_h and w > watermark_w:
            # Semi-transparent dark background
            img_array[:watermark_h, :watermark_w] = (
                img_array[:watermark_h, :watermark_w] * 0.5
            )

            # Add white text (simplified - in real implementation use PIL.ImageDraw)
            # This is just to make it visually distinct
            img_array[10:30, 10:110] = np.clip(img_array[10:30, 10:110] + 100, 0, 255)
