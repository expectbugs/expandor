"""
Base adapter interface for pipeline implementations
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple

from PIL import Image


class BasePipelineAdapter(ABC):
    """Abstract base class for pipeline adapters"""

    @abstractmethod
    def load_pipeline(self, pipeline_type: str, **kwargs) -> Any:
        """Load a specific pipeline type"""

    @abstractmethod
    def get_pipeline(self, pipeline_type: str) -> Optional[Any]:
        """Get loaded pipeline by type"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """Generate image from text prompt"""

    @abstractmethod
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
        **kwargs
    ) -> Image.Image:
        """Inpaint masked region of image"""

    @abstractmethod
    def img2img(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        strength: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """Transform image with img2img"""

    @abstractmethod
    def refine(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        """Run refinement pipeline"""

    @abstractmethod
    def enhance(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        scale_factor: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """Enhance/upscale image"""

    @abstractmethod
    def get_optimal_dimensions(
        self, target_width: int, target_height: int
    ) -> Tuple[int, int]:
        """Get optimal dimensions for the model"""

    @abstractmethod
    def unload_pipeline(self, pipeline_type: str):
        """Unload pipeline to free memory"""

    @abstractmethod
    def get_available_pipelines(self) -> List[str]:
        """List available pipeline types"""

    @abstractmethod
    def supports_inpainting(self) -> bool:
        """Check if adapter supports inpainting"""

    @abstractmethod
    def supports_img2img(self) -> bool:
        """Check if adapter supports img2img"""

    @abstractmethod
    def supports_enhancement(self) -> bool:
        """Check if adapter supports enhancement"""

    @abstractmethod
    def supports_controlnet(self) -> bool:
        """Check if adapter supports ControlNet"""

    @abstractmethod
    def controlnet_inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        control_image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        controlnet_conditioning_scale: Optional[float] = None,
        strength: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """
        Inpaint with ControlNet guidance

        FAIL LOUD: Raise NotImplementedError if ControlNet not supported
        QUALITY OVER ALL: Use ControlNet for structure preservation

        Args:
            image: Source image
            mask: Inpainting mask
            control_image: Control image (canny, depth, etc.)
            prompt: Text prompt
            negative_prompt: Negative prompt
            controlnet_conditioning_scale: ControlNet strength (None = use config default from 'adapters.common.default_controlnet_conditioning_scale')
            strength: Denoising strength (None = use config default from 'adapters.common.default_inpaint_strength')
            num_inference_steps: Number of steps (None = use config default from 'adapters.common.default_num_inference_steps')
            guidance_scale: Guidance scale (None = use config default from 'adapters.common.default_guidance_scale')
            seed: Random seed
            **kwargs: Additional pipeline arguments

        Returns:
            Inpainted image with ControlNet guidance
        """

    @abstractmethod
    def controlnet_img2img(
        self,
        image: Image.Image,
        control_image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        controlnet_conditioning_scale: Optional[float] = None,
        strength: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """
        Image-to-image with ControlNet guidance

        FAIL LOUD: Raise NotImplementedError if ControlNet not supported
        QUALITY OVER ALL: Use ControlNet for structure preservation

        Args:
            image: Source image
            control_image: Control image (canny, depth, etc.)
            prompt: Text prompt
            negative_prompt: Negative prompt
            controlnet_conditioning_scale: ControlNet strength (None = use config default from 'adapters.common.default_controlnet_conditioning_scale')
            strength: Denoising strength (None = use config default from 'adapters.common.default_strength')
            num_inference_steps: Number of steps (None = use config default from 'adapters.common.default_num_inference_steps')
            guidance_scale: Guidance scale (None = use config default from 'adapters.common.default_guidance_scale')
            seed: Random seed
            **kwargs: Additional pipeline arguments

        Returns:
            Transformed image with ControlNet guidance
        """

    @abstractmethod
    def get_controlnet_types(self) -> List[str]:
        """
        Get available ControlNet types

        Returns:
            List of ControlNet types (e.g., ['canny', 'depth', 'openpose'])
        """

    @abstractmethod
    def load_controlnet(
            self,
            controlnet_type: str,
            model_id: Optional[str] = None):
        """
        Load a specific ControlNet model

        Args:
            controlnet_type: Type of ControlNet (canny, depth, etc.)
            model_id: Optional specific model ID
        """

    @abstractmethod
    def estimate_vram(self, operation: str, **kwargs) -> float:
        """Estimate VRAM for an operation in MB"""

    @abstractmethod
    def enable_memory_efficient_mode(self):
        """Enable memory optimizations (CPU offload, etc.)"""

    @abstractmethod
    def clear_cache(self):
        """Clear any cached data/models"""

    def get_config_path(self) -> Optional[Path]:
        """Get associated config path if any"""
        return None

    def get_supported_strategies(self) -> List[str]:
        """Get list of strategies this adapter supports"""
        return ["direct_upscale", "progressive_outpaint", "tiled_expansion"]

    def clear_memory(self):
        """Clear memory/cache if applicable (backward compatibility)"""
        self.clear_cache()

    def enable_cpu_offload(self):
        """Enable CPU offload if supported (backward compatibility)"""
        self.enable_memory_efficient_mode()

    def to(self, device: str):
        """Move to specified device if applicable"""
        return self

    def supports_operation(self, operation: str) -> bool:
        """
        Check if adapter supports a specific operation

        Args:
            operation: Operation name (e.g., 'inpaint', 'img2img', 'controlnet')

        Returns:
            True if operation is supported
        """
        operation_map = {
            "inpaint": self.supports_inpainting,
            "img2img": self.supports_img2img,
            "enhance": self.supports_enhancement,
            "controlnet": self.supports_controlnet,
        }

        check_func = operation_map.get(operation)
        if check_func:
            return check_func()
        return False

    def cleanup(self):
        """Clean up resources (alias for clear_cache)"""
        self.clear_cache()
