"""
Base adapter interface for pipeline implementations
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image


class BasePipelineAdapter(ABC):
    """Abstract base class for pipeline adapters"""

    @abstractmethod
    def load_pipeline(self, pipeline_type: str, **kwargs) -> Any:
        """Load a specific pipeline type"""
        pass

    @abstractmethod
    def get_pipeline(self, pipeline_type: str) -> Optional[Any]:
        """Get loaded pipeline by type"""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """Generate image from text prompt"""
        pass

    @abstractmethod
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
        **kwargs
    ) -> Image.Image:
        """Inpaint masked region of image"""
        pass

    @abstractmethod
    def img2img(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """Transform image with img2img"""
        pass

    @abstractmethod
    def refine(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        """Run refinement pipeline"""
        pass

    @abstractmethod
    def enhance(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        scale_factor: int = 2,
        **kwargs
    ) -> Image.Image:
        """Enhance/upscale image"""
        pass

    @abstractmethod
    def get_optimal_dimensions(
        self, target_width: int, target_height: int
    ) -> Tuple[int, int]:
        """Get optimal dimensions for the model"""
        pass

    @abstractmethod
    def unload_pipeline(self, pipeline_type: str):
        """Unload pipeline to free memory"""
        pass

    @abstractmethod
    def get_available_pipelines(self) -> List[str]:
        """List available pipeline types"""
        pass

    @abstractmethod
    def supports_inpainting(self) -> bool:
        """Check if adapter supports inpainting"""
        pass

    @abstractmethod
    def supports_img2img(self) -> bool:
        """Check if adapter supports img2img"""
        pass

    @abstractmethod
    def supports_enhancement(self) -> bool:
        """Check if adapter supports enhancement"""
        pass

    @abstractmethod
    def supports_controlnet(self) -> bool:
        """Check if adapter supports ControlNet"""
        pass

    @abstractmethod
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
            controlnet_conditioning_scale: ControlNet strength
            strength: Denoising strength
            num_inference_steps: Number of steps
            guidance_scale: Guidance scale
            seed: Random seed
            **kwargs: Additional pipeline arguments

        Returns:
            Inpainted image with ControlNet guidance
        """
        pass

    @abstractmethod
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
            controlnet_conditioning_scale: ControlNet strength
            strength: Denoising strength
            num_inference_steps: Number of steps
            guidance_scale: Guidance scale
            seed: Random seed
            **kwargs: Additional pipeline arguments

        Returns:
            Transformed image with ControlNet guidance
        """
        pass

    @abstractmethod
    def get_controlnet_types(self) -> List[str]:
        """
        Get available ControlNet types

        Returns:
            List of ControlNet types (e.g., ['canny', 'depth', 'openpose'])
        """
        pass

    @abstractmethod
    def load_controlnet(self, controlnet_type: str, model_id: Optional[str] = None):
        """
        Load a specific ControlNet model

        Args:
            controlnet_type: Type of ControlNet (canny, depth, etc.)
            model_id: Optional specific model ID
        """
        pass

    @abstractmethod
    def estimate_vram(self, operation: str, **kwargs) -> float:
        """Estimate VRAM for an operation in MB"""
        pass

    @abstractmethod
    def enable_memory_efficient_mode(self):
        """Enable memory optimizations (CPU offload, etc.)"""
        pass

    @abstractmethod
    def clear_cache(self):
        """Clear any cached data/models"""
        pass

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
