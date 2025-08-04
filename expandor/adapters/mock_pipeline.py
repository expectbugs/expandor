"""
Mock Pipeline Interfaces for Testing
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image


@dataclass
class MockPipelineOutput:
    """Mock output from pipeline"""

    images: List[Image.Image]


class MockInpaintPipeline:
    """Mock inpainting pipeline for testing without real models"""

    def __init__(self):
        self.call_count = 0
        self.last_call_args = {}

    def __call__(
        self,
        prompt: str,
        image: Image.Image,
        mask_image: Image.Image,
        strength: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs
    ) -> MockPipelineOutput:
        """
        Simulate inpainting by blending original with noise in masked areas
        """
        # Get defaults from configuration if not provided
        from ..core.configuration_manager import ConfigurationManager
        config_manager = ConfigurationManager()
        
        if strength is None:
            strength = config_manager.get_value('adapters.common.default_strength')
        if num_inference_steps is None:
            num_inference_steps = config_manager.get_value('adapters.common.default_num_inference_steps')
        if guidance_scale is None:
            guidance_scale = config_manager.get_value('adapters.common.default_guidance_scale')
            
        self.call_count += 1
        self.last_call_args = {
            "prompt": prompt,
            "strength": strength,
            "steps": num_inference_steps,
            "guidance": guidance_scale,
        }

        # Convert to arrays
        img_array = np.array(image)
        mask_array = np.array(mask_image) / 255.0

        # Simulate inpainting: blend with generated pattern
        h, w = img_array.shape[:2]

        # Create a pattern based on prompt hash (deterministic)
        seed = abs(hash(prompt)) % 1000
        np.random.seed(seed)

        # Generate synthetic content
        if "nature" in prompt.lower() or "landscape" in prompt.lower():
            # Green/blue nature pattern
            pattern = np.zeros_like(img_array)
            pattern[:, :, 0] = np.random.randint(50, 100, (h, w))  # R
            pattern[:, :, 1] = np.random.randint(100, 200, (h, w))  # G
            pattern[:, :, 2] = np.random.randint(100, 150, (h, w))  # B
        else:
            # Generic pattern
            pattern = np.random.randint(100, 200, img_array.shape)

        # Apply mask
        for c in range(3):
            img_array[:, :, c] = (
                img_array[:, :, c] * (1 - mask_array) + pattern[:, :, c] * mask_array
            ).astype(np.uint8)

        # Convert back to image
        result = Image.fromarray(img_array)

        # Resize if dimensions specified
        if width and height:
            result = result.resize((width, height), Image.Resampling.LANCZOS)

        return MockPipelineOutput(images=[result])


class MockRefinerPipeline:
    """Mock refinement pipeline"""

    def __call__(
        self, prompt: str, image: Image.Image, strength: Optional[float] = None, **kwargs
    ) -> MockPipelineOutput:
        """
        Simulate refinement by slight sharpening
        """
        # Get default from configuration if not provided
        if strength is None:
            from ..core.configuration_manager import ConfigurationManager
            config_manager = ConfigurationManager()
            strength = config_manager.get_value('adapters.common.default_refiner_strength')
        # Simple sharpening filter
        from PIL import ImageFilter

        refined = image.filter(ImageFilter.SHARPEN)

        return MockPipelineOutput(images=[refined])


class MockImg2ImgPipeline:
    """Mock img2img pipeline"""

    def __call__(
        self, prompt: str, image: Image.Image, strength: Optional[float] = None, **kwargs
    ) -> MockPipelineOutput:
        """
        Simulate img2img by slight modification
        """
        # Get default from configuration if not provided
        if strength is None:
            from ..core.configuration_manager import ConfigurationManager
            config_manager = ConfigurationManager()
            strength = config_manager.get_value('adapters.common.default_strength')
        # Add slight noise based on strength
        img_array = np.array(image)
        config_manager = ConfigurationManager()
        noise_scale = config_manager.get_value('constants.processing.noise_scale_factor')
        noise = np.random.normal(0, strength * noise_scale, img_array.shape)

        max_rgb = config_manager.get_value('constants.image.max_rgb_value')
        result_array = np.clip(img_array + noise, 0, max_rgb).astype(np.uint8)
        result = Image.fromarray(result_array)

        return MockPipelineOutput(images=[result])
