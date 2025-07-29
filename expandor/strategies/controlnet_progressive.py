"""
ControlNet-guided progressive expansion strategy
Uses control signals to maintain structure during expansion
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter

from ..core.config import ExpandorConfig
from ..utils.logging_utils import setup_logger
from ..utils.config_loader import ConfigLoader
from .progressive_outpaint import ProgressiveOutpaintStrategy


class ControlNetProgressiveStrategy(ProgressiveOutpaintStrategy):
    """
    Progressive expansion with ControlNet guidance
    
    Extends ProgressiveOutpaintStrategy to use control signals
    for maintaining structure and coherence during expansion.
    """
    
    VRAM_PRIORITY = 4  # Higher than regular progressive (3) due to ControlNet overhead
    
    def __init__(self, config: ExpandorConfig, logger: Optional[logging.Logger] = None):
        """Initialize ControlNet progressive strategy"""
        super().__init__(config, logger)
        self.logger = logger or setup_logger(__name__)
        
        # Get config directory path
        config_dir = Path(__file__).parent.parent / "config"
        self.config_loader = ConfigLoader(config_dir, logger=self.logger)
        
        # Load ControlNet configuration
        try:
            controlnet_config = self.config_loader.load_config_file("controlnet_config.yaml")
        except FileNotFoundError:
            raise FileNotFoundError(
                "controlnet_config.yaml not found in config directory.\n"
                "This file is REQUIRED for ControlNet strategies.\n"
                "Create the configuration file with strategy settings."
            )
        
        # ControlNet-specific config from strategy params
        if "controlnet_config" not in config.strategy_params:
            raise ValueError(
                "controlnet_config is REQUIRED in strategy_params for ControlNetProgressiveStrategy.\n"
                "Provide it when creating ExpandorConfig:\n"
                "strategy_params={'controlnet_config': {...}}"
            )
        self.controlnet_config = config.strategy_params["controlnet_config"]
        
        # Get required parameters - FAIL LOUD if missing
        if "control_type" not in self.controlnet_config:
            raise ValueError(
                "control_type is REQUIRED in controlnet_config for strategy.\n"
                "Specify the control type (e.g., 'canny', 'depth', 'blur')"
            )
        self.control_type = self.controlnet_config["control_type"]
        
        # Get conditioning scale from multiple sources in priority order
        self.controlnet_strength = self.controlnet_config.get("controlnet_strength")
        if self.controlnet_strength is None:
            # Try quality preset
            quality_config = self.config_loader.load_quality_preset(config.quality_preset)
            cn_settings = quality_config.get("controlnet", {})
            if "controlnet_strength" not in cn_settings:
                raise ValueError(
                    f"controlnet_strength not found in quality preset '{config.quality_preset}' "
                    f"or in strategy_params.controlnet_config.\n"
                    f"Add it to your quality preset under 'controlnet' section or provide it explicitly."
                )
            self.controlnet_strength = cn_settings["controlnet_strength"]
        
        # Get strategy config section from loaded config
        if "strategy" not in controlnet_config:
            raise ValueError(
                "strategy section not found in controlnet_config.yaml\n"
                "This is REQUIRED for ControlNet strategy configuration."
            )
        strategy_config = controlnet_config["strategy"]
        
        # Get extract_at_each_step with config-based default
        if "default_extract_at_each_step" not in strategy_config:
            raise ValueError(
                "default_extract_at_each_step not found in strategy section of controlnet_config.yaml\n"
                "Add: default_extract_at_each_step: true"
            )
        default_extract = strategy_config["default_extract_at_each_step"]
        
        self.extract_at_each_step = self.controlnet_config.get(
            "extract_at_each_step", 
            default_extract
        )
        
        # Initialize extractors if available
        self.extractor = None
        
        # Import extractors dynamically - FAIL LOUD if control type requires them
        # but they're not available
        control_types_requiring_extraction = ["canny", "blur", "depth", "normal"]
        
        if self.control_type in control_types_requiring_extraction:
            try:
                from ..processors.controlnet_extractors import ControlNetExtractor
                self.extractor = ControlNetExtractor(logger=self.logger)
            except ImportError as e:
                raise ImportError(
                    f"Control type '{self.control_type}' requires ControlNet extractors.\n"
                    f"ControlNet extractors require OpenCV (cv2).\n"
                    f"Install with: pip install opencv-python>=4.8.0\n"
                    f"Original error: {e}"
                )
        
        self.logger.info(
            f"Initialized ControlNet progressive strategy: "
            f"type={self.control_type}, strength={self.controlnet_strength}"
        )
    
    def _prepare_control_image(
        self, 
        image: Image.Image, 
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Prepare control image for the given target size
        
        Args:
            image: Source image to extract control from
            target_size: Target (width, height) for control image
            
        Returns:
            Control image at target size
            
        FAIL LOUD: Invalid control types or extraction failures are fatal
        """
        # First resize the image if needed
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # If no extraction needed, return as is
        if not self.extract_at_each_step:
            return image
        
        # Extract control signal based on type
        if self.control_type == "canny":
            if not self.extractor:
                raise RuntimeError(
                    f"Extractor not initialized for control type '{self.control_type}'.\n"
                    "This is an internal error. Please report this bug."
                )
            return self.extractor.extract_canny(image)
            
        elif self.control_type == "blur":
            if not self.extractor:
                raise RuntimeError(
                    f"Extractor not initialized for control type '{self.control_type}'.\n"
                    "This is an internal error. Please report this bug."
                )
            blur_radius = self.controlnet_config.get("blur_radius")
            return self.extractor.extract_blur(image, radius=blur_radius)
            
        elif self.control_type == "depth":
            if not self.extractor:
                raise RuntimeError(
                    f"Extractor not initialized for control type '{self.control_type}'.\n"
                    "This is an internal error. Please report this bug."
                )
            return self.extractor.extract_depth(image)
            
        elif self.control_type == "identity":
            # Use image as-is for control
            return image
            
        else:
            # For other types (user provided control), expect it in controlnet_config
            if "control_image" not in self.controlnet_config:
                raise ValueError(
                    f"Control type '{self.control_type}' requires a pre-computed control image.\n"
                    f"Provide it in controlnet_config['control_image'] as a PIL Image."
                )
            
            control_image = self.controlnet_config["control_image"]
            if not isinstance(control_image, Image.Image):
                raise TypeError(
                    f"control_image must be a PIL Image, got {type(control_image)}"
                )
            
            # Resize to target if needed
            if control_image.size != target_size:
                control_image = control_image.resize(target_size, Image.Resampling.LANCZOS)
            
            return control_image
    
    def _expand_step(
        self, 
        current_image: Image.Image,
        target_size: Tuple[int, int],
        expansion_step: int,
        total_steps: int,
        **kwargs
    ) -> Image.Image:
        """
        Execute one expansion step with ControlNet guidance
        
        Overrides parent method to add control image preparation
        and use ControlNet-specific generation methods.
        """
        self.logger.info(
            f"ControlNet expansion step {expansion_step + 1}/{total_steps}: "
            f"{current_image.size} -> {target_size}"
        )
        
        # Prepare control image for this step
        control_image = self._prepare_control_image(current_image, target_size)
        
        # Determine operation type
        width, height = target_size
        current_width, current_height = current_image.size
        
        # Create canvas and position current image
        canvas = Image.new("RGB", target_size, color="white")
        x_offset = (width - current_width) // 2
        y_offset = (height - current_height) // 2
        canvas.paste(current_image, (x_offset, y_offset))
        
        # Create mask for new content
        mask = Image.new("L", target_size, color=255)  # White = inpaint
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(
            [x_offset, y_offset, x_offset + current_width - 1, y_offset + current_height - 1],
            fill=0  # Black = keep
        )
        
        # Blur mask for smooth transitions
        blur_radius = self._calculate_blur_radius(target_size, current_image.size)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Get generation parameters
        prompt = self.config.prompt
        negative_prompt = self.config.negative_prompt
        seed = self.config.seed
        
        # Adaptive strength based on expansion ratio
        base_strength = kwargs.get("strength", 0.95)
        strength = self._calculate_adaptive_strength(
            current_image.size, target_size, expansion_step, total_steps, base_strength
        )
        
        # Use ControlNet inpainting
        if not hasattr(self.adapter, 'controlnet_inpaint'):
            raise RuntimeError(
                "Adapter does not support ControlNet operations.\n"
                "Ensure you're using DiffusersPipelineAdapter with ControlNet loaded."
            )
        
        # ControlNet-specific parameters
        controlnet_params = {
            "control_type": self.control_type,
            "controlnet_strength": self.controlnet_strength,
        }
        
        # Update with any custom parameters
        controlnet_params.update(kwargs.get("controlnet_params", {}))
        
        try:
            result = self.adapter.controlnet_inpaint(
                image=canvas,
                mask=mask,
                control_image=control_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                seed=seed,
                **controlnet_params
            )
            
            return result
            
        except Exception as e:
            # FAIL LOUD with helpful context
            raise RuntimeError(
                f"ControlNet expansion step failed at {current_image.size} -> {target_size}\n"
                f"Control type: {self.control_type}, Step: {expansion_step + 1}/{total_steps}\n"
                f"Error: {str(e)}\n"
                f"Solutions:\n"
                f"  1. Ensure ControlNet model is loaded for type '{self.control_type}'\n"
                f"  2. Check VRAM availability\n"
                f"  3. Try smaller expansion steps\n"
                f"  4. Verify control image extraction worked correctly"
            ) from e
    
    def get_vram_requirements(self) -> float:
        """
        Get estimated VRAM requirements including ControlNet overhead
        
        Adds ControlNet-specific overhead to base progressive requirements
        """
        base_vram = super().get_vram_requirements()
        
        # Estimate ControlNet overhead from operations
        max_dimension = max(self.config.target_resolution)
        operation = "controlnet_inpaint"  # Primary operation for this strategy
        
        controlnet_vram = self.adapter.estimate_vram(
            operation=operation,
            width=max_dimension,
            height=max_dimension
        )
        
        # Return the larger of the two estimates
        # (ControlNet estimate includes base operation + overhead)
        return max(base_vram, controlnet_vram)
    
    def validate_support(self) -> bool:
        """
        Validate that the adapter supports ControlNet operations
        
        Returns:
            True if ControlNet is supported and loaded
            
        Raises:
            RuntimeError: If ControlNet is not available
        """
        if not hasattr(self.adapter, 'supports_controlnet'):
            raise RuntimeError(
                "Adapter does not have supports_controlnet method.\n"
                "Ensure you're using a ControlNet-capable adapter."
            )
        
        if not self.adapter.supports_controlnet():
            raise RuntimeError(
                f"Adapter does not support ControlNet for model type: {getattr(self.adapter, 'model_type', 'unknown')}\n"
                "ControlNet currently requires SDXL models."
            )
        
        # Check if ControlNet is actually loaded
        if not hasattr(self.adapter, 'get_controlnet_types'):
            raise RuntimeError(
                "Adapter missing get_controlnet_types method.\n"
                "This is required for ControlNet strategies."
            )
        
        loaded_types = self.adapter.get_controlnet_types()
        if self.control_type not in loaded_types:
            raise RuntimeError(
                f"ControlNet type '{self.control_type}' is not loaded.\n"
                f"Available types: {loaded_types}\n"
                f"Load it first: adapter.load_controlnet(model_id, '{self.control_type}')"
            )
        
        return True