"""
ControlNet extractors for various control types
Provides Canny, Depth, Blur, and REQUIRED explicit extraction

FAIL LOUD: This module requires OpenCV. If cv2 is not available,
the module will fail to import, making the requirement explicit.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from ..utils.config_loader import ConfigLoader
from ..utils.logging_utils import setup_logger
from ..core.configuration_manager import ConfigurationManager

# REQUIRED: OpenCV is mandatory for ControlNet extractors
# This import will fail if cv2 is not installed, which is the intended behavior
try:
    import cv2
except ImportError:
    raise ImportError(
        "ControlNet extractors require OpenCV (cv2).\n"
        "This is an optional feature that requires additional dependencies.\n"
        "Install with: pip install opencv-python>=4.8.0\n"
        "Or install expandor with ControlNet support: pip install expandor[controlnet]")


class ControlNetExtractor:
    """Extract control signals from images for ControlNet guidance"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize ControlNet extractor

        Args:
            logger: Logger instance
        """
        self.logger = logger or setup_logger(__name__)

        # Get configuration from ConfigurationManager
        self.config_manager = ConfigurationManager()
        
        # Get processor config
        try:
            self.processor_config = self.config_manager.get_processor_config('controlnet_extractors')
        except ValueError as e:
            # FAIL LOUD
            raise ValueError(
                f"Failed to load controlnet_extractors configuration!\n{str(e)}"
            )

        # Get config directory path for user config
        config_dir = Path(__file__).parent.parent / "config"
        self.config_loader = ConfigLoader(config_dir, logger=self.logger)

        # Lazy-load ControlNet configuration from user config
        self._config = None

    @property
    def config(self) -> Dict[str, Any]:
        """Lazy-load config, auto-create if missing"""
        if self._config is None:
            try:
                self._config = self.config_loader.load_config_file(
                    "controlnet_config.yaml")
            except FileNotFoundError:
                # Auto-create with defaults
                self.logger.info("Creating default controlnet_config.yaml")
                from ..utils.config_defaults import \
                    create_default_controlnet_config
                default_config = create_default_controlnet_config()

                # Save to user config directory
                user_config_dir = Path.home() / ".config" / "expandor"
                user_config_dir.mkdir(parents=True, exist_ok=True)
                config_path = user_config_dir / "controlnet_config.yaml"

                import yaml
                with open(config_path, 'w') as f:
                    yaml.dump(
                        default_config,
                        f,
                        default_flow_style=False,
                        sort_keys=False)

                self._config = default_config
                self.logger.info(
                    f"Created default ControlNet config at: {config_path}")
        return self._config

    @property
    def extractor_config(self) -> Dict[str, Any]:
        """Get extractor config section"""
        # Access config property which auto-creates if needed
        config = self.config
        if "extractors" not in config:
            raise ValueError(
                "extractors section not found in controlnet_config.yaml\n"
                "This should not happen with auto-created config.\n"
                "Please report this bug."
            )
        return config["extractors"]

    def extract_canny(
        self,
        image: Image.Image,
        low_threshold: Optional[int] = None,
        high_threshold: Optional[int] = None,
        dilate: bool = True,
        l2_gradient: bool = False
    ) -> Image.Image:
        """
        Extract Canny edges for structure guidance

        Args:
            image: Input PIL Image
            low_threshold: Lower threshold for edge detection (0-255)
            high_threshold: Upper threshold for edge detection (0-255)
            dilate: Whether to dilate edges for stronger guidance
            l2_gradient: Use L2 norm for gradient calculation (more accurate)

        Returns:
            PIL Image with Canny edges (RGB format)

        Raises:
            ValueError: If thresholds are invalid
        """
        # Get canny config section
        canny_config = self.extractor_config.get("canny", {})

        # Use config defaults if parameters not provided
        if low_threshold is None:
            if "default_low_threshold" not in canny_config:
                raise ValueError(
                    "default_low_threshold not found in canny config!\n"
                    "This value must be explicitly set in controlnet_config.yaml"
                )
            low_threshold = canny_config["default_low_threshold"]
        if high_threshold is None:
            if "default_high_threshold" not in canny_config:
                raise ValueError(
                    "default_high_threshold not found in canny config!\n"
                    "This value must be explicitly set in controlnet_config.yaml"
                )
            high_threshold = canny_config["default_high_threshold"]

        if "low_threshold_min" not in canny_config:
            raise ValueError(
                "low_threshold_min not found in canny config section\n"
                "Add: low_threshold_min: 0"
            )
        low_threshold_min = canny_config["low_threshold_min"]

        if "low_threshold_max" not in canny_config:
            raise ValueError(
                "low_threshold_max not found in canny config section\n"
                "Add: low_threshold_max: 255"
            )
        low_threshold_max = canny_config["low_threshold_max"]

        if "high_threshold_min" not in canny_config:
            raise ValueError(
                "high_threshold_min not found in canny config section\n"
                "Add: high_threshold_min: 0"
            )
        high_threshold_min = canny_config["high_threshold_min"]

        if "high_threshold_max" not in canny_config:
            raise ValueError(
                "high_threshold_max not found in canny config section\n"
                "Add: high_threshold_max: 255"
            )
        high_threshold_max = canny_config["high_threshold_max"]

        # Validate thresholds
        if not low_threshold_min <= low_threshold <= low_threshold_max:
            raise ValueError(
                f"low_threshold {low_threshold} must be between "
                f"{low_threshold_min} and {low_threshold_max}"
            )

        if not high_threshold_min <= high_threshold <= high_threshold_max:
            raise ValueError(
                f"high_threshold {high_threshold} must be between "
                f"{high_threshold_min} and {high_threshold_max}"
            )

        if low_threshold >= high_threshold:
            raise ValueError(
                f"low_threshold ({low_threshold}) must be less than "
                f"high_threshold ({high_threshold})"
            )

        # Convert to grayscale numpy array
        if image.mode != 'RGB':
            image = image.convert('RGB')

        np_image = np.array(image)
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(
            gray,
            low_threshold,
            high_threshold,
            L2gradient=l2_gradient)

        # Optionally dilate edges
        if dilate:
            # Get kernel config values
            if "kernel_size" not in canny_config:
                raise ValueError(
                    "kernel_size not found in canny config section\n"
                    "Add: kernel_size: 3"
                )
            kernel_size = canny_config["kernel_size"]

            if "dilation_iterations" not in canny_config:
                raise ValueError(
                    "dilation_iterations not found in canny config section\n"
                    "Add: dilation_iterations: 1"
                )
            dilation_iterations = canny_config["dilation_iterations"]

            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=dilation_iterations)

        # Convert back to RGB PIL Image
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)

    def extract_blur(
        self,
        image: Image.Image,
        radius: Optional[int] = None,
        blur_type: str = "gaussian"
    ) -> Image.Image:
        """
        Extract blur control map for soft guidance

        Args:
            image: Input PIL Image
            radius: Blur radius (None = config default)
            blur_type: Type of blur ("gaussian", "box", "motion")

        Returns:
            Blurred PIL Image for ControlNet guidance

        Raises:
            ValueError: If blur_type is invalid
        """
        # Get blur config section
        blur_config = self.extractor_config.get("blur", {})

        if "valid_types" not in blur_config:
            raise ValueError(
                "valid_types not found in blur config section\n"
                "Add: valid_types: [gaussian, box, motion]"
            )
        valid_types = blur_config["valid_types"]

        if blur_type not in valid_types:
            raise ValueError(
                f"Invalid blur_type '{blur_type}'. Must be one of: {valid_types}\n"
                f"Add '{blur_type}' to valid_types in config if needed."
            )

        # Use default radius if not provided
        if radius is None:
            if "default_radius" not in blur_config:
                raise ValueError(
                    "default_radius not found in blur config section\n"
                    "Add: default_radius: 5"
                )
            radius = blur_config["default_radius"]

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if blur_type == "gaussian":
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        elif blur_type == "box":
            return image.filter(ImageFilter.BoxBlur(radius=radius))
        elif blur_type == "motion":
            # Motion blur using cv2
            # Get motion blur params from config
            if "motion_kernel_multiplier" not in blur_config:
                raise ValueError(
                    "motion_kernel_multiplier not found in blur config section\n"
                    "Add: motion_kernel_multiplier: 2")
            kernel_multiplier = blur_config["motion_kernel_multiplier"]

            if "motion_kernel_offset" not in blur_config:
                raise ValueError(
                    "motion_kernel_offset not found in blur config section\n"
                    "Add: motion_kernel_offset: 1"
                )
            kernel_offset = blur_config["motion_kernel_offset"]

            # Get minimum radius from processor config
            min_radius = self.processor_config['motion_blur_min_radius']

            # Ensure radius is valid
            radius = max(radius, min_radius)

            np_image = np.array(image)
            size = radius * kernel_multiplier + kernel_offset
            kernel = np.zeros((size, size))
            kernel[int((size - 1) / 2), :] = np.ones(size)
            kernel = kernel / size

            # Apply motion blur to each channel
            output = np.zeros_like(np_image)
            for i in range(self.processor_config['rgb_channels']):
                output[:, :, i] = cv2.filter2D(np_image[:, :, i], -1, kernel)

            return Image.fromarray(output.astype(np.uint8))

    def extract_depth(
        self,
        image: Image.Image,
        model_id: Optional[str] = None,
        normalize: Optional[bool] = None,
        invert: Optional[bool] = None
    ) -> Image.Image:
        """
        Extract depth map using MiDaS/DPT models
        
        IMPORTANT: This is currently a placeholder implementation.
        For actual depth extraction, install transformers:
            pip install transformers
            
        The current implementation returns a blurred grayscale image
        as a placeholder. This will not produce accurate depth-guided
        results.

        Args:
            image: Input PIL Image
            model_id: HuggingFace model ID for depth estimation
            normalize: Normalize depth values to 0-255
            invert: Invert depth map (near/far swap)

        Returns:
            Depth map as PIL Image (placeholder: blurred grayscale)

        Note: This is a placeholder that shows the interface.
        Full depth extraction requires transformers library.
        """
        # Get depth config
        depth_config = self.extractor_config.get("depth", {})

        # Use config defaults if not provided
        if model_id is None:
            if "model_id" not in depth_config:
                raise ValueError(
                    "model_id not found in depth config section\n"
                    "Add: model_id: Intel/dpt-large"
                )
            model_id = depth_config["model_id"]

        if normalize is None:
            if "normalize" not in depth_config:
                raise ValueError(
                    "normalize not found in depth config section\n"
                    "Add: normalize: true"
                )
            normalize = depth_config["normalize"]

        if invert is None:
            if "invert" not in depth_config:
                raise ValueError(
                    "invert not found in depth config section\n"
                    "Add: invert: false"
                )
            invert = depth_config["invert"]

        # This is a placeholder implementation
        # Real implementation would use transformers library
        self.logger.warning(
            "Depth extraction requires transformers library.\n"
            "Install with: pip install transformers\n"
            "Using blur as depth placeholder for now."
        )

        # Get depth blur radius from processor config
        depth_blur_radius = self.processor_config['depth_blur_radius']

        # For now, return a heavily blurred grayscale as placeholder
        gray = image.convert('L')
        blurred = gray.filter(
            ImageFilter.GaussianBlur(
                radius=depth_blur_radius))

        if invert:
            # Invert the values
            np_img = np.array(blurred)
            np_img = self.processor_config['invert_max_value'] - np_img
            blurred = Image.fromarray(np_img)

        # Convert back to RGB
        return blurred.convert('RGB')

    def preprocess_control_image(
        self,
        image: Image.Image,
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Preprocess control image to match target dimensions

        Args:
            image: Control image
            target_size: Target (width, height)

        Returns:
            Resized control image

        FAIL LOUD: No automatic resizing without config
        """
        # Use pre-loaded config
        config = self.config

        if "extractors" not in config:
            raise ValueError(
                "extractors section not found in controlnet_config.yaml\n"
                "Run 'expandor --setup-controlnet' to create config"
            )

        if "resampling" not in config["extractors"]:
            raise ValueError(
                "resampling section not found in extractors config\n"
                "Add: resampling:\n  method: LANCZOS"
            )

        resampling_config = config["extractors"]["resampling"]

        if "method" not in resampling_config:
            raise ValueError(
                "method not found in resampling config section\n"
                "Add: method: LANCZOS"
            )

        resampling_method = resampling_config["method"]

        # Map string to PIL constant
        resampling_map = {
            "NEAREST": Image.Resampling.NEAREST,
            "BOX": Image.Resampling.BOX,
            "BILINEAR": Image.Resampling.BILINEAR,
            "HAMMING": Image.Resampling.HAMMING,
            "BICUBIC": Image.Resampling.BICUBIC,
            "LANCZOS": Image.Resampling.LANCZOS,
        }

        if resampling_method not in resampling_map:
            raise ValueError(
                f"Invalid resampling method '{resampling_method}'\n"
                f"Valid methods: {list(resampling_map.keys())}"
            )

        resampling = resampling_map[resampling_method]

        if image.size != target_size:
            return image.resize(target_size, resampling)
        return image
