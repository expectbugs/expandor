"""
Image utility functions for Expandor
Provides gradient masks, blending, and edge color extraction for seamless expansions.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..core.configuration_manager import ConfigurationManager

logger = logging.getLogger(__name__)

# Get configuration manager instance
_config_manager = ConfigurationManager()


def create_gradient_mask(
        width: int,
        height: int,
        direction: str,
        blur_radius: int,
        fade_start: float = 0.0) -> Image.Image:
    """
    Create gradient mask for smooth blending.

    Args:
        width: Mask width
        height: Mask height
        direction: One of 'left', 'right', 'top', 'bottom'
        blur_radius: Radius for gaussian blur
        fade_start: Where to start fade (0.0-1.0 of dimension)

    Returns:
        Grayscale mask image with gradient
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if direction == "left":
        fade_width = int(blur_radius * (1 - fade_start))
        if fade_width > 0:
            for i in range(fade_width):
                alpha = i / fade_width
                mask[:, i] = int(_config_manager.get_value('image_processing.masks.max_value') * alpha)
        mask[:, fade_width:] = _config_manager.get_value('image_processing.masks.max_value')

    elif direction == "right":
        fade_width = int(blur_radius * (1 - fade_start))
        if fade_width > 0:
            for i in range(fade_width):
                alpha = i / fade_width
                mask[:, -(i + 1)] = int(_config_manager.get_value('image_processing.masks.max_value') * alpha)
        mask[:, :-fade_width] = _config_manager.get_value('image_processing.masks.max_value')

    elif direction == "top":
        fade_height = int(blur_radius * (1 - fade_start))
        if fade_height > 0:
            for i in range(fade_height):
                alpha = i / fade_height
                mask[i, :] = int(_config_manager.get_value('image_processing.masks.max_value') * alpha)
        mask[fade_height:, :] = _config_manager.get_value('image_processing.masks.max_value')

    elif direction == "bottom":
        fade_height = int(blur_radius * (1 - fade_start))
        if fade_height > 0:
            for i in range(fade_height):
                alpha = i / fade_height
                mask[-(i + 1), :] = int(_config_manager.get_value('image_processing.masks.max_value') * alpha)
        mask[:-fade_height, :] = _config_manager.get_value('image_processing.masks.max_value')

    # Convert to PIL and apply gaussian blur
    mask_img = Image.fromarray(mask, mode="L")
    if blur_radius > 0:
        blur_divisor = _config_manager.get_value('image_processing.blur.gaussian_divisor')
        mask_img = mask_img.filter(
            ImageFilter.GaussianBlur(
                radius=blur_radius // blur_divisor))

    return mask_img


def blend_images(
        img1: Image.Image,
        img2: Image.Image,
        mask: Image.Image,
        mode: str = "normal") -> Image.Image:
    """
    Blend two images using mask.

    Args:
        img1: First image (revealed where mask is white)
        img2: Second image (revealed where mask is black)
        mask: Grayscale blend mask
        mode: Blend mode ('normal', 'overlay', 'soft_light')

    Returns:
        Blended image
    """
    # Ensure all images are same size
    if img1.size != img2.size or img1.size != mask.size:
        logger.warning(
            f"Image sizes don't match: {img1.size}, {img2.size}, {mask.size}"
        )
        # Resize to match first image
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        mask = mask.resize(img1.size, Image.Resampling.LANCZOS)

    # Ensure correct modes
    if img1.mode != "RGB":
        img1 = img1.convert("RGB")
    if img2.mode != "RGB":
        img2 = img2.convert("RGB")
    if mask.mode != "L":
        mask = mask.convert("L")

    if mode == "normal":
        return Image.composite(img1, img2, mask)
    else:
        # For other modes, use numpy operations
        arr1 = np.array(img1, dtype=np.float32) / 255.0
        arr2 = np.array(img2, dtype=np.float32) / 255.0
        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        mask_arr = mask_arr[:, :, np.newaxis]  # Add channel dimension

        if mode == "overlay":
            # Overlay blend mode
            result = np.where(
                arr2 < 0.5, 2 * arr1 * arr2, 1 - 2 * (1 - arr1) * (1 - arr2)
            )
        elif mode == "soft_light":
            # Soft light blend mode
            result = (1 - 2 * arr2) * arr1 * arr1 + 2 * arr2 * arr1
        else:
            result = arr1  # Fallback

        # Apply mask
        final = mask_arr * result + (1 - mask_arr) * arr2
        final = np.clip(final * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(final, mode="RGB")


def extract_edge_colors(
    image: Image.Image, edge: str, width: int = 1, method: str = "average"
) -> np.ndarray:
    """
    Extract colors from image edge.

    Args:
        image: Source image
        edge: One of 'left', 'right', 'top', 'bottom'
        width: Number of pixels to sample
        method: 'average', 'median', or 'mode'

    Returns:
        Array of colors [R, G, B] or [height, 3] / [width, 3] for edges
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    arr = np.array(image)

    if edge == "left":
        edge_pixels = arr[:, :width, :]
    elif edge == "right":
        edge_pixels = arr[:, -width:, :]
    elif edge == "top":
        edge_pixels = arr[:width, :, :]
    elif edge == "bottom":
        edge_pixels = arr[-width:, :, :]
    else:
        raise ValueError(f"Invalid edge: {edge}")

    if method == "average":
        # Average across the width dimension
        if edge in ["left", "right"]:
            return np.mean(edge_pixels, axis=1)
        else:
            return np.mean(edge_pixels, axis=0)
    elif method == "median":
        if edge in ["left", "right"]:
            return np.median(edge_pixels, axis=1)
        else:
            return np.median(edge_pixels, axis=0)
    elif method == "mode":
        # Use most common color
        if edge in ["left", "right"]:
            result = np.zeros((edge_pixels.shape[0], 3))
            for i in range(edge_pixels.shape[0]):
                for c in range(3):
                    values, counts = np.unique(
                        edge_pixels[i, :, c], return_counts=True)
                    result[i, c] = values[np.argmax(counts)]
            return result
        else:
            result = np.zeros((edge_pixels.shape[1], 3))
            for i in range(edge_pixels.shape[1]):
                for c in range(3):
                    values, counts = np.unique(
                        edge_pixels[:, i, c], return_counts=True)
                    result[i, c] = values[np.argmax(counts)]
            return result
    else:
        return np.mean(edge_pixels, axis=1 if edge in ["left", "right"] else 0)


def create_noise_pattern(
    width: int,
    height: int,
    scale: Optional[float] = None,
    octaves: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Create Perlin-like noise pattern for texture synthesis.

    Args:
        width: Pattern width
        height: Pattern height
        scale: Noise scale (smaller = finer detail)
        octaves: Number of noise layers
        seed: Random seed

    Returns:
        Noise array in range [0, 1]
    """
    # Load defaults from config if not provided
    if scale is None:
        scale = _config_manager.get_value('image_processing.noise.perlin_scale')
    if octaves is None:
        octaves = _config_manager.get_value('image_processing.noise.perlin_octaves')
    
    if seed is not None:
        np.random.seed(seed)

    # Simple noise implementation
    noise = np.zeros((height, width))

    octave_freq_base = _config_manager.get_value('image_processing.noise.octave_frequency_base')
    octave_amp_base = _config_manager.get_value('image_processing.noise.octave_amplitude_base')
    
    for octave in range(octaves):
        freq = octave_freq_base**octave
        amp = octave_amp_base**octave

        # Generate random gradients
        grid_width = int(width * scale * freq) + 1
        grid_height = int(height * scale * freq) + 1
        gradients = np.random.randn(grid_height, grid_width, 2)

        # Interpolate
        x = np.linspace(0, grid_width - 1, width)
        y = np.linspace(0, grid_height - 1, height)
        xv, yv = np.meshgrid(x, y)

        # Bilinear interpolation (simplified)
        x0 = np.floor(xv).astype(int)
        y0 = np.floor(yv).astype(int)
        x1 = np.minimum(x0 + 1, grid_width - 1)
        y1 = np.minimum(y0 + 1, grid_height - 1)

        fx = xv - x0
        fy = yv - y0

        # Add this octave
        octave_noise = (1 - fx) * (1 - fy) * \
            np.random.randn(height, width) * amp
        noise += octave_noise

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


def create_circular_mask(
    size: Tuple[int, int], center: Tuple[int, int], radius: int, feather: int = 0
) -> Image.Image:
    """
    Create circular mask with optional feathering.

    Args:
        size: (width, height) of mask
        center: (x, y) center of circle
        radius: Circle radius
        feather: Feather amount in pixels

    Returns:
        Grayscale mask image
    """
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    if feather > 0:
        # Draw multiple circles with decreasing opacity
        for i in range(feather):
            alpha = int(255 * (i / feather))
            r = radius + feather - i
            draw.ellipse([center[0] - r, center[1] - r,
                          center[0] + r, center[1] + r], fill=alpha)

    # Draw solid center
    draw.ellipse(
        [
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ],
        fill=255,
    )

    return mask


def gaussian_weights(size: int, sigma: Optional[float] = None) -> np.ndarray:
    """
    Generate 1D Gaussian weights for blending.

    Args:
        size: Number of weights
        sigma: Standard deviation (default: size/4)

    Returns:
        Normalized weights array
    """
    if sigma is None:
        sigma = size / 4.0

    x = np.arange(size)
    center = size / 2.0
    weights = np.exp(-0.5 * ((x - center) / sigma) ** 2)

    # Normalize
    weights = weights / weights.sum()
    return weights
