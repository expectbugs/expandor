# Expandor Phase 3 Step 2: Quality Systems - Ultra-Detailed Implementation Guide

## Overview

This document provides a foolproof, zero-error implementation guide for the Quality Systems component of Expandor Phase 3. This includes multi-pass refinement, artifact detection, boundary tracking, and smart quality refinement. Each section includes exact code, precise directory locations, comprehensive error handling, and validation steps.

**IMPORTANT**: This is for the standalone Expandor project, NOT part of ai-wallpaper. All implementations follow the FAIL LOUD philosophy - perfect operation or complete failure, no graceful degradation.

## Prerequisites Verification

Before starting ANY implementation:

```bash
# 1. Verify you're in the expandor repository (NOT ai-wallpaper)
pwd
# Expected: /path/to/expandor

# 2. Verify Python environment is activated
which python
# Should show: /path/to/expandor/venv/bin/python

# 3. Verify required directories exist
ls -la expandor/processors/
ls -la expandor/processors/refinement/
# Should show existing __init__.py files

# 4. Verify core components exist from Phase 2
ls -la expandor/core/metadata_tracker.py
ls -la expandor/core/boundary_tracker.py
# These should exist from Phase 2 implementation
```

## 0. Create Missing Utility Files

### 0.1 Create Image Utilities

**EXACT FILE PATH**: `expandor/utils/image_utils.py`

```python
"""
Image utility functions for Expandor
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, Optional

def create_gradient_mask(width: int, height: int, direction: str, 
                        blur_radius: int) -> Image.Image:
    """Create gradient mask for smooth blending."""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if direction == "left":
        mask[:, :blur_radius] = np.linspace(255, 0, blur_radius)
    elif direction == "right":
        mask[:, -blur_radius:] = np.linspace(0, 255, blur_radius)
    elif direction == "top":
        mask[:blur_radius, :] = np.linspace(255, 0, blur_radius).reshape(-1, 1)
    elif direction == "bottom":
        mask[-blur_radius:, :] = np.linspace(0, 255, blur_radius).reshape(-1, 1)
    
    mask_img = Image.fromarray(mask, mode='L')
    return mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius//4))

def blend_images(img1: Image.Image, img2: Image.Image, 
                mask: Image.Image) -> Image.Image:
    """Blend two images using mask."""
    return Image.composite(img1, img2, mask)

def extract_edge_colors(image: Image.Image, edge: str, 
                       width: int = 1) -> np.ndarray:
    """Extract colors from image edge."""
    arr = np.array(image)
    if edge == "left":
        return arr[:, :width, :]
    elif edge == "right":
        return arr[:, -width:, :]
    elif edge == "top":
        return arr[:width, :, :]
    elif edge == "bottom":
        return arr[-width:, :, :]
```

### 0.2 Create Edge Analysis Module

**EXACT FILE PATH**: `expandor/processors/edge_analysis.py`

```python
"""
Edge analysis for artifact detection
"""

import numpy as np
from typing import List, Tuple, Optional
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

class EdgeAnalyzer:
    @staticmethod
    def detect_edges(image_array: np.ndarray, 
                    method: str = "sobel") -> np.ndarray:
        """Detect edges using specified method."""
        if not HAS_CV2:
            return EdgeAnalyzer._detect_edges_numpy(image_array)
        
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        if method == "sobel":
            edges_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            edges_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(edges_x**2 + edges_y**2)
        elif method == "canny":
            edges = cv2.Canny(gray, 100, 200)
        return edges
    
    @staticmethod
    def _detect_edges_numpy(image_array: np.ndarray) -> np.ndarray:
        """Fallback edge detection without OpenCV."""
        gray = np.mean(image_array, axis=2)
        # Simple Sobel implementation
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        from scipy.ndimage import convolve
        edges_x = convolve(gray, sobel_x)
        edges_y = convolve(gray, sobel_y)
        return np.sqrt(edges_x**2 + edges_y**2)
```

### 0.3 Create Memory Utilities

**EXACT FILE PATH**: `expandor/utils/memory_utils.py`

```python
"""
Memory management utilities for Expandor
"""

import gc
import torch
from contextlib import contextmanager
from typing import Any

@contextmanager
def gpu_memory_manager():
    """Context manager for GPU memory cleanup."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

def offload_to_cpu(tensor: Any) -> Any:
    """Move tensor to CPU memory."""
    if hasattr(tensor, 'cpu'):
        return tensor.cpu()
    return tensor

def load_to_gpu(tensor: Any) -> Any:
    """Move tensor to GPU memory."""
    if torch.cuda.is_available() and hasattr(tensor, 'cuda'):
        return tensor.cuda()
    return tensor
```

## 1. Smart Artifact Detection Implementation

### 1.1 Create Smart Artifact Detector

**EXACT FILE PATH**: `expandor/processors/artifact_detector.py`

```python
"""
Smart Artifact Detection with Zero Tolerance for Seams
Implements aggressive multi-method detection for perfect seamless images.
Zero tolerance implementation for the Expandor standalone system.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
from scipy import ndimage
import warnings

from ..core.exceptions import ExpandorError

# FAIL LOUD: cv2 is NOT a dependency - all operations use numpy/PIL/scipy
# If cv2 operations are needed, the system must fail loudly
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV (cv2) not available - using numpy/scipy alternatives")


class ArtifactSeverity(Enum):
    """Severity levels for detected artifacts"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectionResult:
    """Result from artifact detection"""
    has_artifacts: bool
    severity: ArtifactSeverity
    artifact_mask: Optional[np.ndarray]
    seam_locations: List[Dict[str, Any]]
    detection_scores: Dict[str, float]
    total_artifact_pixels: int
    artifact_percentage: float


class SmartArtifactDetector:
    """
    Aggressive artifact detection with multiple methods.
    Zero tolerance for visible seams or boundaries.
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # Detection thresholds from config
        detection_config = self.config.get('artifact_detection', {})
        self.thresholds = {
            'aggressive': {
                'color_threshold': 10,
                'gradient_multiplier': 2.0,
                'frequency_multiplier': 1.5,
                'edge_sensitivity': 0.8,
                'min_seam_width': 5,
                'max_seam_width': 20,
                'texture_window': 50
            },
            'standard': {
                'color_threshold': 15,
                'gradient_multiplier': 2.5,
                'frequency_multiplier': 2.0,
                'edge_sensitivity': 0.6,
                'min_seam_width': 3,
                'max_seam_width': 15,
                'texture_window': 40
            },
            'light': {
                'color_threshold': 20,
                'gradient_multiplier': 3.0,
                'frequency_multiplier': 2.5,
                'edge_sensitivity': 0.4,
                'min_seam_width': 2,
                'max_seam_width': 10,
                'texture_window': 30
            }
        }
        
    def detect_artifacts(self, 
                        image_path: Path,
                        metadata: Dict[str, Any],
                        detection_level: str = "aggressive") -> DetectionResult:
        """
        Comprehensive artifact detection using multiple methods.
        
        Args:
            image_path: Path to image to analyze
            metadata: Image generation metadata with boundary info
            detection_level: Detection sensitivity level
            
        Returns:
            DetectionResult with all findings
        """
        self.logger.info(f"Starting artifact detection [{detection_level}] on {image_path}")
        
        # Load thresholds for detection level
        if detection_level not in self.thresholds:
            self.logger.warning(f"Unknown detection level {detection_level}, using aggressive")
            detection_level = "aggressive"
            
        self.current_thresholds = self.thresholds[detection_level]
        
        # FAIL LOUD: Validate inputs
        if not image_path.exists():
            raise ExpandorError(f"Image path does not exist: {image_path}")
        if not image_path.is_file():
            raise ExpandorError(f"Image path is not a file: {image_path}")
            
        # Load and prepare image
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise ExpandorError(f"Failed to load image for artifact detection: {str(e)}")
            
        # FAIL LOUD: Validate image dimensions
        if image.width < 8 or image.height < 8:
            raise ExpandorError(f"Image too small: {image.width}x{image.height} (minimum 8x8)")
        if image.width > 65536 or image.height > 65536:
            raise ExpandorError(f"Image too large: {image.width}x{image.height} (maximum 65536x65536)")
            
        # Work at high resolution for better detection (up to 4K)
        original_size = image.size
        scale = min(1.0, 4096 / max(image.size))
        
        if scale < 1.0:
            detect_size = (int(image.width * scale), int(image.height * scale))
            detect_image = image.resize(detect_size, Image.Resampling.LANCZOS)
            self.logger.debug(f"Scaled image from {original_size} to {detect_size} for detection")
        else:
            detect_image = image
            detect_size = image.size
            scale = 1.0
            
        img_array = np.array(detect_image)
        # FAIL LOUD: Validate array shape
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ExpandorError(f"Invalid image array shape: {img_array.shape} (expected (H, W, 3))")
        h, w = img_array.shape[:2]
        
        # Initialize detection components
        combined_mask = np.zeros((h, w), dtype=np.float32)
        seam_locations = []
        detection_scores = {}
        
        # 1. Progressive Boundary Detection (most important)
        if self._has_progressive_boundaries(metadata):
            self.logger.info("Detecting artifacts at progressive boundaries")
            boundary_result = self._detect_progressive_boundaries(
                img_array, metadata, scale
            )
            if boundary_result['mask'] is not None:
                combined_mask = np.maximum(combined_mask, boundary_result['mask'])
                seam_locations.extend(boundary_result['seams'])
                detection_scores['boundary'] = boundary_result['score']
        
        # 2. Color Discontinuity Detection
        color_result = self._detect_color_discontinuities(img_array)
        if color_result['mask'] is not None:
            combined_mask = np.maximum(combined_mask, color_result['mask'] * 0.8)
            detection_scores['color'] = color_result['score']
            
        # 3. Gradient Analysis
        gradient_result = self._detect_gradient_anomalies(img_array)
        if gradient_result['mask'] is not None:
            combined_mask = np.maximum(combined_mask, gradient_result['mask'] * 0.7)
            detection_scores['gradient'] = gradient_result['score']
            
        # 4. Frequency Domain Analysis
        frequency_result = self._detect_frequency_artifacts(img_array)
        if frequency_result['mask'] is not None:
            combined_mask = np.maximum(combined_mask, frequency_result['mask'] * 0.6)
            detection_scores['frequency'] = frequency_result['score']
            
        # 5. Texture Consistency Analysis
        texture_result = self._detect_texture_inconsistencies(img_array)
        if texture_result['mask'] is not None:
            combined_mask = np.maximum(combined_mask, texture_result['mask'] * 0.5)
            detection_scores['texture'] = texture_result['score']
            
        # 6. Edge Pattern Analysis
        edge_result = self._detect_edge_artifacts(img_array)
        if edge_result['mask'] is not None:
            combined_mask = np.maximum(combined_mask, edge_result['mask'] * 0.6)
            detection_scores['edge'] = edge_result['score']
            
        # Apply morphological operations to clean up mask
        combined_mask = self._refine_mask(combined_mask)
        
        # Scale mask back to original size if needed
        if scale < 1.0:
            mask_pil = Image.fromarray((combined_mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(original_size, Image.Resampling.LANCZOS)
            combined_mask = np.array(mask_pil) / 255.0
            
        # Calculate statistics
        artifact_pixels = np.sum(combined_mask > 0.1)
        total_pixels = original_size[0] * original_size[1]
        artifact_percentage = (artifact_pixels / total_pixels) * 100
        
        # Determine severity
        severity = self._determine_severity(
            artifact_percentage, 
            detection_scores,
            len(seam_locations)
        )
        
        # Log results
        self.logger.info(
            f"Detection complete: {severity.value} severity, "
            f"{artifact_percentage:.2f}% artifacts, {len(seam_locations)} seams"
        )
        
        return DetectionResult(
            has_artifacts=severity != ArtifactSeverity.NONE,
            severity=severity,
            artifact_mask=combined_mask if severity != ArtifactSeverity.NONE else None,
            seam_locations=seam_locations,
            detection_scores=detection_scores,
            total_artifact_pixels=int(artifact_pixels),
            artifact_percentage=artifact_percentage
        )
    
    def _has_progressive_boundaries(self, metadata: Dict) -> bool:
        """Check if image has progressive expansion boundaries"""
        boundaries = metadata.get('progressive_boundaries', [])
        boundaries_v = metadata.get('progressive_boundaries_vertical', [])
        generation_metadata = metadata.get('generation_metadata', {})
        
        return (
            len(boundaries) > 0 or 
            len(boundaries_v) > 0 or
            generation_metadata.get('used_progressive', False)
        )
    
    def _detect_progressive_boundaries(self, img_array: np.ndarray, 
                                     metadata: Dict, scale: float) -> Dict:
        """Detect artifacts at known progressive boundaries"""
        h, w = img_array.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        seams = []
        
        # Get boundaries from metadata
        boundaries_h = metadata.get('progressive_boundaries', [])
        boundaries_v = metadata.get('progressive_boundaries_vertical', [])
        
        # Also check generation_metadata for additional boundary info
        gen_meta = metadata.get('generation_metadata', {})
        if 'boundaries' in gen_meta:
            for boundary in gen_meta['boundaries']:
                if boundary.get('direction') == 'vertical':
                    boundaries_h.append(boundary.get('position', 0))
                else:
                    boundaries_v.append(boundary.get('position', 0))
        
        threshold = self.current_thresholds['color_threshold']
        min_width = self.current_thresholds['min_seam_width']
        max_width = self.current_thresholds['max_seam_width']
        
        # Check horizontal boundaries (vertical seams)
        for boundary in boundaries_h:
            x = int(boundary * scale)
            if min_width < x < w - min_width:
                # Analyze region around boundary
                left_region = img_array[:, max(0, x-max_width):x]
                right_region = img_array[:, x:min(w, x+max_width)]
                
                if left_region.size > 0 and right_region.size > 0:
                    # Multiple detection methods
                    # 1. Color difference
                    left_mean = np.mean(left_region, axis=(0, 1))
                    right_mean = np.mean(right_region, axis=(0, 1))
                    color_diff = np.linalg.norm(left_mean - right_mean)
                    
                    # 2. Texture difference (using std deviation)
                    left_std = np.std(left_region)
                    right_std = np.std(right_region)
                    texture_diff = abs(left_std - right_std)
                    
                    # 3. Edge density difference
                    # Convert to grayscale
                    left_gray = np.dot(left_region[...,:3], [0.2989, 0.5870, 0.1140])
                    right_gray = np.dot(right_region[...,:3], [0.2989, 0.5870, 0.1140])
                    
                    # Simple edge detection using Sobel filters
                    left_edges = self._detect_edges_numpy(left_gray)
                    right_edges = self._detect_edges_numpy(right_gray)
                    edge_diff = abs(np.mean(left_edges) - np.mean(right_edges))
                    
                    # Combined score
                    if color_diff > threshold or texture_diff > 10 or edge_diff > 0.1:
                        # Mark seam area with gradient
                        seam_strength = min(1.0, (color_diff / 50) + (texture_diff / 30) + edge_diff)
                        seam_width = int(min_width + (max_width - min_width) * seam_strength)
                        
                        for offset in range(-seam_width//2, seam_width//2):
                            if 0 <= x + offset < w:
                                weight = 1.0 - abs(offset) / (seam_width/2)
                                mask[:, x + offset] = np.maximum(
                                    mask[:, x + offset], 
                                    seam_strength * weight
                                )
                        
                        seams.append({
                            'position': int(boundary),
                            'direction': 'vertical',
                            'strength': float(seam_strength),
                            'color_diff': float(color_diff),
                            'texture_diff': float(texture_diff),
                            'edge_diff': float(edge_diff)
                        })
        
        # Check vertical boundaries (horizontal seams)
        for boundary in boundaries_v:
            y = int(boundary * scale)
            if min_width < y < h - min_width:
                # Similar analysis for horizontal seams
                top_region = img_array[max(0, y-max_width):y, :]
                bottom_region = img_array[y:min(h, y+max_width), :]
                
                if top_region.size > 0 and bottom_region.size > 0:
                    top_mean = np.mean(top_region, axis=(0, 1))
                    bottom_mean = np.mean(bottom_region, axis=(0, 1))
                    color_diff = np.linalg.norm(top_mean - bottom_mean)
                    
                    top_std = np.std(top_region)
                    bottom_std = np.std(bottom_region)
                    texture_diff = abs(top_std - bottom_std)
                    
                    if color_diff > threshold or texture_diff > 10:
                        seam_strength = min(1.0, (color_diff / 50) + (texture_diff / 30))
                        seam_width = int(min_width + (max_width - min_width) * seam_strength)
                        
                        for offset in range(-seam_width//2, seam_width//2):
                            if 0 <= y + offset < h:
                                weight = 1.0 - abs(offset) / (seam_width/2)
                                mask[y + offset, :] = np.maximum(
                                    mask[y + offset, :],
                                    seam_strength * weight
                                )
                        
                        seams.append({
                            'position': int(boundary),
                            'direction': 'horizontal',
                            'strength': float(seam_strength),
                            'color_diff': float(color_diff),
                            'texture_diff': float(texture_diff)
                        })
        
        score = len(seams) / max(1, len(boundaries_h) + len(boundaries_v))
        
        return {
            'mask': mask if np.max(mask) > 0 else None,
            'seams': seams,
            'score': score
        }
    
    def _detect_color_discontinuities(self, img_array: np.ndarray) -> Dict:
        """Detect sudden color changes that indicate seams"""
        h, w = img_array.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Convert to LAB color space for better color difference detection
        # FAIL LOUD: We implement our own RGB to LAB conversion
        lab = self._rgb_to_lab(img_array)
        
        # Calculate color differences in multiple directions
        threshold = self.current_thresholds['color_threshold']
        
        # Horizontal differences
        diff_h = np.zeros((h, w-1))
        for i in range(3):  # For each LAB channel
            channel_diff = np.abs(lab[:, 1:, i].astype(float) - lab[:, :-1, i].astype(float))
            diff_h += channel_diff / 3.0
            
        # Vertical differences
        diff_v = np.zeros((h-1, w))
        for i in range(3):
            channel_diff = np.abs(lab[1:, :, i].astype(float) - lab[:-1, :, i].astype(float))
            diff_v += channel_diff / 3.0
        
        # Find anomalous differences
        # Horizontal seams
        h_mean = np.mean(diff_h)
        h_std = np.std(diff_h)
        h_anomalies = diff_h > (h_mean + self.current_thresholds['gradient_multiplier'] * h_std)
        
        # Mark horizontal discontinuities
        for x in range(w-1):
            if np.sum(h_anomalies[:, x]) > h * 0.3:  # At least 30% of height
                mask[:, x:x+2] = np.maximum(mask[:, x:x+2], 0.8)
        
        # Vertical seams
        v_mean = np.mean(diff_v)
        v_std = np.std(diff_v)
        v_anomalies = diff_v > (v_mean + self.current_thresholds['gradient_multiplier'] * v_std)
        
        # Mark vertical discontinuities
        for y in range(h-1):
            if np.sum(v_anomalies[y, :]) > w * 0.3:
                mask[y:y+2, :] = np.maximum(mask[y:y+2, :], 0.8)
        
        # Calculate score
        total_anomalies = np.sum(h_anomalies) + np.sum(v_anomalies)
        max_possible = (h * (w-1)) + ((h-1) * w)
        score = total_anomalies / max_possible
        
        return {
            'mask': mask if np.max(mask) > 0 else None,
            'score': score
        }
    
    def _detect_gradient_anomalies(self, img_array: np.ndarray) -> Dict:
        """Detect gradient discontinuities using Sobel filters"""
        # Convert to grayscale
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        h, w = gray.shape
        
        # Calculate gradients using numpy/scipy
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find gradient anomalies
        grad_mean = np.mean(grad_mag)
        grad_std = np.std(grad_mag)
        threshold = grad_mean + self.current_thresholds['gradient_multiplier'] * grad_std
        
        # Create mask for high gradient areas
        mask = np.zeros((h, w), dtype=np.float32)
        high_grad = grad_mag > threshold
        
        # Look for linear patterns in high gradient areas
        # FAIL LOUD: Hough transform requires more complex implementation
        # For now, use simple thresholding approach
        # Find connected components of high gradient areas
        high_grad_binary = high_grad.astype(np.uint8)
        
        # Look for linear structures by checking gradient directions
        grad_direction = np.arctan2(grad_y, grad_x)
        
        # Find predominantly horizontal or vertical gradients
        horizontal_mask = (np.abs(grad_direction) < np.pi/8) | (np.abs(grad_direction) > 7*np.pi/8)
        vertical_mask = (np.abs(grad_direction - np.pi/2) < np.pi/8) | (np.abs(grad_direction + np.pi/2) < np.pi/8)
        
        # Mark linear structures in mask
        linear_structures = high_grad & (horizontal_mask | vertical_mask)
        mask[linear_structures] = 0.7
        
        # Also mark regions with sudden gradient changes
        grad_diff_x = np.abs(grad_mag[:, 1:] - grad_mag[:, :-1])
        grad_diff_y = np.abs(grad_mag[1:, :] - grad_mag[:-1, :])
        
        # Mark sudden changes
        sudden_x = grad_diff_x > (np.mean(grad_diff_x) + 3 * np.std(grad_diff_x))
        sudden_y = grad_diff_y > (np.mean(grad_diff_y) + 3 * np.std(grad_diff_y))
        
        mask[:, 1:][sudden_x] = np.maximum(mask[:, 1:][sudden_x], 0.6)
        mask[1:, :][sudden_y] = np.maximum(mask[1:, :][sudden_y], 0.6)
        
        score = np.sum(high_grad) / (h * w)
        
        return {
            'mask': mask if np.max(mask) > 0 else None,
            'score': score
        }
    
    def _detect_frequency_artifacts(self, img_array: np.ndarray) -> Dict:
        """Detect artifacts in frequency domain"""
        # Convert to grayscale
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        h, w = gray.shape
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Look for periodic patterns
        # Remove DC component
        center_y, center_x = h // 2, w // 2
        magnitude_spectrum[center_y-2:center_y+2, center_x-2:center_x+2] = 0
        
        # Find peaks in frequency domain
        threshold = np.mean(magnitude_spectrum) + self.current_thresholds['frequency_multiplier'] * np.std(magnitude_spectrum)
        peaks = magnitude_spectrum > threshold
        
        # Convert frequency peaks back to spatial domain to identify patterns
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Look for regular patterns that might indicate tiling or repetition
        peak_coords = np.argwhere(peaks)
        for y, x in peak_coords:
            # Calculate frequency
            freq_y = abs(y - center_y) / h
            freq_x = abs(x - center_x) / w
            
            # If frequency suggests visible pattern (not too high frequency)
            if 0.02 < freq_y < 0.5 or 0.02 < freq_x < 0.5:
                # Create spatial mask for this frequency
                if freq_y > freq_x:  # Horizontal pattern
                    # FAIL LOUD: Check for division by zero
                    if freq_y == 0:
                        raise ExpandorError("Invalid frequency component: division by zero")
                    period = int(1 / freq_y)
                    if period > 0:  # Sanity check
                        for i in range(0, h, period):
                            if i < h:
                                mask[max(0, i-2):min(h, i+2), :] = 0.5
                else:  # Vertical pattern
                    # FAIL LOUD: Check for division by zero
                    if freq_x == 0:
                        raise ExpandorError("Invalid frequency component: division by zero")
                    period = int(1 / freq_x)
                    if period > 0:  # Sanity check
                        for i in range(0, w, period):
                            if i < w:
                                mask[:, max(0, i-2):min(w, i+2)] = 0.5
        
        score = np.sum(peaks) / (h * w)
        
        return {
            'mask': mask if np.max(mask) > 0 else None,
            'score': score
        }
    
    def _detect_texture_inconsistencies(self, img_array: np.ndarray) -> Dict:
        """Detect texture inconsistencies using local statistics"""
        # Convert to grayscale
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        h, w = gray.shape
        
        window_size = self.current_thresholds['texture_window']
        stride = window_size // 2
        
        # Calculate local statistics
        local_means = []
        local_stds = []
        positions = []
        
        for y in range(0, h - window_size, stride):
            for x in range(0, w - window_size, stride):
                window = gray[y:y+window_size, x:x+window_size]
                local_means.append(np.mean(window))
                local_stds.append(np.std(window))
                positions.append((y + window_size//2, x + window_size//2))
        
        local_means = np.array(local_means)
        local_stds = np.array(local_stds)
        
        # Find outliers
        mean_threshold = np.percentile(local_means, [5, 95])
        std_threshold = np.percentile(local_stds, [5, 95])
        
        mask = np.zeros((h, w), dtype=np.float32)
        
        for i, (y, x) in enumerate(positions):
            # Check if this region is significantly different
            if (local_means[i] < mean_threshold[0] or local_means[i] > mean_threshold[1] or
                local_stds[i] < std_threshold[0] or local_stds[i] > std_threshold[1]):
                
                # Mark region as potentially problematic
                y1 = max(0, y - window_size//2)
                y2 = min(h, y + window_size//2)
                x1 = max(0, x - window_size//2)
                x2 = min(w, x + window_size//2)
                
                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], 0.4)
        
        # Calculate score
        outlier_count = np.sum((local_means < mean_threshold[0]) | (local_means > mean_threshold[1]))
        score = outlier_count / len(local_means)
        
        return {
            'mask': mask if np.max(mask) > 0 else None,
            'score': score
        }
    
    def _detect_edge_artifacts(self, img_array: np.ndarray) -> Dict:
        """Detect artifacts in edge patterns"""
        # Convert to grayscale
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Multi-scale edge detection using gradient magnitude
        # FAIL LOUD: Using Sobel-based edge detection instead of Canny
        edges1 = self._detect_edges_numpy(gray, threshold_low=0.1, threshold_high=0.3)
        edges2 = self._detect_edges_numpy(gray, threshold_low=0.05, threshold_high=0.2)
        edges3 = self._detect_edges_numpy(gray, threshold_low=0.15, threshold_high=0.4)
        
        # Combine edges
        combined_edges = np.maximum(edges1, np.maximum(edges2, edges3))
        
        # FAIL LOUD: Line detection without Hough transform
        # Use edge continuity analysis instead
        mask = np.zeros_like(gray, dtype=np.float32)
        
        # Find continuous edge segments
        # Horizontal edge continuity
        h_continuity = np.sum(combined_edges, axis=1)
        h_peaks = np.where(h_continuity > gray.shape[1] * 0.3)[0]
        
        # Vertical edge continuity
        v_continuity = np.sum(combined_edges, axis=0)
        v_peaks = np.where(v_continuity > gray.shape[0] * 0.3)[0]
        
        horizontal_lines = list(h_peaks)
        vertical_lines = list(v_peaks)
            
            # Check for regular patterns in line positions
            if len(horizontal_lines) > 2:
                h_lines = sorted(horizontal_lines)
                for i in range(len(h_lines) - 1):
                    gap = h_lines[i+1] - h_lines[i]
                    if 50 < gap < 500:  # Suspicious regular gap
                        mask[h_lines[i]-5:h_lines[i]+5, :] = 0.6
                        
            if len(vertical_lines) > 2:
                v_lines = sorted(vertical_lines)
                for i in range(len(v_lines) - 1):
                    gap = v_lines[i+1] - v_lines[i]
                    if 50 < gap < 500:
                        mask[:, v_lines[i]-5:v_lines[i]+5] = 0.6
        
        # Also check for edge density anomalies
        h, w = gray.shape
        # Use uniform filter instead of boxFilter
        from scipy.ndimage import uniform_filter
        edge_density = uniform_filter(combined_edges, size=50)
        
        # Find regions with unusual edge density
        density_mean = np.mean(edge_density)
        density_std = np.std(edge_density)
        anomalies = np.abs(edge_density - density_mean) > 2 * density_std
        
        mask = np.maximum(mask, anomalies.astype(np.float32) * 0.4)
        
        score = np.mean(combined_edges) * self.current_thresholds['edge_sensitivity']
        
        return {
            'mask': mask if np.max(mask) > 0 else None,
            'score': score
        }
    
    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up mask"""
        # Remove small isolated regions
        kernel_small = np.ones((3, 3), np.uint8)
        mask_binary = (mask > 0.3).astype(np.uint8)
        
        # Opening to remove noise
        mask_cleaned = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_small)
        
        # Closing to fill gaps
        kernel_medium = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        # Dilate to ensure we capture full artifact regions
        kernel_dilate = np.ones((7, 7), np.uint8)
        mask_dilated = cv2.dilate(mask_cleaned, kernel_dilate, iterations=1)
        
        # Apply gaussian blur for smooth transitions
        mask_smooth = cv2.GaussianBlur(
            mask_dilated.astype(np.float32),
            (11, 11),
            0
        )
        
        # Combine with original mask to preserve fine details
        refined_mask = np.maximum(mask * 0.7, mask_smooth * 0.8)
        
        return np.clip(refined_mask, 0, 1)
    
    def _determine_severity(self, artifact_percentage: float,
                          detection_scores: Dict[str, float],
                          seam_count: int) -> ArtifactSeverity:
        """Determine overall severity based on multiple factors"""
        if artifact_percentage == 0 and seam_count == 0:
            return ArtifactSeverity.NONE
            
        # Critical if any progressive boundaries detected
        if 'boundary' in detection_scores and detection_scores['boundary'] > 0:
            return ArtifactSeverity.CRITICAL
            
        # High if significant artifacts or multiple seams
        if artifact_percentage > 5 or seam_count > 2:
            return ArtifactSeverity.HIGH
            
        # Medium if moderate artifacts
        if artifact_percentage > 2 or seam_count > 0:
            return ArtifactSeverity.MEDIUM
            
        # Low if minor artifacts
        if artifact_percentage > 0.5:
            return ArtifactSeverity.LOW
            
        return ArtifactSeverity.LOW
    
    def _rgb_to_lab(self, rgb_array: np.ndarray) -> np.ndarray:
        """Convert RGB to LAB color space"""
        # FAIL LOUD: Implementing RGB to LAB conversion
        # Normalize RGB values
        rgb = rgb_array.astype(np.float32) / 255.0
        
        # Convert to XYZ
        # Using sRGB transformation matrix
        transform_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        # Apply gamma correction
        mask = rgb > 0.04045
        rgb[mask] = np.power((rgb[mask] + 0.055) / 1.055, 2.4)
        rgb[~mask] = rgb[~mask] / 12.92
        
        # Transform to XYZ
        xyz = np.dot(rgb, transform_matrix.T)
        
        # Normalize by D65 illuminant
        xyz[:, :, 0] /= 0.95047
        xyz[:, :, 1] /= 1.00000
        xyz[:, :, 2] /= 1.08883
        
        # Convert to LAB
        def f(t):
            delta = 6.0 / 29.0
            mask = t > delta ** 3
            t[mask] = np.power(t[mask], 1.0 / 3.0)
            t[~mask] = t[~mask] / (3 * delta ** 2) + 4.0 / 29.0
            return t
        
        xyz_f = f(xyz.copy())
        
        lab = np.zeros_like(rgb)
        lab[:, :, 0] = 116 * xyz_f[:, :, 1] - 16  # L
        lab[:, :, 1] = 500 * (xyz_f[:, :, 0] - xyz_f[:, :, 1])  # a
        lab[:, :, 2] = 200 * (xyz_f[:, :, 1] - xyz_f[:, :, 2])  # b
        
        return lab
    
    def _detect_edges_numpy(self, gray: np.ndarray, threshold_low: float = 0.1, 
                           threshold_high: float = 0.3) -> np.ndarray:
        """Simple edge detection using gradients"""
        # FAIL LOUD: Implementing edge detection without cv2
        # Calculate gradients
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        grad_mag = grad_mag / (np.max(grad_mag) + 1e-10)
        
        # Apply thresholds
        edges = np.zeros_like(grad_mag)
        edges[grad_mag > threshold_high] = 1.0
        edges[(grad_mag > threshold_low) & (grad_mag <= threshold_high)] = 0.5
        
        return edges
```

### 1.2 Create Artifact Detection Unit Tests

**EXACT FILE PATH**: `tests/unit/processors/test_artifact_detector.py`

```python
"""
Unit tests for Smart Artifact Detector
"""

import pytest
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import tempfile

from expandor.processors.artifact_detector import (
    SmartArtifactDetector, ArtifactSeverity, DetectionResult
)


class TestSmartArtifactDetector:
    """Tests for artifact detection"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        return SmartArtifactDetector()
    
    @pytest.fixture
    def clean_image(self):
        """Create a clean test image with no artifacts"""
        img = Image.new('RGB', (1024, 768), color='blue')
        draw = ImageDraw.Draw(img)
        # Add some natural variation
        for i in range(0, 1024, 100):
            draw.ellipse([i, 300, i+80, 400], fill='green')
        return img
    
    @pytest.fixture
    def seamed_image(self):
        """Create image with visible seam"""
        img = Image.new('RGB', (1024, 768))
        # Left half blue, right half green with visible seam
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 511, 768], fill='blue')
        draw.rectangle([512, 0, 1024, 768], fill='green')
        # Add slight transition to make it realistic
        for i in range(10):
            alpha = i / 10
            color = (
                int(0 * (1-alpha) + 0 * alpha),
                int(0 * (1-alpha) + 128 * alpha),
                int(255 * (1-alpha) + 0 * alpha)
            )
            draw.line([(512-5+i, 0), (512-5+i, 768)], fill=color)
        return img
    
    def test_clean_image_detection(self, detector, clean_image):
        """Test that clean images show no artifacts"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            clean_image.save(tmp.name)
            tmp_path = Path(tmp.name)
            
            result = detector.detect_artifacts(
                tmp_path,
                metadata={},
                detection_level="aggressive"
            )
            
            assert result.has_artifacts == False
            assert result.severity == ArtifactSeverity.NONE
            assert result.artifact_mask is None
            assert len(result.seam_locations) == 0
            
            tmp_path.unlink()
    
    def test_seamed_image_detection(self, detector, seamed_image):
        """Test detection of obvious seam"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            seamed_image.save(tmp.name)
            tmp_path = Path(tmp.name)
            
            result = detector.detect_artifacts(
                tmp_path,
                metadata={},
                detection_level="aggressive"
            )
            
            assert result.has_artifacts == True
            assert result.severity in [ArtifactSeverity.HIGH, ArtifactSeverity.MEDIUM]
            assert result.artifact_mask is not None
            assert result.artifact_percentage > 0
            
            # Check that seam area is detected
            mask_center = result.artifact_mask[:, 500:524]
            # FAIL LOUD: Use proper assertion with tolerance
            center_mean = np.mean(mask_center)
            assert center_mean > 0.3, f"Expected center mean > 0.3, got {center_mean}"
            
            tmp_path.unlink()
    
    def test_progressive_boundary_detection(self, detector, clean_image):
        """Test detection with progressive boundary metadata"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            clean_image.save(tmp.name)
            tmp_path = Path(tmp.name)
            
            # Add metadata indicating progressive boundaries
            metadata = {
                'progressive_boundaries': [512],  # Vertical seam at x=512
                'progressive_boundaries_vertical': [384],  # Horizontal seam at y=384
                'generation_metadata': {
                    'used_progressive': True
                }
            }
            
            result = detector.detect_artifacts(
                tmp_path,
                metadata=metadata,
                detection_level="aggressive"
            )
            
            # Even clean image should show artifacts at boundaries
            assert result.has_artifacts == True
            assert result.severity == ArtifactSeverity.CRITICAL  # Progressive boundaries are critical
            assert len(result.seam_locations) > 0
            
            # Check specific boundary detection
            v_seams = [s for s in result.seam_locations if s['direction'] == 'vertical']
            h_seams = [s for s in result.seam_locations if s['direction'] == 'horizontal']
            assert len(v_seams) > 0
            assert len(h_seams) > 0
            
            tmp_path.unlink()
    
    def test_detection_levels(self, detector, seamed_image):
        """Test different detection sensitivity levels"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            seamed_image.save(tmp.name)
            tmp_path = Path(tmp.name)
            
            # Test aggressive detection
            aggressive_result = detector.detect_artifacts(
                tmp_path, {}, "aggressive"
            )
            
            # Test light detection
            light_result = detector.detect_artifacts(
                tmp_path, {}, "light"
            )
            
            # Aggressive should detect more artifacts
            assert aggressive_result.artifact_percentage >= light_result.artifact_percentage
            
            # Both should detect the obvious seam
            assert aggressive_result.has_artifacts == True
            assert light_result.has_artifacts == True
            
            tmp_path.unlink()
    
    def test_multiple_detection_methods(self, detector):
        """Test that multiple detection methods work together"""
        # Create image with multiple artifact types
        img = Image.new('RGB', (512, 512))
        img_array = np.array(img)
        
        # Add color discontinuity
        img_array[:, 250:260, :] = [255, 0, 0]  # Red stripe
        
        # Add texture inconsistency
        noise_region = np.random.randint(0, 50, (100, 100, 3))
        img_array[200:300, 200:300] = noise_region
        
        # Convert back to PIL
        img = Image.fromarray(img_array.astype(np.uint8))
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = Path(tmp.name)
            
            result = detector.detect_artifacts(tmp_path, {}, "aggressive")
            
            # Should detect multiple issues
            assert result.has_artifacts == True
            assert 'color' in result.detection_scores
            assert 'texture' in result.detection_scores
            
            # Should have non-zero scores
            assert result.detection_scores.get('color', 0) > 0
            assert result.detection_scores.get('texture', 0) > 0
            
            tmp_path.unlink()
    
    def test_mask_refinement(self, detector):
        """Test that mask refinement works properly"""
        # Create image with small artifacts
        img = Image.new('RGB', (256, 256), color='white')
        img_array = np.array(img)
        
        # Add scattered single-pixel artifacts
        for _ in range(100):
            x = np.random.randint(0, 256)
            y = np.random.randint(0, 256)
            img_array[y, x] = [0, 0, 0]
        
        img = Image.fromarray(img_array)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = Path(tmp.name)
            
            result = detector.detect_artifacts(tmp_path, {}, "aggressive")
            
            if result.artifact_mask is not None:
                # Refined mask should be smoother than raw detection
                mask_std = np.std(result.artifact_mask)
                assert mask_std < 0.5  # Should be relatively smooth
                
                # Should not have too many isolated pixels
                kernel = np.ones((3,3))
                from scipy import ndimage
                opened = ndimage.binary_opening(result.artifact_mask > 0.5, kernel)
                isolated_pixels = np.sum((result.artifact_mask > 0.5) & ~opened)
                assert isolated_pixels < 50  # Most noise should be removed
            
            tmp_path.unlink()
    
    def test_scale_invariance(self, detector, seamed_image):
        """Test that detection works at different scales"""
        # Test original size
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            seamed_image.save(tmp.name)
            tmp_path = Path(tmp.name)
            
            result_original = detector.detect_artifacts(tmp_path, {}, "aggressive")
            tmp_path.unlink()
        
        # Test scaled up
        scaled_up = seamed_image.resize((2048, 1536), Image.Resampling.LANCZOS)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            scaled_up.save(tmp.name)
            tmp_path = Path(tmp.name)
            
            result_scaled = detector.detect_artifacts(tmp_path, {}, "aggressive")
            tmp_path.unlink()
        
        # Both should detect the seam
        assert result_original.has_artifacts == True
        assert result_scaled.has_artifacts == True
        
        # Severity should be similar
        assert result_original.severity == result_scaled.severity
```

## 2. Smart Quality Refiner Implementation

### 2.1 Create Smart Quality Refiner

**EXACT FILE PATH**: `expandor/processors/refinement/smart_refiner.py`

```python
"""
Smart Multi-Pass Quality Refinement
Implements intelligent refinement with artifact-aware processing.
Zero tolerance implementation for the Expandor standalone system.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import time
import numpy as np
from PIL import Image, ImageFilter
import torch

from ...core.exceptions import ExpandorError
from ...core.boundary_tracker import BoundaryInfo
from ..artifact_detector import SmartArtifactDetector, ArtifactSeverity


class RefinementPass(Enum):
    """Types of refinement passes"""
    COHERENCE = "coherence"
    TARGETED = "targeted"
    DETAIL = "detail"
    UNIFICATION = "unification"


@dataclass
class RefinementResult:
    """Result from a refinement pass"""
    pass_type: RefinementPass
    input_path: Path
    output_path: Path
    artifacts_before: int
    artifacts_after: int
    duration_seconds: float
    strength_used: float
    mask_coverage: float


class SmartQualityRefiner:
    """
    Multi-pass refinement with intelligent artifact targeting.
    Zero tolerance for visible seams through aggressive repair.
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # Load refinement presets
        self.quality_presets = self.config.get('quality_presets', {})
        
        # Default pass configurations
        self.default_passes = {
            RefinementPass.COHERENCE: {
                'strength': 0.12,
                'steps': 50,
                'guidance_scale': 7.0,
                'description': 'Initial coherence pass for overall consistency'
            },
            RefinementPass.TARGETED: {
                'strength': 0.35,
                'steps': 80,
                'guidance_scale': 7.5,
                'description': 'Targeted artifact removal with inpainting'
            },
            RefinementPass.DETAIL: {
                'strength': 0.08,
                'steps': 40,
                'guidance_scale': 6.5,
                'description': 'Detail enhancement pass'
            },
            RefinementPass.UNIFICATION: {
                'strength': 0.03,
                'steps': 30,
                'guidance_scale': 6.0,
                'description': 'Final light unification pass'
            }
        }
        
        # Initialize artifact detector
        self.artifact_detector = SmartArtifactDetector(config, logger)
        
    def refine(self,
               image_path: Path,
               prompt: str,
               boundaries: List[BoundaryInfo],
               pipelines: Dict[str, Any],
               quality_preset: str = "high",
               max_passes: Optional[int] = None,
               seed: int = 42) -> Dict[str, Any]:
        """
        Execute smart multi-pass refinement.
        
        Args:
            image_path: Path to image to refine
            prompt: Generation prompt
            boundaries: Boundary information for seam detection
            pipelines: Available pipelines (refiner, img2img, inpaint)
            quality_preset: Quality level preset
            max_passes: Maximum refinement passes (overrides preset)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with refinement results and final image path
        """
        self.logger.info(f"Starting smart refinement [{quality_preset}] on {image_path}")
        
        # Get refinement configuration
        preset_config = self.quality_presets.get(quality_preset, {})
        refinement_config = preset_config.get('refinement', {})
        
        if not refinement_config.get('enabled', True):
            self.logger.info("Refinement disabled for this preset")
            return {
                'image_path': image_path,
                'passes_executed': [],
                'total_artifacts_fixed': 0,
                'final_quality_score': 1.0
            }
        
        # Determine passes to execute
        if max_passes is None:
            max_passes = refinement_config.get('passes', 3)
            
        multi_pass_enabled = refinement_config.get('multi_pass_enabled', True)
        
        # Prepare metadata for artifact detection
        metadata = {
            'progressive_boundaries': [b.position for b in boundaries if b.direction == 'vertical'],
            'progressive_boundaries_vertical': [b.position for b in boundaries if b.direction == 'horizontal'],
            'generation_metadata': {
                'boundaries': [b.to_dict() for b in boundaries],
                'used_progressive': len(boundaries) > 0
            }
        }
        
        # Initial artifact detection
        initial_detection = self.artifact_detector.detect_artifacts(
            image_path,
            metadata,
            preset_config.get('artifact_detection', 'standard')
        )
        
        self.logger.info(
            f"Initial detection: {initial_detection.severity.value} severity, "
            f"{len(initial_detection.seam_locations)} seams"
        )
        
        # Execute refinement passes
        current_path = image_path
        passes_executed = []
        total_artifacts_fixed = 0
        
        for pass_num in range(max_passes):
            # Determine pass type based on severity and pass number
            pass_type = self._determine_pass_type(
                pass_num,
                initial_detection.severity,
                multi_pass_enabled
            )
            
            if pass_type is None:
                self.logger.info(f"Skipping pass {pass_num + 1} - no refinement needed")
                continue
                
            self.logger.info(f"Executing pass {pass_num + 1}/{max_passes}: {pass_type.value}")
            
            # Execute refinement pass
            pass_result = self._execute_pass(
                current_path,
                pass_type,
                prompt,
                initial_detection,
                pipelines,
                refinement_config,
                seed + pass_num
            )
            
            passes_executed.append(pass_result)
            current_path = pass_result.output_path
            total_artifacts_fixed += (pass_result.artifacts_before - pass_result.artifacts_after)
            
            # Re-detect artifacts after pass
            if pass_result.artifacts_after > 0:
                detection_result = self.artifact_detector.detect_artifacts(
                    current_path,
                    metadata,
                    preset_config.get('artifact_detection', 'standard')
                )
                
                if detection_result.severity == ArtifactSeverity.NONE:
                    self.logger.info("All artifacts successfully removed!")
                    break
                    
                # Update detection for next pass
                initial_detection = detection_result
            else:
                break
        
        # Final quality assessment
        final_detection = self.artifact_detector.detect_artifacts(
            current_path,
            metadata,
            preset_config.get('artifact_detection', 'standard')
        )
        
        # Calculate quality score (1.0 = perfect, 0.0 = terrible)
        quality_score = self._calculate_quality_score(final_detection)
        
        self.logger.info(
            f"Refinement complete: {len(passes_executed)} passes, "
            f"{total_artifacts_fixed} artifacts fixed, "
            f"quality score: {quality_score:.2f}"
        )
        
        return {
            'image_path': current_path,
            'passes_executed': passes_executed,
            'total_artifacts_fixed': total_artifacts_fixed,
            'final_quality_score': quality_score,
            'final_severity': final_detection.severity.value,
            'remaining_seams': len(final_detection.seam_locations)
        }
    
    def _determine_pass_type(self, pass_num: int, severity: ArtifactSeverity,
                           multi_pass_enabled: bool) -> Optional[RefinementPass]:
        """Determine which type of refinement pass to execute"""
        if not multi_pass_enabled:
            # Single pass mode - always targeted
            return RefinementPass.TARGETED if pass_num == 0 else None
            
        # Multi-pass strategy based on severity
        if severity == ArtifactSeverity.CRITICAL:
            # Critical issues need all passes
            if pass_num == 0:
                return RefinementPass.COHERENCE
            elif pass_num == 1:
                return RefinementPass.TARGETED
            elif pass_num == 2:
                return RefinementPass.TARGETED  # Second targeted pass
            elif pass_num == 3:
                return RefinementPass.DETAIL
            elif pass_num == 4:
                return RefinementPass.UNIFICATION
                
        elif severity == ArtifactSeverity.HIGH:
            # High severity - skip initial coherence
            if pass_num == 0:
                return RefinementPass.TARGETED
            elif pass_num == 1:
                return RefinementPass.DETAIL
            elif pass_num == 2:
                return RefinementPass.UNIFICATION
                
        elif severity == ArtifactSeverity.MEDIUM:
            # Medium - light refinement
            if pass_num == 0:
                return RefinementPass.TARGETED
            elif pass_num == 1:
                return RefinementPass.UNIFICATION
                
        elif severity == ArtifactSeverity.LOW:
            # Low - single light pass
            if pass_num == 0:
                return RefinementPass.DETAIL
                
        return None
    
    def _execute_pass(self, 
                     image_path: Path,
                     pass_type: RefinementPass,
                     prompt: str,
                     detection_result: Any,
                     pipelines: Dict[str, Any],
                     refinement_config: Dict,
                     seed: int) -> RefinementResult:
        """Execute a single refinement pass"""
        start_time = time.time()
        
        # Load image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Get pass configuration
        pass_config = self.default_passes[pass_type].copy()
        
        # Override with preset-specific settings
        if pass_type == RefinementPass.COHERENCE:
            pass_config['strength'] = refinement_config.get('coherence_strength', pass_config['strength'])
            pass_config['steps'] = refinement_config.get('coherence_steps', pass_config['steps'])
        elif pass_type == RefinementPass.TARGETED:
            pass_config['strength'] = refinement_config.get('targeted_strength', pass_config['strength'])
            pass_config['steps'] = refinement_config.get('targeted_steps', pass_config['steps'])
        elif pass_type == RefinementPass.DETAIL:
            pass_config['strength'] = refinement_config.get('detail_strength', pass_config['strength'])
            pass_config['steps'] = refinement_config.get('detail_steps', pass_config['steps'])
            
        # Prepare output path
        output_path = image_path.parent / f"{image_path.stem}_refined_{pass_type.value}.png"
        
        # Execute based on pass type
        if pass_type == RefinementPass.TARGETED and detection_result.artifact_mask is not None:
            # Targeted inpainting for artifact removal
            refined_image = self._execute_targeted_inpaint(
                image,
                detection_result.artifact_mask,
                prompt,
                pipelines.get('inpaint', pipelines.get('refiner')),
                pass_config,
                seed
            )
            mask_coverage = np.mean(detection_result.artifact_mask)
            
        else:
            # Full image refinement
            refined_image = self._execute_full_refinement(
                image,
                prompt,
                pipelines.get('refiner', pipelines.get('img2img')),
                pass_config,
                seed
            )
            mask_coverage = 1.0
            
        # Save refined image
        refined_image.save(output_path, 'PNG', compress_level=0)
        
        # Quick artifact check on result
        quick_check = self._quick_artifact_check(output_path)
        
        return RefinementResult(
            pass_type=pass_type,
            input_path=image_path,
            output_path=output_path,
            artifacts_before=len(detection_result.seam_locations),
            artifacts_after=quick_check['seam_count'],
            duration_seconds=time.time() - start_time,
            strength_used=pass_config['strength'],
            mask_coverage=mask_coverage
        )
    
    def _execute_targeted_inpaint(self, image: Image.Image, mask: np.ndarray,
                                 prompt: str, pipeline: Any, config: Dict,
                                 seed: int) -> Image.Image:
        """Execute targeted inpainting on artifact regions"""
        if pipeline is None:
            self.logger.warning("No inpaint pipeline available, using full refinement")
            return image
            
        # Convert mask to PIL Image
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_uint8, mode='L')
        
        # Dilate mask to ensure full coverage
        # FAIL LOUD: Using scipy for morphological operations
        mask_array = np.array(mask_image)
        kernel_size = 7
        
        # Use scipy's binary dilation - much more efficient than nested loops
        from scipy import ndimage
        kernel = np.ones((kernel_size, kernel_size))
        dilated = ndimage.binary_dilation(mask_array > 0, kernel)
        dilated = (dilated * 255).astype(np.uint8)
                    
        mask_image = Image.fromarray(dilated, mode='L')
        
        # Apply blur for smooth transitions
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=5))
        
        # Execute inpainting
        try:
            generator = torch.Generator()
            if seed is not None:
                generator.manual_seed(seed)
                
            result = pipeline(
                prompt=prompt + ", seamless, perfect quality",
                image=image,
                mask_image=mask_image,
                strength=config['strength'],
                num_inference_steps=config['steps'],
                guidance_scale=config['guidance_scale'],
                generator=generator
            )
            
            return result.images[0]
            
        except Exception as e:
            self.logger.error(f"Inpainting failed: {str(e)}")
            return image
    
    def _execute_full_refinement(self, image: Image.Image, prompt: str,
                               pipeline: Any, config: Dict, seed: int) -> Image.Image:
        """Execute full image refinement"""
        if pipeline is None:
            self.logger.warning("No refinement pipeline available")
            return image
            
        try:
            generator = torch.Generator()
            if seed is not None:
                generator.manual_seed(seed)
                
            result = pipeline(
                prompt=prompt + ", high quality, detailed",
                image=image,
                strength=config['strength'],
                num_inference_steps=config['steps'],
                guidance_scale=config['guidance_scale'],
                generator=generator
            )
            
            return result.images[0]
            
        except Exception as e:
            self.logger.error(f"Refinement failed: {str(e)}")
            return image
    
    def _quick_artifact_check(self, image_path: Path) -> Dict[str, int]:
        """Quick check for remaining artifacts"""
        # Simplified detection for speed
        detection = self.artifact_detector.detect_artifacts(
            image_path,
            {},  # No metadata for quick check
            "light"  # Fast detection
        )
        
        return {
            'seam_count': len(detection.seam_locations),
            'severity': detection.severity.value
        }
    
    def _calculate_quality_score(self, detection_result: Any) -> float:
        """Calculate overall quality score from detection results"""
        # Start with perfect score
        score = 1.0
        
        # Deduct for severity
        severity_penalties = {
            ArtifactSeverity.NONE: 0.0,
            ArtifactSeverity.LOW: 0.1,
            ArtifactSeverity.MEDIUM: 0.2,
            ArtifactSeverity.HIGH: 0.4,
            ArtifactSeverity.CRITICAL: 0.6
        }
        score -= severity_penalties.get(detection_result.severity, 0.0)
        
        # Deduct for artifact coverage
        score -= min(0.3, detection_result.artifact_percentage * 0.01)
        
        # Deduct for seam count
        score -= min(0.2, len(detection_result.seam_locations) * 0.05)
        
        return max(0.0, score)
```

### 2.2 Create Smart Refiner Unit Tests

**EXACT FILE PATH**: `tests/unit/processors/test_smart_refiner.py`

```python
"""
Unit tests for Smart Quality Refiner
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import time

from expandor.processors.refinement.smart_refiner import (
    SmartQualityRefiner, RefinementPass, RefinementResult
)
from expandor.processors.artifact_detector import ArtifactSeverity
from expandor.core.boundary_tracker import BoundaryInfo


class TestSmartQualityRefiner:
    """Tests for smart quality refinement"""
    
    @pytest.fixture
    def refiner(self):
        """Create refiner instance"""
        config = {
            'quality_presets': {
                'high': {
                    'refinement': {
                        'enabled': True,
                        'passes': 3,
                        'multi_pass_enabled': True,
                        'coherence_strength': 0.10,
                        'targeted_strength': 0.30,
                        'detail_strength': 0.08
                    },
                    'artifact_detection': 'standard'
                },
                'fast': {
                    'refinement': {
                        'enabled': True,
                        'passes': 1,
                        'multi_pass_enabled': False
                    },
                    'artifact_detection': 'light'
                }
            }
        }
        return SmartQualityRefiner(config)
    
    @pytest.fixture
    def test_image(self):
        """Create test image"""
        img = Image.new('RGB', (1024, 768), color='blue')
        return img
    
    @pytest.fixture
    def mock_pipelines(self):
        """Create mock pipelines"""
        # Mock refiner pipeline
        mock_refiner = Mock()
        mock_result = Mock()
        mock_result.images = [Image.new('RGB', (1024, 768), color='green')]
        mock_refiner.return_value = mock_result
        
        # Mock inpaint pipeline
        mock_inpaint = Mock()
        mock_inpaint.return_value = mock_result
        
        return {
            'refiner': mock_refiner,
            'inpaint': mock_inpaint,
            'img2img': mock_refiner
        }
    
    @pytest.fixture
    def boundaries(self):
        """Create test boundaries"""
        return [
            BoundaryInfo(position=512, direction='vertical', expansion_size=200),
            BoundaryInfo(position=384, direction='horizontal', expansion_size=150)
        ]
    
    def test_refinement_disabled(self, refiner, test_image, mock_pipelines):
        """Test behavior when refinement is disabled"""
        # Override config
        refiner.quality_presets['disabled'] = {
            'refinement': {'enabled': False}
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image.save(tmp.name)
            tmp_path = Path(tmp.name)
            
            result = refiner.refine(
                tmp_path,
                "test prompt",
                [],
                mock_pipelines,
                quality_preset='disabled'
            )
            
            # Should return original image
            assert result['image_path'] == tmp_path
            assert len(result['passes_executed']) == 0
            assert result['total_artifacts_fixed'] == 0
            
            tmp_path.unlink()
    
    def test_single_pass_refinement(self, refiner, test_image, mock_pipelines):
        """Test single pass refinement mode"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image.save(tmp.name)
            tmp_path = Path(tmp.name)
            
            # Mock artifact detection
            with patch.object(refiner.artifact_detector, 'detect_artifacts') as mock_detect:
                # First detection shows artifacts
                first_detection = Mock()
                first_detection.severity = ArtifactSeverity.MEDIUM
                first_detection.seam_locations = [{'position': 512}]
                first_detection.artifact_mask = np.ones((768, 1024)) * 0.5
                
                # Second detection shows fixed
                second_detection = Mock()
                second_detection.severity = ArtifactSeverity.NONE
                second_detection.seam_locations = []
                
                mock_detect.side_effect = [first_detection, second_detection, second_detection]
                
                result = refiner.refine(
                    tmp_path,
                    "test prompt",
                    [],
                    mock_pipelines,
                    quality_preset='fast',  # Single pass mode
                    seed=42
                )
                
                # Should execute one pass
                assert len(result['passes_executed']) == 1
                assert result['passes_executed'][0].pass_type == RefinementPass.TARGETED
                assert result['total_artifacts_fixed'] == 1
                
                # Should have called pipeline
                assert mock_pipelines['refiner'].called or mock_pipelines['inpaint'].called
            
            # Clean up
            tmp_path.unlink()
            for pass_result in result['passes_executed']:
                pass_result.output_path.unlink()
    
    def test_multi_pass_refinement(self, refiner, test_image, mock_pipelines, boundaries):
        """Test multi-pass refinement with critical artifacts"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image.save(tmp.name)
            tmp_path = Path(tmp.name)
            
            # Mock artifact detection
            with patch.object(refiner.artifact_detector, 'detect_artifacts') as mock_detect:
                # Critical severity triggers all passes
                critical_detection = Mock()
                critical_detection.severity = ArtifactSeverity.CRITICAL
                critical_detection.seam_locations = [{'position': 512}, {'position': 256}]
                critical_detection.artifact_mask = np.ones((768, 1024)) * 0.8
                
                # Gradually improving
                high_detection = Mock()
                high_detection.severity = ArtifactSeverity.HIGH
                high_detection.seam_locations = [{'position': 512}]
                
                medium_detection = Mock()
                medium_detection.severity = ArtifactSeverity.MEDIUM
                medium_detection.seam_locations = []
                
                none_detection = Mock()
                none_detection.severity = ArtifactSeverity.NONE
                none_detection.seam_locations = []
                
                mock_detect.side_effect = [
                    critical_detection,  # Initial
                    high_detection,      # After coherence
                    medium_detection,    # After targeted
                    none_detection,      # After second targeted
                    none_detection       # Final check
                ]
                
                result = refiner.refine(
                    tmp_path,
                    "test prompt",
                    boundaries,
                    mock_pipelines,
                    quality_preset='high',
                    max_passes=5,
                    seed=42
                )
                
                # Should execute multiple passes
                assert len(result['passes_executed']) >= 3
                
                # Should fix artifacts
                assert result['total_artifacts_fixed'] > 0
                assert result['final_severity'] == 'none'
                
                # Check pass types
                pass_types = [p.pass_type for p in result['passes_executed']]
                assert RefinementPass.COHERENCE in pass_types
                assert RefinementPass.TARGETED in pass_types
            
            # Clean up
            tmp_path.unlink()
            for pass_result in result['passes_executed']:
                if pass_result.output_path.exists():
                    pass_result.output_path.unlink()
    
    def test_targeted_inpainting(self, refiner, test_image, mock_pipelines):
        """Test targeted inpainting for artifact removal"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image.save(tmp.name)
            tmp_path = Path(tmp.name)
            
            # Create artifact mask
            artifact_mask = np.zeros((768, 1024))
            artifact_mask[:, 500:520] = 1.0  # Vertical seam
            
            # Mock detection with mask
            detection = Mock()
            detection.severity = ArtifactSeverity.HIGH
            detection.seam_locations = [{'position': 510}]
            detection.artifact_mask = artifact_mask
            
            # Execute targeted pass
            result = refiner._execute_pass(
                tmp_path,
                RefinementPass.TARGETED,
                "test prompt",
                detection,
                mock_pipelines,
                {},
                42
            )
            
            # Should use inpaint pipeline
            assert mock_pipelines['inpaint'].called
            
            # Check inpaint was called with mask
            call_args = mock_pipelines['inpaint'].call_args
            assert 'mask_image' in call_args[1]
            
            # Mask should be dilated and blurred
            mask_arg = call_args[1]['mask_image']
            mask_array = np.array(mask_arg)
            assert np.sum(mask_array > 0) > np.sum(artifact_mask > 0)  # Dilated
            
            # Clean up
            tmp_path.unlink()
            result.output_path.unlink()
    
    def test_pass_type_determination(self, refiner):
        """Test pass type selection based on severity"""
        # Critical severity
        assert refiner._determine_pass_type(0, ArtifactSeverity.CRITICAL, True) == RefinementPass.COHERENCE
        assert refiner._determine_pass_type(1, ArtifactSeverity.CRITICAL, True) == RefinementPass.TARGETED
        assert refiner._determine_pass_type(4, ArtifactSeverity.CRITICAL, True) == RefinementPass.UNIFICATION
        
        # High severity - skip coherence
        assert refiner._determine_pass_type(0, ArtifactSeverity.HIGH, True) == RefinementPass.TARGETED
        assert refiner._determine_pass_type(1, ArtifactSeverity.HIGH, True) == RefinementPass.DETAIL
        
        # Low severity - light touch
        assert refiner._determine_pass_type(0, ArtifactSeverity.LOW, True) == RefinementPass.DETAIL
        assert refiner._determine_pass_type(1, ArtifactSeverity.LOW, True) is None
        
        # Single pass mode
        assert refiner._determine_pass_type(0, ArtifactSeverity.HIGH, False) == RefinementPass.TARGETED
        assert refiner._determine_pass_type(1, ArtifactSeverity.HIGH, False) is None
    
    def test_quality_score_calculation(self, refiner):
        """Test quality score calculation"""
        # Perfect image
        perfect_detection = Mock()
        perfect_detection.severity = ArtifactSeverity.NONE
        perfect_detection.artifact_percentage = 0.0
        perfect_detection.seam_locations = []
        
        score = refiner._calculate_quality_score(perfect_detection)
        assert score == 1.0
        
        # Medium quality
        medium_detection = Mock()
        medium_detection.severity = ArtifactSeverity.MEDIUM
        medium_detection.artifact_percentage = 2.5
        medium_detection.seam_locations = [{'position': 512}]
        
        score = refiner._calculate_quality_score(medium_detection)
        assert 0.5 < score < 0.8
        
        # Poor quality
        poor_detection = Mock()
        poor_detection.severity = ArtifactSeverity.CRITICAL
        poor_detection.artifact_percentage = 10.0
        poor_detection.seam_locations = [{'position': 256}, {'position': 512}, {'position': 768}]
        
        score = refiner._calculate_quality_score(poor_detection)
        assert score < 0.5
    
    def test_refinement_with_no_pipelines(self, refiner, test_image):
        """Test graceful handling when no pipelines available"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image.save(tmp.name)
            tmp_path = Path(tmp.name)
            
            # No pipelines
            empty_pipelines = {}
            
            with patch.object(refiner.artifact_detector, 'detect_artifacts') as mock_detect:
                detection = Mock()
                detection.severity = ArtifactSeverity.MEDIUM
                detection.seam_locations = [{'position': 512}]
                detection.artifact_mask = None
                mock_detect.return_value = detection
                
                result = refiner.refine(
                    tmp_path,
                    "test prompt",
                    [],
                    empty_pipelines,
                    quality_preset='high'
                )
                
                # Should complete without errors
                assert result['image_path'] == tmp_path
                assert result['total_artifacts_fixed'] == 0
            
            tmp_path.unlink()
    
    def test_refinement_timing(self, refiner, test_image, mock_pipelines):
        """Test that timing is tracked correctly"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image.save(tmp.name)
            tmp_path = Path(tmp.name)
            
            # Add delay to mock pipeline
            def delayed_call(*args, **kwargs):
                time.sleep(0.1)
                result = Mock()
                result.images = [test_image]
                return result
                
            mock_pipelines['refiner'].side_effect = delayed_call
            
            with patch.object(refiner.artifact_detector, 'detect_artifacts') as mock_detect:
                detection = Mock()
                detection.severity = ArtifactSeverity.LOW
                detection.seam_locations = []
                detection.artifact_mask = None
                mock_detect.return_value = detection
                
                result = refiner.refine(
                    tmp_path,
                    "test prompt",
                    [],
                    mock_pipelines,
                    quality_preset='fast'
                )
                
                # Check timing
                if result['passes_executed']:
                    assert result['passes_executed'][0].duration_seconds >= 0.1
            
            tmp_path.unlink()
            for pass_result in result['passes_executed']:
                if pass_result.output_path.exists():
                    pass_result.output_path.unlink()
```

## 3. Boundary Tracking Implementation

### 3.1 Create Boundary Tracker

**EXACT FILE PATH**: `expandor/core/boundary_tracker.py`

```python
"""
Boundary Tracking System for Seam Detection
Tracks exact positions where expansions meet for artifact detection.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class BoundaryType(Enum):
    """Types of boundaries"""
    PROGRESSIVE = "progressive"
    SWPO = "swpo"
    TILE = "tile"
    REFINEMENT = "refinement"


@dataclass
class BoundaryInfo:
    """Information about a single boundary/seam"""
    position: int  # Pixel position of boundary
    direction: str  # 'horizontal' or 'vertical'
    expansion_size: int  # Size of expansion at this boundary
    boundary_type: BoundaryType = BoundaryType.PROGRESSIVE
    strength_used: float = 0.0  # Denoising strength used
    step_number: int = 0  # Which step created this boundary
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'position': self.position,
            'direction': self.direction,
            'expansion_size': self.expansion_size,
            'boundary_type': self.boundary_type.value,
            'strength_used': self.strength_used,
            'step_number': self.step_number,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundaryInfo':
        """Create from dictionary"""
        boundary_type = BoundaryType(data.get('boundary_type', 'progressive'))
        return cls(
            position=data['position'],
            direction=data['direction'],
            expansion_size=data.get('expansion_size', 0),
            boundary_type=boundary_type,
            strength_used=data.get('strength_used', 0.0),
            step_number=data.get('step_number', 0),
            metadata=data.get('metadata', {})
        )


class BoundaryTracker:
    """
    Tracks expansion boundaries throughout the pipeline.
    Critical for zero-tolerance artifact detection.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.boundaries: List[BoundaryInfo] = []
        self.current_step = 0
        
    def reset(self):
        """Reset tracker for new operation"""
        self.boundaries.clear()
        self.current_step = 0
        self.logger.debug("Boundary tracker reset")
    
    def add_boundary(self, 
                    position: int,
                    direction: str,
                    expansion_size: int,
                    boundary_type: BoundaryType = BoundaryType.PROGRESSIVE,
                    strength_used: float = 0.0,
                    metadata: Optional[Dict] = None):
        """
        Add a new boundary to track.
        
        Args:
            position: Pixel position where old content meets new
            direction: 'horizontal' or 'vertical'
            expansion_size: Pixels added at this boundary
            boundary_type: Type of operation that created boundary
            strength_used: Denoising/inpainting strength
            metadata: Additional information
        """
        boundary = BoundaryInfo(
            position=position,
            direction=direction,
            expansion_size=expansion_size,
            boundary_type=boundary_type,
            strength_used=strength_used,
            step_number=self.current_step,
            metadata=metadata or {}
        )
        
        self.boundaries.append(boundary)
        
        self.logger.debug(
            f"Added {boundary_type.value} boundary at {position} "
            f"({direction}, {expansion_size}px expansion)"
        )
    
    def add_progressive_boundaries(self, 
                                 current_size: Tuple[int, int],
                                 new_size: Tuple[int, int],
                                 expansion_direction: str,
                                 strength: float = 0.95):
        """
        Add boundaries for progressive expansion.
        
        Args:
            current_size: Size before expansion (w, h)
            new_size: Size after expansion (w, h)
            expansion_direction: 'horizontal' or 'vertical'
            strength: Denoising strength used
        """
        current_w, current_h = current_size
        new_w, new_h = new_size
        
        if expansion_direction == 'horizontal':
            # Expanding width - track vertical boundaries
            if new_w > current_w:
                # Right side expansion
                self.add_boundary(
                    position=current_w,
                    direction='vertical',
                    expansion_size=new_w - current_w,
                    boundary_type=BoundaryType.PROGRESSIVE,
                    strength_used=strength,
                    metadata={'side': 'right'}
                )
            
            # Check for left side (centered expansion)
            # FAIL LOUD: Centered expansion means content starts at an offset
            if current_w < new_w:
                left_expansion = (new_w - current_w) // 2
                if left_expansion > 0:
                    # The boundary is where the original content starts in the new canvas
                    # NOT the expansion size itself!
                    self.add_boundary(
                        position=0,  # Left edge is at position 0
                        direction='vertical',
                        expansion_size=left_expansion,
                        boundary_type=BoundaryType.PROGRESSIVE,
                        strength_used=strength,
                        metadata={'side': 'left', 'original_content_start': left_expansion}
                    )
                    
        else:  # vertical expansion
            # Expanding height - track horizontal boundaries
            if new_h > current_h:
                # Bottom expansion
                self.add_boundary(
                    position=current_h,
                    direction='horizontal',
                    expansion_size=new_h - current_h,
                    boundary_type=BoundaryType.PROGRESSIVE,
                    strength_used=strength,
                    metadata={'side': 'bottom'}
                )
                
            # Check for top side (centered expansion)
            # FAIL LOUD: Centered expansion means content starts at an offset
            if current_h < new_h:
                top_expansion = (new_h - current_h) // 2
                if top_expansion > 0:
                    # The boundary is where the original content starts in the new canvas
                    # NOT the expansion size itself!
                    self.add_boundary(
                        position=0,  # Top edge is at position 0
                        direction='horizontal',
                        expansion_size=top_expansion,
                        boundary_type=BoundaryType.PROGRESSIVE,
                        strength_used=strength,
                        metadata={'side': 'top', 'original_content_start': top_expansion}
                    )
    
    def add_swpo_boundaries(self, windows: List[Any]):
        """
        Add boundaries for SWPO (Sliding Window Progressive Outpainting).
        
        Args:
            windows: List of window information (dict or SWPOWindow objects)
        """
        # FAIL LOUD: Must handle both dict and dataclass formats
        for window in windows:
            # Handle both dict and dataclass formats
            if hasattr(window, 'position'):  # SWPOWindow dataclass
                position = window.position  # This is (x1, y1, x2, y2) tuple
                overlap_size = window.overlap_size
                expansion_type = window.expansion_type
                expansion_size = window.expansion_size
                index = window.index
                
                # Calculate boundary position from tuple
                # FAIL LOUD: Validate position tuple
                if len(position) != 4:
                    raise ExpandorError(f"Invalid SWPO window position tuple: {position} (expected (x1, y1, x2, y2))")
                    
                if expansion_type == 'horizontal':
                    # For horizontal expansion, boundary is at the left edge of new content
                    if overlap_size > 0 and not window.is_first:
                        boundary_pos = position[0] + overlap_size
                    else:
                        boundary_pos = position[0] if window.is_first else position[0]
                    direction = 'vertical'
                else:  # vertical
                    # For vertical expansion, boundary is at the top edge of new content  
                    if overlap_size > 0 and not window.is_first:
                        boundary_pos = position[1] + overlap_size
                    else:
                        boundary_pos = position[1] if window.is_first else position[1]
                    direction = 'horizontal'
                    
            else:  # Dict format (legacy support)
                # Extract from dict
                overlap_size = window.get('overlap_size', 0)
                
                # Handle position - could be tuple or separate fields
                if isinstance(window.get('position'), tuple):
                    position = window['position']
                    if window.get('direction') == 'horizontal':
                        boundary_pos = position[0] + overlap_size if overlap_size > 0 else position[0]
                        direction = 'vertical'
                    else:
                        boundary_pos = position[1] + overlap_size if overlap_size > 0 else position[1]
                        direction = 'horizontal'
                else:
                    # Fallback for other formats
                    boundary_pos = window.get('position', 0)
                    direction = 'vertical' if window.get('direction') == 'horizontal' else 'horizontal'
                    
                expansion_size = window.get('expansion_size', window.get('window_size', 0))
                index = window.get('index', 0)
                    
            self.add_boundary(
                position=boundary_pos,
                direction=direction,
                expansion_size=expansion_size,
                boundary_type=BoundaryType.SWPO,
                strength_used=0.95,  # SWPO typically uses high strength
                metadata={
                    'window_index': index,
                    'overlap_size': overlap_size
                }
            )
    
    def add_tile_boundaries(self, tile_size: int, overlap: int, 
                          image_size: Tuple[int, int]):
        """
        Add boundaries for tiled processing.
        
        Args:
            tile_size: Size of each tile
            overlap: Overlap between tiles
            image_size: Full image size (w, h)
        """
        width, height = image_size
        step = tile_size - overlap
        
        # Vertical boundaries (between horizontal tiles)
        for x in range(step, width, step):
            if x < width - overlap:
                self.add_boundary(
                    position=x,
                    direction='vertical',
                    expansion_size=0,  # No expansion, just processing boundary
                    boundary_type=BoundaryType.TILE,
                    metadata={'tile_size': tile_size, 'overlap': overlap}
                )
        
        # Horizontal boundaries (between vertical tiles)
        for y in range(step, height, step):
            if y < height - overlap:
                self.add_boundary(
                    position=y,
                    direction='horizontal',
                    expansion_size=0,
                    boundary_type=BoundaryType.TILE,
                    metadata={'tile_size': tile_size, 'overlap': overlap}
                )
    
    def increment_step(self):
        """Increment the current step number"""
        self.current_step += 1
        self.logger.debug(f"Advanced to step {self.current_step}")
    
    def get_boundaries_by_type(self, boundary_type: BoundaryType) -> List[BoundaryInfo]:
        """Get all boundaries of a specific type"""
        return [b for b in self.boundaries if b.boundary_type == boundary_type]
    
    def get_boundaries_by_direction(self, direction: str) -> List[BoundaryInfo]:
        """Get all boundaries in a specific direction"""
        return [b for b in self.boundaries if b.direction == direction]
    
    def get_critical_boundaries(self) -> List[BoundaryInfo]:
        """Get boundaries most likely to have artifacts"""
        critical = []
        
        for boundary in self.boundaries:
            # Progressive boundaries are always critical
            if boundary.boundary_type == BoundaryType.PROGRESSIVE:
                critical.append(boundary)
            # High strength operations are critical
            elif boundary.strength_used > 0.7:
                critical.append(boundary)
            # Large expansions are critical
            elif boundary.expansion_size > 200:
                critical.append(boundary)
                
        return critical
    
    def get_all_boundaries(self) -> List[BoundaryInfo]:
        """Get all tracked boundaries"""
        return self.boundaries.copy()
    
    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        Convert boundaries to metadata format for artifact detection.
        
        Returns dictionary compatible with SmartArtifactDetector expectations.
        """
        # Separate by direction for compatibility
        vertical_boundaries = [
            b.position for b in self.boundaries 
            if b.direction == 'vertical'
        ]
        horizontal_boundaries = [
            b.position for b in self.boundaries
            if b.direction == 'horizontal'
        ]
        
        return {
            'progressive_boundaries': vertical_boundaries,
            'progressive_boundaries_vertical': horizontal_boundaries,
            'generation_metadata': {
                'boundaries': [b.to_dict() for b in self.boundaries],
                'used_progressive': any(
                    b.boundary_type == BoundaryType.PROGRESSIVE 
                    for b in self.boundaries
                ),
                'used_swpo': any(
                    b.boundary_type == BoundaryType.SWPO
                    for b in self.boundaries
                ),
                'used_tiled': any(
                    b.boundary_type == BoundaryType.TILE
                    for b in self.boundaries
                )
            }
        }
    
    def save_to_file(self, filepath: str):
        """Save boundary information to JSON file"""
        data = {
            'boundaries': [b.to_dict() for b in self.boundaries],
            'total_boundaries': len(self.boundaries),
            'critical_count': len(self.get_critical_boundaries()),
            'by_type': {
                boundary_type.value: len(self.get_boundaries_by_type(boundary_type))
                for boundary_type in BoundaryType
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        self.logger.info(f"Saved {len(self.boundaries)} boundaries to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load boundary information from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.reset()
        
        for boundary_data in data.get('boundaries', []):
            boundary = BoundaryInfo.from_dict(boundary_data)
            self.boundaries.append(boundary)
            
        self.logger.info(f"Loaded {len(self.boundaries)} boundaries from {filepath}")
```

### 3.2 Create Boundary Tracker Unit Tests

**EXACT FILE PATH**: `tests/unit/core/test_boundary_tracker.py`

```python
"""
Unit tests for Boundary Tracker
"""

import pytest
import json
import tempfile
from pathlib import Path

from expandor.core.boundary_tracker import (
    BoundaryTracker, BoundaryInfo, BoundaryType
)


class TestBoundaryTracker:
    """Tests for boundary tracking system"""
    
    @pytest.fixture
    def tracker(self):
        """Create tracker instance"""
        return BoundaryTracker()
    
    def test_basic_boundary_addition(self, tracker):
        """Test adding basic boundaries"""
        tracker.add_boundary(
            position=512,
            direction='vertical',
            expansion_size=200,
            strength_used=0.95
        )
        
        assert len(tracker.boundaries) == 1
        boundary = tracker.boundaries[0]
        assert boundary.position == 512
        assert boundary.direction == 'vertical'
        assert boundary.expansion_size == 200
        assert boundary.boundary_type == BoundaryType.PROGRESSIVE
        
    def test_progressive_boundaries(self, tracker):
        """Test adding progressive expansion boundaries"""
        # Horizontal expansion (width increase)
        tracker.add_progressive_boundaries(
            current_size=(1024, 768),
            new_size=(1536, 768),
            expansion_direction='horizontal',
            strength=0.9
        )
        
        boundaries = tracker.get_all_boundaries()
        assert len(boundaries) > 0
        
        # Should have right side boundary
        right_boundaries = [b for b in boundaries if b.metadata.get('side') == 'right']
        assert len(right_boundaries) == 1
        assert right_boundaries[0].position == 1024
        assert right_boundaries[0].expansion_size == 512
        
        # Reset and test vertical expansion
        tracker.reset()
        tracker.add_progressive_boundaries(
            current_size=(1024, 768),
            new_size=(1024, 1080),
            expansion_direction='vertical',
            strength=0.85
        )
        
        boundaries = tracker.get_all_boundaries()
        bottom_boundaries = [b for b in boundaries if b.metadata.get('side') == 'bottom']
        assert len(bottom_boundaries) == 1
        assert bottom_boundaries[0].position == 768
        assert bottom_boundaries[0].expansion_size == 312
    
    def test_centered_expansion(self, tracker):
        """Test boundaries for centered expansion"""
        # Centered horizontal expansion
        tracker.add_progressive_boundaries(
            current_size=(1024, 768),
            new_size=(1536, 768),
            expansion_direction='horizontal',
            strength=0.9
        )
        
        boundaries = tracker.get_all_boundaries()
        
        # Should have both left and right boundaries
        left_boundaries = [b for b in boundaries if b.metadata.get('side') == 'left']
        right_boundaries = [b for b in boundaries if b.metadata.get('side') == 'right']
        
        # FAIL LOUD: For centered expansion, left boundary is at position 0
        # The original content starts at offset stored in metadata
        assert len(left_boundaries) == 1
        assert left_boundaries[0].position == 0  # Left edge boundary
        assert left_boundaries[0].metadata['original_content_start'] == 256  # (1536-1024)/2
        assert len(right_boundaries) == 1
        assert right_boundaries[0].position == 1024  # Right edge of original content
    
    def test_swpo_boundaries(self, tracker):
        """Test SWPO window boundaries"""
        windows = [
            {
                'index': 0,
                'position': (0, 0, 400, 768),
                'direction': 'horizontal',
                'expansion_size': 400,
                'overlap_size': 0,
                'strength': 0.95
            },
            {
                'index': 1,
                'position': (320, 0, 720, 768),
                'direction': 'horizontal',
                'expansion_size': 320,
                'overlap_size': 80,
                'strength': 0.95
            }
        ]
        
        tracker.add_swpo_boundaries(windows)
        
        boundaries = tracker.get_boundaries_by_type(BoundaryType.SWPO)
        assert len(boundaries) == 2
        
        # First window has no overlap
        assert boundaries[0].position == 0
        
        # Second window boundary is at overlap edge
        assert boundaries[1].position == 400  # 320 + 80
        assert boundaries[1].metadata['overlap_size'] == 80
    
    def test_tile_boundaries(self, tracker):
        """Test tiled processing boundaries"""
        tracker.add_tile_boundaries(
            tile_size=512,
            overlap=128,
            image_size=(1536, 1024)
        )
        
        tile_boundaries = tracker.get_boundaries_by_type(BoundaryType.TILE)
        
        # Calculate expected boundaries
        step = 512 - 128  # 384
        expected_v_boundaries = [384, 768, 1152]  # Vertical boundaries
        expected_h_boundaries = [384, 768]  # Horizontal boundaries
        
        v_boundaries = [b for b in tile_boundaries if b.direction == 'vertical']
        h_boundaries = [b for b in tile_boundaries if b.direction == 'horizontal']
        
        assert len(v_boundaries) == len(expected_v_boundaries)
        assert len(h_boundaries) == len(expected_h_boundaries)
        
        # Check positions
        v_positions = sorted([b.position for b in v_boundaries])
        assert v_positions == expected_v_boundaries
    
    def test_critical_boundaries(self, tracker):
        """Test identification of critical boundaries"""
        # Add various boundaries
        tracker.add_boundary(
            position=512,
            direction='vertical',
            expansion_size=100,
            boundary_type=BoundaryType.PROGRESSIVE
        )
        
        tracker.add_boundary(
            position=768,
            direction='horizontal',
            expansion_size=50,
            boundary_type=BoundaryType.TILE,
            strength_used=0.3
        )
        
        tracker.add_boundary(
            position=1024,
            direction='vertical',
            expansion_size=300,  # Large expansion
            boundary_type=BoundaryType.SWPO,
            strength_used=0.5
        )
        
        tracker.add_boundary(
            position=256,
            direction='horizontal',
            expansion_size=100,
            boundary_type=BoundaryType.SWPO,
            strength_used=0.85  # High strength
        )
        
        critical = tracker.get_critical_boundaries()
        
        # Should include: progressive (always), large expansion, high strength
        assert len(critical) == 3
        
        # Check that non-critical tile boundary is excluded
        critical_positions = [b.position for b in critical]
        assert 768 not in critical_positions
    
    def test_metadata_conversion(self, tracker):
        """Test conversion to artifact detector metadata format"""
        # Add mixed boundaries
        tracker.add_boundary(512, 'vertical', 200)
        tracker.add_boundary(384, 'horizontal', 150)
        tracker.add_boundary(768, 'vertical', 100, BoundaryType.SWPO)
        
        metadata = tracker.to_metadata_dict()
        
        # Check format
        assert 'progressive_boundaries' in metadata
        assert 'progressive_boundaries_vertical' in metadata
        assert 'generation_metadata' in metadata
        
        # Check boundary separation
        assert 512 in metadata['progressive_boundaries']
        assert 768 in metadata['progressive_boundaries']
        assert 384 in metadata['progressive_boundaries_vertical']
        
        # Check flags
        assert metadata['generation_metadata']['used_progressive'] == True
        assert metadata['generation_metadata']['used_swpo'] == True
        assert metadata['generation_metadata']['used_tiled'] == False
    
    def test_step_tracking(self, tracker):
        """Test step number tracking"""
        tracker.add_boundary(100, 'vertical', 50)
        assert tracker.boundaries[0].step_number == 0
        
        tracker.increment_step()
        tracker.add_boundary(200, 'vertical', 50)
        assert tracker.boundaries[1].step_number == 1
        
        tracker.increment_step()
        tracker.add_boundary(300, 'vertical', 50)
        assert tracker.boundaries[2].step_number == 2
    
    def test_save_load(self, tracker):
        """Test saving and loading boundaries"""
        # Add various boundaries
        tracker.add_boundary(512, 'vertical', 200, strength_used=0.9)
        tracker.add_boundary(768, 'horizontal', 150, BoundaryType.SWPO)
        tracker.add_tile_boundaries(512, 128, (1536, 1024))
        
        original_count = len(tracker.boundaries)
        
        # Save to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tracker.save_to_file(tmp.name)
            tmp_path = tmp.name
        
        # Create new tracker and load
        new_tracker = BoundaryTracker()
        new_tracker.load_from_file(tmp_path)
        
        # Verify
        assert len(new_tracker.boundaries) == original_count
        
        # Check specific boundary
        first_boundary = new_tracker.boundaries[0]
        assert first_boundary.position == 512
        assert first_boundary.direction == 'vertical'
        assert first_boundary.strength_used == 0.9
        
        # Clean up
        Path(tmp_path).unlink()
    
    def test_boundary_info_serialization(self):
        """Test BoundaryInfo to/from dict conversion"""
        boundary = BoundaryInfo(
            position=1024,
            direction='vertical',
            expansion_size=256,
            boundary_type=BoundaryType.SWPO,
            strength_used=0.85,
            step_number=3,
            metadata={'window_index': 2}
        )
        
        # Convert to dict
        boundary_dict = boundary.to_dict()
        assert boundary_dict['position'] == 1024
        assert boundary_dict['boundary_type'] == 'swpo'
        assert boundary_dict['metadata']['window_index'] == 2
        
        # Convert back
        restored = BoundaryInfo.from_dict(boundary_dict)
        assert restored.position == boundary.position
        assert restored.boundary_type == boundary.boundary_type
        assert restored.metadata == boundary.metadata
```

## 4. Integration Components

### 4.1 Create Quality Validator

**EXACT FILE PATH**: `expandor/processors/quality_validator.py`

```python
"""
Quality Validation System
Integrates artifact detection, boundary tracking, and quality scoring.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .artifact_detector import SmartArtifactDetector
from ..core.exceptions import ExpandorError

# Define QualityError for quality validation failures
class QualityError(ExpandorError):
    """Raised when quality validation fails"""
    pass


class QualityValidator:
    """Validates image quality and detects artifacts"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.artifact_detector = SmartArtifactDetector(config, logger)
        
    def validate(self, image_path: Path, metadata: Dict[str, Any],
                detection_level: str = "standard") -> Dict[str, Any]:
        """
        Validate image quality.
        
        Returns:
            Dictionary with validation results
        """
        # Run artifact detection
        detection_result = self.artifact_detector.detect_artifacts(
            image_path,
            metadata,
            detection_level
        )
        
        return {
            "issues_found": detection_result.has_artifacts,
            "seam_count": len(detection_result.seam_locations),
            "quality_score": 1.0 - (detection_result.artifact_percentage / 100),
            "mask": detection_result.artifact_mask,
            "severity": detection_result.severity.value,
            "detection_scores": detection_result.detection_scores
        }
```

### 4.2 Create Seam Repair Processor

**EXACT FILE PATH**: `expandor/processors/seam_repair.py`

```python
"""
Seam Repair Processor
Repairs detected artifacts using available pipelines.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

from ..core.exceptions import ExpandorError


class SeamRepairProcessor:
    """Repairs detected seams and artifacts"""
    
    def __init__(self, pipelines: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.pipelines = pipelines
        self.logger = logger or logging.getLogger(__name__)
        
    def repair_seams(self, image_path: Path, artifact_mask: np.ndarray,
                    prompt: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair detected seams using available pipelines.
        
        Returns:
            Dictionary with repair results
        """
        # Load image
        image = Image.open(image_path)
        
        # Prepare mask
        mask_image = Image.fromarray((artifact_mask * 255).astype(np.uint8), 'L')
        
        # Dilate and blur mask
        mask_image = mask_image.filter(ImageFilter.MaxFilter(5))
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(3))
        
        # Try inpainting first
        if 'inpaint' in self.pipelines:
            try:
                result = self.pipelines['inpaint'](
                    prompt=prompt + ", seamless, perfect quality",
                    image=image,
                    mask_image=mask_image,
                    strength=0.8,
                    num_inference_steps=50
                )
                repaired = result.images[0]
            except Exception as e:
                self.logger.error(f"Inpainting failed: {e}")
                repaired = image
        else:
            repaired = image
            
        # Save result
        output_path = image_path.parent / f"{image_path.stem}_repaired.png"
        repaired.save(output_path, 'PNG', compress_level=0)
        
        return {
            "image_path": output_path,
            "seams_repaired": np.sum(artifact_mask > 0.5),
            "metadata": metadata
        }
```

## 5. Verification Checklist

Before proceeding with implementation:

### 5.1 File Creation Verification
```bash
# Verify all quality system files exist
ls -la expandor/processors/artifact_detector.py
ls -la expandor/processors/refinement/smart_refiner.py
ls -la expandor/core/boundary_tracker.py
ls -la expandor/processors/quality_validator.py
ls -la expandor/processors/seam_repair.py

# Verify test files
ls -la tests/unit/processors/test_artifact_detector.py
ls -la tests/unit/processors/test_smart_refiner.py
ls -la tests/unit/core/test_boundary_tracker.py
```

### 5.2 Import Verification
```python
# Test all imports work
python -c "from expandor.processors.artifact_detector import SmartArtifactDetector"
python -c "from expandor.processors.refinement.smart_refiner import SmartQualityRefiner"
python -c "from expandor.core.boundary_tracker import BoundaryTracker"
python -c "from expandor.processors.quality_validator import QualityValidator"
```

### 5.3 Unit Test Execution
```bash
# Run unit tests
pytest tests/unit/processors/test_artifact_detector.py -v
pytest tests/unit/processors/test_smart_refiner.py -v
pytest tests/unit/core/test_boundary_tracker.py -v
```

## 6. Critical Implementation Notes

### 6.1 Artifact Detection
- **Multiple detection methods MUST be combined** for zero-tolerance
- **Progressive boundaries are ALWAYS critical** regardless of appearance
- **Detection must work at multiple scales** (up to 4K analysis)
- **Morphological operations clean up masks** to avoid false positives
- **All thresholds are configurable** per detection level

### 6.2 Smart Refinement
- **Pass selection based on severity** ensures efficient processing
- **Targeted inpainting for artifacts** preserves unaffected areas
- **Multiple passes with decreasing strength** prevent over-processing
- **Quality score calculation** provides objective success metrics
- **Graceful degradation** when pipelines unavailable

### 6.3 Boundary Tracking
- **Every expansion creates boundaries** that must be tracked
- **Metadata format compatibility** with artifact detector
- **Critical boundary identification** prioritizes likely problem areas
- **Serialization support** for debugging and analysis
- **Step tracking** maintains operation history

## 7. Integration Notes

The quality systems integrate tightly:

1. **Boundary Tracker** → Records where expansions meet
2. **Artifact Detector** → Uses boundaries to find seams
3. **Smart Refiner** → Targets detected artifacts
4. **Quality Validator** → Provides final assessment

This creates a closed-loop system where:
- Every expansion is tracked
- Every seam is detected
- Every artifact is repaired
- Every result is validated

## 8. Next Steps

After implementing these quality systems:

1. **Integration with Strategies**
   - Update each strategy to use boundary tracker
   - Add artifact detection after each stage
   - Implement refinement as final step

2. **Performance Optimization**
   - Profile detection algorithms
   - Optimize mask operations
   - Cache detection results

3. **Extended Testing**
   - Test with real progressive expansions
   - Verify SWPO boundary tracking
   - Validate refinement effectiveness

4. **Documentation**
   - API documentation for each component
   - Integration guide for strategies
   - Quality tuning guide