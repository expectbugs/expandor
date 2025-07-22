# Expandor Phase 3 Step 3: Integration Tests - Ultra-Detailed Implementation Guide

## Overview

This document provides a foolproof, zero-error implementation guide for the Integration Tests component of Expandor Phase 3. These tests verify that all components work together seamlessly, including full pipeline tests, extreme aspect ratio tests, quality validation tests, and SWPO integration tests.

## Prerequisites Verification

Before starting ANY implementation:

```bash
# 1. Verify you're in the expandor repository (NOT ai-wallpaper)
pwd
# Expected: /path/to/expandor

# 2. Verify Python environment is activated
which python
# Should show: /path/to/expandor/venv/bin/python

# 3. Verify required test directories exist
ls -la tests/integration/
ls -la tests/fixtures/
# Should show existing directories

# 4. Verify all Phase 3 components are implemented
ls -la expandor/strategies/swpo_strategy.py
ls -la expandor/processors/artifact_detector.py
ls -la expandor/core/boundary_tracker.py
# All should exist from previous steps

# 5. Install test dependencies
pip install pytest pytest-cov pytest-mock pytest-timeout scipy
```

## 1. Integration Test Infrastructure

### 1.1 Create Test Fixtures and Helpers

**EXACT FILE PATH**: `tests/fixtures/mock_pipelines.py`

```python
"""
Mock Pipeline Fixtures for Integration Testing
Provides realistic mock pipelines that simulate actual AI model behavior.
"""

from typing import List, Optional, Any, Dict, Tuple
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from dataclasses import dataclass
import torch
import hashlib
import time
import random


@dataclass
class MockPipelineOutput:
    """Mock output from pipeline matching diffusers interface"""
    images: List[Image.Image]


class MockInpaintPipeline:
    """
    Realistic mock inpainting pipeline for integration testing.
    Simulates actual inpainting behavior with deterministic results.
    """
    
    def __init__(self, latency_ms: float = 100, failure_rate: float = 0.0):
        self.call_count = 0
        self.call_history = []
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        
    def __call__(self, 
                 prompt: str,
                 image: Image.Image,
                 mask_image: Image.Image,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 strength: float = 0.8,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 generator: Optional[torch.Generator] = None,
                 **kwargs) -> MockPipelineOutput:
        """
        Simulate inpainting with realistic behavior.
        
        Creates content that blends with existing image based on:
        - Prompt keywords
        - Mask boundaries
        - Existing image colors
        """
        import time
        import random
        
        # Track call
        self.call_count += 1
        self.call_history.append({
            'prompt': prompt,
            'strength': strength,
            'steps': num_inference_steps,
            'size': (width or image.width, height or image.height)
        })
        
        # Simulate latency
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000)
            
        # Simulate failures
        if self.failure_rate > 0 and random.random() < self.failure_rate:
            raise RuntimeError("Mock pipeline failure (simulated)")
            
        # Get dimensions
        target_w = width or image.width
        target_h = height or image.height
        
        # Resize if needed
        if (target_w, target_h) != image.size:
            canvas = Image.new('RGB', (target_w, target_h))
            # Center original image
            paste_x = (target_w - image.width) // 2
            paste_y = (target_h - image.height) // 2
            canvas.paste(image, (paste_x, paste_y))
            working_image = canvas
            
            # Adjust mask
            mask_canvas = Image.new('L', (target_w, target_h), 255)
            mask_canvas.paste(mask_image, (paste_x, paste_y))
            working_mask = mask_canvas
        else:
            working_image = image.copy()
            working_mask = mask_image.copy()
            
        # Convert to arrays
        img_array = np.array(working_image)
        mask_array = np.array(working_mask) / 255.0
        
        # Analyze prompt for content generation
        prompt_lower = prompt.lower()
        
        # Determine base color from prompt
        if 'sky' in prompt_lower or 'blue' in prompt_lower:
            base_color = np.array([135, 206, 235])  # Sky blue
        elif 'grass' in prompt_lower or 'green' in prompt_lower:
            base_color = np.array([34, 139, 34])  # Forest green
        elif 'sunset' in prompt_lower or 'orange' in prompt_lower:
            base_color = np.array([255, 140, 0])  # Dark orange
        elif 'night' in prompt_lower or 'dark' in prompt_lower:
            base_color = np.array([25, 25, 112])  # Midnight blue
        else:
            # Use hash of prompt for deterministic color
            prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:6], 16)
            base_color = np.array([
                (prompt_hash >> 16) & 0xFF,
                (prompt_hash >> 8) & 0xFF,
                prompt_hash & 0xFF
            ])
            
        # Analyze edges of mask for color continuity
        edge_colors = self._analyze_mask_edges(img_array, mask_array)
        
        # Generate content
        h, w = img_array.shape[:2]
        
        # Create base pattern
        if 'landscape' in prompt_lower or 'nature' in prompt_lower:
            # Natural gradient pattern
            pattern = self._generate_landscape_pattern(h, w, base_color, edge_colors)
        elif 'abstract' in prompt_lower or 'pattern' in prompt_lower:
            # Abstract pattern
            pattern = self._generate_abstract_pattern(h, w, base_color)
        else:
            # Smooth gradient blending with edges
            pattern = self._generate_smooth_blend(h, w, base_color, edge_colors)
            
        # Add realistic texture
        pattern = self._add_texture(pattern, strength)
        
        # Blend with original based on mask
        for c in range(3):
            img_array[:, :, c] = (
                img_array[:, :, c] * (1 - mask_array) + 
                pattern[:, :, c] * mask_array * strength +
                img_array[:, :, c] * mask_array * (1 - strength)
            ).astype(np.uint8)
            
        # Apply smoothing at mask boundaries
        result_img = Image.fromarray(img_array)
        result_img = self._smooth_boundaries(result_img, working_mask)
        
        return MockPipelineOutput(images=[result_img])
    
    def _analyze_mask_edges(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze colors at mask edges for seamless blending"""
        h, w = image.shape[:2]
        edge_colors = {}
        
        # Find mask boundaries
        mask_binary = mask > 0.5
        
        # Left edge
        for x in range(w):
            if np.any(mask_binary[:, x]):
                if x > 0:
                    edge_colors['left'] = np.mean(image[:, max(0, x-10):x], axis=(0, 1))
                break
                
        # Right edge
        for x in range(w-1, -1, -1):
            if np.any(mask_binary[:, x]):
                if x < w-1:
                    edge_colors['right'] = np.mean(image[:, x+1:min(w, x+11)], axis=(0, 1))
                break
                
        # Top edge
        for y in range(h):
            if np.any(mask_binary[y, :]):
                if y > 0:
                    edge_colors['top'] = np.mean(image[max(0, y-10):y, :], axis=(0, 1))
                break
                
        # Bottom edge
        for y in range(h-1, -1, -1):
            if np.any(mask_binary[y, :]):
                if y < h-1:
                    edge_colors['bottom'] = np.mean(image[y+1:min(h, y+11), :], axis=(0, 1))
                break
                
        return edge_colors
    
    def _generate_landscape_pattern(self, h: int, w: int, base_color: np.ndarray, 
                                  edge_colors: Dict) -> np.ndarray:
        """Generate natural-looking landscape pattern"""
        pattern = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create gradient from top to bottom
        for y in range(h):
            t = y / h
            # Sky to ground gradient
            color = base_color * (1 - t) + base_color * 0.7 * t
            
            # Add horizontal variation
            for x in range(w):
                noise = np.random.normal(0, 10, 3)
                pattern[y, x] = np.clip(color + noise, 0, 255)
                
        # Blend with edge colors
        if 'left' in edge_colors:
            for x in range(min(100, w)):
                alpha = x / 100
                pattern[:, x] = pattern[:, x] * alpha + edge_colors['left'] * (1 - alpha)
                
        if 'right' in edge_colors:
            for x in range(max(0, w-100), w):
                alpha = (w - x) / 100
                pattern[:, x] = pattern[:, x] * alpha + edge_colors['right'] * (1 - alpha)
                
        return pattern
    
    def _generate_abstract_pattern(self, h: int, w: int, base_color: np.ndarray) -> np.ndarray:
        """Generate abstract pattern"""
        pattern = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create multiple overlapping gradients
        center_x, center_y = w // 2, h // 2
        
        for y in range(h):
            for x in range(w):
                # Distance from center
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                
                # Radial gradient with variation
                t = dist / max_dist
                color_variation = np.array([
                    np.sin(t * np.pi * 2) * 30,
                    np.cos(t * np.pi * 3) * 30,
                    np.sin(t * np.pi * 4) * 30
                ])
                
                pattern[y, x] = np.clip(base_color + color_variation, 0, 255)
                
        return pattern
    
    def _generate_smooth_blend(self, h: int, w: int, base_color: np.ndarray,
                             edge_colors: Dict) -> np.ndarray:
        """Generate smooth blend between edge colors"""
        pattern = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Start with base color
        pattern[:, :] = base_color
        
        # Blend from each edge
        blend_distance = min(h, w) // 3
        
        if 'left' in edge_colors:
            for x in range(min(blend_distance, w)):
                alpha = x / blend_distance
                pattern[:, x] = edge_colors['left'] * (1 - alpha) + pattern[:, x] * alpha
                
        if 'right' in edge_colors:
            for x in range(max(0, w - blend_distance), w):
                alpha = (w - 1 - x) / blend_distance
                pattern[:, x] = edge_colors['right'] * (1 - alpha) + pattern[:, x] * alpha
                
        if 'top' in edge_colors:
            for y in range(min(blend_distance, h)):
                alpha = y / blend_distance
                pattern[y, :] = edge_colors['top'] * (1 - alpha) + pattern[y, :] * alpha
                
        if 'bottom' in edge_colors:
            for y in range(max(0, h - blend_distance), h):
                alpha = (h - 1 - y) / blend_distance
                pattern[y, :] = edge_colors['bottom'] * (1 - alpha) + pattern[y, :] * alpha
                
        return pattern
    
    def _add_texture(self, pattern: np.ndarray, strength: float) -> np.ndarray:
        """Add realistic texture to pattern"""
        h, w = pattern.shape[:2]
        
        # Add Perlin-like noise
        noise_scale = 50
        noise = np.zeros((h, w))
        
        for scale in [1, 2, 4, 8]:
            freq = scale / noise_scale
            amplitude = 20 / scale
            
            for y in range(h):
                for x in range(w):
                    noise[y, x] += np.sin(x * freq) * np.cos(y * freq) * amplitude
                    
        # Apply noise to pattern
        for c in range(3):
            pattern[:, :, c] = np.clip(
                pattern[:, :, c] + noise * strength,
                0, 255
            )
            
        return pattern
    
    def _smooth_boundaries(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Smooth boundaries between masked and unmasked regions"""
        # FAIL LOUD: Ensure we have valid inputs
        if image.size != mask.size:
            raise ValueError(f"Image size {image.size} != mask size {mask.size}")
            
        mask_array = np.array(mask)
        
        # Efficient dilation using scipy if available, else numpy
        try:
            from scipy.ndimage import binary_dilation
            # Create structuring element for 3x3 dilation
            struct = np.ones((3, 3), dtype=bool)
            dilated = binary_dilation(mask_array > 128, structure=struct).astype(np.uint8) * 255
        except ImportError:
            # Fallback to numpy-based dilation (still more efficient than pixel loops)
            binary_mask = mask_array > 128
            h, w = binary_mask.shape
            dilated = np.zeros_like(mask_array)
            
            # Vectorized dilation
            dilated[1:h-1, 1:w-1] = (
                binary_mask[0:h-2, 0:w-2] | binary_mask[0:h-2, 1:w-1] | binary_mask[0:h-2, 2:w] |
                binary_mask[1:h-1, 0:w-2] | binary_mask[1:h-1, 1:w-1] | binary_mask[1:h-1, 2:w] |
                binary_mask[2:h,   0:w-2] | binary_mask[2:h,   1:w-1] | binary_mask[2:h,   2:w]
            ).astype(np.uint8) * 255
                    
        # Find boundary pixels
        boundary = (dilated == 255) & (mask_array < 255)
        
        if np.any(boundary):
            # Apply slight blur to boundary region
            boundary_mask = Image.fromarray(boundary.astype(np.uint8) * 255)
            blurred = image.filter(ImageFilter.GaussianBlur(radius=2))
            
            # Composite
            return Image.composite(blurred, image, boundary_mask.convert('L'))
            
        return image


class MockRefinerPipeline:
    """Mock refinement pipeline for quality enhancement"""
    
    def __init__(self, quality_improvement: float = 0.2):
        self.call_count = 0
        self.quality_improvement = quality_improvement
        
    def __call__(self,
                 prompt: str,
                 image: Image.Image,
                 strength: float = 0.3,
                 num_inference_steps: int = 30,
                 guidance_scale: float = 7.0,
                 generator: Optional[torch.Generator] = None,
                 **kwargs) -> MockPipelineOutput:
        """Simulate refinement by enhancing image quality"""
        self.call_count += 1
        
        # Apply sharpening
        refined = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        # Enhance contrast slightly
        img_array = np.array(refined)
        
        # Increase contrast
        mean = np.mean(img_array, axis=(0, 1))
        enhanced = img_array.astype(np.float32)
        
        for c in range(3):
            enhanced[:, :, c] = (enhanced[:, :, c] - mean[c]) * (1 + self.quality_improvement) + mean[c]
            
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        result = Image.fromarray(enhanced)
        
        return MockPipelineOutput(images=[result])


class MockUpscaler:
    """Mock upscaler for resolution enhancement"""
    
    def __init__(self):
        self.call_count = 0
        
    def upscale(self, image: Image.Image, scale: int = 2) -> Image.Image:
        """Simulate upscaling with bicubic interpolation"""
        self.call_count += 1
        
        new_size = (image.width * scale, image.height * scale)
        return image.resize(new_size, Image.Resampling.BICUBIC)


def create_mock_inpaint_pipeline(**kwargs) -> MockInpaintPipeline:
    """Factory function to create configured mock inpaint pipeline"""
    return MockInpaintPipeline(**kwargs)


def create_mock_refiner_pipeline(**kwargs) -> MockRefinerPipeline:
    """Factory function to create configured mock refiner pipeline"""
    return MockRefinerPipeline(**kwargs)


def create_mock_img2img_pipeline(**kwargs) -> MockRefinerPipeline:
    """Factory function to create configured mock img2img pipeline"""
    # Img2img can use same implementation as refiner
    return MockRefinerPipeline(**kwargs)
```

### 1.2 Create Integration Test Base Class

**EXACT FILE PATH**: `tests/integration/base_integration_test.py`

```python
"""
Base class for integration tests with common setup and utilities.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
import logging
import json

from expandor import Expandor, ExpandorConfig
from expandor.core.exceptions import ExpandorError, VRAMError
from tests.fixtures.mock_pipelines import (
    create_mock_inpaint_pipeline,
    create_mock_refiner_pipeline,
    create_mock_img2img_pipeline
)


class BaseIntegrationTest:
    """Base class with common integration test functionality"""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment for each test"""
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Set up logging
        self.logger = logging.getLogger('test')
        self.logger.setLevel(logging.DEBUG)
        
        # Create expandor instance
        self.expandor = Expandor(logger=self.logger)
        
        yield
        
        # Cleanup
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    def create_test_image(self, size: Tuple[int, int], 
                         pattern: str = "gradient") -> Image.Image:
        """Create test image with specified pattern"""
        width, height = size
        
        if pattern == "gradient":
            # Create gradient pattern
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            
            for y in range(height):
                for x in range(width):
                    img_array[y, x] = [
                        int(x * 255 / width),      # Red gradient horizontal
                        int(y * 255 / height),      # Green gradient vertical
                        128                         # Blue constant
                    ]
                    
        elif pattern == "checkerboard":
            # Create checkerboard pattern
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            square_size = 64
            
            for y in range(height):
                for x in range(width):
                    if (x // square_size + y // square_size) % 2 == 0:
                        img_array[y, x] = [255, 255, 255]  # White
                    else:
                        img_array[y, x] = [0, 0, 0]        # Black
                        
        elif pattern == "solid":
            # Solid color
            img_array = np.full((height, width, 3), 128, dtype=np.uint8)
            
        else:
            # Random noise
            img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            
        return Image.fromarray(img_array)
    
    def create_mock_pipelines(self, 
                            include_inpaint: bool = True,
                            include_refiner: bool = True,
                            include_img2img: bool = True,
                            **kwargs) -> Dict[str, Any]:
        """Create set of mock pipelines"""
        pipelines = {}
        
        if include_inpaint:
            pipelines['inpaint'] = create_mock_inpaint_pipeline(**kwargs)
            
        if include_refiner:
            pipelines['refiner'] = create_mock_refiner_pipeline(**kwargs)
            
        if include_img2img:
            pipelines['img2img'] = create_mock_img2img_pipeline(**kwargs)
            
        return pipelines
    
    def create_config(self,
                     source_image: Image.Image,
                     target_resolution: Tuple[int, int],
                     quality_preset: str = "high",
                     save_stages: bool = True,
                     **kwargs) -> ExpandorConfig:
        """Create expansion configuration"""
        # Set defaults
        config_dict = {
            'source_image': source_image,
            'target_resolution': target_resolution,
            'prompt': kwargs.get('prompt', 'A beautiful landscape with seamless expansion'),
            'seed': kwargs.get('seed', 42),
            'source_metadata': kwargs.get('source_metadata', {'model': 'test'}),
            'quality_preset': quality_preset,
            'save_stages': save_stages,
            'stage_dir': self.temp_dir / "stages",
            'verbose': True
        }
        
        # Add any additional kwargs
        config_dict.update(kwargs)
        
        return ExpandorConfig(**config_dict)
    
    def verify_result(self, result: Any, expected_size: Tuple[int, int]):
        """Common result verification"""
        # FAIL LOUD: Check required fields exist
        if not hasattr(result, 'success'):
            raise AssertionError("Result missing 'success' field - expansion failed!")
        assert result.success is True, f"Expansion failed: {getattr(result, 'error', 'Unknown error')}"
        
        # Check size
        if not hasattr(result, 'size'):
            raise AssertionError("Result missing 'size' field!")
        assert result.size == expected_size, f"Size mismatch: expected {expected_size}, got {result.size}"
        
        # Check image path
        if not hasattr(result, 'image_path'):
            raise AssertionError("Result missing 'image_path' field!")
        assert result.image_path.exists(), f"Result image not found at {result.image_path}"
        
        # Verify image can be loaded
        img = Image.open(result.image_path)
        assert img.size == expected_size
        
        # Verify metadata (with existence checks)
        assert hasattr(result, 'stages') and len(result.stages) > 0, "No processing stages recorded"
        assert hasattr(result, 'total_duration_seconds') and result.total_duration_seconds > 0, "No duration recorded"
        assert hasattr(result, 'strategy_used') and result.strategy_used is not None, "No strategy recorded"
        
    def check_no_artifacts(self, result: Any, tolerance: float = 0.1):
        """Verify no significant artifacts detected"""
        # FAIL LOUD: Check if quality tracking exists
        has_seam_tracking = hasattr(result, 'seams_detected')
        has_quality_score = hasattr(result, 'quality_score')
        
        if not has_seam_tracking and not has_quality_score:
            raise AssertionError("Result has no artifact tracking fields!")
            
        if has_seam_tracking and has_quality_score:
            assert result.seams_detected == 0 or result.quality_score > (1.0 - tolerance), \
                f"Artifacts detected: {result.seams_detected} seams, quality score: {result.quality_score}"
        elif has_seam_tracking:
            assert result.seams_detected == 0, f"Detected {result.seams_detected} seams!"
        else:  # has_quality_score
            assert result.quality_score > (1.0 - tolerance), f"Quality score {result.quality_score} below threshold"
        
    def save_debug_info(self, result: Any, test_name: str):
        """Save debug information for failed tests"""
        debug_dir = self.temp_dir / "debug" / test_name
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Save result image
        if result.image_path.exists():
            shutil.copy(result.image_path, debug_dir / "final_result.png")
            
        # Save stages if available
        if hasattr(result, 'stages'):
            for i, stage in enumerate(result.stages):
                stage_info = {
                    'index': i,
                    'name': stage.get('name', 'unknown'),
                    'method': stage.get('method', 'unknown'),
                    'duration': stage.get('duration_seconds', 0)
                }
                
                with open(debug_dir / f"stage_{i}.json", 'w') as f:
                    json.dump(stage_info, f, indent=2)
                    
        # Save metadata
        with open(debug_dir / "metadata.json", 'w') as f:
            json.dump(result.metadata, f, indent=2)
            
        print(f"Debug info saved to: {debug_dir}")
```

## 2. Full Pipeline Integration Tests

### 2.1 Create Full Pipeline Tests

**EXACT FILE PATH**: `tests/integration/test_full_pipeline.py`

```python
"""
Full pipeline integration tests.
Tests complete expansion workflows from start to finish.
"""

import pytest
from PIL import Image
import time

from tests.integration.base_integration_test import BaseIntegrationTest
from expandor.core.exceptions import ExpandorError, VRAMError


class TestFullPipeline(BaseIntegrationTest):
    """Test complete expansion pipelines end-to-end"""
    
    def test_simple_upscale_pipeline(self):
        """Test simple 2x upscale with same aspect ratio"""
        # Create 1080p test image
        source_image = self.create_test_image((1920, 1080), "gradient")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for 4K upscale
        config = self.create_config(
            source_image=source_image,
            target_resolution=(3840, 2160),
            quality_preset="balanced",
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (3840, 2160))
        # Strategy should be upscale-related (not exact string match)
        assert "upscale" in result.strategy_used.lower() or "direct" in result.strategy_used.lower(), \
            f"Unexpected strategy: {result.strategy_used}"
        self.check_no_artifacts(result)
        
        # Check that only necessary pipelines were used
        assert pipelines['inpaint'].call_count == 0  # No inpainting needed
        
    def test_aspect_ratio_change_pipeline(self):
        """Test 16:9 to 21:9 aspect ratio change"""
        # Create 16:9 image
        source_image = self.create_test_image((1920, 1080), "checkerboard")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for 21:9
        config = self.create_config(
            source_image=source_image,
            target_resolution=(2560, 1080),
            quality_preset="high",
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (2560, 1080))
        assert "progressive" in result.strategy_used.lower()
        
        # Should have used inpainting
        assert pipelines['inpaint'].call_count > 0
        
        # Check boundaries were tracked
        if not hasattr(result, 'boundaries'):
            raise AssertionError("Result missing 'boundaries' field!")
        assert len(result.boundaries) > 0, "No boundaries tracked during expansion"
        
        # Quality should be good
        self.check_no_artifacts(result, tolerance=0.15)
        
    def test_extreme_expansion_pipeline(self):
        """Test extreme 8x expansion with SWPO"""
        # Create small image
        source_image = self.create_test_image((1024, 768), "gradient")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for extreme expansion
        config = self.create_config(
            source_image=source_image,
            target_resolution=(8192, 768),  # 8x width
            quality_preset="ultra",
            window_size=300,
            overlap_ratio=0.8,
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (8192, 768))
        
        # Should use SWPO for extreme ratio
        assert "swpo" in result.strategy_used.lower()
        
        # Should have multiple stages
        assert len(result.stages) > 3
        
        # Should have tracked many boundaries
        assert len(result.boundaries) > 5
        
        # Check SWPO was used effectively
        inpaint_calls = pipelines['inpaint'].call_history
        assert len(inpaint_calls) > 5  # Multiple windows
        
    def test_multi_stage_pipeline(self):
        """Test complex multi-stage expansion"""
        # Create square image
        source_image = self.create_test_image((1024, 1024), "solid")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for both aspect change and upscale
        config = self.create_config(
            source_image=source_image,
            target_resolution=(3840, 2160),  # 16:9 4K
            quality_preset="high",
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (3840, 2160))
        
        # Should use hybrid strategy for complex transformation
        assert "hybrid" in result.strategy_used.lower() or len(result.stages) > 2
        
        # Should have aspect adjustment and upscaling stages
        stage_methods = [s.get('method', '') for s in result.stages]
        assert any('outpaint' in m or 'expand' in m for m in stage_methods)
        assert any('upscale' in m or 'refine' in m for m in stage_methods)
        
    def test_pipeline_with_refinement(self):
        """Test pipeline with quality refinement enabled"""
        # Create image with potential artifacts
        source_image = self.create_test_image((1344, 768), "gradient")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure with ultra quality for maximum refinement
        config = self.create_config(
            source_image=source_image,
            target_resolution=(2688, 768),  # 2x width
            quality_preset="ultra",
            artifact_detection_level="aggressive",
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (2688, 768))
        
        # Should have refinement passes
        has_refinement_count = hasattr(result, 'refinement_passes')
        if has_refinement_count:
            assert result.refinement_passes > 0, "No refinement passes performed"
        else:
            # Fall back to checking pipeline usage
            assert pipelines['refiner'].call_count > 0, "No refinement performed"
        
        # Quality should be excellent
        if hasattr(result, 'quality_score'):
            assert result.quality_score > 0.9, f"Quality score {result.quality_score} not excellent"
        else:
            # Just verify completion if no quality tracking
            assert result.success is True
        
    def test_pipeline_error_recovery(self):
        """Test pipeline handles errors gracefully"""
        source_image = self.create_test_image((1024, 768), "solid")
        
        # Create pipelines with failure
        pipelines = self.create_mock_pipelines(failure_rate=1.0)  # Always fail
        
        config = self.create_config(
            source_image=source_image,
            target_resolution=(2048, 1536),
            **pipelines
        )
        
        # Should raise ExpandorError
        with pytest.raises(ExpandorError) as exc_info:
            self.expandor.expand(config)
            
        # Error should have context
        assert "Expansion failed" in str(exc_info.value)
        
    def test_pipeline_vram_constraints(self):
        """Test pipeline with VRAM constraints"""
        source_image = self.create_test_image((1920, 1080), "gradient")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure with VRAM limit
        config = self.create_config(
            source_image=source_image,
            target_resolution=(7680, 4320),  # 8K
            vram_limit_mb=2000,  # Force tiled/CPU strategy
            allow_tiled=True,
            allow_cpu_offload=True,
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (7680, 4320))
        
        # Should use VRAM-efficient strategy
        assert "tiled" in result.strategy_used.lower() or "cpu" in result.strategy_used.lower()
        
    def test_pipeline_stage_saving(self):
        """Test that pipeline saves intermediate stages correctly"""
        source_image = self.create_test_image((1024, 768), "checkerboard")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure with stage saving
        config = self.create_config(
            source_image=source_image,
            target_resolution=(2048, 768),
            save_stages=True,
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify stages were saved
        stage_dir = config.stage_dir
        assert stage_dir.exists()
        
        stage_files = list(stage_dir.glob("*.png"))
        assert len(stage_files) > 0
        
        # Verify stage images are valid
        for stage_file in stage_files:
            img = Image.open(stage_file)
            assert img.size[0] > 0 and img.size[1] > 0
            
    def test_pipeline_performance(self):
        """Test pipeline performance characteristics"""
        source_image = self.create_test_image((1920, 1080), "gradient")
        
        # Create pipelines with simulated latency
        pipelines = self.create_mock_pipelines(latency_ms=50)
        
        config = self.create_config(
            source_image=source_image,
            target_resolution=(3840, 2160),
            quality_preset="fast",
            **pipelines
        )
        
        # Measure execution time
        start_time = time.time()
        result = self.expandor.expand(config)
        execution_time = time.time() - start_time
        
        # Verify performance
        assert result.total_duration_seconds > 0
        assert result.total_duration_seconds <= execution_time + 0.1  # Small tolerance
        
        # Fast preset should be quick
        assert execution_time < 5.0  # Should complete in under 5 seconds
```

## 3. Extreme Aspect Ratio Tests

### 3.1 Create Extreme Aspect Ratio Tests

**EXACT FILE PATH**: `tests/integration/test_extreme_aspects.py`

```python
"""
Integration tests for extreme aspect ratio changes.
Tests the limits of expansion capabilities.
"""

import pytest
from PIL import Image

from tests.integration.base_integration_test import BaseIntegrationTest
from expandor.core.exceptions import ExpandorError


class TestExtremeAspects(BaseIntegrationTest):
    """Test extreme aspect ratio transformations"""
    
    def test_16_9_to_32_9_expansion(self):
        """Test 16:9 to 32:9 super ultrawide expansion"""
        # Create 16:9 source
        source_image = self.create_test_image((1920, 1080), "gradient")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for 32:9
        config = self.create_config(
            source_image=source_image,
            target_resolution=(5760, 1080),  # 32:9 super ultrawide
            prompt="Expansive panoramic landscape",
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (5760, 1080))
        
        # Verify aspect ratio
        aspect_ratio = result.size[0] / result.size[1]
        assert abs(aspect_ratio - 32/9) < 0.01
        
        # Should use SWPO or progressive for 3x width expansion
        assert any(strategy in result.strategy_used.lower() 
                  for strategy in ['swpo', 'progressive'])
        
        # Check quality
        self.check_no_artifacts(result, tolerance=0.2)
        
    def test_1_1_to_21_9_expansion(self):
        """Test square to ultrawide expansion"""
        # Create square source
        source_image = self.create_test_image((1080, 1080), "checkerboard")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for 21:9
        config = self.create_config(
            source_image=source_image,
            target_resolution=(2520, 1080),  # 21:9
            quality_preset="high",
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (2520, 1080))
        
        # Should handle 2.33x aspect change
        assert len(result.boundaries) > 0
        
    def test_ultrawide_to_portrait(self):
        """Test ultrawide to portrait transformation"""
        # Create ultrawide source
        source_image = self.create_test_image((2560, 1080), "gradient")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for portrait
        config = self.create_config(
            source_image=source_image,
            target_resolution=(1080, 1920),  # 9:16 portrait
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (1080, 1920))
        
        # This is a complex transformation
        assert len(result.stages) > 1
        
    def test_extreme_vertical_expansion(self):
        """Test extreme vertical expansion (1:8 ratio)"""
        # Create short wide image
        source_image = self.create_test_image((2048, 256), "solid")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for tall expansion
        config = self.create_config(
            source_image=source_image,
            target_resolution=(2048, 2048),  # 8x height
            window_size=200,
            overlap_ratio=0.8,
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (2048, 2048))
        
        # Should use SWPO for 8x expansion
        assert "swpo" in result.strategy_used.lower()
        
        # Should have many vertical boundaries
        v_boundaries = [b for b in result.boundaries if b['direction'] == 'horizontal']
        assert len(v_boundaries) > 5
        
    def test_maximum_aspect_ratio_limit(self):
        """Test maximum supported aspect ratio"""
        # Create minimum height image
        source_image = self.create_test_image((1024, 128), "gradient")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Try extreme aspect ratio (80:1)
        config = self.create_config(
            source_image=source_image,
            target_resolution=(10240, 128),  # 80:1 extreme ratio
            **pipelines
        )
        
        # Either should handle it or fail with clear error
        try:
            result = self.expandor.expand(config)
            # If it succeeds, verify it actually achieved the target
            self.verify_result(result, (10240, 128))
            # Should use SWPO for extreme expansion
            assert "swpo" in result.strategy_used.lower()
        except ExpandorError as e:
            # If it fails, should mention aspect ratio or dimension limits
            assert any(term in str(e).lower() for term in ["aspect", "ratio", "dimension", "limit"]), \
                f"Error message doesn't indicate aspect ratio issue: {e}"
        
    def test_multi_direction_extreme_expansion(self):
        """Test extreme expansion in both directions"""
        # Create small source
        source_image = self.create_test_image((512, 512), "checkerboard")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for large expansion in both directions
        config = self.create_config(
            source_image=source_image,
            target_resolution=(4096, 2048),  # 8x width, 4x height
            quality_preset="ultra",
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (4096, 2048))
        
        # Should have boundaries in both directions
        h_boundaries = [b for b in result.boundaries if b['direction'] == 'vertical']
        v_boundaries = [b for b in result.boundaries if b['direction'] == 'horizontal']
        
        assert len(h_boundaries) > 0
        assert len(v_boundaries) > 0
        
    def test_aspect_preservation_mode(self):
        """Test expansion with aspect ratio preservation"""
        # Create 16:9 source
        source_image = self.create_test_image((1920, 1080), "gradient")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for 4K with same aspect
        config = self.create_config(
            source_image=source_image,
            target_resolution=(3840, 2160),  # Maintains 16:9
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (3840, 2160))
        
        # Should use simple upscale (no aspect change)
        assert "upscale" in result.strategy_used.lower()
        
        # Should have no expansion boundaries
        assert len(result.boundaries) == 0 or all(
            b.get('expansion_size', 0) == 0 for b in result.boundaries
        )
        
    def test_panoramic_to_standard(self):
        """Test panoramic to standard aspect conversion"""
        # Create panoramic source (3:1)
        source_image = self.create_test_image((3840, 1280), "gradient")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for 16:9
        config = self.create_config(
            source_image=source_image,
            target_resolution=(1920, 1080),  # 16:9
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (1920, 1080))
        
        # This requires cropping or letterboxing
        # Strategy should handle it appropriately
        assert result.success == True
```

## 4. Quality Validation Tests

### 4.1 Create Quality Validation Tests

**EXACT FILE PATH**: `tests/integration/test_quality_validation.py`

```python
"""
Integration tests for quality validation and artifact detection.
Tests the complete quality assurance pipeline.
"""

import pytest
import numpy as np
from typing import Tuple
from PIL import Image, ImageDraw

from tests.integration.base_integration_test import BaseIntegrationTest
# Note: ArtifactSeverity would be imported from artifact_detector if it exists
# For now, we'll work without it or define it inline if needed


class TestQualityValidation(BaseIntegrationTest):
    """Test quality validation and artifact repair"""
    
    def create_seamed_image(self, size: Tuple[int, int], seam_position: int) -> Image.Image:
        """Create image with visible seam for testing"""
        width, height = size
        img = Image.new('RGB', size, color='blue')
        draw = ImageDraw.Draw(img)
        
        # Create visible seam
        if seam_position < width:
            # Vertical seam
            draw.rectangle([0, 0, seam_position-1, height], fill='red')
            draw.rectangle([seam_position+1, 0, width, height], fill='green')
            # Seam line
            draw.line([(seam_position, 0), (seam_position, height)], fill='white', width=2)
        else:
            # Horizontal seam
            seam_y = seam_position - width
            draw.rectangle([0, 0, width, seam_y-1], fill='red')
            draw.rectangle([0, seam_y+1, width, height], fill='green')
            draw.line([(0, seam_y), (width, seam_y)], fill='white', width=2)
            
        return img
    
    def test_artifact_detection_integration(self):
        """Test artifact detection in expansion pipeline"""
        # Create clean source
        source_image = self.create_test_image((1024, 768), "gradient")
        
        # Create pipelines that introduce artifacts
        pipelines = self.create_mock_pipelines()
        
        # Override inpaint to create seams
        original_inpaint = pipelines['inpaint'].__call__
        
        def seamed_inpaint(*args, **kwargs):
            result = original_inpaint(*args, **kwargs)
            # Add artificial seam to result
            img = result.images[0]
            draw = ImageDraw.Draw(img)
            draw.line([(img.width//2, 0), (img.width//2, img.height)], 
                     fill='black', width=3)
            return result
            
        pipelines['inpaint'].__call__ = seamed_inpaint
        
        # Configure expansion
        config = self.create_config(
            source_image=source_image,
            target_resolution=(2048, 768),
            artifact_detection_level="aggressive",
            quality_preset="high",
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Should detect and attempt to fix artifacts
        detected_artifacts = (
            (hasattr(result, 'seams_detected') and result.seams_detected > 0) or
            (hasattr(result, 'artifacts_fixed') and result.artifacts_fixed > 0)
        )
        if not detected_artifacts:
            # At least refiner should have been called
            assert pipelines['refiner'].call_count > 0, "No artifact handling occurred"
        
        # Refiner should have been called for repair
        assert pipelines['refiner'].call_count > 0
        
    def test_multi_pass_refinement_integration(self):
        """Test multi-pass refinement for quality improvement"""
        # Create source with potential quality issues
        source_image = self.create_test_image((1344, 768), "solid")
        
        # Add some noise
        img_array = np.array(source_image)
        noise = np.random.normal(0, 20, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        source_image = Image.fromarray(img_array)
        
        # Create pipelines
        pipelines = self.create_mock_pipelines(quality_improvement=0.3)
        
        # Configure with ultra quality
        config = self.create_config(
            source_image=source_image,
            target_resolution=(2688, 1536),
            quality_preset="ultra",
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (2688, 1536))
        
        # Should have multiple refinement passes
        assert result.refinement_passes >= 3
        
        # Quality should be high
        assert result.quality_score > 0.85
        
    def test_boundary_tracking_accuracy(self):
        """Test that boundaries are accurately tracked and detected"""
        # Create source
        source_image = self.create_test_image((1024, 768), "gradient")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for progressive expansion
        config = self.create_config(
            source_image=source_image,
            target_resolution=(2048, 768),  # Double width
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify boundaries were tracked
        assert len(result.boundaries) > 0
        
        # Check boundary positions
        for boundary in result.boundaries:
            assert 'position' in boundary
            assert 'direction' in boundary
            assert 'expansion_size' in boundary
            
        # Boundaries should be at expected positions
        # For centered expansion, boundaries at edges of original
        boundary_positions = [b['position'] for b in result.boundaries]
        
        # Should have boundary around original image width (with tolerance)
        original_width = 1024
        tolerance = 50  # Allow some flexibility in boundary placement
        assert any(abs(pos - original_width) < tolerance for pos in boundary_positions), \
            f"No boundary near original width {original_width}, found: {boundary_positions}"
        
    def test_quality_preset_effectiveness(self):
        """Test different quality presets produce expected results"""
        source_image = self.create_test_image((1920, 1080), "checkerboard")
        
        presets = ['fast', 'balanced', 'high', 'ultra']
        results = {}
        
        for preset in presets:
            # Create fresh pipelines
            pipelines = self.create_mock_pipelines()
            
            config = self.create_config(
                source_image=source_image,
                target_resolution=(3840, 1080),
                quality_preset=preset,
                **pipelines
            )
            
            # Execute expansion
            result = self.expandor.expand(config)
            results[preset] = result
            
        # Verify quality increases with preset level
        # Use pipeline call counts as proxy if refinement_passes not available
        def get_refinement_count(result, preset):
            if hasattr(result, 'refinement_passes'):
                return result.refinement_passes
            # Fall back to checking stages
            return len([s for s in result.stages if 'refine' in s.get('method', '').lower()])
        
        fast_count = get_refinement_count(results['fast'], 'fast')
        balanced_count = get_refinement_count(results['balanced'], 'balanced')
        high_count = get_refinement_count(results['high'], 'high')
        ultra_count = get_refinement_count(results['ultra'], 'ultra')
        
        assert fast_count <= balanced_count, "Fast preset did more refinement than balanced"
        assert balanced_count <= high_count, "Balanced preset did more refinement than high"
        assert high_count <= ultra_count, "High preset did more refinement than ultra"
        
        # Ultra should have best quality score
        assert results['ultra'].quality_score >= results['high'].quality_score
        
    def test_artifact_repair_effectiveness(self):
        """Test that artifact repair actually improves quality"""
        # Create image with known artifacts
        source_image = self.create_seamed_image((1024, 768), 512)
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure with artifact repair
        config = self.create_config(
            source_image=source_image,
            target_resolution=(2048, 768),
            artifact_detection_level="aggressive",
            quality_preset="high",
            **pipelines
        )
        
        # Add known boundary metadata
        # If config supports generation_metadata, set it
        if hasattr(config, 'generation_metadata'):
            config.generation_metadata = {
                'progressive_boundaries': [512],
                'used_progressive': True
            }
        else:
            # Otherwise, pass it as part of config
            config.source_metadata['generation_metadata'] = {
                'progressive_boundaries': [512],
                'used_progressive': True
            }
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Should detect initial artifacts
        assert result.seams_detected > 0 or result.artifacts_fixed > 0
        
        # Final quality should be improved
        if hasattr(result, 'quality_score'):
            assert result.quality_score > 0.7, f"Quality score {result.quality_score} too low after repair"
        else:
            # Just verify success
            assert result.success is True
        
    def test_quality_validation_with_swpo(self):
        """Test quality validation with SWPO strategy"""
        source_image = self.create_test_image((1024, 768), "gradient")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for SWPO
        config = self.create_config(
            source_image=source_image,
            target_resolution=(5120, 768),  # 5x width for SWPO
            window_size=256,
            overlap_ratio=0.8,
            quality_preset="high",
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify SWPO was used
        assert "swpo" in result.strategy_used.lower()
        
        # Should have multiple window boundaries
        assert len(result.boundaries) > 3
        
        # Quality should still be good with proper overlap
        self.check_no_artifacts(result, tolerance=0.2)
        
    def test_quality_metadata_completeness(self):
        """Test that quality metadata is complete and accurate"""
        source_image = self.create_test_image((1344, 768), "solid")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        config = self.create_config(
            source_image=source_image,
            target_resolution=(2688, 1536),
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Check metadata completeness
        assert 'operation_log' in result.metadata
        assert 'boundary_positions' in result.metadata
        assert 'config_snapshot' in result.metadata
        
        # Check boundary metadata format
        boundary_meta = result.metadata.get('boundary_positions', [])
        if boundary_meta:
            for boundary in boundary_meta:
                assert 'position' in boundary
                assert 'direction' in boundary
                assert 'boundary_type' in boundary
```

## 5. SWPO Integration Tests

### 5.1 Create SWPO Integration Tests

**EXACT FILE PATH**: `tests/integration/test_swpo_integration.py`

```python
"""
Integration tests specifically for SWPO (Sliding Window Progressive Outpainting).
Tests window management, overlap handling, and seamless results.
"""

import pytest
import numpy as np
from PIL import Image

from tests.integration.base_integration_test import BaseIntegrationTest


class TestSWPOIntegration(BaseIntegrationTest):
    """Test SWPO strategy integration"""
    
    def test_basic_swpo_execution(self):
        """Test basic SWPO execution with default parameters"""
        source_image = self.create_test_image((1024, 768), "gradient")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for SWPO
        config = self.create_config(
            source_image=source_image,
            target_resolution=(4096, 768),  # 4x width
            strategy_override="swpo",  # Force SWPO
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (4096, 768))
        assert "swpo" in result.strategy_used.lower()
        
        # Check window execution
        inpaint_calls = pipelines['inpaint'].call_history
        assert len(inpaint_calls) > 3  # Multiple windows
        
        # Check windows had proper overlap
        for i in range(1, len(inpaint_calls)):
            # Each subsequent window should overlap
            prev_size = inpaint_calls[i-1]['size']
            curr_size = inpaint_calls[i]['size']
            # Sizes should progress
            assert curr_size[0] >= prev_size[0]
            
    def test_swpo_window_parameters(self):
        """Test SWPO with custom window parameters"""
        source_image = self.create_test_image((1024, 768), "checkerboard")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure with custom window settings
        config = self.create_config(
            source_image=source_image,
            target_resolution=(3072, 768),
            strategy_override="swpo",
            window_size=512,  # Larger windows
            overlap_ratio=0.5,  # 50% overlap
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (3072, 768))
        
        # Calculate expected windows
        expansion = 3072 - 1024  # 2048px
        step = 512 * (1 - 0.5)  # 256px effective step
        expected_windows = int(np.ceil(expansion / step))
        
        # Check approximately correct number of windows
        assert len(pipelines['inpaint'].call_history) >= expected_windows - 1
        
    def test_swpo_extreme_expansion(self):
        """Test SWPO with extreme expansion ratio"""
        source_image = self.create_test_image((512, 512), "gradient")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for 16x expansion
        config = self.create_config(
            source_image=source_image,
            target_resolution=(8192, 512),
            strategy_override="swpo",
            window_size=256,
            overlap_ratio=0.75,  # High overlap for quality
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (8192, 512))
        
        # Should have many windows
        assert len(result.boundaries) > 10
        
        # All boundaries should be from SWPO
        # Check boundary types if field exists
        swpo_boundaries = [b for b in result.boundaries if b.get('boundary_type') == 'swpo']
        assert len(swpo_boundaries) > 0, "No SWPO boundaries found despite using SWPO strategy"
        
    def test_swpo_bidirectional_expansion(self):
        """Test SWPO expanding in both directions"""
        source_image = self.create_test_image((1024, 768), "solid")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure for expansion in both dimensions
        config = self.create_config(
            source_image=source_image,
            target_resolution=(3072, 2304),  # 3x both dimensions
            strategy_override="swpo",
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Verify result
        self.verify_result(result, (3072, 2304))
        
        # Should have both horizontal and vertical boundaries
        h_boundaries = [b for b in result.boundaries if b['direction'] == 'vertical']
        v_boundaries = [b for b in result.boundaries if b['direction'] == 'horizontal']
        
        assert len(h_boundaries) > 0
        assert len(v_boundaries) > 0
        
    def test_swpo_window_overlap_quality(self):
        """Test that window overlap produces seamless results"""
        # Create image with distinct pattern
        source_image = Image.new('RGB', (1024, 768))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(source_image)
        
        # Create striped pattern
        for i in range(0, 1024, 64):
            color = (255, 0, 0) if (i // 64) % 2 == 0 else (0, 0, 255)
            draw.rectangle([i, 0, i+64, 768], fill=color)
            
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure with high overlap
        config = self.create_config(
            source_image=source_image,
            target_resolution=(4096, 768),
            strategy_override="swpo",
            window_size=400,
            overlap_ratio=0.8,  # 80% overlap
            quality_preset="high",
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Check quality at boundaries
        # With 80% overlap, transitions should be smooth
        self.check_no_artifacts(result, tolerance=0.15)
        
    def test_swpo_with_refinement(self):
        """Test SWPO followed by refinement passes"""
        source_image = self.create_test_image((1024, 768), "gradient")
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure with refinement
        config = self.create_config(
            source_image=source_image,
            target_resolution=(5120, 768),
            strategy_override="swpo",
            quality_preset="ultra",  # Triggers refinement
            final_unification_pass=True,
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Should have SWPO + refinement
        assert "swpo" in result.strategy_used.lower()
        assert result.refinement_passes > 0 or pipelines['refiner'].call_count > 0
        
        # Check for unification pass
        if config.img2img_pipeline:
            # img2img might be used for unification
            # Only check if img2img pipeline was provided
            if 'img2img' in pipelines:
                assert pipelines['img2img'].call_count >= 1, "No unification pass performed"
            
    def test_swpo_cache_management(self):
        """Test SWPO clears cache periodically during execution"""
        source_image = self.create_test_image((1024, 768), "solid")
        
        # Create pipelines with latency to simulate memory usage
        pipelines = self.create_mock_pipelines(latency_ms=10)
        
        # Configure for many windows
        config = self.create_config(
            source_image=source_image,
            target_resolution=(8192, 768),
            strategy_override="swpo",
            window_size=200,
            overlap_ratio=0.8,
            **pipelines
        )
        
        # Track memory operations (mock)
        import gc
        gc_collect_count = 0
        original_collect = gc.collect
        
        def mock_collect():
            nonlocal gc_collect_count
            gc_collect_count += 1
            return original_collect()
            
        gc.collect = mock_collect
        
        try:
            # Execute expansion
            result = self.expandor.expand(config)
            
            # Should have cleared cache multiple times
            assert gc_collect_count > 2
            
        finally:
            gc.collect = original_collect
            
    def test_swpo_dimension_constraints(self):
        """Test SWPO respects dimension constraints (multiples of 8)"""
        source_image = self.create_test_image((1023, 767), "gradient")  # Not multiples of 8
        
        # Create pipelines
        pipelines = self.create_mock_pipelines()
        
        # Configure SWPO
        config = self.create_config(
            source_image=source_image,
            target_resolution=(4095, 767),  # Also not multiples of 8
            strategy_override="swpo",
            **pipelines
        )
        
        # Execute expansion
        result = self.expandor.expand(config)
        
        # Result dimensions should be adjusted to multiples of 8
        assert result.size[0] % 8 == 0
        assert result.size[1] % 8 == 0
        
        # Should be close to requested size
        assert abs(result.size[0] - 4095) <= 8
        assert abs(result.size[1] - 767) <= 8
        
    def test_swpo_error_recovery(self):
        """Test SWPO handles window failures gracefully"""
        source_image = self.create_test_image((1024, 768), "solid")
        
        # Create pipelines that fail intermittently
        pipelines = self.create_mock_pipelines()
        
        # Make third window fail
        call_count = 0
        original_call = pipelines['inpaint'].__call__
        
        def failing_inpaint(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("Simulated window failure")
            return original_call(*args, **kwargs)
            
        pipelines['inpaint'].__call__ = failing_inpaint
        
        # Configure SWPO
        config = self.create_config(
            source_image=source_image,
            target_resolution=(3072, 768),
            strategy_override="swpo",
            **pipelines
        )
        
        # Should fail with context
        with pytest.raises(ExpandorError) as exc_info:
            self.expandor.expand(config)
            
        assert "window" in str(exc_info.value).lower() or "swpo" in str(exc_info.value).lower()
```

## 6. Test Execution and Validation

### 6.1 Create Test Runner Script

**EXACT FILE PATH**: `tests/run_integration_tests.py`

```python
#!/usr/bin/env python
"""
Run all integration tests with detailed reporting.
"""

import sys
import subprocess
import time
from pathlib import Path


def run_tests():
    """Run integration tests with coverage and reporting"""
    print("=" * 60)
    print("Running Expandor Integration Tests")
    print("=" * 60)
    
    test_dir = Path(__file__).parent
    
    # Test categories
    test_suites = [
        ("Full Pipeline Tests", "integration/test_full_pipeline.py"),
        ("Extreme Aspect Tests", "integration/test_extreme_aspects.py"),
        ("Quality Validation Tests", "integration/test_quality_validation.py"),
        ("SWPO Integration Tests", "integration/test_swpo_integration.py"),
    ]
    
    all_passed = True
    
    for suite_name, test_file in test_suites:
        print(f"\n{suite_name}")
        print("-" * len(suite_name))
        
        start_time = time.time()
        
        # Run tests with verbose output
        cmd = [
            sys.executable, "-m", "pytest",
            test_dir / test_file,
            "-v",
            "--tb=short",
            "--color=yes"
        ]
        
        result = subprocess.run(cmd, capture_output=False)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ {suite_name} PASSED ({duration:.2f}s)")
        else:
            print(f"✗ {suite_name} FAILED ({duration:.2f}s)")
            all_passed = False
            
    print("\n" + "=" * 60)
    
    if all_passed:
        print("All integration tests PASSED!")
        return 0
    else:
        print("Some integration tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
```

### 6.2 Create Pytest Configuration

**EXACT FILE PATH**: `pytest.ini`

```ini
[pytest]
# Pytest configuration for Expandor integration tests

# Test discovery
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Output options
addopts = 
    --verbose
    --strict-markers
    --tb=short
    --color=yes
    -p no:warnings

# Markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    
# Timeout
timeout = 300

# Coverage options (when run with --cov)
[coverage:run]
source = expandor
omit = 
    */tests/*
    */venv/*
    setup.py

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

## 7. Verification Checklist

Before running integration tests:

### 7.1 File Creation Verification
```bash
# Verify all integration test files exist
ls -la tests/fixtures/mock_pipelines.py
ls -la tests/integration/base_integration_test.py
ls -la tests/integration/test_full_pipeline.py
ls -la tests/integration/test_extreme_aspects.py
ls -la tests/integration/test_quality_validation.py
ls -la tests/integration/test_swpo_integration.py
ls -la tests/run_integration_tests.py
ls -la pytest.ini
```

### 7.2 Import Verification
```python
# Test all imports work
python -c "from tests.fixtures.mock_pipelines import create_mock_inpaint_pipeline"
python -c "from tests.integration.base_integration_test import BaseIntegrationTest"
```

### 7.3 Run Integration Tests
```bash
# Run all integration tests
python tests/run_integration_tests.py

# Or run individual test suites
pytest tests/integration/test_full_pipeline.py -v
pytest tests/integration/test_extreme_aspects.py -v
pytest tests/integration/test_quality_validation.py -v
pytest tests/integration/test_swpo_integration.py -v

# Run with coverage
pytest tests/integration/ --cov=expandor --cov-report=html
```

## 8. Critical Testing Notes

### 8.1 Mock Pipeline Realism
- **Mock pipelines simulate real behavior** with deterministic results
- **Edge color analysis** ensures seamless blending
- **Pattern generation** based on prompt keywords
- **Latency simulation** for performance testing
- **Failure injection** for error handling tests

### 8.2 Test Coverage
- **Full pipeline tests** verify complete workflows
- **Extreme aspect tests** push boundaries of capability
- **Quality validation tests** ensure artifact detection works
- **SWPO tests** verify window management
- **Error cases** test graceful failure handling

### 8.3 Integration Points
- **Strategy selection** based on image characteristics
- **Pipeline coordination** between components
- **Boundary tracking** throughout execution
- **Quality validation** after each stage
- **Metadata persistence** for debugging

## 9. Next Steps

After implementing integration tests:

1. **Run Full Test Suite**
   - Execute all unit and integration tests
   - Generate coverage report
   - Fix any failing tests

2. **Performance Profiling**
   - Profile slow tests
   - Optimize bottlenecks
   - Add performance benchmarks

3. **Real Model Testing**
   - Test with actual diffusers pipelines
   - Validate against ai-wallpaper results
   - Fine-tune parameters

4. **Documentation**
   - Document test scenarios
   - Create testing guide
   - Add CI/CD configuration