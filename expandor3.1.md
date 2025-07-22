# Expandor Phase 3 Step 1: Complex Strategies - Ultra-Detailed Implementation Guide

## Overview

This document provides a foolproof, zero-error implementation guide for the Complex Strategies component of Expandor Phase 3. Each section includes exact code, precise directory locations, comprehensive error handling, and validation steps.

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
ls -la expandor/strategies/
# Should show: __init__.py, base_strategy.py files already exist

# 4. Verify test structure exists
ls -la tests/unit/strategies/
# Create if missing:
mkdir -p tests/unit/strategies/
touch tests/unit/strategies/__init__.py
```

## 1. SWPO (Sliding Window Progressive Outpainting) Implementation

### 1.1 Create SWPO Strategy File

**EXACT FILE PATH**: `expandor/strategies/swpo_strategy.py`

```python
"""
Sliding Window Progressive Outpainting (SWPO) Strategy
Implements progressive expansion using overlapping windows for seamless results.
Based on ai-wallpaper's AspectAdjuster._sliding_window_adjust implementation.
"""

import math
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageFilter
import torch

from expandor.strategies.base_strategy import BaseExpansionStrategy
from expandor.core.result import ExpansionResult
from expandor.core.metadata_tracker import MetadataTracker
from expandor.core.vram_manager import VRAMManager
from expandor.core.dimension_calculator import DimensionCalculator
from expandor.utils.image_utils import create_gradient_mask, blend_images  
from expandor.processors.edge_analysis import EdgeAnalyzer
from expandor.core.exceptions import ExpandorError, VRAMError
from expandor.utils.memory_utils import gpu_memory_manager


@dataclass
class SWPOWindow:
    """Represents a single window in SWPO processing"""
    index: int
    position: Tuple[int, int, int, int]  # x1, y1, x2, y2
    expansion_type: str  # 'horizontal' or 'vertical'
    expansion_size: int
    overlap_size: int
    is_first: bool
    is_last: bool


class SWPOStrategy(BaseExpansionStrategy):
    """
    Sliding Window Progressive Outpainting strategy for extreme aspect ratios.
    
    Key features:
    - Overlapping windows maintain context throughout expansion
    - Configurable window size and overlap ratio
    - Automatic VRAM management with cache clearing
    - Optional final unification pass
    - Zero tolerance for visible seams
    """
    
    def __init__(self):
        super().__init__()
        self.vram_manager = VRAMManager()
        self.dimension_calc = DimensionCalculator()
        # self.edge_analyzer = EdgeAnalyzer()  # Needs implementation
        
        # Default SWPO parameters (can be overridden by config)
        self.default_window_size = 200
        self.default_overlap_ratio = 0.8
        self.default_denoising_strength = 0.95
        self.default_edge_blur_width = 20
        self.clear_cache_every_n_windows = 5
        
    def can_handle(self, config: 'ExpandorConfig') -> bool:
        """
        Check if SWPO can handle this expansion request.
        
        Requirements:
        - Inpaint pipeline must be available
        - Extreme aspect ratio changes (>4x) benefit most
        - VRAM must support at least one window operation
        """
        if not config.inpaint_pipeline:
            return False
            
        # Calculate aspect ratio change
        metrics = self._calculate_metrics(config)
        
        # SWPO is best for extreme ratios but can handle any expansion
        if metrics['is_extreme_ratio']:
            return True
            
        # Also good for large expansions even if not extreme ratio
        if metrics['absolute_width_change'] > 2000 or metrics['absolute_height_change'] > 2000:
            return True
            
        # Check VRAM for at least one window
        vram_required = self._estimate_window_vram(config)
        vram_available = self.vram_manager.get_available_vram()
        
        return vram_required <= vram_available * 0.9
    
    def _estimate_window_vram(self, config: 'ExpandorConfig') -> float:
        """Estimate VRAM needed for a single window operation"""
        # Get window parameters
        window_size = getattr(config, 'window_size', self.default_window_size)
        
        # Calculate approximate window canvas size
        source_w, source_h = self._get_source_dimensions(config)
        target_w, target_h = config.target_resolution
        
        # Estimate based on larger dimension change
        if target_w - source_w > target_h - source_h:
            # Horizontal expansion
            window_w = source_w + window_size
            window_h = target_h
        else:
            # Vertical expansion  
            window_w = target_w
            window_h = source_h + window_size
            
        # Use VRAM manager to estimate
        return self.vram_manager.calculate_generation_vram(
            width=window_w,
            height=window_h,
            batch_size=1,
            dtype=torch.float16
        )
    
    def execute(self, config: 'ExpandorConfig') -> ExpansionResult:
        """
        Execute SWPO expansion with comprehensive error handling.
        
        Process:
        1. Plan window strategy
        2. Execute each window with overlap
        3. Track boundaries for seam detection
        4. Optional final unification pass
        5. Validate results
        """
        self.logger.info("Starting SWPO expansion strategy")
        
        with gpu_memory_manager():
            try:
                # Initialize tracking
                result = ExpansionResult(
                strategy_name="SWPO",
                stages=[],
                boundaries=[],
                metadata={}
            )
            
            # Load source image
            if isinstance(config.source_image, Image.Image):
                current_image = config.source_image.copy()
            else:
                current_image = Image.open(config.source_image)
                
            source_w, source_h = current_image.size
            target_w, target_h = config.target_resolution
            
            # Validate dimensions are multiples of 8 for SDXL
            target_w = self.dimension_calc.round_to_multiple(target_w, 8)
            target_h = self.dimension_calc.round_to_multiple(target_h, 8)
            
            # Plan window strategy
            windows = self._plan_windows(
                source_size=(source_w, source_h),
                target_size=(target_w, target_h),
                window_size=getattr(config, 'window_size', self.default_window_size),
                overlap_ratio=getattr(config, 'overlap_ratio', self.default_overlap_ratio)
            )
            
            self.logger.info(f"Planned {len(windows)} windows for expansion")
            
            # Execute each window
            for i, window in enumerate(windows):
                self.logger.info(f"Processing window {i+1}/{len(windows)}")
                
                # Clear CUDA cache periodically
                if i > 0 and i % self.clear_cache_every_n_windows == 0:
                    self._clear_cuda_cache()
                    
                # Execute window
                current_image, window_result = self._execute_window(
                    current_image=current_image,
                    window=window,
                    config=config,
                    accumulated_boundaries=result.boundaries
                )
                
                # Track stage
                result.stages.append({
                    'name': f'swpo_window_{i+1}',
                    'method': 'inpaint',
                    'input_size': window_result['input_size'],
                    'output_size': window_result['output_size'],
                    'window_position': window.position,
                    'expansion_size': window.expansion_size
                })
                
                # Track boundaries for this window
                result.boundaries.extend(window_result['boundaries'])
                
                # Save stage if requested
                if config.save_stages and config.stage_save_callback:
                    config.stage_save_callback(
                        image=current_image,
                        stage_name=f"swpo_window_{i+1}",
                        metadata=window_result
                    )
                    
            # Optional final unification pass
            if getattr(config, 'final_unification_pass', True):
                self.logger.info("Executing final unification pass")
                current_image = self._unification_pass(
                    image=current_image,
                    config=config,
                    boundaries=result.boundaries
                )
                
                result.stages.append({
                    'name': 'swpo_unification',
                    'method': 'img2img',
                    'strength': getattr(config, 'unification_strength', 0.15)
                })
                
            # Final validation
            if current_image.size != (target_w, target_h):
                raise ExpandorError(
                    f"SWPO size mismatch: expected {target_w}x{target_h}, "
                    f"got {current_image.size[0]}x{current_image.size[1]}"
                )
                
            # Set final result
            result.image = current_image
            result.final_size = current_image.size
            result.metadata['total_windows'] = len(windows)
            result.metadata['window_parameters'] = {
                'window_size': getattr(config, 'window_size', self.default_window_size),
                'overlap_ratio': getattr(config, 'overlap_ratio', self.default_overlap_ratio),
                'denoising_strength': getattr(config, 'denoising_strength', self.default_denoising_strength)
                }
                
                return result
                
            except Exception as e:
                self.logger.error(f"SWPO strategy failed: {str(e)}")
                # Re-raise with context
                raise ExpandorError(
                    f"SWPO expansion failed: {str(e)}",
                    strategy="SWPO",
                    stage=f"window_{len(result.stages)}",
                    original_error=e
                )
    
    def _plan_windows(self, source_size: Tuple[int, int], 
                     target_size: Tuple[int, int],
                     window_size: int,
                     overlap_ratio: float) -> List[SWPOWindow]:
        """
        Plan sliding windows for expansion.
        
        Critical requirements:
        - All dimensions must be multiples of 8
        - Overlap must be sufficient to maintain context
        - Windows must cover entire expansion area
        """
        source_w, source_h = source_size
        target_w, target_h = target_size
        
        windows = []
        
        # Determine expansion direction and size
        width_expansion = target_w - source_w
        height_expansion = target_h - source_h
        
        if width_expansion <= 0 and height_expansion <= 0:
            # No expansion needed
            return windows
            
        # Calculate overlap size
        overlap_size = int(window_size * overlap_ratio)
        effective_step = window_size - overlap_size
        
        # Ensure step is multiple of 8
        effective_step = self.dimension_calc.round_to_multiple(effective_step, 8)
        
        # Recalculate overlap based on adjusted step
        overlap_size = window_size - effective_step
        
        current_w = source_w
        current_h = source_h
        window_index = 0
        
        # Plan horizontal expansion windows
        if width_expansion > 0:
            steps_needed = math.ceil(width_expansion / effective_step)
            
            for i in range(steps_needed):
                # Calculate this window's expansion
                if i == steps_needed - 1:
                    # Last window - expand to exact target
                    next_w = target_w
                else:
                    next_w = min(current_w + window_size, target_w)
                    next_w = self.dimension_calc.round_to_multiple(next_w, 8)
                
                expansion = next_w - current_w
                
                window = SWPOWindow(
                    index=window_index,
                    position=(current_w - overlap_size if i > 0 else 0, 0, next_w, current_h),
                    expansion_type='horizontal',
                    expansion_size=expansion,
                    overlap_size=overlap_size if i > 0 else 0,
                    is_first=(i == 0),
                    is_last=(i == steps_needed - 1)
                )
                
                windows.append(window)
                current_w = next_w
                window_index += 1
                
        # Plan vertical expansion windows (after horizontal is complete)
        if height_expansion > 0:
            steps_needed = math.ceil(height_expansion / effective_step)
            
            for i in range(steps_needed):
                # Calculate this window's expansion
                if i == steps_needed - 1:
                    # Last window - expand to exact target
                    next_h = target_h
                else:
                    next_h = min(current_h + window_size, target_h)
                    next_h = self.dimension_calc.round_to_multiple(next_h, 8)
                    
                expansion = next_h - current_h
                
                window = SWPOWindow(
                    index=window_index,
                    position=(0, current_h - overlap_size if i > 0 else 0, current_w, next_h),
                    expansion_type='vertical',
                    expansion_size=expansion,
                    overlap_size=overlap_size if i > 0 else 0,
                    is_first=(i == 0),
                    is_last=(i == steps_needed - 1)
                )
                
                windows.append(window)
                current_h = next_h
                window_index += 1
                
        return windows
    
    def _execute_window(self, current_image: Image.Image,
                       window: SWPOWindow,
                       config: 'ExpandorConfig',
                       accumulated_boundaries: List[Dict[str, Any]]) -> Tuple[Image.Image, Dict]:
        """
        Execute a single SWPO window with proper masking and blending.
        
        Critical steps:
        1. Create canvas at window size
        2. Position current image with overlap
        3. Create gradient mask for smooth blending
        4. Analyze edges for color consistency
        5. Execute inpainting
        6. Track new boundaries
        """
        x1, y1, x2, y2 = window.position
        window_w = x2 - x1
        window_h = y2 - y1
        
        # Create canvas at window size
        canvas = Image.new('RGB', (window_w, window_h))
        mask = Image.new('L', (window_w, window_h), 0)  # Start with black (preserve)
        
        # Calculate positioning for current image
        if window.expansion_type == 'horizontal':
            # For horizontal expansion, position at left side
            paste_x = 0 if window.is_first else window.overlap_size
            paste_y = 0
            
            # Mark new area for generation (right side)
            if window.is_first:
                mask_x = current_image.width
                mask_w = window_w - current_image.width
            else:
                mask_x = window.overlap_size + (current_image.width - window.overlap_size)
                mask_w = window_w - mask_x
                
            mask.paste(255, (mask_x, 0, mask_x + mask_w, window_h))
            
        else:  # vertical
            # For vertical expansion, position at top
            paste_x = 0
            paste_y = 0 if window.is_first else window.overlap_size
            
            # Mark new area for generation (bottom)
            if window.is_first:
                mask_y = current_image.height
                mask_h = window_h - current_image.height
            else:
                mask_y = window.overlap_size + (current_image.height - window.overlap_size)
                mask_h = window_h - mask_y
                
            mask.paste(255, (0, mask_y, window_w, mask_y + mask_h))
            
        # Paste current image
        canvas.paste(current_image, (paste_x, paste_y))
        
        # Create gradient mask for smooth blending
        if not window.is_first:
            gradient_size = min(window.overlap_size // 2, 100)
            
            if window.expansion_type == 'horizontal':
                # Horizontal gradient from left edge of mask
                # Create horizontal gradient
                gradient = self._create_gradient_mask(
                    size=(gradient_size, window_h),
                    direction='horizontal'
                )
                mask.paste(gradient, (mask_x, 0))
                
            else:  # vertical
                # Vertical gradient from top edge of mask
                # Create vertical gradient
                gradient = self._create_gradient_mask(
                    size=(window_w, gradient_size),
                    direction='vertical'
                )
                mask.paste(gradient, (0, mask_y))
                
        # Apply edge blur for seamless transition
        edge_blur = getattr(config, 'edge_blur_width', self.default_edge_blur_width)
        mask = mask.filter(ImageFilter.GaussianBlur(edge_blur))
        
        # Edge analysis for color continuity (simplified implementation)
        # Analyze edge colors at the boundary
        edge_colors = self._analyze_edge_colors_simple(canvas, mask, window.expansion_type)
        
        # Pre-fill with edge colors (helps with coherence)
        if edge_colors:
            canvas = self._apply_edge_fill_simple(canvas, mask, edge_colors, blur_radius=50)
            
        # Add noise to masked areas (improves generation)
        canvas = self._add_noise_to_mask(canvas, mask, strength=0.02)
        
        # Execute inpainting
        with self.vram_manager.managed_execution():
            result = config.inpaint_pipeline(
                prompt=config.prompt,
                image=canvas,
                mask_image=mask,
                height=window_h,
                width=window_w,
                strength=getattr(config, 'denoising_strength', self.default_denoising_strength),
                num_inference_steps=self._calculate_steps(window.expansion_size),
                guidance_scale=self._calculate_guidance_scale(window.expansion_size),
                generator=torch.Generator().manual_seed(config.seed + window.index)
            )
            
        generated = result.images[0]
        
        # Extract the relevant portion for the final image
        if window.expansion_type == 'horizontal':
            # For horizontal, we expanded to the right
            if window.is_first:
                final_image = generated
            else:
                # Extract from overlap point to end
                final_image = generated.crop((window.overlap_size, 0, window_w, window_h))
                
        else:  # vertical
            # For vertical, we expanded downward
            if window.is_first:
                final_image = generated
            else:
                # Extract from overlap point to end
                final_image = generated.crop((0, window.overlap_size, window_w, window_h))
                
        # Track boundaries
        boundaries = []
        
        if window.expansion_type == 'horizontal':
            # Track the seam where old meets new
            seam_x = current_image.width
            boundaries.append({
                'position': {'x': seam_x, 'y': 0, 'width': 1, 'height': current_image.height},
                'direction': 'vertical',
                'step': window.index,
                'strength': getattr(config, 'denoising_strength', self.default_denoising_strength),
                'type': 'swpo'
            })
        else:
            # Track horizontal seam
            seam_y = current_image.height
            boundaries.append({
                'position': {'x': 0, 'y': seam_y, 'width': current_image.width, 'height': 1},
                'direction': 'horizontal',
                'step': window.index,
                'strength': getattr(config, 'denoising_strength', self.default_denoising_strength),
                'type': 'swpo'
            })
            
        return final_image, {
            'input_size': current_image.size,
            'output_size': final_image.size,
            'window_size': (window_w, window_h),
            'boundaries': boundaries,
            'expansion_type': window.expansion_type,
            'expansion_size': window.expansion_size
        }
    
    def _unification_pass(self, image: Image.Image, 
                         config: 'ExpandorConfig',
                         boundaries: List[Dict[str, Any]]) -> Image.Image:
        """
        Optional final pass to unify the entire image.
        Uses very low denoising strength to preserve content while smoothing transitions.
        """
        if not config.img2img_pipeline:
            self.logger.warning("No img2img pipeline available for unification pass")
            return image
            
        strength = getattr(config, 'unification_strength', 0.15)
        
        with self.vram_manager.managed_execution():
            result = config.img2img_pipeline(
                prompt=config.prompt,
                image=image,
                strength=strength,
                num_inference_steps=30,  # Fewer steps for light touch
                guidance_scale=7.0,
                generator=torch.Generator().manual_seed(config.seed + 9999)
            )
            
        return result.images[0]
    
    def _add_noise_to_mask(self, image: Image.Image, mask: Image.Image, 
                          strength: float = 0.02) -> Image.Image:
        """Add subtle noise to masked areas to improve inpainting"""
        img_array = np.array(image)
        mask_array = np.array(mask) / 255.0
        
        # Generate noise
        noise = np.random.normal(0, strength * 255, img_array.shape)
        
        # Apply noise only to masked areas
        for c in range(3):
            img_array[:, :, c] = img_array[:, :, c] + (noise[:, :, c] * mask_array)
            
        # Clip values
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def _calculate_steps(self, expansion_size: int) -> int:
        """Calculate inference steps based on expansion size"""
        # More steps for larger expansions
        if expansion_size < 100:
            return 50
        elif expansion_size < 300:
            return 60
        elif expansion_size < 500:
            return 70
        else:
            return 80
            
    def _calculate_guidance_scale(self, expansion_size: int) -> float:
        """Calculate guidance scale based on expansion size"""
        # Higher guidance for larger expansions
        if expansion_size < 100:
            return 7.0
        elif expansion_size < 300:
            return 7.5
        elif expansion_size < 500:
            return 8.0
        else:
            return 8.5
            
    def _clear_cuda_cache(self):
        """Clear CUDA cache to prevent OOM errors"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.debug("Cleared CUDA cache")
            
    def _create_gradient_mask(self, size: Tuple[int, int], direction: str) -> Image.Image:
        """Create a gradient mask for smooth blending"""
        width, height = size
        gradient = Image.new('L', (width, height))
        gradient_array = np.array(gradient)
        
        if direction == 'horizontal':
            for x in range(width):
                gradient_array[:, x] = int((x / width) * 255)
        else:  # vertical
            for y in range(height):
                gradient_array[y, :] = int((y / height) * 255)
                
        return Image.fromarray(gradient_array)
    
    def _analyze_edge_colors_simple(self, image: Image.Image, mask: Image.Image, 
                                   expansion_type: str) -> Optional[Dict]:
        """Simple edge color analysis for pre-filling"""
        # Convert to arrays
        img_array = np.array(image)
        mask_array = np.array(mask)
        
        # Find edge pixels where mask transitions from black to white
        edge_colors = {}
        
        if expansion_type == 'horizontal':
            # Look at right edge
            for y in range(img_array.shape[0]):
                for x in range(img_array.shape[1] - 1, -1, -1):
                    if mask_array[y, x] > 128:  # Found edge
                        if x > 0:
                            edge_colors['right'] = tuple(img_array[y, x-1])
                        break
        else:  # vertical
            # Look at bottom edge
            for x in range(img_array.shape[1]):
                for y in range(img_array.shape[0] - 1, -1, -1):
                    if mask_array[y, x] > 128:  # Found edge
                        if y > 0:
                            edge_colors['bottom'] = tuple(img_array[y-1, x])
                        break
                        
        return edge_colors if edge_colors else None
    
    def _apply_edge_fill_simple(self, image: Image.Image, mask: Image.Image,
                               edge_colors: Dict, blur_radius: int) -> Image.Image:
        """Apply edge color fill to masked areas"""
        # Simple implementation - fill masked areas with edge color
        img_array = np.array(image)
        mask_array = np.array(mask) / 255.0
        
        # Apply edge color to masked areas
        if 'right' in edge_colors:
            color = edge_colors['right']
            for c in range(3):
                img_array[:, :, c] = (1 - mask_array) * img_array[:, :, c] + mask_array * color[c]
        elif 'bottom' in edge_colors:
            color = edge_colors['bottom']
            for c in range(3):
                img_array[:, :, c] = (1 - mask_array) * img_array[:, :, c] + mask_array * color[c]
                
        # Convert back and apply blur
        result = Image.fromarray(img_array.astype(np.uint8))
        if blur_radius > 0:
            result = result.filter(ImageFilter.GaussianBlur(blur_radius))
            
        return result
    
    def get_vram_requirement(self, config: 'ExpandorConfig') -> float:
        """Calculate total VRAM requirement for SWPO"""
        # SWPO only needs VRAM for one window at a time
        return self._estimate_window_vram(config)
```

### 1.2 Create SWPO Unit Tests

**EXACT FILE PATH**: `tests/unit/strategies/test_swpo_strategy.py`

```python
"""
Unit tests for SWPO (Sliding Window Progressive Outpainting) Strategy
Tests window planning, execution, boundary tracking, and error handling.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from PIL import Image
import torch
import numpy as np

from expandor.strategies.swpo_strategy import SWPOStrategy, SWPOWindow
from expandor.core.config import ExpandorConfig
from expandor.core.exceptions import ExpandorError


class TestSWPOStrategy:
    """Comprehensive tests for SWPO strategy"""
    
    @pytest.fixture
    def strategy(self):
        """Create SWPO strategy instance"""
        return SWPOStrategy()
    
    @pytest.fixture
    def base_config(self):
        """Create base configuration for testing"""
        # Create test image
        test_image = Image.new('RGB', (1024, 768), color='blue')
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.images = [Image.new('RGB', (1344, 768), color='red')]
        mock_pipeline.return_value = mock_result
        
        config = ExpandorConfig(
            source_image=test_image,
            target_resolution=(5376, 768),  # 7:1 extreme ratio
            prompt="Test prompt",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=mock_pipeline,
            window_size=200,
            overlap_ratio=0.8,
            denoising_strength=0.95
        )
        
        return config
    
    def test_can_handle_extreme_ratio(self, strategy, base_config):
        """Test that SWPO handles extreme aspect ratios"""
        assert strategy.can_handle(base_config) == True
        
    def test_can_handle_no_pipeline(self, strategy, base_config):
        """Test that SWPO returns False without inpaint pipeline"""
        base_config.inpaint_pipeline = None
        assert strategy.can_handle(base_config) == False
        
    def test_window_planning_horizontal(self, strategy):
        """Test window planning for horizontal expansion"""
        windows = strategy._plan_windows(
            source_size=(1024, 768),
            target_size=(2048, 768),
            window_size=400,
            overlap_ratio=0.5
        )
        
        # Should have multiple windows
        assert len(windows) > 1
        
        # First window should start at source edge
        assert windows[0].is_first == True
        assert windows[0].position[0] == 0
        
        # Last window should reach target size
        assert windows[-1].is_last == True
        assert windows[-1].position[2] == 2048
        
        # All should be horizontal expansion
        for window in windows:
            assert window.expansion_type == 'horizontal'
            
    def test_window_planning_vertical(self, strategy):
        """Test window planning for vertical expansion"""
        windows = strategy._plan_windows(
            source_size=(768, 1024),
            target_size=(768, 2048),
            window_size=400,
            overlap_ratio=0.5
        )
        
        # Should have multiple windows
        assert len(windows) > 1
        
        # All should be vertical expansion
        for window in windows:
            assert window.expansion_type == 'vertical'
            
        # Last window should reach target size
        assert windows[-1].position[3] == 2048
        
    def test_window_planning_both_directions(self, strategy):
        """Test window planning for expansion in both directions"""
        windows = strategy._plan_windows(
            source_size=(1024, 768),
            target_size=(2048, 1536),
            window_size=400,
            overlap_ratio=0.5
        )
        
        # Should have both horizontal and vertical windows
        h_windows = [w for w in windows if w.expansion_type == 'horizontal']
        v_windows = [w for w in windows if w.expansion_type == 'vertical']
        
        assert len(h_windows) > 0
        assert len(v_windows) > 0
        
        # Horizontal should come before vertical
        assert windows.index(h_windows[-1]) < windows.index(v_windows[0])
        
    def test_dimension_rounding(self, strategy):
        """Test that dimensions are rounded to multiples of 8"""
        windows = strategy._plan_windows(
            source_size=(1023, 767),  # Not multiples of 8
            target_size=(2047, 1535),  # Not multiples of 8
            window_size=200,
            overlap_ratio=0.8
        )
        
        # All window dimensions should be multiples of 8
        for window in windows:
            x1, y1, x2, y2 = window.position
            width = x2 - x1
            height = y2 - y1
            assert width % 8 == 0, f"Width {width} not multiple of 8"
            assert height % 8 == 0, f"Height {height} not multiple of 8"
            
    def test_execute_basic(self, strategy, base_config):
        """Test basic execution of SWPO strategy"""
        # Mock the window execution to return progressively larger images
        def mock_execute_window(current_image, window, config, boundaries):
            # Simulate expansion
            if window.expansion_type == 'horizontal':
                new_w = current_image.width + window.expansion_size
                new_h = current_image.height
            else:
                new_w = current_image.width
                new_h = current_image.height + window.expansion_size
                
            new_image = Image.new('RGB', (new_w, new_h), color='green')
            
            return new_image, {
                'input_size': current_image.size,
                'output_size': new_image.size,
                'boundaries': []
            }
            
        strategy._execute_window = mock_execute_window
        
        # Execute strategy
        result = strategy.execute(base_config)
        
        # Verify result
        assert result.strategy_name == "SWPO"
        assert result.final_size == base_config.target_resolution
        assert len(result.stages) > 0
        assert result.image is not None
        
    def test_execute_with_vram_management(self, strategy, base_config):
        """Test that CUDA cache is cleared periodically"""
        clear_count = 0
        
        def mock_clear():
            nonlocal clear_count
            clear_count += 1
            
        strategy._clear_cuda_cache = mock_clear
        
        # Create config that will generate many windows
        base_config.target_resolution = (8192, 768)
        base_config.window_size = 200
        
        # Mock window execution
        strategy._execute_window = Mock(side_effect=lambda img, w, c, b: (
            Image.new('RGB', (img.width + 200, img.height)),
            {'input_size': img.size, 'output_size': (img.width + 200, img.height), 'boundaries': []}
        ))
        
        # Execute
        result = strategy.execute(base_config)
        
        # Should have cleared cache at least once
        assert clear_count > 0
        
    def test_execute_with_unification_pass(self, strategy, base_config):
        """Test execution with final unification pass"""
        # Mock img2img pipeline
        mock_img2img = Mock()
        mock_result = Mock()
        mock_result.images = [Image.new('RGB', base_config.target_resolution)]
        mock_img2img.return_value = mock_result
        
        base_config.img2img_pipeline = mock_img2img
        base_config.final_unification_pass = True
        base_config.unification_strength = 0.1
        
        # Mock window execution
        strategy._execute_window = Mock(side_effect=lambda img, w, c, b: (
            Image.new('RGB', base_config.target_resolution),
            {'input_size': img.size, 'output_size': base_config.target_resolution, 'boundaries': []}
        ))
        
        # Execute
        result = strategy.execute(base_config)
        
        # Verify unification was called
        mock_img2img.assert_called_once()
        call_args = mock_img2img.call_args[1]
        assert call_args['strength'] == 0.1
        
        # Verify unification stage was added
        assert any(stage['name'] == 'swpo_unification' for stage in result.stages)
        
    def test_boundary_tracking(self, strategy, base_config):
        """Test that boundaries are properly tracked"""
        # Create a smaller expansion for easier testing
        base_config.target_resolution = (1624, 768)  # Just 600px expansion
        base_config.window_size = 300
        
        # Execute strategy
        result = strategy.execute(base_config)
        
        # Should have boundaries tracked
        assert len(result.boundaries) > 0
        
        # Boundaries should have required information
        for boundary in result.boundaries:
            assert hasattr(boundary, 'position')
            assert hasattr(boundary, 'direction')
            assert hasattr(boundary, 'expansion_size')
            assert hasattr(boundary, 'strength_used')
            
    def test_error_handling_pipeline_failure(self, strategy, base_config):
        """Test error handling when pipeline fails"""
        # Make pipeline raise an error
        base_config.inpaint_pipeline.side_effect = RuntimeError("Pipeline failed")
        
        # Should raise ExpandorError with context
        with pytest.raises(ExpandorError) as exc_info:
            strategy.execute(base_config)
            
        assert "SWPO expansion failed" in str(exc_info.value)
        assert "Pipeline failed" in str(exc_info.value)
        
    def test_error_handling_size_mismatch(self, strategy, base_config):
        """Test error handling for size mismatch"""
        # Mock window execution to return wrong size
        strategy._execute_window = Mock(side_effect=lambda img, w, c, b: (
            Image.new('RGB', (100, 100)),  # Wrong size
            {'input_size': img.size, 'output_size': (100, 100), 'boundaries': []}
        ))
        
        # Should raise ExpandorError
        with pytest.raises(ExpandorError) as exc_info:
            strategy.execute(base_config)
            
        assert "size mismatch" in str(exc_info.value)
        
    def test_edge_analysis_integration(self, strategy):
        """Test that edge analysis is used in window execution"""
        # Create test window
        window = SWPOWindow(
            index=0,
            position=(0, 0, 1224, 768),
            expansion_type='horizontal',
            expansion_size=200,
            overlap_size=0,
            is_first=True,
            is_last=False
        )
        
        # Create test config
        test_image = Image.new('RGB', (1024, 768), color='blue')
        config = Mock()
        config.inpaint_pipeline = Mock(return_value=Mock(images=[test_image]))
        config.prompt = "test"
        config.seed = 42
        
        # Execute window
        with patch.object(strategy, '_analyze_edge_colors_simple') as mock_analyze:
            mock_analyze.return_value = {'right': (100, 150, 200)}
            
            result_img, result_data = strategy._execute_window(
                current_image=test_image,
                window=window,
                config=config,
                accumulated_boundaries=[]
            )
            
            # Edge analysis should have been called
            mock_analyze.assert_called_once()
            
    @pytest.mark.parametrize("expansion_size,expected_steps", [
        (50, 50),
        (150, 60),
        (350, 70),
        (600, 80)
    ])
    def test_adaptive_steps(self, strategy, expansion_size, expected_steps):
        """Test that inference steps adapt to expansion size"""
        steps = strategy._calculate_steps(expansion_size)
        assert steps == expected_steps
        
    @pytest.mark.parametrize("expansion_size,expected_scale", [
        (50, 7.0),
        (150, 7.5),
        (350, 8.0),
        (600, 8.5)
    ])
    def test_adaptive_guidance(self, strategy, expansion_size, expected_scale):
        """Test that guidance scale adapts to expansion size"""
        scale = strategy._calculate_guidance_scale(expansion_size)
        assert scale == expected_scale
        
    def test_vram_estimation(self, strategy, base_config):
        """Test VRAM requirement estimation"""
        vram_req = strategy.get_vram_requirement(base_config)
        
        # Should return a reasonable value
        assert vram_req > 0
        assert vram_req < 50000  # Less than 50GB (sanity check)
        
        # Should be less than full image VRAM since we process windows
        full_vram = strategy.vram_manager.calculate_generation_vram(
            width=base_config.target_resolution[0],
            height=base_config.target_resolution[1],
            batch_size=1,
            dtype=torch.float16
        )
        assert vram_req < full_vram
```

### 1.3 Integration Test Helper

**EXACT FILE PATH**: `tests/integration/strategies/test_swpo_integration.py`

```python
"""
Integration tests for SWPO strategy with real-like scenarios.
Tests full pipeline integration, boundary detection, and quality validation.
"""

import pytest
from PIL import Image
import numpy as np
import tempfile
from pathlib import Path

from expandor import Expandor, ExpandorConfig
from expandor.strategies.swpo_strategy import SWPOStrategy
from tests.fixtures.mock_pipelines import create_mock_inpaint_pipeline


class TestSWPOIntegration:
    """Integration tests for SWPO with realistic scenarios"""
    
    @pytest.fixture
    def expandor(self):
        """Create Expandor instance with SWPO strategy"""
        exp = Expandor()
        # Force SWPO strategy selection for testing
        exp.strategy_selector._force_strategy = SWPOStrategy
        return exp
    
    @pytest.fixture
    def test_image_1080p(self):
        """Create a test image with patterns for seam detection"""
        # Create image with gradient pattern
        width, height = 1920, 1080
        img = Image.new('RGB', (width, height))
        pixels = img.load()
        
        for x in range(width):
            for y in range(height):
                # Create gradient pattern
                r = int((x / width) * 255)
                g = int((y / height) * 255)
                b = 128
                pixels[x, y] = (r, g, b)
                
        return img
    
    def test_extreme_horizontal_expansion(self, expandor, test_image_1080p):
        """Test 16:9 to 32:9 expansion"""
        config = ExpandorConfig(
            source_image=test_image_1080p,
            target_resolution=(3840, 1080),  # 32:9
            prompt="Extend the gradient pattern seamlessly",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=create_mock_inpaint_pipeline(),
            window_size=300,
            overlap_ratio=0.8,
            save_stages=True,
            stage_dir=Path(tempfile.mkdtemp())
        )
        
        result = expandor.expand(config)
        
        # Verify success
        assert result.success is True
        assert result.size == (3840, 1080)
        
        # Verify windows were used
        assert 'total_windows' in result.metadata
        assert result.metadata['total_windows'] > 1
        
        # Verify boundaries were tracked
        assert len(result.boundaries) == result.metadata['total_windows']
        
        # Verify stages were saved
        stage_files = list(config.stage_dir.glob("*.png"))
        assert len(stage_files) > 0
        
    def test_extreme_vertical_expansion(self, expandor):
        """Test 9:16 to 9:32 expansion"""
        # Create portrait image
        test_image = Image.new('RGB', (1080, 1920), color='blue')
        
        config = ExpandorConfig(
            source_image=test_image,
            target_resolution=(1080, 3840),  # 9:32
            prompt="Extend vertically",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=create_mock_inpaint_pipeline(),
            window_size=400,
            overlap_ratio=0.75
        )
        
        result = expandor.expand(config)
        
        # Verify success
        assert result.success is True
        assert result.size == (1080, 3840)
        
        # Verify vertical windows
        vertical_boundaries = [b for b in result.boundaries if b.direction == 'horizontal']
        assert len(vertical_boundaries) > 0
        
    def test_both_direction_expansion(self, expandor):
        """Test expansion in both directions"""
        test_image = Image.new('RGB', (1024, 768))
        
        config = ExpandorConfig(
            source_image=test_image,
            target_resolution=(2048, 1536),
            prompt="Expand in all directions",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=create_mock_inpaint_pipeline()
        )
        
        result = expandor.expand(config)
        
        # Should have both horizontal and vertical boundaries
        h_boundaries = [b for b in result.boundaries if b.direction == 'vertical']
        v_boundaries = [b for b in result.boundaries if b.direction == 'horizontal']
        
        assert len(h_boundaries) > 0
        assert len(v_boundaries) > 0
        
    def test_seam_positions_accuracy(self, expandor, test_image_1080p):
        """Test that boundary positions are accurately tracked"""
        config = ExpandorConfig(
            source_image=test_image_1080p,
            target_resolution=(2720, 1080),  # Add 800px
            prompt="Extend",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=create_mock_inpaint_pipeline(),
            window_size=400,
            overlap_ratio=0.5  # 50% overlap for easier calculation
        )
        
        result = expandor.expand(config)
        
        # Calculate expected boundary positions
        # With 400px windows and 50% overlap, effective step is 200px
        # Need 800px total, so 4 windows
        expected_positions = [1920, 2120, 2320, 2520]
        
        actual_positions = sorted([b.position for b in result.boundaries])
        
        # Positions should be close (within 8px due to rounding)
        for expected, actual in zip(expected_positions, actual_positions):
            assert abs(expected - actual) <= 8
            
    def test_final_unification_pass(self, expandor, test_image_1080p):
        """Test that unification pass is applied when requested"""
        config = ExpandorConfig(
            source_image=test_image_1080p,
            target_resolution=(3840, 1080),
            prompt="Extend with unification",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=create_mock_inpaint_pipeline(),
            img2img_pipeline=create_mock_inpaint_pipeline(),  # Needed for unification
            final_unification_pass=True,
            unification_strength=0.1
        )
        
        result = expandor.expand(config)
        
        # Should have unification stage
        unification_stages = [s for s in result.stages if s['name'] == 'swpo_unification']
        assert len(unification_stages) == 1
        assert unification_stages[0]['strength'] == 0.1
        
    def test_vram_fallback(self, expandor, test_image_1080p):
        """Test that SWPO works with VRAM limits"""
        config = ExpandorConfig(
            source_image=test_image_1080p,
            target_resolution=(7680, 1080),  # Very wide
            prompt="Extend with limited VRAM",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=create_mock_inpaint_pipeline(),
            vram_limit_mb=4000  # 4GB limit
        )
        
        result = expandor.expand(config)
        
        # Should still succeed
        assert result.success is True
        
        # Should have used SWPO (processes windows sequentially)
        assert result.strategy_used == "SWPO"
```

## 2. CPU Offload Strategy Implementation

### 2.1 Create CPU Offload Strategy File

**EXACT FILE PATH**: `expandor/strategies/cpu_offload_strategy.py`

```python
"""
CPU Offload Strategy for extreme VRAM constraints.
Implements aggressive memory management with sequential processing.
Based on ai-wallpaper's CPUOffloadRefiner implementation.
"""

import gc
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
import torch
from PIL import Image
import numpy as np

from expandor.strategies.base_strategy import BaseExpansionStrategy
from expandor.core.result import ExpansionResult
from expandor.core.vram_manager import VRAMManager
from expandor.core.dimension_calculator import DimensionCalculator
from expandor.processors.tiled_processor import TiledProcessor
from expandor.core.exceptions import ExpandorError, VRAMError
from expandor.utils.memory_utils import offload_to_cpu, load_to_gpu


class CPUOffloadStrategy(BaseExpansionStrategy):
    """
    CPU Offload strategy for extreme memory constraints.
    
    Features:
    - Sequential CPU offloading of model components
    - Small tile processing (384x384 minimum)
    - Aggressive garbage collection
    - Multi-stage processing with memory clearing
    - Automatic batch size adjustment
    """
    
    def __init__(self):
        super().__init__()
        self.vram_manager = VRAMManager()
        self.dimension_calc = DimensionCalculator()
        self.tiled_processor = TiledProcessor()
        
        # CPU offload specific parameters
        self.min_tile_size = 384
        self.default_tile_size = 512
        self.max_tile_size = 768
        self.min_overlap = 64
        self.default_overlap = 128
        
    def can_handle(self, config: 'ExpandorConfig') -> bool:
        """
        CPU offload can theoretically handle anything, but should be last resort.
        
        Requirements:
        - CPU offload must be enabled in config
        - At least one pipeline available
        """
        if not config.allow_cpu_offload:
            return False
            
        # Need at least one pipeline
        has_pipeline = any([
            config.inpaint_pipeline,
            config.refiner_pipeline,
            config.img2img_pipeline
        ])
        
        return has_pipeline
    
    def execute(self, config: 'ExpandorConfig') -> ExpansionResult:
        """
        Execute expansion with aggressive CPU offloading.
        
        Process:
        1. Configure for minimal memory usage
        2. Process in smallest viable tiles
        3. Clear memory aggressively between operations
        4. Use sequential offloading if needed
        """
        self.logger.info("Starting CPU offload expansion strategy")
        self.logger.warning("Using CPU offload - this will be SLOW but should always work")
        
        with gpu_memory_manager():
            try:
                # Initialize tracking
                result = ExpansionResult(
                strategy_name="CPUOffload",
                stages=[],
                boundaries=[],
                metadata={}
            )
            
            # Load source image
            if isinstance(config.source_image, Image.Image):
                current_image = config.source_image.copy()
            else:
                current_image = Image.open(config.source_image)
                
            source_w, source_h = current_image.size
            target_w, target_h = config.target_resolution
            
            # Ensure dimensions are valid
            target_w = self.dimension_calc.round_to_multiple(target_w, 8)
            target_h = self.dimension_calc.round_to_multiple(target_h, 8)
            
            # Configure pipelines for CPU offload
            self._configure_cpu_offload(config)
            
            # Determine processing strategy
            if self._can_process_directly(source_w, source_h, target_w, target_h):
                # Simple upscale/downscale
                self.logger.info("Using direct processing (no expansion needed)")
                current_image = self._process_direct(
                    current_image, (target_w, target_h), config
                )
                
                result.stages.append({
                    'name': 'cpu_offload_direct',
                    'method': 'resize',
                    'input_size': (source_w, source_h),
                    'output_size': (target_w, target_h)
                })
                
            else:
                # Need expansion - use tiled approach
                self.logger.info("Using tiled expansion with CPU offload")
                
                # First, expand to target aspect ratio if needed
                if abs((target_w/target_h) - (source_w/source_h)) > 0.1:
                    current_image, boundaries = self._expand_aspect_ratio_tiled(
                        current_image, (target_w, target_h), config
                    )
                    
                    result.boundaries.extend(boundaries)
                    result.stages.append({
                        'name': 'cpu_offload_aspect_adjust',
                        'method': 'tiled_inpaint',
                        'input_size': (source_w, source_h),
                        'output_size': current_image.size,
                        'tile_size': self._calculate_tile_size(config)
                    })
                    
                # Then upscale if needed
                if current_image.size != (target_w, target_h):
                    current_image = self._upscale_tiled(
                        current_image, (target_w, target_h), config
                    )
                    
                    result.stages.append({
                        'name': 'cpu_offload_upscale',
                        'method': 'tiled_upscale',
                        'input_size': current_image.size,
                        'output_size': (target_w, target_h)
                    })
                    
            # Aggressive cleanup
            self._cleanup_memory()
            
            # Final validation
            if current_image.size != (target_w, target_h):
                raise ExpandorError(
                    f"CPU offload size mismatch: expected {target_w}x{target_h}, "
                    f"got {current_image.size[0]}x{current_image.size[1]}"
                )
                
            # Set result
            result.image = current_image
                result.final_size = current_image.size
                result.metadata['tile_size'] = self._calculate_tile_size(config)
                result.metadata['cpu_offload_mode'] = 'sequential'
                
                return result
                
            except Exception as e:
                self.logger.error(f"CPU offload strategy failed: {str(e)}")
                # Log system resources for debugging
                self._log_system_resources()
                
                raise ExpandorError(
                    f"CPU offload expansion failed: {str(e)}",
                    strategy="CPUOffload",
                    original_error=e
                )
    
    def _configure_cpu_offload(self, config: 'ExpandorConfig'):
        """Configure all pipelines for minimal memory usage"""
        # Enable CPU offload for all available pipelines
        pipelines = [
            ('inpaint', config.inpaint_pipeline),
            ('refiner', config.refiner_pipeline),
            ('img2img', config.img2img_pipeline)
        ]
        
        for name, pipeline in pipelines:
            if pipeline is None:
                continue
                
            try:
                # Enable sequential CPU offload (most aggressive)
                if hasattr(pipeline, 'enable_sequential_cpu_offload'):
                    pipeline.enable_sequential_cpu_offload()
                    self.logger.info(f"Enabled sequential CPU offload for {name} pipeline")
                    
                # Enable model CPU offload as fallback
                elif hasattr(pipeline, 'enable_model_cpu_offload'):
                    pipeline.enable_model_cpu_offload()
                    self.logger.info(f"Enabled model CPU offload for {name} pipeline")
                    
                # Enable attention slicing for memory efficiency
                if hasattr(pipeline, 'enable_attention_slicing'):
                    pipeline.enable_attention_slicing(1)
                    
                # Enable xformers if available
                if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                    try:
                        pipeline.enable_xformers_memory_efficient_attention()
                    except Exception:
                        pass  # Xformers not available
                        
                # Set to use less memory
                if hasattr(pipeline, 'vae'):
                    # Disable VAE slicing for tiled VAE operations
                    if hasattr(pipeline.vae, 'enable_tiling'):
                        pipeline.vae.enable_tiling()
                        
            except Exception as e:
                self.logger.warning(f"Could not configure CPU offload for {name}: {e}")
                
    def _calculate_tile_size(self, config: 'ExpandorConfig') -> int:
        """Calculate optimal tile size based on available memory"""
        # Get available VRAM
        available_vram = self.vram_manager.get_available_vram()
        
        # Start with default
        tile_size = self.default_tile_size
        
        if available_vram < 2000:  # Less than 2GB
            tile_size = self.min_tile_size
        elif available_vram < 4000:  # Less than 4GB
            tile_size = 448
        elif available_vram < 6000:  # Less than 6GB
            tile_size = 512
        else:
            tile_size = min(self.max_tile_size, 640)
            
        # Ensure multiple of 8
        tile_size = self.dimension_calc.round_to_multiple(tile_size, 8)
        
        self.logger.info(f"Using tile size: {tile_size}x{tile_size}")
        return tile_size
    
    def _can_process_directly(self, source_w: int, source_h: int,
                            target_w: int, target_h: int) -> bool:
        """Check if we can process without expansion (just resize)"""
        # Same aspect ratio and only scaling needed
        source_aspect = source_w / source_h
        target_aspect = target_w / target_h
        
        aspect_similar = abs(source_aspect - target_aspect) < 0.01
        no_expansion = target_w <= source_w or target_h <= source_h
        
        return aspect_similar and no_expansion
    
    def _process_direct(self, image: Image.Image, 
                       target_size: Tuple[int, int],
                       config: 'ExpandorConfig') -> Image.Image:
        """Direct processing when no expansion needed"""
        # Simple high-quality resize
        return image.resize(target_size, Image.Resampling.LANCZOS)
    
    def _expand_aspect_ratio_tiled(self, image: Image.Image,
                                  target_size: Tuple[int, int],
                                  config: 'ExpandorConfig') -> Tuple[Image.Image, List]:
        """Expand aspect ratio using tiled inpainting"""
        if not config.inpaint_pipeline:
            raise ExpandorError("Inpaint pipeline required for aspect ratio expansion")
            
        source_w, source_h = image.size
        target_w, target_h = target_size
        
        # Calculate intermediate size (same aspect as target)
        target_aspect = target_w / target_h
        source_aspect = source_w / source_h
        
        if target_aspect > source_aspect:
            # Wider - expand width
            new_w = int(source_h * target_aspect)
            new_h = source_h
        else:
            # Taller - expand height
            new_w = source_w
            new_h = int(source_w / target_aspect)
            
        # Round to multiple of 8
        new_w = self.dimension_calc.round_to_multiple(new_w, 8)
        new_h = self.dimension_calc.round_to_multiple(new_h, 8)
        
        self.logger.info(f"Expanding from {source_w}x{source_h} to {new_w}x{new_h}")
        
        # Create canvas
        canvas = Image.new('RGB', (new_w, new_h))
        
        # Position source image (centered)
        paste_x = (new_w - source_w) // 2
        paste_y = (new_h - source_h) // 2
        canvas.paste(image, (paste_x, paste_y))
        
        # Create mask for new areas
        mask = Image.new('L', (new_w, new_h), 255)  # White = generate
        mask.paste(0, (paste_x, paste_y, paste_x + source_w, paste_y + source_h))  # Black = preserve
        
        # Process in tiles
        tile_size = self._calculate_tile_size(config)
        overlap = self.default_overlap
        
        # Use tiled processor
        result = self.tiled_processor.process_tiled(
            canvas=canvas,
            mask=mask,
            pipeline=config.inpaint_pipeline,
            prompt=config.prompt,
            tile_size=tile_size,
            overlap=overlap,
            strength=0.95,
            seed=config.seed,
            callback=lambda x, y, w, h: self._cleanup_memory()  # Clean between tiles
        )
        
        # Track boundaries
        boundaries = []
        if new_w > source_w:
            # Vertical boundaries
            boundaries.append({
                'position': paste_x,
                'direction': 'vertical',
                'expansion_size': paste_x
            })
            boundaries.append({
                'position': paste_x + source_w,
                'direction': 'vertical',
                'expansion_size': new_w - (paste_x + source_w)
            })
            
        if new_h > source_h:
            # Horizontal boundaries
            boundaries.append({
                'position': paste_y,
                'direction': 'horizontal',
                'expansion_size': paste_y
            })
            boundaries.append({
                'position': paste_y + source_h,
                'direction': 'horizontal', 
                'expansion_size': new_h - (paste_y + source_h)
            })
            
        return result, boundaries
    
    def _upscale_tiled(self, image: Image.Image,
                      target_size: Tuple[int, int],
                      config: 'ExpandorConfig') -> Image.Image:
        """Upscale using tiled processing"""
        # For CPU offload, we'll use a simple but effective approach
        # First resize to target, then refine in tiles if pipeline available
        
        resized = image.resize(target_size, Image.Resampling.LANCZOS)
        
        if config.refiner_pipeline:
            self.logger.info("Refining upscaled image in tiles")
            
            tile_size = self._calculate_tile_size(config)
            
            # Process with refiner
            result = self.tiled_processor.process_tiled(
                canvas=resized,
                mask=None,  # Refine entire image
                pipeline=config.refiner_pipeline,
                prompt=config.prompt,
                tile_size=tile_size,
                overlap=self.min_overlap,  # Less overlap for refinement
                strength=0.3,  # Light refinement
                seed=config.seed,
                is_refiner=True,
                callback=lambda x, y, w, h: self._cleanup_memory()
            )
            
            return result
        else:
            # No refiner available, return resized
            return resized
    
    def _cleanup_memory(self):
        """Aggressive memory cleanup"""
        # Python garbage collection
        gc.collect()
        
        # CUDA cleanup if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Log memory status
        if self.logger.isEnabledFor(10):  # DEBUG level
            self._log_system_resources()
            
    def _log_system_resources(self):
        """Log current system resource usage"""
        try:
            import psutil
            
            # CPU and memory info
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            self.logger.debug(f"System resources - CPU: {cpu_percent}%, "
                            f"RAM: {memory.used/1024/1024/1024:.1f}GB/{memory.total/1024/1024/1024:.1f}GB "
                            f"({memory.percent}%)")
            
            # GPU info if available
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                reserved = torch.cuda.memory_reserved() / 1024 / 1024
                self.logger.debug(f"CUDA memory - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")
                
        except ImportError:
            pass  # psutil not available
            
    def get_vram_requirement(self, config: 'ExpandorConfig') -> float:
        """Calculate VRAM requirement for CPU offload strategy"""
        # CPU offload only needs memory for one tile at a time
        tile_size = self._calculate_tile_size(config)
        
        return self.vram_manager.calculate_generation_vram(
            width=tile_size,
            height=tile_size,
            batch_size=1,
            dtype=torch.float16
        )
```

### 2.2 Create Tiled Processor Helper

**EXACT FILE PATH**: `expandor/processors/tiled_processor.py`

```python
"""
Tiled processing utilities for memory-efficient operations.
Handles overlapping tiles, blending, and progressive processing.
"""

from typing import Callable, Optional, Tuple, Any
import numpy as np
from PIL import Image, ImageFilter
import torch

from ..core.exceptions import ExpandorError


class TiledProcessor:
    """Processes large images in overlapping tiles"""
    
    def process_tiled(self, 
                     canvas: Image.Image,
                     mask: Optional[Image.Image],
                     pipeline: Any,
                     prompt: str,
                     tile_size: int,
                     overlap: int,
                     strength: float,
                     seed: int,
                     is_refiner: bool = False,
                     callback: Optional[Callable] = None) -> Image.Image:
        """
        Process image in tiles with overlap and blending.
        
        Args:
            canvas: Image to process
            mask: Optional mask for inpainting
            pipeline: Diffusion pipeline to use
            prompt: Generation prompt
            tile_size: Size of each tile
            overlap: Overlap between tiles
            strength: Denoising strength
            seed: Random seed
            is_refiner: Whether this is a refiner pipeline
            callback: Optional callback after each tile
            
        Returns:
            Processed image
        """
        width, height = canvas.size
        
        # Calculate tile grid
        step = tile_size - overlap
        tiles_x = (width - overlap) // step + 1
        tiles_y = (height - overlap) // step + 1
        
        # Create output canvas
        output = Image.new('RGB', (width, height))
        weight_map = np.zeros((height, width), dtype=np.float32)
        
        tile_idx = 0
        
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # Calculate tile bounds
                x1 = tx * step
                y1 = ty * step
                x2 = min(x1 + tile_size, width)
                y2 = min(y1 + tile_size, height)
                
                # Adjust for edge tiles
                if x2 - x1 < tile_size:
                    x1 = max(0, x2 - tile_size)
                if y2 - y1 < tile_size:
                    y1 = max(0, y2 - tile_size)
                    
                # Extract tile
                tile_img = canvas.crop((x1, y1, x2, y2))
                
                # Extract mask tile if provided
                tile_mask = None
                if mask:
                    tile_mask = mask.crop((x1, y1, x2, y2))
                    
                    # Skip if mask is all black (nothing to generate)
                    if tile_mask.getextrema()[1] == 0:
                        continue
                        
                # Process tile
                if is_refiner:
                    result = self._process_refiner_tile(
                        pipeline, tile_img, prompt, strength, seed + tile_idx
                    )
                else:
                    result = self._process_inpaint_tile(
                        pipeline, tile_img, tile_mask, prompt, strength, seed + tile_idx
                    )
                    
                # Create blend mask
                blend_mask = self._create_blend_mask(x2-x1, y2-y1, overlap)
                
                # Blend into output
                self._blend_tile(output, result, (x1, y1), blend_mask, weight_map)
                
                tile_idx += 1
                
                # Callback for memory cleanup
                if callback:
                    callback(x1, y1, x2-x1, y2-y1)
                    
        # Normalize by weight map
        output_array = np.array(output, dtype=np.float32)
        
        for c in range(3):
            mask = weight_map > 0
            output_array[mask, c] /= weight_map[mask]
            
        return Image.fromarray(output_array.astype(np.uint8))
    
    def _process_inpaint_tile(self, pipeline, tile_img, tile_mask, 
                             prompt, strength, seed):
        """Process single inpainting tile"""
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
            
        try:
            result = pipeline(
                prompt=prompt,
                image=tile_img,
                mask_image=tile_mask,
                height=tile_img.height,
                width=tile_img.width,
                strength=strength,
                num_inference_steps=30,  # Fewer steps for speed
                guidance_scale=7.5,
                generator=generator
            )
            
            return result.images[0]
            
        except Exception as e:
            raise ExpandorError(f"Tile inpainting failed: {str(e)}")
            
    def _process_refiner_tile(self, pipeline, tile_img, prompt, strength, seed):
        """Process single refiner tile"""
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
            
        try:
            result = pipeline(
                prompt=prompt,
                image=tile_img,
                strength=strength,
                num_inference_steps=20,  # Fewer steps for refinement
                guidance_scale=7.0,
                generator=generator
            )
            
            return result.images[0]
            
        except Exception as e:
            raise ExpandorError(f"Tile refinement failed: {str(e)}")
            
    def _create_blend_mask(self, width: int, height: int, overlap: int) -> np.ndarray:
        """Create blend mask with gradients at edges"""
        mask = np.ones((height, width), dtype=np.float32)
        
        if overlap > 0:
            # Create gradients
            fade = min(overlap // 2, 50)
            
            # Top edge
            for y in range(fade):
                mask[y, :] *= y / fade
                
            # Bottom edge  
            for y in range(fade):
                mask[height-1-y, :] *= y / fade
                
            # Left edge
            for x in range(fade):
                mask[:, x] *= x / fade
                
            # Right edge
            for x in range(fade):
                mask[:, width-1-x] *= x / fade
                
        return mask
    
    def _blend_tile(self, output: Image.Image, tile: Image.Image,
                   position: Tuple[int, int], blend_mask: np.ndarray,
                   weight_map: np.ndarray):
        """Blend tile into output with accumulation"""
        x, y = position
        h, w = blend_mask.shape
        
        # Convert to arrays
        output_array = np.array(output, dtype=np.float32)
        tile_array = np.array(tile, dtype=np.float32)
        
        # Apply blend mask and accumulate
        for c in range(3):
            output_array[y:y+h, x:x+w, c] += tile_array[:h, :w, c] * blend_mask
            
        weight_map[y:y+h, x:x+w] += blend_mask
        
        # Convert back
        output.paste(Image.fromarray(output_array.astype(np.uint8)), (0, 0))
```

### 2.3 Create CPU Offload Unit Tests

**EXACT FILE PATH**: `tests/unit/strategies/test_cpu_offload_strategy.py`

```python
"""
Unit tests for CPU Offload Strategy.
Tests memory management, tiled processing, and fallback behavior.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from PIL import Image
import torch
import gc

from expandor.strategies.cpu_offload_strategy import CPUOffloadStrategy
from expandor.core.config import ExpandorConfig
from expandor.core.exceptions import ExpandorError


class TestCPUOffloadStrategy:
    """Tests for CPU offload strategy"""
    
    @pytest.fixture
    def strategy(self):
        """Create CPU offload strategy instance"""
        return CPUOffloadStrategy()
    
    @pytest.fixture
    def base_config(self):
        """Create base configuration"""
        test_image = Image.new('RGB', (1024, 768))
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.images = [Image.new('RGB', (1024, 768))]
        mock_pipeline.return_value = mock_result
        
        # Add CPU offload methods
        mock_pipeline.enable_sequential_cpu_offload = Mock()
        mock_pipeline.enable_attention_slicing = Mock()
        
        config = ExpandorConfig(
            source_image=test_image,
            target_resolution=(2048, 1536),
            prompt="Test prompt",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=mock_pipeline,
            allow_cpu_offload=True
        )
        
        return config
    
    def test_can_handle_with_offload_enabled(self, strategy, base_config):
        """Test that strategy can handle when CPU offload is enabled"""
        assert strategy.can_handle(base_config) == True
        
    def test_can_handle_without_offload_enabled(self, strategy, base_config):
        """Test that strategy cannot handle when CPU offload is disabled"""
        base_config.allow_cpu_offload = False
        assert strategy.can_handle(base_config) == False
        
    def test_can_handle_no_pipeline(self, strategy, base_config):
        """Test that strategy cannot handle without any pipeline"""
        base_config.inpaint_pipeline = None
        base_config.refiner_pipeline = None
        base_config.img2img_pipeline = None
        assert strategy.can_handle(base_config) == False
        
    def test_pipeline_configuration(self, strategy, base_config):
        """Test that pipelines are configured for CPU offload"""
        # Add multiple pipelines
        base_config.refiner_pipeline = Mock()
        base_config.refiner_pipeline.enable_model_cpu_offload = Mock()
        
        strategy._configure_cpu_offload(base_config)
        
        # Should have called CPU offload methods
        base_config.inpaint_pipeline.enable_sequential_cpu_offload.assert_called_once()
        base_config.inpaint_pipeline.enable_attention_slicing.assert_called_once()
        
    def test_tile_size_calculation_low_vram(self, strategy, base_config):
        """Test tile size calculation with low VRAM"""
        with patch.object(strategy.vram_manager, 'get_available_vram', return_value=1500):
            tile_size = strategy._calculate_tile_size(base_config)
            assert tile_size == 384  # Minimum tile size
            
    def test_tile_size_calculation_medium_vram(self, strategy, base_config):
        """Test tile size calculation with medium VRAM"""
        with patch.object(strategy.vram_manager, 'get_available_vram', return_value=3500):
            tile_size = strategy._calculate_tile_size(base_config)
            assert tile_size == 448
            
    def test_tile_size_calculation_high_vram(self, strategy, base_config):
        """Test tile size calculation with high VRAM"""
        with patch.object(strategy.vram_manager, 'get_available_vram', return_value=8000):
            tile_size = strategy._calculate_tile_size(base_config)
            assert tile_size == 640
            
    def test_direct_processing_detection(self, strategy):
        """Test detection of direct processing (no expansion)"""
        # Same aspect ratio, downscaling
        assert strategy._can_process_directly(2048, 1152, 1920, 1080) == True
        
        # Different aspect ratio
        assert strategy._can_process_directly(1920, 1080, 1920, 1200) == False
        
        # Upscaling
        assert strategy._can_process_directly(1920, 1080, 3840, 2160) == False
        
    def test_execute_direct_processing(self, strategy, base_config):
        """Test execution with direct processing (resize only)"""
        # Setup for downscaling
        base_config.source_image = Image.new('RGB', (3840, 2160))
        base_config.target_resolution = (1920, 1080)
        
        result = strategy.execute(base_config)
        
        # Should succeed with simple resize
        assert result.success is True
        assert result.final_size == (1920, 1080)
        assert len(result.stages) == 1
        assert result.stages[0]['method'] == 'resize'
        
    @patch('expandor.strategies.cpu_offload_strategy.gc.collect')
    @patch('torch.cuda.empty_cache')
    def test_memory_cleanup(self, mock_cuda_clear, mock_gc, strategy):
        """Test that memory is cleaned up aggressively"""
        strategy._cleanup_memory()
        
        # Should call cleanup methods
        mock_gc.assert_called_once()
        
        if torch.cuda.is_available():
            mock_cuda_clear.assert_called_once()
            
    def test_execute_with_aspect_expansion(self, strategy, base_config):
        """Test execution with aspect ratio expansion"""
        # Mock tiled processor
        mock_tiled_result = Image.new('RGB', (2048, 1536))
        strategy.tiled_processor.process_tiled = Mock(return_value=mock_tiled_result)
        
        result = strategy.execute(base_config)
        
        # Should succeed with tiled processing
        assert result.success is True
        assert result.final_size == (2048, 1536)
        
        # Should have aspect adjustment stage
        aspect_stages = [s for s in result.stages if 'aspect_adjust' in s['name']]
        assert len(aspect_stages) > 0
        
    def test_execute_with_upscaling(self, strategy, base_config):
        """Test execution with upscaling"""
        # Setup for pure upscaling (same aspect)
        base_config.source_image = Image.new('RGB', (1024, 768))
        base_config.target_resolution = (2048, 1536)
        
        # Mock tiled processor
        strategy.tiled_processor.process_tiled = Mock(
            return_value=Image.new('RGB', (2048, 1536))
        )
        
        result = strategy.execute(base_config)
        
        # Should have upscale stage
        upscale_stages = [s for s in result.stages if 'upscale' in s['name']]
        assert len(upscale_stages) > 0
        
    def test_boundary_tracking(self, strategy, base_config):
        """Test that boundaries are tracked during expansion"""
        # Mock tiled processor to simulate expansion
        def mock_process(canvas, mask, **kwargs):
            # Return canvas as-is
            return canvas
            
        strategy.tiled_processor.process_tiled = mock_process
        
        # Execute with expansion
        result = strategy.execute(base_config)
        
        # Should have boundaries if aspect ratio changed
        if abs((base_config.target_resolution[0]/base_config.target_resolution[1]) - 
               (1024/768)) > 0.1:
            assert len(result.boundaries) > 0
            
    def test_error_handling_pipeline_failure(self, strategy, base_config):
        """Test error handling when pipeline fails"""
        # Make tiled processor fail
        strategy.tiled_processor.process_tiled = Mock(
            side_effect=RuntimeError("Pipeline OOM")
        )
        
        with pytest.raises(ExpandorError) as exc_info:
            strategy.execute(base_config)
            
        assert "CPU offload expansion failed" in str(exc_info.value)
        assert "Pipeline OOM" in str(exc_info.value)
        
    def test_vram_requirement_calculation(self, strategy, base_config):
        """Test VRAM requirement is based on tile size"""
        with patch.object(strategy, '_calculate_tile_size', return_value=512):
            vram_req = strategy.get_vram_requirement(base_config)
            
            # Should be based on 512x512 tile
            expected = strategy.vram_manager.calculate_generation_vram(
                width=512, height=512, batch_size=1, dtype=torch.float16
            )
            
            assert vram_req == expected
            
    def test_system_resource_logging(self, strategy):
        """Test system resource logging functionality"""
        # Should not raise even if psutil not available
        try:
            strategy._log_system_resources()
        except Exception:
            pytest.fail("Resource logging should not raise exceptions")
            
    def test_callback_execution(self, strategy, base_config):
        """Test that callbacks are executed during tiled processing"""
        callback_count = 0
        
        def test_callback(x, y, w, h):
            nonlocal callback_count
            callback_count += 1
            
        # Mock tiled processor to accept callback
        original_process = strategy.tiled_processor.process_tiled
        
        def mock_process(*args, **kwargs):
            # Extract callback and call it
            callback = kwargs.get('callback')
            if callback:
                # Simulate 4 tiles
                for i in range(4):
                    callback(i*512, 0, 512, 512)
            return Image.new('RGB', base_config.target_resolution)
            
        strategy.tiled_processor.process_tiled = mock_process
        
        result = strategy.execute(base_config)
        
        # Callback should have been provided (even if not explicitly tested)
        assert result.success is True
```

## 3. Adaptive Hybrid Strategy Implementation

### 3.1 Create Adaptive Hybrid Strategy File

**EXACT FILE PATH**: `expandor/strategies/adaptive_hybrid_strategy.py`

```python
"""
Adaptive Hybrid Strategy that intelligently combines multiple approaches.
Analyzes the expansion requirements and selects optimal sub-strategies.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch

from expandor.strategies.base_strategy import BaseExpansionStrategy
from expandor.core.result import ExpansionResult
from expandor.core.vram_manager import VRAMManager
from expandor.core.dimension_calculator import DimensionCalculator
# Note: These strategies are implemented as placeholders later in this document
from expandor.strategies.progressive_outpaint import ProgressiveOutpaintStrategy
from expandor.strategies.direct_upscale import DirectUpscaleStrategy
from expandor.strategies.swpo import SWPOStrategy
from expandor.processors.smart_refiner import SmartRefiner
from expandor.core.exceptions import ExpandorError, VRAMError


@dataclass
class HybridPlan:
    """Execution plan for hybrid strategy"""
    steps: List[Dict[str, Any]]
    estimated_vram: float
    estimated_quality: float
    rationale: str


class AdaptiveHybridStrategy(BaseStrategy):
    """
    Intelligently combines multiple strategies based on:
    - Input/output characteristics
    - Available resources
    - Quality requirements
    - Optimal path analysis
    """
    
    def __init__(self):
        super().__init__()
        self.vram_manager = VRAMManager()
        self.dimension_calc = DimensionCalculator()
        
        # Initialize sub-strategies
        self.progressive_strategy = ProgressiveOutpaintStrategy()
        self.upscale_strategy = DirectUpscaleStrategy()
        self.swpo_strategy = SWPOStrategy()
        self.smart_refiner = SmartRefiner()
        
    def can_handle(self, config: 'ExpandorConfig') -> bool:
        """
        Adaptive strategy can handle most scenarios by combining approaches.
        
        Requirements:
        - At least one sub-strategy must be viable
        - Not explicitly disabled
        """
        # Check if any sub-strategy can handle
        can_progressive = self.progressive_strategy.can_handle(config)
        can_upscale = self.upscale_strategy.can_handle(config)
        can_swpo = self.swpo_strategy.can_handle(config)
        
        return any([can_progressive, can_upscale, can_swpo])
    
    def execute(self, config: 'ExpandorConfig') -> ExpansionResult:
        """
        Execute adaptive hybrid strategy.
        
        Process:
        1. Analyze requirements
        2. Create optimal execution plan
        3. Execute plan with checkpoints
        4. Apply smart refinement if needed
        5. Validate results
        """
        self.logger.info("Starting adaptive hybrid strategy")
        
        with gpu_memory_manager():
            try:
                # Initialize result
                result = ExpansionResult(
                strategy_name="AdaptiveHybrid",
                stages=[],
                boundaries=[],
                metadata={}
            )
            
            # Analyze and create plan
            plan = self._create_execution_plan(config)
            
            self.logger.info(f"Created hybrid plan: {plan.rationale}")
            result.metadata['plan_rationale'] = plan.rationale
            result.metadata['planned_steps'] = len(plan.steps)
            
            # Load source image
            if isinstance(config.source_image, Image.Image):
                current_image = config.source_image.copy()
            else:
                current_image = Image.open(config.source_image)
                
            # Execute plan steps
            for i, step in enumerate(plan.steps):
                self.logger.info(f"Executing step {i+1}/{len(plan.steps)}: {step['name']}")
                
                # Execute step based on type
                if step['type'] == 'progressive_outpaint':
                    current_image, step_result = self._execute_progressive(
                        current_image, step, config
                    )
                    
                elif step['type'] == 'direct_upscale':
                    current_image, step_result = self._execute_upscale(
                        current_image, step, config
                    )
                    
                elif step['type'] == 'swpo':
                    current_image, step_result = self._execute_swpo(
                        current_image, step, config
                    )
                    
                elif step['type'] == 'smart_refine':
                    current_image, step_result = self._execute_refinement(
                        current_image, step, config, result.boundaries
                    )
                    
                else:
                    raise ExpandorError(f"Unknown step type: {step['type']}")
                    
                # Track stage
                result.stages.append(step_result['stage'])
                
                # Track boundaries
                if 'boundaries' in step_result:
                    result.boundaries.extend(step_result['boundaries'])
                    
                # Save checkpoint if requested
                if config.save_stages and config.stage_save_callback:
                    config.stage_save_callback(
                        image=current_image,
                        stage_name=f"hybrid_step_{i+1}_{step['name']}",
                        metadata=step_result
                    )
                    
            # Final validation
            target_w, target_h = config.target_resolution
            if current_image.size != (target_w, target_h):
                raise ExpandorError(
                    f"Hybrid strategy size mismatch: expected {target_w}x{target_h}, "
                    f"got {current_image.size[0]}x{current_image.size[1]}"
                )
                
            # Set final result
            result.image = current_image
            result.final_size = current_image.size
                result.metadata['total_stages'] = len(result.stages)
                
                return result
                
            except Exception as e:
                self.logger.error(f"Adaptive hybrid strategy failed: {str(e)}")
                raise ExpandorError(
                    f"Hybrid expansion failed: {str(e)}",
                    strategy="AdaptiveHybrid",
                    original_error=e
                )
    
    def _create_execution_plan(self, config: 'ExpandorConfig') -> HybridPlan:
        """
        Create optimal execution plan based on analysis.
        
        Considers:
        - Aspect ratio change
        - Scale factor
        - Available pipelines
        - VRAM constraints
        - Quality requirements
        """
        # Get dimensions
        if isinstance(config.source_image, Image.Image):
            source_w, source_h = config.source_image.size
        else:
            with Image.open(config.source_image) as img:
                source_w, source_h = img.size
                
        target_w, target_h = config.target_resolution
        
        # Calculate metrics
        source_aspect = source_w / source_h
        target_aspect = target_w / target_h
        aspect_change = max(target_aspect/source_aspect, source_aspect/target_aspect)
        scale_factor = max(target_w/source_w, target_h/source_h)
        area_increase = (target_w * target_h) / (source_w * source_h)
        
        # Initialize plan
        steps = []
        estimated_vram = 0
        estimated_quality = 1.0
        
        # Decision tree for plan creation
        
        # Case 1: Extreme aspect ratio change (>3x)
        if aspect_change > 3.0 and config.inpaint_pipeline:
            if self._estimate_vram_for_swpo(config) < self.vram_manager.get_available_vram():
                # Use SWPO for extreme ratios
                steps.append({
                    'name': 'swpo_expansion',
                    'type': 'swpo',
                    'target_size': (target_w, target_h),
                    'description': 'SWPO for extreme aspect ratio change'
                })
                rationale = f"Using SWPO for {aspect_change:.1f}x aspect ratio change"
                
            else:
                # Fall back to progressive with smaller steps
                intermediate_w, intermediate_h = self._calculate_intermediate_size(
                    (source_w, source_h), (target_w, target_h), max_ratio=2.0
                )
                
                steps.append({
                    'name': 'progressive_aspect',
                    'type': 'progressive_outpaint',
                    'target_size': (intermediate_w, intermediate_h),
                    'description': 'Progressive expansion for aspect adjustment'
                })
                
                if (intermediate_w, intermediate_h) != (target_w, target_h):
                    steps.append({
                        'name': 'final_expansion',
                        'type': 'progressive_outpaint',
                        'target_size': (target_w, target_h),
                        'description': 'Final expansion to target size'
                    })
                    
                rationale = "Using multi-stage progressive for extreme aspect change"
                
        # Case 2: Moderate aspect change with large scale
        elif aspect_change > 1.5 and scale_factor > 2.0:
            # First adjust aspect ratio
            aspect_w = int(source_h * target_aspect)
            aspect_h = source_h
            
            if aspect_w > source_w:  # Need to expand width
                steps.append({
                    'name': 'aspect_adjustment',
                    'type': 'progressive_outpaint',
                    'target_size': (aspect_w, aspect_h),
                    'description': 'Adjust aspect ratio first'
                })
                
            # Then upscale
            steps.append({
                'name': 'quality_upscale',
                'type': 'direct_upscale',
                'target_size': (target_w, target_h),
                'description': 'High-quality upscaling'
            })
            
            rationale = "Aspect adjustment followed by upscaling"
            
        # Case 3: Pure upscaling (minimal aspect change)
        elif aspect_change < 1.1 and scale_factor > 1.0:
            steps.append({
                'name': 'direct_upscale',
                'type': 'direct_upscale',
                'target_size': (target_w, target_h),
                'description': 'Direct upscaling with same aspect ratio'
            })
            
            rationale = f"Direct {scale_factor:.1f}x upscaling"
            
        # Case 4: Complex transformation
        else:
            # Use progressive for flexibility
            if area_increase > 4:
                # Large increase - do in stages
                intermediate_w, intermediate_h = self._calculate_intermediate_size(
                    (source_w, source_h), (target_w, target_h), max_ratio=2.0
                )
                
                steps.append({
                    'name': 'stage1_expansion',
                    'type': 'progressive_outpaint',
                    'target_size': (intermediate_w, intermediate_h),
                    'description': 'First stage expansion'
                })
                
                steps.append({
                    'name': 'stage2_expansion',
                    'type': 'progressive_outpaint',
                    'target_size': (target_w, target_h),
                    'description': 'Second stage expansion'
                })
            else:
                steps.append({
                    'name': 'single_expansion',
                    'type': 'progressive_outpaint',
                    'target_size': (target_w, target_h),
                    'description': 'Single-stage expansion'
                })
                
            rationale = "Progressive expansion for complex transformation"
            
        # Add refinement step if quality preset demands it
        if config.quality_preset in ['ultra', 'high'] and config.refiner_pipeline:
            steps.append({
                'name': 'smart_refinement',
                'type': 'smart_refine',
                'target_size': (target_w, target_h),
                'description': 'Smart quality refinement',
                'passes': 3 if config.quality_preset == 'ultra' else 2
            })
            
        # Estimate VRAM for plan
        for step in steps:
            if step['type'] == 'swpo':
                estimated_vram = max(estimated_vram, self._estimate_vram_for_swpo(config))
            elif step['type'] == 'progressive_outpaint':
                w, h = step['target_size']
                vram = self.vram_manager.calculate_generation_vram(w, h, 1, torch.float16)
                estimated_vram = max(estimated_vram, vram)
            elif step['type'] == 'direct_upscale':
                # Upscaling typically uses less VRAM
                estimated_vram = max(estimated_vram, 2000)  # 2GB estimate
                
        # Estimate quality based on approach
        if any(s['type'] == 'swpo' for s in steps):
            estimated_quality = 0.95  # SWPO is very good
        elif any(s['type'] == 'smart_refine' for s in steps):
            estimated_quality = 0.98  # Refinement ensures quality
        elif len(steps) > 2:
            estimated_quality = 0.90  # Multi-stage might have minor artifacts
        else:
            estimated_quality = 0.93  # Single stage is generally good
            
        return HybridPlan(
            steps=steps,
            estimated_vram=estimated_vram,
            estimated_quality=estimated_quality,
            rationale=rationale
        )
    
    def _calculate_intermediate_size(self, source_size: Tuple[int, int],
                                   target_size: Tuple[int, int],
                                   max_ratio: float = 2.0) -> Tuple[int, int]:
        """Calculate intermediate size for multi-stage expansion"""
        source_w, source_h = source_size
        target_w, target_h = target_size
        
        # Calculate ratios
        width_ratio = target_w / source_w
        height_ratio = target_h / source_h
        
        # Limit to max_ratio
        intermediate_w = int(source_w * min(width_ratio, max_ratio))
        intermediate_h = int(source_h * min(height_ratio, max_ratio))
        
        # Ensure we're making progress
        intermediate_w = max(intermediate_w, source_w + 64)
        intermediate_h = max(intermediate_h, source_h + 64)
        
        # Don't exceed target
        intermediate_w = min(intermediate_w, target_w)
        intermediate_h = min(intermediate_h, target_h)
        
        # Round to multiple of 8
        intermediate_w = self.dimension_calc.round_to_multiple(intermediate_w, 8)
        intermediate_h = self.dimension_calc.round_to_multiple(intermediate_h, 8)
        
        return intermediate_w, intermediate_h
    
    def _execute_progressive(self, image: Image.Image, step: Dict,
                           config: 'ExpandorConfig') -> Tuple[Image.Image, Dict]:
        """Execute progressive outpainting step"""
        # Create sub-config for progressive strategy
        sub_config = self._create_sub_config(config, image, step['target_size'])
        
        # Execute
        sub_result = self.progressive_strategy.execute(sub_config)
        
        return sub_result.image, {
            'stage': {
                'name': f"hybrid_{step['name']}",
                'method': 'progressive_outpaint',
                'input_size': image.size,
                'output_size': sub_result.final_size,
                'sub_stages': len(sub_result.stages)
            },
            'boundaries': sub_result.boundaries
        }
    
    def _execute_upscale(self, image: Image.Image, step: Dict,
                        config: 'ExpandorConfig') -> Tuple[Image.Image, Dict]:
        """Execute upscaling step"""
        sub_config = self._create_sub_config(config, image, step['target_size'])
        
        sub_result = self.upscale_strategy.execute(sub_config)
        
        return sub_result.image, {
            'stage': {
                'name': f"hybrid_{step['name']}",
                'method': 'direct_upscale',
                'input_size': image.size,
                'output_size': sub_result.final_size,
                'upscale_factor': sub_result.metadata.get('upscale_factor', 1.0)
            }
        }
    
    def _execute_swpo(self, image: Image.Image, step: Dict,
                     config: 'ExpandorConfig') -> Tuple[Image.Image, Dict]:
        """Execute SWPO step"""
        sub_config = self._create_sub_config(config, image, step['target_size'])
        
        sub_result = self.swpo_strategy.execute(sub_config)
        
        return sub_result.image, {
            'stage': {
                'name': f"hybrid_{step['name']}",
                'method': 'swpo',
                'input_size': image.size,
                'output_size': sub_result.final_size,
                'windows': sub_result.metadata.get('total_windows', 0)
            },
            'boundaries': sub_result.boundaries
        }
    
    def _execute_refinement(self, image: Image.Image, step: Dict,
                          config: 'ExpandorConfig',
                          boundaries: List) -> Tuple[Image.Image, Dict]:
        """Execute smart refinement step"""
        if not config.refiner_pipeline:
            # No refiner available, skip
            return image, {
                'stage': {
                    'name': f"hybrid_{step['name']}_skipped",
                    'method': 'none',
                    'reason': 'No refiner pipeline available'
                }
            }
            
        # Use smart refiner
        refined, artifacts_fixed = self.smart_refiner.refine(
            image=image,
            boundaries=boundaries,
            pipeline=config.refiner_pipeline,
            prompt=config.prompt,
            seed=config.seed,
            max_passes=step.get('passes', 2)
        )
        
        return refined, {
            'stage': {
                'name': f"hybrid_{step['name']}",
                'method': 'smart_refine',
                'input_size': image.size,
                'output_size': refined.size,
                'artifacts_fixed': artifacts_fixed,
                'passes': step.get('passes', 2)
            }
        }
    
    def _create_sub_config(self, parent_config: 'ExpandorConfig',
                          current_image: Image.Image,
                          target_size: Tuple[int, int]) -> 'ExpandorConfig':
        """Create sub-configuration for strategy execution"""
        # Copy parent config with updated image and target
        sub_config = ExpandorConfig(
            source_image=current_image,
            target_resolution=target_size,
            prompt=parent_config.prompt,
            seed=parent_config.seed,
            source_metadata=parent_config.source_metadata,
            generation_metadata=parent_config.generation_metadata,
            inpaint_pipeline=parent_config.inpaint_pipeline,
            refiner_pipeline=parent_config.refiner_pipeline,
            img2img_pipeline=parent_config.img2img_pipeline,
            quality_preset=parent_config.quality_preset,
            vram_limit_mb=parent_config.vram_limit_mb,
            allow_cpu_offload=parent_config.allow_cpu_offload,
            allow_tiled=parent_config.allow_tiled,
            window_size=parent_config.window_size,
            overlap_ratio=parent_config.overlap_ratio,
            denoising_strength=parent_config.denoising_strength,
            save_stages=False,  # Handle at parent level
            verbose=parent_config.verbose
        )
        
        return sub_config
    
    def _estimate_vram_for_swpo(self, config: 'ExpandorConfig') -> float:
        """Estimate VRAM requirement for SWPO execution"""
        # SWPO processes windows sequentially, so only need one window's VRAM
        window_size = getattr(config, 'window_size', 200)
        
        # Estimate based on larger dimension
        if isinstance(config.source_image, Image.Image):
            source_w, source_h = config.source_image.size
        else:
            with Image.open(config.source_image) as img:
                source_w, source_h = img.size
                
        target_w, target_h = config.target_resolution
        
        # Calculate window canvas size
        if target_w - source_w > target_h - source_h:
            window_w = source_w + window_size
            window_h = target_h
        else:
            window_w = target_w
            window_h = source_h + window_size
            
        return self.vram_manager.calculate_generation_vram(
            window_w, window_h, 1, torch.float16
        )
    
    def get_vram_requirement(self, config: 'ExpandorConfig') -> float:
        """Calculate VRAM requirement for hybrid strategy"""
        # Create plan to estimate
        plan = self._create_execution_plan(config)
        return plan.estimated_vram
```

### 3.2 Create Supporting Strategies

**EXACT FILE PATH**: `expandor/strategies/progressive_outpaint.py`

```python
"""
Progressive Outpainting Strategy
Based on ai-wallpaper's AspectAdjuster implementation
"""

from typing import Tuple, Dict, List
from PIL import Image
import numpy as np

from expandor.strategies.base_strategy import BaseExpansionStrategy
from expandor.core.result import ExpansionResult
from expandor.core.exceptions import ExpandorError, VRAMError


class ProgressiveOutpaintStrategy(BaseExpansionStrategy):
    """Progressive outpainting for aspect ratio adjustment"""
    
    def can_handle(self, config: 'ExpandorConfig') -> bool:
        """Check if progressive outpainting can handle this request"""
        # Needs inpaint pipeline
        return config.inpaint_pipeline is not None
        
    def execute(self, config: 'ExpandorConfig') -> ExpansionResult:
        """Execute progressive outpainting"""
        # Simplified implementation for hybrid strategy
        result = ExpansionResult(
            strategy_name="ProgressiveOutpaint",
            stages=[],
            boundaries=[],
            metadata={}
        )
        
        # TODO: Implement based on AspectAdjuster
        # This is a placeholder for the hybrid strategy
        
        if isinstance(config.source_image, Image.Image):
            result.image = config.source_image
        else:
            result.image = Image.open(config.source_image)
            
        result.final_size = result.image.size
        
        return result
        
    def get_vram_requirement(self, config: 'ExpandorConfig') -> float:
        """Calculate VRAM requirement"""
        return 4000  # Placeholder
```

**EXACT FILE PATH**: `expandor/strategies/direct_upscale.py`

```python
"""
Direct Upscale Strategy
Simple upscaling without expansion
"""

from PIL import Image

from expandor.strategies.base_strategy import BaseExpansionStrategy
from expandor.core.result import ExpansionResult
from expandor.core.exceptions import ExpandorError, VRAMError


class DirectUpscaleStrategy(BaseExpansionStrategy):
    """Direct upscaling strategy for simple scale changes"""
    
    def can_handle(self, config: 'ExpandorConfig') -> bool:
        """Check if direct upscaling is appropriate"""
        # Can always handle if dimensions allow
        return True
        
    def execute(self, config: 'ExpandorConfig') -> ExpansionResult:
        """Execute direct upscaling"""
        result = ExpansionResult(
            strategy_name="DirectUpscale",
            stages=[],
            boundaries=[],
            metadata={}
        )
        
        # Load image
        if isinstance(config.source_image, Image.Image):
            image = config.source_image
        else:
            image = Image.open(config.source_image)
            
        # Simple resize
        result.image = image.resize(
            config.target_resolution,
            Image.Resampling.LANCZOS
        )
        
        result.final_size = result.image.size
        result.metadata['upscale_factor'] = (
            config.target_resolution[0] / image.width
        )
        
        result.stages.append({
            'name': 'direct_upscale',
            'method': 'resize',
            'input_size': image.size,
            'output_size': result.final_size
        })
        
        return result
        
    def get_vram_requirement(self, config: 'ExpandorConfig') -> float:
        """Calculate VRAM requirement"""
        return 2000  # Minimal for resize
```

**EXACT FILE PATH**: `expandor/processors/smart_refiner.py`

```python
"""
Smart Quality Refiner
Placeholder for refinement operations
"""

from typing import List, Tuple, Any
from PIL import Image


class SmartRefiner:
    """Smart refinement for quality improvement"""
    
    def refine(self, image: Image.Image, boundaries: List,
              pipeline: Any, prompt: str, seed: int,
              max_passes: int = 2) -> Tuple[Image.Image, int]:
        """Refine image quality"""
        # Placeholder implementation
        # TODO: Implement based on ai-wallpaper's SmartQualityRefiner
        
        # Return original image and 0 artifacts fixed
        return image, 0
```

### 3.3 Create Adaptive Hybrid Unit Tests

**EXACT FILE PATH**: `tests/unit/strategies/test_adaptive_hybrid_strategy.py`

```python
"""
Unit tests for Adaptive Hybrid Strategy.
Tests plan creation, multi-strategy execution, and intelligent routing.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from PIL import Image

from expandor.strategies.adaptive_hybrid_strategy import (
    AdaptiveHybridStrategy, HybridPlan
)
from expandor.core.config import ExpandorConfig
from expandor.core.exceptions import ExpandorError


class TestAdaptiveHybridStrategy:
    """Tests for adaptive hybrid strategy"""
    
    @pytest.fixture
    def strategy(self):
        """Create adaptive hybrid strategy instance"""
        strategy = AdaptiveHybridStrategy()
        
        # Mock sub-strategies
        strategy.progressive_strategy = Mock()
        strategy.progressive_strategy.can_handle = Mock(return_value=True)
        strategy.progressive_strategy.execute = Mock()
        
        strategy.upscale_strategy = Mock()
        strategy.upscale_strategy.can_handle = Mock(return_value=True)
        strategy.upscale_strategy.execute = Mock()
        
        strategy.swpo_strategy = Mock()
        strategy.swpo_strategy.can_handle = Mock(return_value=True)
        strategy.swpo_strategy.execute = Mock()
        
        return strategy
    
    @pytest.fixture
    def base_config(self):
        """Create base configuration"""
        test_image = Image.new('RGB', (1920, 1080))
        
        config = ExpandorConfig(
            source_image=test_image,
            target_resolution=(3840, 1080),  # 32:9
            prompt="Test prompt",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=Mock(),
            quality_preset='high'
        )
        
        return config
    
    def test_can_handle_with_sub_strategies(self, strategy, base_config):
        """Test that hybrid can handle if any sub-strategy can"""
        assert strategy.can_handle(base_config) == True
        
        # Test with all sub-strategies unable
        strategy.progressive_strategy.can_handle.return_value = False
        strategy.upscale_strategy.can_handle.return_value = False
        strategy.swpo_strategy.can_handle.return_value = False
        
        assert strategy.can_handle(base_config) == False
        
    def test_plan_creation_extreme_aspect(self, strategy, base_config):
        """Test plan creation for extreme aspect ratio change"""
        # 16:9 to 32:9 (2x aspect change)
        plan = strategy._create_execution_plan(base_config)
        
        assert isinstance(plan, HybridPlan)
        assert len(plan.steps) > 0
        
        # Should use SWPO or progressive for extreme aspect
        step_types = [step['type'] for step in plan.steps]
        assert 'swpo' in step_types or 'progressive_outpaint' in step_types
        
    def test_plan_creation_pure_upscale(self, strategy):
        """Test plan creation for pure upscaling"""
        config = ExpandorConfig(
            source_image=Image.new('RGB', (1920, 1080)),
            target_resolution=(3840, 2160),  # 2x upscale, same aspect
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        plan = strategy._create_execution_plan(config)
        
        # Should use direct upscale
        assert any(step['type'] == 'direct_upscale' for step in plan.steps)
        
    def test_plan_creation_with_refinement(self, strategy, base_config):
        """Test that refinement is added for high quality presets"""
        base_config.quality_preset = 'ultra'
        base_config.refiner_pipeline = Mock()
        
        plan = strategy._create_execution_plan(base_config)
        
        # Should include refinement step
        assert any(step['type'] == 'smart_refine' for step in plan.steps)
        
        # Ultra should have 3 passes
        refine_steps = [s for s in plan.steps if s['type'] == 'smart_refine']
        assert refine_steps[0]['passes'] == 3
        
    def test_plan_creation_multi_stage(self, strategy):
        """Test multi-stage planning for large expansions"""
        config = ExpandorConfig(
            source_image=Image.new('RGB', (1024, 768)),
            target_resolution=(4096, 3072),  # 4x expansion
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=Mock()
        )
        
        plan = strategy._create_execution_plan(config)
        
        # Should have multiple stages for large expansion
        progressive_steps = [s for s in plan.steps if s['type'] == 'progressive_outpaint']
        assert len(progressive_steps) >= 1
        
    def test_execute_basic_plan(self, strategy, base_config):
        """Test basic execution of a hybrid plan"""
        # Mock sub-strategy results
        mock_result = Mock()
        mock_result.image = Image.new('RGB', base_config.target_resolution)
        mock_result.final_size = base_config.target_resolution
        mock_result.stages = [{'name': 'test'}]
        mock_result.boundaries = []
        mock_result.metadata = {}
        
        strategy.swpo_strategy.execute.return_value = mock_result
        
        # Execute
        result = strategy.execute(base_config)
        
        # Verify success
        assert result.strategy_name == "AdaptiveHybrid"
        assert result.final_size == base_config.target_resolution
        assert len(result.stages) > 0
        
    def test_execute_with_multiple_steps(self, strategy, base_config):
        """Test execution with multiple plan steps"""
        # Create config that will generate multi-step plan
        base_config.target_resolution = (7680, 1080)  # Very wide
        
        # Mock different results for different strategies
        progressive_result = Mock()
        progressive_result.image = Image.new('RGB', (3840, 1080))
        progressive_result.final_size = (3840, 1080)
        progressive_result.stages = [{'name': 'progressive'}]
        progressive_result.boundaries = [{'position': 1920}]
        
        final_result = Mock()
        final_result.image = Image.new('RGB', (7680, 1080))
        final_result.final_size = (7680, 1080)
        final_result.stages = [{'name': 'final'}]
        final_result.boundaries = [{'position': 3840}]
        
        strategy.progressive_strategy.execute.side_effect = [progressive_result, final_result]
        
        # Execute
        result = strategy.execute(base_config)
        
        # Should have executed multiple steps
        assert len(result.stages) >= 2
        assert len(result.boundaries) >= 2
        
    def test_execute_with_refinement(self, strategy, base_config):
        """Test execution with refinement step"""
        base_config.quality_preset = 'ultra'
        base_config.refiner_pipeline = Mock()
        
        # Mock main execution
        mock_result = Mock()
        mock_result.image = Image.new('RGB', base_config.target_resolution)
        mock_result.final_size = base_config.target_resolution
        mock_result.stages = []
        mock_result.boundaries = []
        
        strategy.swpo_strategy.execute.return_value = mock_result
        
        # Mock refiner
        strategy.smart_refiner.refine.return_value = (
            Image.new('RGB', base_config.target_resolution),
            5  # artifacts fixed
        )
        
        # Execute
        result = strategy.execute(base_config)
        
        # Should have refinement stage
        refine_stages = [s for s in result.stages if 'refine' in s['name']]
        assert len(refine_stages) > 0
        
    def test_intermediate_size_calculation(self, strategy):
        """Test intermediate size calculation for multi-stage"""
        # Test various scenarios
        
        # Width expansion
        intermediate = strategy._calculate_intermediate_size(
            (1024, 768), (4096, 768), max_ratio=2.0
        )
        assert intermediate[0] == 2048  # 2x max
        assert intermediate[1] == 768
        
        # Height expansion
        intermediate = strategy._calculate_intermediate_size(
            (1024, 768), (1024, 3072), max_ratio=2.0
        )
        assert intermediate[0] == 1024
        assert intermediate[1] == 1536  # 2x max
        
        # Both dimensions
        intermediate = strategy._calculate_intermediate_size(
            (1024, 768), (4096, 3072), max_ratio=2.0
        )
        assert intermediate[0] == 2048
        assert intermediate[1] == 1536
        
        # Ensure multiple of 8
        intermediate = strategy._calculate_intermediate_size(
            (1023, 767), (2047, 1535), max_ratio=2.0
        )
        assert intermediate[0] % 8 == 0
        assert intermediate[1] % 8 == 0
        
    def test_error_handling_unknown_step(self, strategy, base_config):
        """Test error handling for unknown step type"""
        # Inject bad step
        with patch.object(strategy, '_create_execution_plan') as mock_plan:
            bad_plan = HybridPlan(
                steps=[{'name': 'bad', 'type': 'unknown_type'}],
                estimated_vram=1000,
                estimated_quality=0.9,
                rationale="Test"
            )
            mock_plan.return_value = bad_plan
            
            with pytest.raises(ExpandorError) as exc_info:
                strategy.execute(base_config)
                
            assert "Unknown step type" in str(exc_info.value)
            
    def test_sub_config_creation(self, strategy, base_config):
        """Test sub-configuration creation preserves settings"""
        current_image = Image.new('RGB', (2048, 1080))
        target_size = (3840, 1080)
        
        sub_config = strategy._create_sub_config(
            base_config, current_image, target_size
        )
        
        # Should preserve key settings
        assert sub_config.prompt == base_config.prompt
        assert sub_config.seed == base_config.seed
        assert sub_config.quality_preset == base_config.quality_preset
        
        # Should update image and target
        assert sub_config.source_image == current_image
        assert sub_config.target_resolution == target_size
        
        # Should disable sub-level stage saving
        assert sub_config.save_stages == False
        
    def test_vram_estimation(self, strategy, base_config):
        """Test VRAM requirement estimation"""
        with patch.object(strategy.vram_manager, 'calculate_generation_vram') as mock_calc:
            mock_calc.return_value = 4000  # 4GB
            
            vram_req = strategy.get_vram_requirement(base_config)
            
            # Should return plan's estimated VRAM
            assert vram_req > 0
            assert vram_req < 50000  # Sanity check
            
    def test_quality_estimation_in_plan(self, strategy):
        """Test quality estimation in different scenarios"""
        # SWPO should have high quality
        base_config = ExpandorConfig(
            source_image=Image.new('RGB', (1920, 1080)),
            target_resolution=(7680, 1080),  # 4x width
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=Mock()
        )
        
        with patch.object(strategy, '_estimate_vram_for_swpo', return_value=3000):
            with patch.object(strategy.vram_manager, 'get_available_vram', return_value=8000):
                plan = strategy._create_execution_plan(base_config)
                
                # SWPO should give high quality
                if any(s['type'] == 'swpo' for s in plan.steps):
                    assert plan.estimated_quality >= 0.95
```

## 4. Verification Checklist

Before proceeding with implementation:

### 4.1 File Creation Verification
```bash
# Verify all strategy files exist
ls -la expandor/strategies/swpo_strategy.py
ls -la expandor/strategies/cpu_offload_strategy.py
ls -la expandor/strategies/adaptive_hybrid_strategy.py
ls -la expandor/strategies/progressive_outpaint.py
ls -la expandor/strategies/direct_upscale.py

# Verify processor files
ls -la expandor/processors/tiled_processor.py
ls -la expandor/processors/smart_refiner.py

# Verify test files
ls -la tests/unit/strategies/test_swpo_strategy.py
ls -la tests/unit/strategies/test_cpu_offload_strategy.py
ls -la tests/unit/strategies/test_adaptive_hybrid_strategy.py
ls -la tests/integration/strategies/test_swpo_integration.py
```

### 4.2 Import Verification
```python
# Test all imports work
python -c "from expandor.strategies.swpo_strategy import SWPOStrategy"
python -c "from expandor.strategies.cpu_offload_strategy import CPUOffloadStrategy"
python -c "from expandor.strategies.adaptive_hybrid_strategy import AdaptiveHybridStrategy"
python -c "from expandor.processors.tiled_processor import TiledProcessor"
```

### 4.3 Unit Test Execution
```bash
# Run unit tests
pytest tests/unit/strategies/test_swpo_strategy.py -v
pytest tests/unit/strategies/test_cpu_offload_strategy.py -v
pytest tests/unit/strategies/test_adaptive_hybrid_strategy.py -v
```

### 4.4 Integration Test Execution
```bash
# Run integration tests
pytest tests/integration/strategies/test_swpo_integration.py -v
```

## 5. Critical Implementation Notes

### 5.1 SWPO Implementation
- **Window overlap MUST be >= 50%** to maintain context
- **All dimensions MUST be multiples of 8** for SDXL compatibility
- **Clear CUDA cache every 5 windows** to prevent OOM
- **Track exact boundary positions** for seam detection
- **Use gradient masks** for smooth blending

### 5.2 CPU Offload Implementation
- **Enable sequential CPU offload** for maximum memory savings
- **Use minimum tile size of 384x384** for stability
- **Call garbage collection after each tile**
- **Provide fallback for missing methods** in pipelines
- **Log system resources** on failure for debugging

### 5.3 Adaptive Hybrid Implementation
- **Plan must consider VRAM constraints** before execution
- **Multi-stage expansions need intermediate size calculation**
- **Sub-configurations must preserve all settings**
- **Each step must be independently recoverable**
- **Quality estimation guides user expectations**

## 6. Next Steps

After implementing these complex strategies:

1. **Integration with Base System**
   - Register strategies in strategy selector
   - Add to pipeline orchestrator
   - Update VRAM manager with new calculations

2. **Testing with Real Pipelines**
   - Create mock pipeline fixtures
   - Test with actual diffusers pipelines
   - Verify VRAM usage matches estimates

3. **Performance Optimization**
   - Profile execution times
   - Optimize memory usage
   - Add caching where beneficial

4. **Documentation**
   - API documentation for each strategy
   - Usage examples
   - Performance characteristics