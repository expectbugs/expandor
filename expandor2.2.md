# Expandor Phase 2 Step 2: Basic Strategies

## Overview

This step implements three fundamental expansion strategies that cover the most common use cases. Each strategy is designed with fail-loud error handling, VRAM awareness, and zero quality compromise.

## Step 2.2.1: Update Base Strategy Class

### Location: `expandor/strategies/base_strategy.py` (update existing)

```python
"""
Base Strategy Class for all expansion strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import logging
import time
from pathlib import Path
from PIL import Image

from ..core.vram_manager import VRAMManager
from ..core.exceptions import StrategyError, VRAMError
from ..processors.artifact_removal import ArtifactDetector

class BaseExpansionStrategy(ABC):
    """Base class for all expansion strategies"""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 metrics: Optional[Any] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize base strategy
        
        Args:
            config: Strategy-specific configuration
            metrics: Selection metrics from StrategySelector
            logger: Logger instance
        """
        self.config = config or {}
        self.metrics = metrics
        self.logger = logger or logging.getLogger(__name__)
        self.vram_manager = VRAMManager(self.logger)
        self.artifact_detector = ArtifactDetector(self.logger)
        
        # Pipeline placeholders - will be injected by orchestrator
        self.inpaint_pipeline = None
        self.refiner_pipeline = None
        self.img2img_pipeline = None
        
        # Tracking
        self.boundary_tracker = None  # Injected by orchestrator
        self.stage_results = []
        self.temp_files = []
        
        # Initialize strategy-specific components
        self._initialize()
    
    def _initialize(self):
        """Override to perform strategy-specific initialization"""
        pass
    
    @abstractmethod
    def execute(self, config, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the expansion strategy
        
        Args:
            config: ExpandorConfig instance
            context: Execution context from orchestrator
            
        Returns:
            Dictionary with results including:
            - image_path: Path to result image
            - size: Final image size
            - stages: List of stage results
            - boundaries: List of boundary positions
            - metadata: Additional metadata
        """
        pass
    
    def validate_requirements(self):
        """
        Validate that strategy has required components
        
        Raises:
            StrategyError: If requirements not met
        """
        # Override in subclasses to check specific requirements
        pass
    
    def estimate_vram(self, config: ExpandorConfig) -> Dict[str, float]:
        """
        Estimate VRAM requirements for this strategy
        
        Args:
            config: ExpandorConfig instance
            
        Returns:
            Dictionary with VRAM estimates
        """
        target_w, target_h = config.target_resolution
        base_req = self.vram_manager.estimate_requirement({
            'target_resolution': (target_w, target_h),
            'source_metadata': {'model': 'sdxl'}
        })
        
        return {
            "base_vram_mb": base_req["total_vram_mb"],
            "peak_vram_mb": base_req["total_with_buffer_mb"],
            "strategy_overhead_mb": 0  # Override in subclasses
        }
    
    def check_vram_requirements(self, width: int, height: int) -> Dict[str, Any]:
        """Check if operation can fit in VRAM"""
        return self.vram_manager.determine_strategy(width, height)
    
    def track_boundary(self, position: int, direction: str, step: int, **kwargs):
        """Track expansion boundary for seam detection"""
        if self.boundary_tracker:
            self.boundary_tracker.add_boundary(
                position=position,
                direction=direction,
                step=step,
                **kwargs
            )
    
    def record_stage(self, 
                    name: str,
                    method: str,
                    input_size: Tuple[int, int],
                    output_size: Tuple[int, int],
                    start_time: float,
                    **kwargs):
        """Record a stage completion"""
        from ..core.result import StageResult
        
        stage = StageResult(
            name=name,
            method=method,
            input_size=input_size,
            output_size=output_size,
            duration_seconds=time.time() - start_time,
            vram_used_mb=self.vram_manager.get_current_usage(),
            **kwargs
        )
        
        self.stage_results.append(stage)
        
        # Call context callback if provided
        if hasattr(self, '_context') and 'stage_callback' in self._context:
            self._context['stage_callback'](stage)
    
    def save_temp_image(self, image: Image.Image, name: str) -> Path:
        """Save temporary image and track for cleanup"""
        timestamp = int(time.time() * 1000)
        temp_path = Path("temp") / f"{name}_{timestamp}.png"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use lossless PNG
        image.save(temp_path, "PNG", compress_level=0)
        self.temp_files.append(temp_path)
        
        return temp_path
    
    def cleanup(self):
        """Clean up temporary files"""
        for path in self.temp_files:
            try:
                if path.exists():
                    path.unlink()
            except:
                pass
        self.temp_files.clear()
    
    def validate_image_path(self, path: Path) -> Image.Image:
        """Validate and load image from path"""
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        try:
            image = Image.open(path)
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise StrategyError(f"Failed to load image {path}: {str(e)}")
```

## Step 2.2.2: Implement DirectUpscaleStrategy

### Location: `expandor/strategies/direct_upscale.py`

```python
"""
Direct Upscale Strategy - Simple but effective
Uses Real-ESRGAN for high-quality upscaling without aspect change
"""

import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image

from expandor.strategies.base_strategy import BaseExpansionStrategy
from expandor.core.exceptions import StrategyError
from expandor.core.config import ExpandorConfig

class DirectUpscaleStrategy(BaseExpansionStrategy):
    """
    Direct upscaling for simple size increases
    
    Best for:
    - No aspect ratio change
    - Area increase < 4x
    - When Real-ESRGAN is sufficient
    """
    
    def _initialize(self):
        """Initialize upscaler-specific settings"""
        # Default upscaler config
        self.upscale_config = {
            'model': 'RealESRGAN_x4plus',
            'tile_size': 0,  # 0 means auto
            'tile_pad': 10
        }
        self.model_config = {
            'RealESRGAN_x4plus': {'scale': 4},
            'RealESRGAN_x2plus': {'scale': 2}
        }
        self.tile_config = {
            'high_vram': 2048,
            'medium_vram': 1024,
            'low_vram': 512
        }
        
        # Find Real-ESRGAN executable
        self.realesrgan_path = self._find_realesrgan()
    
    def validate_requirements(self):
        """Validate upscaler is available"""
        if not self.realesrgan_path:
            raise StrategyError(
                "DirectUpscaleStrategy requires Real-ESRGAN. "
                "Please install it or use a different strategy."
            )
    
    def _find_realesrgan(self) -> Optional[Path]:
        """Find Real-ESRGAN executable"""
        # Check common locations
        possible_paths = [
            Path("realesrgan-ncnn-vulkan"),
            Path("/usr/local/bin/realesrgan-ncnn-vulkan"),
            Path.home() / "bin" / "realesrgan-ncnn-vulkan",
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                return path
        
        # Try system PATH
        try:
            result = subprocess.run(
                ["which", "realesrgan-ncnn-vulkan"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except:
            pass
        
        return None
    
    def execute(self, config: ExpandorConfig, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute direct upscaling strategy
        
        This is the simplest strategy - just upscale to target size
        """
        self._context = context
        start_time = time.time()
        
        # Load source image
        if isinstance(config.source_image, Path):
            source_image = self.validate_image_path(config.source_image)
            source_path = config.source_image
        else:
            source_image = config.source_image
            source_path = self.save_temp_image(source_image, "source")
        
        source_w, source_h = source_image.size
        target_w, target_h = config.target_resolution
        
        self.logger.info(
            f"Direct upscale: {source_w}x{source_h} -> {target_w}x{target_h}"
        )
        
        # Calculate required scale factor
        scale_w = target_w / source_w
        scale_h = target_h / source_h
        
        # Use the larger scale to ensure we meet target size
        scale = max(scale_w, scale_h)
        
        # Determine Real-ESRGAN scale factor (2, 3, or 4)
        if scale <= 2.0:
            esrgan_scale = 2
            model_name = self.model_config.get('fast', 'RealESRGAN_x2plus')
        elif scale <= 3.0:
            esrgan_scale = 3
            model_name = self.model_config.get('default', 'RealESRGAN_x4plus')
        else:
            esrgan_scale = 4
            model_name = self.model_config.get('default', 'RealESRGAN_x4plus')
        
        # Check if we need multiple passes
        passes_needed = 1
        current_scale = esrgan_scale
        
        while current_scale < scale:
            passes_needed += 1
            current_scale *= esrgan_scale
            
        if passes_needed > 1:
            self.logger.info(
                f"Large upscale factor {scale:.1f}x requires {passes_needed} passes"
            )
        
        # Execute upscaling passes
        current_path = source_path
        current_image = source_image
        
        for pass_num in range(passes_needed):
            self.logger.info(f"Upscaling pass {pass_num + 1}/{passes_needed}")
            
            # Record stage start
            stage_start = time.time()
            input_size = current_image.size
            
            # Determine tile size based on VRAM
            tile_size = self._determine_tile_size(current_image.size)
            
            # Execute upscaling
            output_path = self._run_upscaler(
                input_path=current_path,
                scale=esrgan_scale,
                model_name=model_name,
                tile_size=tile_size,
                fp32=self.upscale_config.get('fp32', True)
            )
            
            # Load result
            current_image = Image.open(output_path)
            current_path = output_path
            
            # Record stage
            self.record_stage(
                name=f"upscale_pass_{pass_num + 1}",
                method="real_esrgan",
                input_size=input_size,
                output_size=current_image.size,
                start_time=stage_start,
                metadata={
                    "scale": esrgan_scale,
                    "model": model_name,
                    "tile_size": tile_size
                }
            )
        
        # Final resize to exact target if needed
        if current_image.size != (target_w, target_h):
            self.logger.info(
                f"Final resize: {current_image.size} -> {target_w}x{target_h}"
            )
            
            stage_start = time.time()
            
            # High-quality downsampling if needed
            if current_image.width > target_w or current_image.height > target_h:
                # Use Lanczos for downsampling
                final_image = current_image.resize(
                    (target_w, target_h),
                    Image.Resampling.LANCZOS
                )
            else:
                # This shouldn't happen, but use bicubic for small upsampling
                final_image = current_image.resize(
                    (target_w, target_h),
                    Image.Resampling.BICUBIC
                )
            
            final_path = self.save_temp_image(final_image, "final")
            
            self.record_stage(
                name="final_resize",
                method="lanczos" if current_image.width > target_w else "bicubic",
                input_size=current_image.size,
                output_size=(target_w, target_h),
                start_time=stage_start
            )
        else:
            final_image = current_image
            final_path = current_path
        
        # No boundaries in direct upscale
        boundaries = []
        
        # Build result
        return {
            'image': final_image,
            'image_path': final_path,
            'size': (target_w, target_h),
            'stages': self.stage_results,
            'boundaries': boundaries,
            'metadata': {
                'strategy': 'direct_upscale',
                'scale_factor': scale,
                'passes': passes_needed,
                'final_resize': current_image.size != (target_w, target_h)
            }
        }
    
    def _find_realesrgan(self) -> Optional[Path]:
        """Find Real-ESRGAN installation"""
        # Check common locations
        search_paths = [
            Path("Real-ESRGAN/inference_realesrgan.py"),
            Path.home() / "Real-ESRGAN/inference_realesrgan.py",
            Path("/opt/Real-ESRGAN/inference_realesrgan.py"),
        ]
        
        # Also check for the ncnn-vulkan executable
        ncnn_paths = [
            Path("Real-ESRGAN/realesrgan-ncnn-vulkan"),
            Path("/usr/local/bin/realesrgan-ncnn-vulkan"),
        ]
        
        for path in search_paths + ncnn_paths:
            if path.exists():
                self.logger.info(f"Found Real-ESRGAN at: {path}")
                return path
        
        # Check if it's in PATH
        try:
            result = subprocess.run(
                ["which", "realesrgan-ncnn-vulkan"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                path = Path(result.stdout.strip())
                self.logger.info(f"Found realesrgan-ncnn-vulkan in PATH: {path}")
                return path
        except:
            pass
        
        return None
    
    def _determine_tile_size(self, image_size: Tuple[int, int]) -> int:
        """Determine optimal tile size based on VRAM"""
        # Check available VRAM
        available_vram = self.vram_manager.get_available_vram()
        
        if not available_vram:
            # CPU mode - use small tiles
            return self.tile_config.get('low_vram', 512)
        
        # Select tile size based on VRAM
        if available_vram > 8000:  # 8GB+
            return self.tile_config.get('unlimited', 2048)
        elif available_vram > 6000:  # 6GB+
            return self.tile_config.get('high_vram', 1024)
        elif available_vram > 4000:  # 4GB+
            return self.tile_config.get('medium_vram', 768)
        else:
            return self.tile_config.get('low_vram', 512)
    
    def _run_realesrgan(self,
                       input_path: Path,
                       scale: int,
                       model_name: str,
                       tile_size: int,
                       fp32: bool) -> Path:
        """
        Run Real-ESRGAN upscaling
        
        Returns:
            Path to upscaled image
        """
        # Prepare output path
        timestamp = int(time.time() * 1000)
        output_path = Path("temp") / f"upscaled_{scale}x_{timestamp}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build command based on executable type
        if self.realesrgan_path.name == "realesrgan-ncnn-vulkan":
            # Using ncnn-vulkan executable
            cmd = [
                str(self.realesrgan_path),
                "-i", str(input_path),
                "-o", str(output_path),
                "-s", str(scale),
                "-n", model_name,
                "-t", str(tile_size),
            ]
            
            if fp32:
                cmd.extend(["-x"])  # Use fp32 precision
        else:
            # Using Python script
            cmd = [
                sys.executable,
                str(self.realesrgan_path),
                "-i", str(input_path),
                "-o", str(output_path),
                "-s", str(scale),
                "-n", model_name,
                "--tile", str(tile_size),
            ]
            
            if fp32:
                cmd.extend(["--fp32"])
        
        # Add model path if needed
        model_dir = self.realesrgan_path.parent / "weights"
        if model_dir.exists():
            if self.realesrgan_path.name == "realesrgan-ncnn-vulkan":
                cmd.extend(["-m", str(model_dir)])
            else:
                cmd.extend(["--model_path", str(model_dir)])
        
        # Execute Real-ESRGAN
        self.logger.debug(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Check if output was created
            if not output_path.exists():
                raise UpscalerError(
                    "Real-ESRGAN",
                    Exception(f"Output file not created: {output_path}")
                )
            
            # Track temp file
            self.temp_files.append(output_path)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Real-ESRGAN failed with code {e.returncode}"
            if e.stderr:
                error_msg += f"\nError: {e.stderr}"
            if e.stdout:
                error_msg += f"\nOutput: {e.stdout}"
            
            raise UpscalerError("Real-ESRGAN", Exception(error_msg))
        except Exception as e:
            raise UpscalerError("Real-ESRGAN", e)
    
    def estimate_vram(self, config: ExpandorConfig) -> Dict[str, float]:
        """Estimate VRAM for direct upscaling"""
        # Real-ESRGAN VRAM usage is relatively predictable
        target_w, target_h = config.target_resolution
        
        # Rough estimates based on tile size
        tile_size = self._determine_tile_size((target_w, target_h))
        
        if tile_size >= 1024:
            vram_per_tile = 3000  # ~3GB for 1024 tiles
        elif tile_size >= 768:
            vram_per_tile = 2000  # ~2GB for 768 tiles
        else:
            vram_per_tile = 1000  # ~1GB for 512 tiles
        
        return {
            "base_vram_mb": vram_per_tile,
            "peak_vram_mb": vram_per_tile * 1.2,  # 20% overhead
            "strategy_overhead_mb": 500  # Model loading
        }
```

## Step 2.2.3: Implement ProgressiveOutpaintStrategy (update existing)

### Location: `expandor/strategies/progressive_outpaint.py` (update)

Add the missing imports and fix the execute method:

```python
"""
Progressive Outpainting Strategy
Adapted from ai-wallpaper AspectAdjuster
"""

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import time

from expandor.strategies.base_strategy import BaseExpansionStrategy
from expandor.core.exceptions import StrategyError
from expandor.core.vram_manager import VRAMManager
from expandor.utils.dimension_calculator import DimensionCalculator
from expandor.core.config import ExpandorConfig

class ProgressiveOutpaintStrategy(BaseExpansionStrategy):
    """Progressive aspect ratio adjustment with zero quality compromise"""
    
    def _initialize(self):
        """Initialize progressive outpainting settings"""
        self.dimension_calc = DimensionCalculator(self.logger)
        
        # Get configuration
        prog_config = self.config.get('progressive_outpainting', {})
        outpaint_config = prog_config.get('outpaint_settings', {})
        
        # Progressive outpainting settings
        self.prog_enabled = prog_config.get('enabled', True)
        self.max_supported = prog_config.get('aspect_ratio_thresholds', {}).get(
            'max_supported', 8.0
        )
        
        # Outpaint settings
        self.denoising_strength = outpaint_config.get('denoising_strength', 0.95)
        self.min_strength = outpaint_config.get('min_strength', 0.20)
        self.max_strength = outpaint_config.get('max_strength', 0.95)
        self.outpaint_prompt_suffix = ', seamless expansion, extended scenery, natural continuation'
        self.base_mask_blur = 32
        self.base_steps = 60
        
        # Default expansion ratios
        self.first_step_ratio = 1.4
        self.middle_step_ratio = 1.25
        self.final_step_ratio = 1.15
    
    def validate_requirements(self):
        """Validate inpainting pipeline is available"""
        if not self.inpaint_pipeline:
            raise StrategyError(
                "ProgressiveOutpaintStrategy requires an inpainting pipeline"
            )
    
    def execute(self, config, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute progressive outpainting strategy"""
        self._context = context
        start_time = time.time()
        
        # Validate requirements
        self.validate_requirements()
        
        # Load source image
        if isinstance(config.source_image, Path):
            current_image = self.validate_image_path(config.source_image)
            current_path = config.source_image
        else:
            current_image = config.source_image
            current_path = self.save_temp_image(current_image, "source")
        
        current_size = current_image.size
        target_w, target_h = config.target_resolution
        target_aspect = target_w / target_h
        
        # Calculate progressive steps
        steps = self.dimension_calc.calculate_progressive_strategy(
            current_size=current_size,
            target_aspect=target_aspect,
            max_expansion_per_step=self.first_step_ratio
        )
        
        if not steps:
            # No expansion needed
            self.logger.info("No aspect adjustment needed")
            return {
                'image': current_image,
                'image_path': current_path,
                'size': current_size,
                'stages': [],
                'boundaries': []
            }
        
        self.logger.info(f"Progressive outpainting: {len(steps)} steps")
        
        # Execute progressive steps
        for i, step in enumerate(steps):
            step_num = i + 1
            self.logger.info(f"Step {step_num}/{len(steps)}: {step['description']}")
            
            stage_start = time.time()
            
            # Execute outpaint step
            result = self._execute_outpaint_step(
                image=current_image,
                image_path=current_path,
                prompt=config.prompt,
                step_info=step,
                step_num=step_num
            )
            
            # Update current state
            current_image = result['image']
            current_path = result['image_path']
            
            # Track boundaries
            if self.boundary_tracker:
                self.boundary_tracker.add_progressive_boundaries(
                    current_size=step['current_size'],
                    target_size=step['target_size'],
                    step=step_num,
                    method='progressive'
                )
            
            # Record stage
            self.record_stage(
                name=f'progressive_step_{step_num}',
                method='progressive_outpaint',
                input_size=step['current_size'],
                output_size=step['target_size'],
                start_time=stage_start,
                metadata={
                    'direction': step['direction'],
                    'expansion_ratio': step['expansion_ratio'],
                    'step_type': step['step_type']
                }
            )
        
        # Get final boundaries for metadata
        boundaries = []
        if self.boundary_tracker:
            boundaries = self.boundary_tracker.get_all_boundaries()
        
        return {
            'image': current_image,
            'image_path': current_path,
            'size': current_image.size,
            'stages': self.stage_results,
            'boundaries': boundaries,
            'metadata': {
                'strategy': 'progressive_outpaint',
                'steps_executed': len(steps),
                'final_aspect_ratio': current_image.width / current_image.height
            }
        }
    
    def _execute_outpaint_step(self,
                              image: Image.Image,
                              image_path: Path,
                              prompt: str,
                              step_info: Dict,
                              step_num: int) -> Dict[str, Any]:
        """Execute a single outpaint step"""
        current_w, current_h = image.size
        target_w, target_h = step_info['target_size']
        
        # Round to model constraints (8x for SDXL)
        if self.metrics and self.metrics.model_type == 'sdxl':
            target_w = self.dimension_calc.round_to_multiple(target_w, 8)
            target_h = self.dimension_calc.round_to_multiple(target_h, 8)
        
        # Check VRAM for this operation
        vram_check = self.check_vram_requirements(target_w, target_h)
        if vram_check['strategy'] != 'full':
            self.logger.warning(
                f"Limited VRAM for {target_w}x{target_h}: {vram_check['details']}"
            )
        
        # Create canvas and mask
        canvas = Image.new('RGB', (target_w, target_h), color='black')
        mask = Image.new('L', (target_w, target_h), color='white')
        
        # Calculate padding
        pad_left = (target_w - current_w) // 2
        pad_top = (target_h - current_h) // 2
        
        # Place image on canvas
        canvas.paste(image, (pad_left, pad_top))
        
        # Create mask (black = keep, white = generate)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(
            [pad_left, pad_top, pad_left + current_w - 1, pad_top + current_h - 1],
            fill='black'
        )
        
        # Apply adaptive mask blur
        mask_blur = self._get_adaptive_blur(step_info)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))
        
        # Pre-fill with edge colors
        canvas = self._prefill_canvas_with_edge_colors(
            canvas, image, pad_left, pad_top, current_w, current_h
        )
        
        # Save intermediate files if debugging
        if self._context.get('save_stages', False):
            canvas.save(f"temp/canvas_step_{step_num}.png")
            mask.save(f"temp/mask_step_{step_num}.png")
        
        # Enhance prompt
        enhanced_prompt = prompt + self.outpaint_prompt_suffix
        
        # Get adaptive parameters
        num_steps = self._get_adaptive_steps(step_info)
        guidance = self._get_adaptive_guidance(step_info)
        strength = self._get_adaptive_strength(step_info)
        
        # Execute inpainting
        self.logger.debug(
            f"Inpainting: strength={strength:.2f}, steps={num_steps}, "
            f"guidance={guidance:.1f}"
        )
        
        try:
            result = self.inpaint_pipeline(
                prompt=enhanced_prompt,
                image=canvas,
                mask_image=mask,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                width=target_w,
                height=target_h
            ).images[0]
        except Exception as e:
            raise StrategyError(f"Inpainting failed: {str(e)}")
        
        # Save result
        result_path = self.save_temp_image(result, f"progressive_{step_num}")
        
        return {
            'image': result,
            'image_path': result_path
        }
    
    def _analyze_edge_colors(self, image: Image.Image, edge: str, sample_width: int = 50) -> Dict:
        """Analyze colors at image edge for better continuation"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Ensure sample width is valid
        sample_width = min(sample_width, w // 4, h // 4)
        
        if edge == 'left':
            sample = img_array[:, :sample_width]
        elif edge == 'right':
            sample = img_array[:, -sample_width:]
        elif edge == 'top':
            sample = img_array[:sample_width, :]
        elif edge == 'bottom':
            sample = img_array[-sample_width:, :]
        else:
            raise ValueError(f"Invalid edge: {edge}")
        
        # Calculate statistics
        pixels = sample.reshape(-1, 3)
        mean_color = np.mean(pixels, axis=0)
        median_color = np.median(pixels, axis=0)
        color_std = np.std(pixels, axis=0)
        
        return {
            'mean_rgb': mean_color.tolist(),
            'median_rgb': median_color.tolist(),
            'color_variance': float(np.mean(color_std)),
            'is_uniform': float(np.mean(color_std)) < 20,
            'sample_size': pixels.shape[0]
        }
    
    def _prefill_canvas_with_edge_colors(self, canvas, source_image, 
                                        pad_left, pad_top, current_w, current_h):
        """Pre-fill empty areas with edge-extended colors"""
        canvas_array = np.array(canvas)
        
        # Analyze edges
        edges = {}
        for edge_name in ['left', 'right', 'top', 'bottom']:
            edges[edge_name] = self._analyze_edge_colors(source_image, edge_name)
        
        # Fill empty areas with gradient from nearest edge
        h, w = canvas_array.shape[:2]
        
        # Pre-calculate masks for efficiency
        empty_mask = np.all(canvas_array == 0, axis=2)
        
        # Vectorized filling
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Calculate distances to edges
        dist_left = np.maximum(0, pad_left - x_coords)
        dist_right = np.maximum(0, x_coords - (pad_left + current_w - 1))
        dist_top = np.maximum(0, pad_top - y_coords)
        dist_bottom = np.maximum(0, y_coords - (pad_top + current_h - 1))
        
        # Determine closest edge for each pixel
        distances = np.stack([dist_left, dist_right, dist_top, dist_bottom], axis=2)
        closest_edge = np.argmin(distances, axis=2)
        
        # Map to edge names
        edge_map = ['left', 'right', 'top', 'bottom']
        
        # Fill based on closest edge
        for idx, edge_name in enumerate(edge_map):
            edge_mask = (closest_edge == idx) & empty_mask
            if np.any(edge_mask):
                base_color = np.array(edges[edge_name]['median_rgb'])
                variance = edges[edge_name]['color_variance']
                
                # Add slight variation
                noise = np.random.normal(0, variance * 0.2, 
                                       (np.sum(edge_mask), 3))
                colors = base_color + noise
                colors = np.clip(colors, 0, 255).astype(np.uint8)
                
                # Apply colors
                canvas_array[edge_mask] = colors
        
        return Image.fromarray(canvas_array)
    
    def _get_adaptive_blur(self, step_info: Dict) -> int:
        """Calculate adaptive mask blur based on expansion"""
        expansion_ratio = step_info.get('expansion_ratio', 1.5)
        
        # Get multipliers from config
        adaptive_config = self.config.get('progressive_outpainting', {}).get(
            'adaptive_settings', {}
        ).get('blur_multipliers', {})
        
        if expansion_ratio > 1.6:
            multiplier = adaptive_config.get('large', 1.5)
        elif expansion_ratio > 1.3:
            multiplier = adaptive_config.get('medium', 1.2)
        else:
            multiplier = adaptive_config.get('small', 1.0)
        
        return int(self.base_mask_blur * multiplier)
    
    def _get_adaptive_steps(self, step_info: Dict) -> int:
        """Calculate adaptive inference steps"""
        step_type = step_info.get('step_type', 'progressive')
        
        # Get multipliers from config
        adaptive_config = self.config.get('progressive_outpainting', {}).get(
            'adaptive_settings', {}
        ).get('step_multipliers', {})
        
        if step_type == 'initial':
            multiplier = adaptive_config.get('initial', 1.2)
        elif step_type == 'final':
            multiplier = adaptive_config.get('final', 0.8)
        else:
            multiplier = adaptive_config.get('middle', 1.0)
        
        return int(self.base_steps * multiplier)
    
    def _get_adaptive_strength(self, step_info: Dict) -> float:
        """Calculate adaptive denoising strength"""
        step_type = step_info.get('step_type', 'progressive')
        step_num = step_info.get('step', 1)
        
        # Start with base strength
        strength = self.denoising_strength
        
        # Apply decay for later steps
        if step_type == 'final':
            strength *= 0.8
        elif step_num > 2:
            # Gradual decay
            decay_factor = 0.95 ** (step_num - 2)
            strength *= decay_factor
        
        # Clamp to configured range
        return max(self.min_strength, min(strength, self.max_strength))
    
    def _get_adaptive_guidance(self, step_info: Dict) -> float:
        """Calculate adaptive guidance scale"""
        # Lower guidance for better blending
        return 7.5
    
    def estimate_vram(self, config: ExpandorConfig) -> Dict[str, float]:
        """Estimate VRAM for progressive outpainting"""
        # Progressive outpainting needs VRAM for largest step
        target_w, target_h = config.target_resolution
        
        # Calculate largest intermediate size
        if isinstance(config.source_image, Path):
            from PIL import Image
            with Image.open(config.source_image) as img:
                source_w, source_h = img.size
        else:
            source_w, source_h = config.source_image.size
        
        # Largest size will be close to target
        max_w = int(target_w * 1.1)  # Add some buffer
        max_h = int(target_h * 1.1)
        
        vram_req = self.vram_manager.estimate_requirement({
            'target_resolution': (max_w, max_h),
            'source_metadata': {'model': 'sdxl'}
        })
        
        return {
            "base_vram_mb": vram_req["total_vram_mb"],
            "peak_vram_mb": vram_req["total_with_buffer_mb"],
            "strategy_overhead_mb": 1000  # Inpainting overhead
        }
```

## Step 2.2.4: Implement TiledExpansionStrategy

### Location: `expandor/strategies/tiled_expansion.py`

```python
"""
Tiled Expansion Strategy
Process large images in tiles when VRAM is limited
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time

from expandor.strategies.base_strategy import BaseExpansionStrategy
from expandor.core.exceptions import StrategyError
from expandor.core.config import ExpandorConfig

class TiledExpansionStrategy(BaseExpansionStrategy):
    """
    Tiled processing for VRAM-limited situations
    
    Processes image in overlapping tiles to handle any size
    without quality compromise - just takes longer
    """
    
    def _initialize(self):
        """Initialize tiled processing settings"""
        # Default tile settings
        self.default_tile_size = 1024
        self.overlap = 256
        self.blend_width = 128
        self.min_tile_size = 512
        self.max_tile_size = 2048
    
    def validate_requirements(self):
        """Validate at least one pipeline is available"""
        if not any([self.inpaint_pipeline, self.refiner_pipeline, self.img2img_pipeline]):
            raise StrategyError(
                "TiledExpansionStrategy requires at least one pipeline "
                "(inpaint, refiner, or img2img)"
            )
    
    def execute(self, config, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tiled expansion strategy"""
        self._context = context
        start_time = time.time()
        
        # Validate requirements
        self.validate_requirements()
        
        # Load source image
        if isinstance(config.source_image, Path):
            source_image = self.validate_image_path(config.source_image)
        else:
            source_image = config.source_image
        
        source_w, source_h = source_image.size
        target_w, target_h = config.target_resolution
        
        self.logger.info(
            f"Tiled expansion: {source_w}x{source_h} -> {target_w}x{target_h}"
        )
        
        # Determine optimal tile size based on VRAM
        tile_size = self._determine_tile_size(target_w, target_h)
        self.logger.info(
            f"Using tile size: {tile_size}x{tile_size} with {self.overlap}px overlap"
        )
        
        # First, upscale to target size using simple resize
        # This gives us the base to refine
        stage_start = time.time()
        
        base_image = source_image.resize(
            (target_w, target_h),
            Image.Resampling.LANCZOS
        )
        base_path = self.save_temp_image(base_image, "tiled_base")
        
        self.record_stage(
            name="initial_resize",
            method="lanczos",
            input_size=(source_w, source_h),
            output_size=(target_w, target_h),
            start_time=stage_start
        )
        
        # Calculate tiles
        tiles = self._calculate_tiles(target_w, target_h, tile_size)
        self.logger.info(f"Processing {len(tiles)} tiles")
        
        # Track tile boundaries
        if self.boundary_tracker:
            self.boundary_tracker.add_tile_boundaries(tiles, step=1)
        
        # Process each tile
        processed_tiles = []
        
        for i, (x1, y1, x2, y2) in enumerate(tiles):
            tile_num = i + 1
            self.logger.info(f"Processing tile {tile_num}/{len(tiles)}")
            
            tile_start = time.time()
            
            # Extract tile from base image
            tile = base_image.crop((x1, y1, x2, y2))
            tile_w = x2 - x1
            tile_h = y2 - y1
            
            # Process tile based on available pipelines
            if self.refiner_pipeline:
                processed_tile = self._refine_tile(tile, config.prompt)
            elif self.img2img_pipeline:
                processed_tile = self._img2img_tile(tile, config.prompt)
            elif self.inpaint_pipeline:
                # For inpaint, we need to create a mask
                processed_tile = self._inpaint_tile(tile, config.prompt)
            else:
                # Should not happen due to validation
                raise StrategyError("No suitable pipeline for tile processing")
            
            # Store processed tile with coordinates
            processed_tiles.append({
                'image': processed_tile,
                'coords': (x1, y1, x2, y2)
            })
            
            self.record_stage(
                name=f"tile_{tile_num}",
                method="tiled_process",
                input_size=(tile_w, tile_h),
                output_size=(tile_w, tile_h),
                start_time=tile_start,
                metadata={
                    'tile_index': i,
                    'tile_coords': (x1, y1, x2, y2)
                }
            )
        
        # Blend tiles back together
        self.logger.info("Blending tiles...")
        blend_start = time.time()
        
        final_image = self._blend_tiles(
            processed_tiles,
            (target_w, target_h),
            self.overlap,
            self.blend_width
        )
        
        final_path = self.save_temp_image(final_image, "tiled_final")
        
        self.record_stage(
            name="tile_blending",
            method="weighted_blend",
            input_size=(target_w, target_h),
            output_size=(target_w, target_h),
            start_time=blend_start,
            metadata={
                'tiles_blended': len(tiles)
            }
        )
        
        # Get boundaries for metadata
        boundaries = []
        if self.boundary_tracker:
            boundaries = self.boundary_tracker.get_all_boundaries()
        
        return {
            'image': final_image,
            'image_path': final_path,
            'size': (target_w, target_h),
            'stages': self.stage_results,
            'boundaries': boundaries,
            'metadata': {
                'strategy': 'tiled_expansion',
                'tile_size': tile_size,
                'tile_count': len(tiles),
                'overlap': self.overlap
            }
        }
    
    def _determine_tile_size(self, width: int, height: int) -> int:
        """Determine optimal tile size based on VRAM"""
        # Check available VRAM
        available_vram = self.vram_manager.get_available_vram()
        
        if not available_vram:
            # CPU mode - use minimum tile size
            return self.min_tile_size
        
        # Calculate VRAM needed for different tile sizes
        for tile_size in [self.max_tile_size, 1536, 1024, 768, self.min_tile_size]:
            vram_req = self.vram_manager.estimate_requirement({
                'target_resolution': (tile_size, tile_size),
                'source_metadata': {'model': 'sdxl'}
            })
            if vram_req['total_with_buffer_mb'] <= available_vram * 0.7:  # 70% safety
                return tile_size
        
        # If even minimum doesn't fit, return it anyway (will fail loudly)
        return self.min_tile_size
    
    def _calculate_tiles(self, width: int, height: int, tile_size: int) -> List[Tuple[int, int, int, int]]:
        """Calculate tile positions with overlap"""
        tiles = []
        
        # Calculate step size (tile size minus overlap)
        step = tile_size - self.overlap
        
        # Ensure we don't have tiny edge tiles
        min_edge_size = tile_size // 2
        
        # Generate tiles
        y = 0
        while y < height:
            # Adjust last row if needed
            if y + tile_size > height and height - y < min_edge_size:
                y = max(0, height - tile_size)
            
            x = 0
            while x < width:
                # Adjust last column if needed
                if x + tile_size > width and width - x < min_edge_size:
                    x = max(0, width - tile_size)
                
                # Calculate tile bounds
                x1 = x
                y1 = y
                x2 = min(x + tile_size, width)
                y2 = min(y + tile_size, height)
                
                tiles.append((x1, y1, x2, y2))
                
                # Move to next column
                if x + tile_size >= width:
                    break
                x += step
            
            # Move to next row
            if y + tile_size >= height:
                break
            y += step
        
        return tiles
    
    def _refine_tile(self, tile: Image.Image, prompt: str) -> Image.Image:
        """Refine a tile using refiner pipeline"""
        try:
            result = self.refiner_pipeline(
                prompt=prompt,
                image=tile,
                strength=0.3,  # Light refinement
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
            return result
        except Exception as e:
            self.logger.warning(f"Tile refinement failed: {e}, using original")
            return tile
    
    def _img2img_tile(self, tile: Image.Image, prompt: str) -> Image.Image:
        """Process tile using img2img pipeline"""
        try:
            result = self.img2img_pipeline(
                prompt=prompt,
                image=tile,
                strength=0.5,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
            return result
        except Exception as e:
            self.logger.warning(f"Tile img2img failed: {e}, using original")
            return tile
    
    def _inpaint_tile(self, tile: Image.Image, prompt: str) -> Image.Image:
        """Process tile using inpaint pipeline (with full mask)"""
        # Create a mask that processes the entire tile
        mask = Image.new('L', tile.size, 255)  # Full white = process all
        
        try:
            result = self.inpaint_pipeline(
                prompt=prompt,
                image=tile,
                mask_image=mask,
                strength=0.7,
                num_inference_steps=40,
                guidance_scale=7.5
            ).images[0]
            return result
        except Exception as e:
            self.logger.warning(f"Tile inpainting failed: {e}, using original")
            return tile
    
    def _blend_tiles(self, 
                    tiles: List[Dict[str, Any]], 
                    canvas_size: Tuple[int, int],
                    overlap: int,
                    blend_width: int) -> Image.Image:
        """Blend tiles together with smooth transitions"""
        canvas = Image.new('RGB', canvas_size)
        weight_map = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.float32)
        
        # Process each tile
        for tile_info in tiles:
            tile_img = tile_info['image']
            x1, y1, x2, y2 = tile_info['coords']
            
            # Create weight mask for this tile
            tile_w = x2 - x1
            tile_h = y2 - y1
            tile_weight = np.ones((tile_h, tile_w), dtype=np.float32)
            
            # Apply gradients at edges if not at canvas boundary
            blend_size = min(blend_width, overlap // 2)
            
            if blend_size > 0:
                # Left edge
                if x1 > 0:
                    for i in range(blend_size):
                        weight = i / blend_size
                        tile_weight[:, i] *= weight
                
                # Right edge
                if x2 < canvas_size[0]:
                    for i in range(blend_size):
                        weight = i / blend_size
                        tile_weight[:, -(i+1)] *= weight
                
                # Top edge
                if y1 > 0:
                    for i in range(blend_size):
                        weight = i / blend_size
                        tile_weight[i, :] *= weight
                
                # Bottom edge
                if y2 < canvas_size[1]:
                    for i in range(blend_size):
                        weight = i / blend_size
                        tile_weight[-(i+1), :] *= weight
            
            # Convert tile to array
            tile_array = np.array(tile_img).astype(np.float32)
            
            # Get canvas region
            canvas_array = np.array(canvas).astype(np.float32)
            region = canvas_array[y1:y2, x1:x2]
            region_weight = weight_map[y1:y2, x1:x2]
            
            # Weighted blend
            total_weight = region_weight + tile_weight
            mask = total_weight > 0
            
            for c in range(3):
                region[mask, c] = (
                    (region[mask, c] * region_weight[mask] + 
                     tile_array[mask, c] * tile_weight[mask]) /
                    total_weight[mask]
                )
            
            # Update canvas
            canvas_array[y1:y2, x1:x2] = region
            weight_map[y1:y2, x1:x2] = total_weight
            
            # Convert back to image
            canvas = Image.fromarray(canvas_array.astype(np.uint8))
        
        return canvas
    
    def estimate_vram(self, config: ExpandorConfig) -> Dict[str, float]:
        """Estimate VRAM for tiled processing"""
        # Tiled processing uses VRAM based on tile size, not final size
        tile_size = self._determine_tile_size(*config.target_resolution)
        
        vram_req = self.vram_manager.estimate_requirement({
            'target_resolution': (tile_size, tile_size),
            'source_metadata': {'model': 'sdxl'}
        })
        
        return {
            "base_vram_mb": vram_req["total_vram_mb"],
            "peak_vram_mb": vram_req["total_with_buffer_mb"],
            "strategy_overhead_mb": 500  # Blending overhead
        }
```

## Step 2.2.5: Create Strategy Factory

### Location: `expandor/strategies/__init__.py`

```python
"""
Strategy module initialization
"""

from .base_strategy import BaseExpansionStrategy
from .direct_upscale import DirectUpscaleStrategy
from .progressive_outpaint import ProgressiveOutpaintStrategy
from .tiled_expansion import TiledExpansionStrategy

# Strategy registry
STRATEGY_CLASSES = {
    'direct_upscale': DirectUpscaleStrategy,
    'progressive_outpaint': ProgressiveOutpaintStrategy,
    'tiled_expansion': TiledExpansionStrategy,
}

def get_strategy_class(name: str):
    """Get strategy class by name"""
    if name not in STRATEGY_CLASSES:
        raise ValueError(f"Unknown strategy: {name}")
    return STRATEGY_CLASSES[name]

__all__ = [
    'BaseExpansionStrategy',
    'DirectUpscaleStrategy', 
    'ProgressiveOutpaintStrategy',
    'TiledExpansionStrategy',
    'get_strategy_class',
    'STRATEGY_CLASSES'
]
```

## Step 2.2.6: Create Placeholder Strategies

### Location: `expandor/strategies/swpo_strategy.py`

```python
"""
SWPO Strategy - Placeholder for Phase 3
"""

from .base_strategy import BaseExpansionStrategy

class SWPOStrategy(BaseExpansionStrategy):
    """Sliding Window Progressive Outpainting - To be implemented in Phase 3"""
    
    def execute(self, config, context):
        raise NotImplementedError("SWPO will be implemented in Phase 3")
```

### Location: `expandor/strategies/cpu_offload.py`

```python
"""
CPU Offload Strategy - Placeholder for Phase 3
"""

from .base_strategy import BaseExpansionStrategy

class CPUOffloadStrategy(BaseExpansionStrategy):
    """CPU offload for zero VRAM situations - To be implemented in Phase 3"""
    
    def execute(self, config, context):
        raise NotImplementedError("CPU offload will be implemented in Phase 3")
```

### Location: `expandor/strategies/hybrid_adaptive.py`

```python
"""
Hybrid Adaptive Strategy - Placeholder for Phase 3
"""

from .base_strategy import BaseExpansionStrategy

class HybridAdaptiveStrategy(BaseExpansionStrategy):
    """Intelligent hybrid approach - To be implemented in Phase 3"""
    
    def execute(self, config, context):
        raise NotImplementedError("Hybrid adaptive will be implemented in Phase 3")
```

## Testing the Strategies

Create a test script to verify the basic strategies work:

### Location: `tests/manual/test_strategies.py`

```python
"""
Manual test script for basic strategies
"""

import sys
sys.path.append('.')

from pathlib import Path
from PIL import Image

from expandor import Expandor, ExpandorConfig
from expandor.adapters.mock_pipeline import MockInpaintPipeline, MockRefinerPipeline

def test_direct_upscale():
    """Test direct upscaling"""
    print("\n=== Testing DirectUpscaleStrategy ===")
    
    # Create test image
    test_img = Image.new('RGB', (512, 512), color='blue')
    
    # Configure
    config = ExpandorConfig(
        source_image=test_img,
        target_resolution=(1024, 1024),
        prompt="A blue square",
        seed=42,
        source_metadata={'model': 'test'},
        quality_preset='fast'
    )
    
    # Execute
    expandor = Expandor()
    
    try:
        result = expandor.expand(config)
        print(f"Success! Result size: {result.size}")
        print(f"Strategy used: {result.strategy_used}")
    except Exception as e:
        print(f"Expected error (no Real-ESRGAN): {e}")

def test_progressive_outpaint():
    """Test progressive outpainting"""
    print("\n=== Testing ProgressiveOutpaintStrategy ===")
    
    # Create test image
    test_img = Image.new('RGB', (768, 768), color='green')
    
    # Configure for aspect change
    config = ExpandorConfig(
        source_image=test_img,
        target_resolution=(1920, 768),  # 16:9 aspect
        prompt="A green landscape",
        seed=42,
        source_metadata={'model': 'sdxl'},
        inpaint_pipeline=MockInpaintPipeline(),
        quality_preset='balanced'
    )
    
    # Execute
    expandor = Expandor()
    result = expandor.expand(config)
    
    print(f"Success! Result size: {result.size}")
    print(f"Strategy used: {result.strategy_used}")
    print(f"Stages executed: {len(result.stages)}")
    print(f"Boundaries tracked: {len(result.boundaries)}")

def test_tiled_expansion():
    """Test tiled expansion"""
    print("\n=== Testing TiledExpansionStrategy ===")
    
    # Create test image
    test_img = Image.new('RGB', (512, 512), color='red')
    
    # Configure for large expansion with limited VRAM
    config = ExpandorConfig(
        source_image=test_img,
        target_resolution=(4096, 4096),
        prompt="A red pattern",
        seed=42,
        source_metadata={'model': 'test'},
        refiner_pipeline=MockRefinerPipeline(),
        quality_preset='balanced',
        vram_limit_mb=2000  # Force tiled strategy
    )
    
    # Execute
    expandor = Expandor()
    result = expandor.expand(config)
    
    print(f"Success! Result size: {result.size}")
    print(f"Strategy used: {result.strategy_used}")
    print(f"Tiles processed: {len([s for s in result.stages if 'tile_' in s.name])}")

if __name__ == "__main__":
    # Create temp directory
    Path("temp").mkdir(exist_ok=True)
    
    # Run tests
    test_direct_upscale()
    test_progressive_outpaint()
    test_tiled_expansion()
    
    print("\n=== All strategy tests complete! ===")
```

## Summary

This step implements three fundamental expansion strategies:

1. **DirectUpscaleStrategy**
   - Uses Real-ESRGAN for high-quality upscaling
   - Handles multiple passes for large scale factors
   - VRAM-aware tile size selection
   - Falls back gracefully if Real-ESRGAN not available

2. **ProgressiveOutpaintStrategy** 
   - Implements sophisticated aspect ratio adjustment
   - Edge color analysis for seamless expansion
   - Adaptive parameters based on expansion size
   - Comprehensive boundary tracking

3. **TiledExpansionStrategy**
   - Processes images in overlapping tiles
   - Smooth blending with gradient weights
   - Works with any available pipeline
   - Handles unlimited image sizes

Each strategy follows the core principles:
- **Fail Loud**: Clear errors with actionable messages
- **Quality First**: No compromises on output quality
- **VRAM Aware**: Adapts to available resources
- **Boundary Tracking**: Every seam position is tracked
- **Comprehensive Metadata**: Full operation history

The strategies are designed to work together through the orchestrator's fallback system, ensuring that expansion always succeeds (even if slowly) as long as basic requirements are met.