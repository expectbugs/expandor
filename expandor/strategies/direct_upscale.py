"""
Direct Upscale Strategy - Simple but effective
Uses Real-ESRGAN for high-quality upscaling without aspect change
"""

import time
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

from .base_strategy import BaseExpansionStrategy
from ..core.exceptions import StrategyError, UpscalerError
from ..core.config import ExpandorConfig

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
    
    def execute(self, config: ExpandorConfig, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute direct upscaling strategy
        
        This is the simplest strategy - just upscale to target size
        """
        self._context = context or {}
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
            output_path = self._run_realesrgan(
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
        
        self.logger.debug(f"Running: {' '.join(cmd)}")
        
        # Execute upscaling
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if not output_path.exists():
                raise UpscalerError(
                    f"Real-ESRGAN failed to create output file",
                    tool_name="realesrgan",
                    exit_code=result.returncode
                )
            
            self.logger.debug("Real-ESRGAN completed successfully")
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Real-ESRGAN failed with exit code {e.returncode}"
            if e.stderr:
                error_msg += f"\nError: {e.stderr}"
            raise UpscalerError(
                error_msg,
                tool_name="realesrgan",
                exit_code=e.returncode
            )
        except Exception as e:
            raise UpscalerError(
                f"Failed to run Real-ESRGAN: {str(e)}",
                tool_name="realesrgan"
            )
    
    def _calculate_upscale_passes(self, source_size: Tuple[int, int], 
                                 target_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Calculate required upscaling passes"""
        source_w, source_h = source_size
        target_w, target_h = target_size
        
        # Calculate scale factors
        scale_w = target_w / source_w
        scale_h = target_h / source_h
        scale = max(scale_w, scale_h)
        
        passes = []
        current_scale = 1.0
        pass_num = 0
        
        while current_scale < scale:
            pass_num += 1
            # Determine optimal scale for this pass
            remaining_scale = scale / current_scale
            
            if remaining_scale <= 2.0:
                pass_scale = 2
            elif remaining_scale <= 3.0:
                pass_scale = 3
            else:
                pass_scale = 4
            
            current_scale *= pass_scale
            
            passes.append({
                'pass_num': pass_num,
                'scale': pass_scale,
                'cumulative_scale': current_scale,
                'model': 'RealESRGAN_x4plus' if pass_scale >= 3 else 'RealESRGAN_x2plus'
            })
        
        return passes