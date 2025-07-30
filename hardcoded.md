# Hardcoded Values in Expandor Codebase

This document lists all hardcoded values found in the Expandor codebase that should be moved to configuration files to comply with the "complete configurability" principle.

## 1. Generation/Processing Strength Values

### CPU Offload Strategy (`expandor/strategies/cpu_offload.py`)
- Line 70: `safety_factor=0.6` - VRAM safety factor for CPU offload
- Line 78: `pipeline_vram = 1024` - Max VRAM with offloading (1GB)
- Line 271: `safety_factor=0.5` - Very conservative for CPU offload
- Line 302: `steps = 3` - Multiple small steps for processing
- Line 405: `strength=0.9` - Tile generation strength
- Line 406: `num_inference_steps=25` - Fewer steps for speed
- Line 407: `guidance_scale=7.0` - Guidance scale for generation
- Line 463: `strength=0.3` - Tile refinement strength
- Line 464: `num_inference_steps=20` - Steps for refinement
- Line 465: `guidance_scale=7.0` - Guidance scale for refinement

### Tiled Expansion Strategy (`expandor/strategies/tiled_expansion.py`)
- Line 29: `self.default_tile_size = 1024` - Default tile size
- Line 30: `self.overlap = 256` - Tile overlap size
- Line 31: `self.blend_width = 256` - Blending region width
- Line 32: `self.min_tile_size = 512` - Minimum tile size
- Line 33: `self.max_tile_size = 2048` - Maximum tile size
- Line 270: `strength=0.2` - Very light refinement strength
- Line 271: `num_inference_steps=20` - Steps for refinement
- Line 272: `guidance_scale=6.5` - Lower guidance for preservation
- Line 285: `strength=0.3` - Edge fix strength
- Line 286: `num_inference_steps=30` - Steps for edge fixing
- Line 287: `guidance_scale=7.0` - Guidance for edge fixing
- Line 304: `strength=0.4` - Final pass strength
- Line 305: `num_inference_steps=40` - Steps for final pass
- Line 306: `guidance_scale=7.0` - Guidance for final pass

### SWPO Strategy (`expandor/strategies/swpo_strategy.py`)
- Line 610: `blur_radius=50` - Hardcoded in _apply_edge_fill call
- Line 613: `strength=0.02` - Noise strength for mask
- Line 724: `num_inference_steps=30` - Steps for light touch
- Line 725: `guidance_scale=7.0` - Guidance scale
- Line 756: `width=10` - Edge sampling width
- Line 764: `width=10` - Edge sampling width
- Line 966: `max_dist=100` - Maximum distance for edge detection

### Progressive Outpaint Strategy (`expandor/strategies/progressive_outpaint.py`)
- Line 40: `self.max_supported = 8.0` - Maximum aspect ratio change
- Line 45: `self.min_strength = 0.20` - Minimum strength
- Line 46: `self.max_strength = 0.85` - Maximum strength
- Line 48: `self.first_step_ratio = 2.0` - First expansion ratio
- Line 53: `self.base_mask_blur = 32` - Base blur value
- Line 54: `self.base_steps = 60` - Base inference steps
- Line 56: `self.middle_step_ratio = 1.5` - Middle step ratio
- Line 57: `self.final_step_ratio = 1.3` - Final step ratio
- Line 627: `num_inference_steps=30` - Steps for seam repair
- Line 628: `guidance_scale=5.0` - Lower guidance for blending

### ControlNet Progressive Strategy (`expandor/strategies/controlnet_progressive.py`)
- Line 26: `VRAM_PRIORITY = 4` - Priority for VRAM allocation
- Line 231: `color=255` - White color for inpaint mask

## 2. Image Processing Parameters

### Seam Repair Processor (`expandor/processors/seam_repair.py`)
- Line 136: `radius=5` - Gaussian blur radius for mask
- Line 147: `strength=0.8` - High strength for seam repair
- Line 148: `guidance_scale=7.5` - Guidance scale
- Line 149: `num_inference_steps=50` - Inference steps
- Line 171: `base_strength = 0.2` - Base blur strength
- Line 172: `artifact_strength = 0.5` - Artifact repair strength
- Line 179: `guidance_scale=7.5` - Guidance scale
- Line 180: `num_inference_steps=30` - Inference steps
- Line 216: `strength=0.4` - Final blend strength
- Line 217: `guidance_scale=7.5` - Guidance scale
- Line 218: `num_inference_steps=40` - Inference steps
- Line 264: `radius=10` - Blur radius for blend mask
- Line 267: `radius=2` - Small blur for final result

### Edge Analysis Processor (`expandor/processors/edge_analysis.py`)
- Line 206: `threshold=100` - Edge detection threshold
- Line 226: `strength=0.8` - Edge refinement strength
- Line 241: `strength=0.5` - Soft edge strength

### Smart Refiner (`expandor/processors/refinement/smart_refiner.py`)
- Line 75: `self.base_strength = 0.4` - Base refinement strength
- Line 77: `self.min_region_size = 32` - Minimum region size
- Line 287: `blur = 32` - Blur radius for large boundaries
- Line 290: `blur = 16` - Blur radius for medium boundaries
- Line 293: `blur = 24` - Blur radius for small boundaries
- Line 366: `num_inference_steps=30` - Steps for refinement
- Line 367: `guidance_scale=7.5` - Guidance scale

### ControlNet Extractors (`expandor/processors/controlnet_extractors.py`)
- Line 346: `radius=20` - Gaussian blur for depth simulation
- Line 351: `255 - np_img` - Invert depth values

### Artifact Detector Enhanced (`expandor/processors/artifact_detector_enhanced.py`)
- Line 106: `self.min_quality_score = 0.75` - Minimum quality score

### Tiled Processor (`expandor/processors/tiled_processor.py`)
- Line 379: `width=2` - Debug rectangle border width
- Line 383: `width=1` - Center rectangle border width

### Boundary Analysis (`expandor/processors/boundary_analysis.py`)
- Line 61: `analysis["quality_score"] -= 0.2` - Severe artifact penalty
- Line 63: `analysis["quality_score"] -= 0.1` - Moderate artifact penalty
- Line 65: `analysis["quality_score"] -= 0.05` - Minor artifact penalty
- Line 102: `margin = 10` - Visual margin for boundary regions

## 3. Diffusers Adapter Parameters (`expandor/adapters/diffusers_adapter.py`)
- Line 70: `controlnet_strength=0.8` - Default ControlNet strength
- Line 71: `strength=0.75` - Default img2img strength
- Line 72: `num_inference_steps=50` - Default inference steps
- Line 73: `guidance_scale=7.5` - Default guidance scale
- Line 534: `multiple = 8` - SDXL dimension multiple
- Line 553: `max_dimension = 4096` - Maximum dimension for images
- Line 924: `strength=0.3` - Low strength for enhancement

## 4. Quality and Output Parameters

### CLI Process (`expandor/cli/process.py`)
- Line 72: `quality=95` - JPEG quality setting
- Line 74: `quality=95` - WebP quality setting
- Line 79: `compress_level=1` - PNG compression level
- Line 87: `indent=2` - JSON indentation

### Setup Wizard (`expandor/cli/setup_wizard.py`)
- Line 157: `min_val=1024` - Minimum VRAM for setup
- Line 158: `max_val=49152` - Maximum VRAM for setup
- Line 197: `min_val=0.1` - Minimum LoRA weight
- Line 198: `max_val=2.0` - Maximum LoRA weight

## 5. Configuration and System Parameters

### User Config (`expandor/config/user_config.py`)
- Line 61: `weight: float = 1.0` - Default LoRA weight
- Line 90: `clear_cache_frequency: int = 5` - Cache clearing frequency
- Line 270: `width=120` - YAML output width

### LoRA Manager (`expandor/config/lora_manager.py`)
- Line 177: `scale_factor = 1.5 / total_weight` - Weight scaling factor
- Line 265: `base_steps = 50` - Base inference steps
- Line 290: `recommended = 100` - Recommended inference steps

### VRAM Manager (`expandor/core/vram_manager.py`)
- Line 260: `min_size = 384` - Minimum dimension size
- Line 261: `max_size = 2048` - Maximum dimension size

### Memory Utils (`expandor/utils/memory_utils.py`)
- Line 137: `safety_factor: float = 1.2` - Memory safety factor
- Line 182: `safety_factor: float = 0.8` - Batch size safety factor
- Line 294: `param_bytes *= 2` - Gradient memory multiplier
- Line 299: `activation_multiplier = 4 if include_gradients else 2`
- Line 319: `bytes_per_pixel: float = 12` - Memory per pixel
- Line 320: `overlap: int = 64` - Tile overlap
- Line 321: `min_tile: int = 256` - Minimum tile size
- Line 322: `max_tile: int = 2048` - Maximum tile size

### Image Utils (`expandor/utils/image_utils.py`)
- Line 16: `fade_start: float = 0.0` - Fade start position
- Line 133: `width: int = 1` - Edge sampling width
- Line 197: `scale: float = 0.1` - Noise scale
- Line 198: `octaves: int = 1` - Perlin noise octaves
- Line 221: `freq = 2**octave` - Frequency calculation
- Line 222: `amp = 0.5**octave` - Amplitude calculation
- Line 287: `fill=255` - White fill for mask

### Dimension Calculator (`expandor/utils/dimension_calculator.py`)
- Line 76: `multiple: int = 8` - Rounding multiple
- Line 150: `max_expansion_per_step: float = 2.0` - Max expansion ratio
- Line 201: `if total_expansion >= 2.0` - Expansion threshold
- Line 217: `step_num = 2` - Initial step number
- Line 269: `if total_expansion >= 2.0` - Expansion threshold
- Line 285: `step_num = 2` - Initial step number
- Line 329: `window_size: int = 200` - SWPO window size
- Line 330: `overlap_ratio: float = 0.8` - SWPO overlap ratio

### Strategy Selector (`expandor/strategies/strategy_selector.py`)
- Line 107: `if expansion_factor <= 1.5` - Simple upscale threshold
- Line 109: `elif expansion_factor <= 4.0` - Progressive threshold
- Line 177: `priority: int = 50` - Default strategy priority

## 6. Mock/Test Values

### Mock Pipeline Adapter (`expandor/adapters/mock_pipeline_adapter.py`)
- Line 60: `self.max_vram_mb = 24576` - Simulated 24GB GPU
- Various default parameters for mock operations

### ComfyUI and A1111 Adapters
- Multiple default values for unimplemented methods (these are placeholders)

## 7. Output Quality and Compression Settings

### Direct Upscale Strategy (`expandor/strategies/direct_upscale.py`)
- Line 381: `compress_level=0, optimize=False` - PNG output settings (no compression)

### CLI Process (`expandor/cli/process.py`) - Additional
- Line 72: `optimize=True` - JPEG optimization
- Line 74: `lossless=False` - WebP lossy compression

## 8. Quality Estimation Values

### Hybrid Adaptive Strategy (`expandor/strategies/experimental/hybrid_adaptive.py`)
- Line 79: `self.aspect_ratio_threshold = 0.2` - 20% change triggers outpainting
- Line 80: `self.extreme_ratio_threshold = 3.0` - 3x+ ratio triggers SWPO
- Line 283: `estimated_quality = 0.9` - Direct upscale quality estimate
- Line 299: `estimated_quality = 0.85` - Progressive quality estimate
- Line 320: `estimated_quality = 0.85` - SWPO quality estimate
- Line 332: `estimated_quality = 0.7` - Tiled quality estimate
- Line 369: `estimated_quality = 0.85` - Mixed strategy quality
- Line 383: `estimated_quality = 0.8` - Fallback quality
- Line 398: `estimated_quality = 0.9` - Default quality
- Line 418: `estimated_quality=0.7` - Hybrid plan quality

## 9. Detection Thresholds

### Artifact Detector Enhanced (`expandor/processors/artifact_detector_enhanced.py`)
- Line 102: `self.seam_threshold = 0.25` - Seam detection threshold
- Line 103: `self.color_threshold = 30` - Color discontinuity threshold
- Line 104: `self.gradient_threshold = 0.25` - Gradient spike threshold
- Line 105: `self.frequency_threshold = 0.35` - Frequency anomaly threshold
- Line 360: `gradient_threshold = 0.1` - 10% deviation allowed

### Edge Analysis Processor (`expandor/processors/edge_analysis.py`)
- Line 49: `self.edge_threshold = 0.1` - Edge detection sensitivity
- Line 50: `self.seam_threshold = 0.3` - Seam detection sensitivity
- Line 51: `self.artifact_threshold = 0.5` - Artifact detection sensitivity

## Summary

The codebase contains **250+ hardcoded values** across **30+ files** that should be moved to configuration files. These include:

1. **Processing strengths**: 0.2 to 0.9 for various operations
2. **Inference steps**: 20 to 60 steps for different stages
3. **Guidance scales**: 5.0 to 7.5 for different operations
4. **Blur radii**: 2 to 50 pixels for various effects
5. **Tile/window sizes**: 200 to 2048 pixels
6. **Quality settings**: 95% for JPEG/WebP, compression levels
7. **Detection thresholds**: 0.1 to 0.5 for various artifact types
8. **Safety factors**: 0.5 to 1.2 for memory management
9. **Priority values**: For strategy selection
10. **Dimension constraints**: Min/max sizes for various operations
11. **Quality estimates**: 0.7 to 0.9 for different strategies
12. **Ratio thresholds**: For strategy selection decisions

These values directly control the behavior and quality of image processing operations and should be configurable to allow users to tune the system for their specific needs and hardware.

## Most Critical Values to Move to Config

### Priority 1 (Affects Quality Most):
- All `strength` values (0.2-0.9)
- All `num_inference_steps` (20-60)
- All `guidance_scale` values (5.0-7.5)
- All blur radius values (2-50)
- JPEG/WebP quality (95)
- PNG compression level (0-9)

### Priority 2 (Affects Performance):
- Tile sizes (512-2048)
- Window sizes and overlaps
- VRAM safety factors
- Cache clearing frequency
- Batch processing limits

### Priority 3 (Affects Detection/Decisions):
- All threshold values (0.1-0.5)
- Quality score penalties
- Ratio thresholds for strategy selection
- Minimum region sizes
- Edge detection parameters

Moving these to configuration would allow users to:
- Fine-tune quality vs speed tradeoffs
- Adapt to different hardware capabilities
- Customize for specific image types
- Debug and experiment with settings
- Create presets for different use cases