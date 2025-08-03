# Optimal Settings Guide for Expandor (August 2025)

> **Last Updated**: August 2025  
> **Research Focus**: Maximum quality image expansion using latest AI models and techniques  
> **Hardware Target**: NVIDIA RTX 3090 (24GB VRAM)

## 🎯 Executive Summary

This document contains comprehensively researched optimal settings for every component of the Expandor image expansion system, based on the latest 2025 best practices and benchmarks.

### Critical Changes from Defaults
1. **Denoising Strength**: Reduce from 0.95 to 0.6-0.75 for expansions
2. **Tile Overlap**: Increase to 48-128 pixels (not 256)
3. **Guidance Scale**: 7-8 (not 10+)
4. **Inference Steps**: 30-50 (not 80)
5. **ControlNet Weight**: 0.25-0.5 (not 1.0)

---

## 📊 SDXL Optimal Settings

### Base Generation Parameters

```yaml
# For initial generation and expansion
denoising_strength: 0.6-0.9  # Critical: Higher values destroy original content
guidance_scale: 7-8          # 91% success rate at this range
num_inference_steps: 30-50   # Diminishing returns above 50
scheduler: "Euler"           # Or "Euler Ancestral" for sharpest results
batch_size: 2-4              # For quality comparison
```

### Resolution Handling

```yaml
base_resolution: 1024x1024   # SDXL native resolution
aspect_ratios:               # Maintain multiples of 8
  - 1024x1024
  - 1152x896
  - 896x1152
  - 1216x832
  - 832x1216
```

### Outpainting Specific Settings

```yaml
outpainting:
  direction: "one_at_a_time"     # Best practice: expand one direction only
  denoising_strength: 0.6-0.75   # Lower than generation
  mask_mode: "fill"              # Uses average color under mask
  mask_blur: 40                  # Percentage of new content dimension
  refiner_usage: true            # Use SDXL refiner for details
```

---

## 🔲 Tiled Generation Best Practices

### Tile Configuration

```yaml
tile_sizes:
  # Based on VRAM availability
  24GB_VRAM:
    latent_tile_size: 128    # In latent space
    pixel_tile_size: 1024    # In pixel space
  16GB_VRAM:
    latent_tile_size: 96
    pixel_tile_size: 768
  8GB_VRAM:
    latent_tile_size: 64
    pixel_tile_size: 512
```

### Overlap Settings

```yaml
overlap_settings:
  method: "mixture_of_diffusers"  # Better than multidiffusion
  overlap_pixels: 48              # Default, can reduce to 8
  overlap_ratio: 0.125-0.25       # 12.5-25% of tile size
  gaussian_smoothing: true        # For seamless blending
```

### Progressive Strength Reduction

```yaml
# For large tile counts (>50)
strength_reduction:
  enabled: true
  base_strength: 0.75
  reduction_per_tile: 0.003
  minimum_strength: 0.4
```

---

## 🎮 ControlNet Optimal Configuration

### Weight Settings

```yaml
controlnet:
  control_weight: 0.25           # Starting point for balanced results
  conditioning_scale: 0.5        # For good generalization
  guidance_start: 0.0            # When to start applying
  guidance_end: 1.0              # When to stop applying
```

### Model Selection

```yaml
canny_edge:
  model: "diffusers_xl_canny_full"  # Best quality despite size
  low_threshold: 100                 # Discard below this
  high_threshold: 200                # Always keep above this

depth_estimation:
  moderate_detail: "depth_zoe"       # Balanced option
  maximum_detail: "depth_leres++"    # Most detailed
  
blur_extraction:
  kernel_size: 31                    # For soft guidance
  sigma: 10.0                        # Blur intensity
```

### Performance Flags

```yaml
performance:
  flags:
    - "--medvram-sdxl"      # For 8-16GB VRAM
    - "--xformers"          # Memory efficient attention
    - "--no-half-vae"       # Faster generation (critical!)
  cpu_offload: true         # For low VRAM situations
```

---

## 🔍 Real-ESRGAN Upscaling Settings

### Model Selection

```yaml
upscaling_models:
  realistic: "RealESRGAN_x4plus"        # Best for photos
  anime: "RealESRGAN_x4plus_anime_6B"  # For anime/artwork
  universal: "4x-UltraSharp"            # Good all-rounder
```

### Optimal Parameters

```yaml
upscaling:
  scale_factor: 4              # Native 4x works best
  tile_size: 512               # For VRAM management
  tile_overlap: 32             # Prevent seams
  denoising_strength: 0.1-0.3  # Preserve detail
  face_enhancement: true       # When applicable
```

### Quality Enhancement

```yaml
enhancement:
  noise_inversion:
    enabled: true
    steps: 50
    renoise_strength: 0.3     # Adds detail
  color_fix:
    enabled: true
    mode: "fast_encoder"      # Prevents washing out
```

---

## 💾 VRAM Optimization (RTX 3090 24GB)

### Memory Management

```yaml
vram_optimization:
  precision: "fp16"            # 50% memory reduction
  tf32: false                  # Disable for 7% savings
  attention: "sdpa"            # For Ada generation
  gradient_checkpointing: true # 22% slower but saves memory
```

### Batch Processing

```yaml
batch_config:
  24GB_settings:
    max_batch_size: 4
    resolution_limit: "1024x1024"
    enable_tiling: true
    clear_cache_frequency: 10  # Every 10 batches
```

### TensorRT Optimization (2025)

```yaml
tensorrt:
  enabled: true               # For SD 3.5 models
  benefits:
    speed_increase: "2x"
    memory_reduction: "40%"
  fp8_quantization: true      # Latest optimization
```

---

## 🔧 Strategy-Specific Optimizations

### Progressive Outpainting

```yaml
progressive_outpaint:
  max_steps: 3                        # Don't exceed
  strength_decay: [0.75, 0.65, 0.55]  # Per step
  overlap_ratio: 0.3                  # 30% overlap
  edge_blend_width: 128               # Pixels
```

### SWPO (Sliding Window)

```yaml
swpo:
  window_size: 256-512         # Based on content
  overlap_ratio: 0.7-0.8       # 70-80% overlap
  final_unification: true      # Blend all windows
  strength_per_window: 0.6     # Consistent strength
```

### Tiled Expansion

```yaml
tiled_expansion:
  max_tiles: 50                # Reduce strength after
  refinement_strength: 0.15    # For seam repair
  edge_fix_strength: 0.25      # Border corrections
  final_pass_strength: 0.35    # Global coherence
```

---

## ⚠️ Critical Quality Thresholds

### Detection Settings

```yaml
quality_validation:
  seam_detection:
    threshold: 0.03           # More sensitive for ultra
    methods:
      - color_discontinuity
      - gradient_analysis
      - frequency_domain
  
  texture_corruption:
    enabled: true
    threshold: 0.15
    high_freq_ratio: 0.3      # Minimum acceptable
  
  artifact_detection:
    sensitivity: "high"
    multipass_threshold: 0.1
```

### Failure Conditions

```yaml
failure_triggers:
  vram_usage: 0.95            # 95% usage = abort
  seam_count: 10              # Too many seams
  corruption_score: 0.3       # High corruption
  generation_time: 600        # 10 minute timeout
```

---

## 📋 Preset Configurations

### Ultra Quality (Fixed)

```yaml
ultra_preset:
  # Expansion settings
  expansion:
    denoising_strength: 0.70    # Reduced from 0.95
    guidance_scale: 8.5
    num_inference_steps: 50     # Reduced from 80
  
  # Tiling adjustments
  tiling:
    strength_multiplier: 0.8    # Reduce for tiles
    max_tile_denoising: 0.65    # Cap strength
  
  # Quality checks
  validation:
    enable_all_checks: true
    texture_validation: true
    progressive_reduction: true
```

### Balanced Preset

```yaml
balanced_preset:
  expansion:
    denoising_strength: 0.55
    guidance_scale: 7.5
    num_inference_steps: 30
  
  performance:
    tile_batch_size: 8
    enable_caching: true
    progressive_loading: true
```

### Fast Preset

```yaml
fast_preset:
  expansion:
    denoising_strength: 0.45
    guidance_scale: 7.0
    num_inference_steps: 20
  
  optimizations:
    scheduler: "DPM++ 2M"       # Faster convergence
    enable_tensorrt: true
    skip_refinement: true
```

---

## 🚀 Implementation Priority

### Immediate Fixes (Critical)
1. Change ultra preset denoising from 0.95 to 0.70
2. Fix division by zero in HybridAdaptiveStrategy
3. Implement texture corruption detection
4. Add progressive strength reduction for tiles

### Short-term Improvements
1. Switch to Mixture of Diffusers for blending
2. Implement proper ControlNet weights
3. Add TensorRT optimization support
4. Improve pipeline state management

### Long-term Enhancements
1. Adaptive tile sizing based on content
2. Multi-stage refinement pipeline
3. AI-based quality assessment
4. Dynamic strategy selection

---

## 📝 Usage Examples

### High-Quality 4K Expansion
```bash
expandor --resolution 3840x2160 \
  --quality balanced \
  --denoising-strength 0.65 \
  --guidance-scale 7.5 \
  --tile-overlap 48 \
  --method mixture_of_diffusers
```

### Extreme Aspect Ratio
```bash
expandor --resolution 21600x2160 \
  --strategy progressive \
  --max-steps 3 \
  --strength-decay "0.75,0.65,0.55" \
  --enable-refinement
```

### VRAM-Constrained Operation
```bash
expandor --resolution 3840x2160 \
  --quality fast \
  --cpu-offload \
  --tile-size 512 \
  --gradient-checkpointing \
  --fp16
```

---

## 🔬 Testing & Validation

### Test Matrix
- Resolutions: 1920x1080, 3840x2160, 5120x2880, 7680x4320
- Aspect ratios: 16:9, 21:9, 32:9, 1:1
- Content types: Landscapes, portraits, abstract, detailed
- Expansion ratios: 2x, 4x, 8x, 16x

### Success Metrics
- No visible seams or artifacts
- Texture preservation > 85%
- Color accuracy > 95%
- Processing time < 5 minutes for 4K
- VRAM usage < 22GB peak

---

## 📚 References

This guide synthesizes research from:
- Latest SDXL documentation and benchmarks (2025)
- TensorRT optimization guides for SD 3.5
- Community best practices from RunComfy, Stable Diffusion Art
- Hardware benchmarks for RTX 3090 and newer cards
- Academic papers on tiled diffusion and seamless generation

*Note: These settings are optimized for quality over speed, aligning with the Expandor project philosophy.*