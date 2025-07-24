# Expandor Configuration Guide

## Table of Contents

1. [Overview](#overview)
2. [Configuration Hierarchy](#configuration-hierarchy)
3. [Configuration Files](#configuration-files)
4. [User Configuration](#user-configuration)
5. [Quality Presets](#quality-presets)
6. [Strategy Configuration](#strategy-configuration)
7. [VRAM Strategies](#vram-strategies)
8. [Model Configuration](#model-configuration)
9. [LoRA Configuration](#lora-configuration)
10. [Pipeline Parameters](#pipeline-parameters)
11. [Performance Tuning](#performance-tuning)
12. [Environment Variables](#environment-variables)
13. [Examples](#examples)

## Overview

Expandor uses a hierarchical configuration system that allows fine-grained control over all aspects of image expansion. Configuration can be specified through:

1. Configuration files (YAML)
2. Command-line arguments
3. Environment variables
4. Python API parameters

## Configuration Hierarchy

Configuration sources are applied in the following order (later sources override earlier ones):

1. **System defaults** - Built-in defaults
2. **User configuration** - `~/.config/expandor/config.yaml`
3. **Custom config file** - Specified with `--config`
4. **Environment variables** - `EXPANDOR_*` variables
5. **Command-line arguments** - Direct CLI options
6. **API parameters** - Direct Python API calls

## Configuration Files

### Default Locations

```
~/.config/expandor/
├── config.yaml          # Main user configuration
├── loras.yaml          # LoRA definitions
├── models.yaml         # Model configurations
└── presets/           # Custom presets
    ├── quality.yaml
    └── strategies.yaml
```

### File Format

All configuration files use YAML format:

```yaml
# Comments are supported
key: value
nested:
  key: value
  list:
    - item1
    - item2
```

## User Configuration

### Basic Structure

Create `~/.config/expandor/config.yaml`:

```yaml
# Model Configuration
models:
  sdxl:
    model_id: stabilityai/stable-diffusion-xl-base-1.0
    dtype: float16
    device: cuda
    custom_pipeline_params:
      variant: fp16
      use_safetensors: true
  
  sd15:
    model_id: runwayml/stable-diffusion-v1-5
    dtype: float32
    device: cuda

# Default Settings
default_model: sdxl
default_quality: balanced
default_strategy: auto

# User Preferences
preferences:
  save_metadata: true
  metadata_format: json
  auto_select_strategy: true
  preserve_original_prompt: false
  verbose_errors: true

# Performance Settings
performance:
  max_workers: 4
  tile_size: 1024
  tile_overlap: 128
  cache_size: 2048
  clear_cache_frequency: 10

# Output Settings
output:
  format: png
  quality: 95
  optimize: true
  save_stages: false
  stage_format: png
```

### Model Configuration

```yaml
models:
  custom_sdxl:
    # HuggingFace model ID or local path
    model_id: /path/to/local/model
    
    # Data type: float32, float16, bfloat16
    dtype: float16
    
    # Device: cuda, cpu, mps
    device: cuda
    
    # Custom pipeline loading parameters
    custom_pipeline_params:
      variant: fp16
      use_safetensors: true
      torch_dtype: float16
      safety_checker: null
      requires_safety_checker: false
      
    # Model-specific constraints
    constraints:
      min_size: 512
      max_size: 2048
      size_multiple: 8
      
    # Recommended settings
    recommended:
      inference_steps: 50
      guidance_scale: 7.5
```

### Preferences

```yaml
preferences:
  # Metadata handling
  save_metadata: true
  metadata_format: json  # json, yaml, or xml
  embed_metadata: true   # Embed in image EXIF
  
  # Strategy selection
  auto_select_strategy: true
  prefer_quality_over_speed: true
  adaptive_parameters: true
  
  # Generation settings
  preserve_original_prompt: false
  enhance_prompts: true
  use_negative_defaults: true
  
  # Error handling
  verbose_errors: true
  fail_on_warning: false
  retry_on_oom: true
  max_retries: 3
  
  # UI/UX
  show_progress: true
  progress_style: bar  # bar, dots, or percentage
  confirm_overwrites: true
```

## Quality Presets

### Built-in Presets

Located in `expandor/config/quality_presets.yaml`:

```yaml
quality_presets:
  fast:
    inference_steps: 20
    cfg_scale: 5.0
    denoise_strength: 0.7
    sampler: "DPM++ 2M"
    scheduler: "karras"
    
  balanced:
    inference_steps: 40
    cfg_scale: 7.0
    denoise_strength: 0.8
    sampler: "DPM++ 2M SDE"
    scheduler: "karras"
    
  high:
    inference_steps: 60
    cfg_scale: 7.5
    denoise_strength: 0.85
    sampler: "DPM++ 2M SDE"
    scheduler: "exponential"
    
  ultra:
    inference_steps: 100
    cfg_scale: 8.0
    denoise_strength: 0.95
    sampler: "DPM++ 3M SDE"
    scheduler: "exponential"
    enable_refinement: true
    refinement_steps: 20
```

### Custom Quality Presets

Create `~/.config/expandor/presets/quality.yaml`:

```yaml
quality_presets:
  my_custom:
    inference_steps: 80
    cfg_scale: 9.0
    denoise_strength: 0.9
    sampler: "Euler a"
    scheduler: "normal"
    
    # Advanced settings
    eta: 0.0
    churn: 0.0
    churn_tmin: 0.0
    churn_tmax: inf
    sigma_min: 0.0
    sigma_max: inf
```

## Strategy Configuration

### Strategy Definitions

Located in `expandor/config/strategies.yaml`:

```yaml
strategies:
  direct_upscale:
    description: "Single-step upscaling"
    parameters:
      method: "latent"
      upscale_factor: 2.0
      denoise_strength: 0.7
    vram_requirement: "low"
    supported_factors: [1.0, 4.0]
    
  progressive_outpaint:
    description: "Multi-step progressive expansion"
    parameters:
      initial_expansion: 1.4
      middle_expansion: 1.25
      final_expansion: 1.15
      denoise_strength: 0.95
      mask_blur_radius: 0.4
      prefill_mode: "edge_extend"
    vram_requirement: "medium"
    supported_factors: [1.5, 8.0]
    
  swpo:
    description: "Sliding Window Progressive Outpaint"
    parameters:
      window_size: 256
      overlap_ratio: 0.8
      direction: "horizontal"
      unification_pass: true
      unification_strength: 0.3
    vram_requirement: "high"
    supported_factors: [2.0, 16.0]
    best_for: "extreme_aspect_ratios"
```

### Custom Strategies

Create custom strategy configurations:

```yaml
strategies:
  my_custom_strategy:
    description: "Custom expansion strategy"
    base_strategy: "progressive"  # Inherit from base
    parameters:
      initial_expansion: 1.5
      denoise_strength: 0.9
      custom_param: "value"
    vram_requirement: "medium"
    
    # Conditions for auto-selection
    conditions:
      min_factor: 2.0
      max_factor: 6.0
      aspect_ratio_change: 0.5
```

## VRAM Strategies

### VRAM Profiles

Located in `expandor/config/vram_strategies.yaml`:

```yaml
vram_profiles:
  high_vram:
    min_vram: 20000  # 20GB+
    strategies:
      preferred_strategy: "progressive"
      fallback_strategies: ["swpo", "direct"]
    parameters:
      batch_size: 4
      enable_attention_slicing: false
      enable_cpu_offload: false
      cache_size: 4096
      
  medium_vram:
    min_vram: 8000   # 8GB+
    strategies:
      preferred_strategy: "direct"
      fallback_strategies: ["tiled", "progressive"]
    parameters:
      batch_size: 2
      enable_attention_slicing: true
      enable_cpu_offload: false
      cache_size: 2048
      
  low_vram:
    min_vram: 4000   # 4GB+
    strategies:
      preferred_strategy: "tiled"
      fallback_strategies: ["cpu_offload"]
    parameters:
      batch_size: 1
      enable_attention_slicing: true
      enable_cpu_offload: true
      sequential_cpu_offload: true
      tile_size: 512
      cache_size: 512
```

### Memory Optimization

```yaml
memory_optimization:
  # Automatic optimization triggers
  auto_optimize:
    enable: true
    vram_threshold: 0.9  # Trigger at 90% VRAM usage
    
  # Optimization techniques
  techniques:
    attention_slicing:
      enable: auto  # auto, true, false
      slice_size: auto  # auto, 1, 2, 4, 8
      
    vae_slicing:
      enable: true
      slice_size: auto
      
    cpu_offload:
      enable: false
      sequential: false
      offload_params: true
      offload_buffers: true
      
    gradient_checkpointing:
      enable: false
      
  # Cache management
  cache:
    max_size: 2048  # MB
    clear_frequency: 10  # Clear every N operations
    clear_on_error: true
```

## Model Configuration

### Model Constraints

Located in `expandor/config/model_constraints.yaml`:

```yaml
model_constraints:
  sdxl:
    family: "stable-diffusion-xl"
    min_size: 512
    max_size: 2048
    optimal_size: 1024
    size_multiple: 8
    aspect_ratios:
      min: 0.5
      max: 2.0
    vae_scale_factor: 8
    min_vram: 10000
    recommended_vram: 16000
    
  sd15:
    family: "stable-diffusion"
    min_size: 256
    max_size: 1024
    optimal_size: 512
    size_multiple: 8
    aspect_ratios:
      min: 0.5
      max: 2.0
    vae_scale_factor: 8
    min_vram: 4000
    recommended_vram: 8000
    
  flux:
    family: "flux"
    min_size: 256
    max_size: 2048
    optimal_size: 1024
    size_multiple: 16
    aspect_ratios:
      min: 0.25
      max: 4.0
    vae_scale_factor: 8
    min_vram: 24000
    recommended_vram: 40000
```

### Model-Specific Parameters

```yaml
model_parameters:
  sdxl:
    default_scheduler: "DPMSolverMultistepScheduler"
    scheduler_config:
      use_karras_sigmas: true
      algorithm_type: "dpmsolver++"
    pipeline_config:
      add_watermarker: false
      force_zeros_for_empty_prompt: true
      
  sd15:
    default_scheduler: "DDIMScheduler"
    scheduler_config:
      beta_schedule: "scaled_linear"
      clip_sample: false
      set_alpha_to_one: false
```

## LoRA Configuration

### LoRA Definitions

Create `~/.config/expandor/loras.yaml`:

```yaml
loras:
  # Style LoRAs
  anime_style:
    path: /models/loras/anime_style_v2.safetensors
    type: style
    default_weight: 0.7
    compatible_models: ["sdxl", "sd15"]
    trigger_words: ["anime style", "manga"]
    description: "Anime/manga art style"
    
  photorealistic:
    path: /models/loras/photorealistic_v3.safetensors
    type: style
    default_weight: 0.8
    compatible_models: ["sdxl"]
    trigger_words: ["photorealistic", "photography"]
    conflicts_with: ["anime_style"]
    
  # Detail LoRAs
  detail_enhancer:
    path: /models/loras/detail_tweaker.safetensors
    type: detail
    default_weight: 0.6
    compatible_models: ["sdxl", "sd15"]
    recommended_steps: 60
    
  # Subject LoRAs
  face_enhance:
    path: /models/loras/face_detail.safetensors
    type: subject
    default_weight: 0.5
    compatible_models: ["sdxl"]
    target: "faces"
    
  # Quality LoRAs
  quality_boost:
    path: /models/loras/quality_enhancer.safetensors
    type: quality
    default_weight: 0.4
    compatible_models: ["sdxl", "sd15"]
    min_steps: 40
```

### LoRA Stacking Rules

```yaml
lora_rules:
  # Maximum weights
  max_total_weight: 1.5
  max_per_type:
    style: 1.0
    detail: 0.8
    subject: 0.6
    quality: 0.5
    
  # Type compatibility
  incompatible_combinations:
    - ["anime_style", "photorealistic"]
    - ["cartoon_style", "hyperrealistic"]
    
  # Auto-adjustment rules
  weight_scaling:
    enabled: true
    method: "proportional"  # proportional, equal, priority
    
  # Recommended combinations
  presets:
    portrait_enhancement:
      loras:
        - name: photorealistic
          weight: 0.8
        - name: face_enhance
          weight: 0.6
        - name: detail_enhancer
          weight: 0.4
```

## Pipeline Parameters

### Default Pipeline Parameters

```yaml
pipeline_defaults:
  # Generation parameters
  guidance_scale: 7.5
  num_inference_steps: 40
  eta: 0.0
  generator_seed: -1  # -1 for random
  
  # Scheduler settings
  scheduler: "DPMSolverMultistepScheduler"
  scheduler_kwargs:
    use_karras_sigmas: true
    algorithm_type: "dpmsolver++"
    solver_order: 2
    
  # Safety and quality
  safety_checker: false
  watermark: false
  nsfw_filter: false
  
  # Advanced settings
  clip_skip: 0
  control_guidance_start: 0.0
  control_guidance_end: 1.0
```

### Per-Operation Parameters

```yaml
operation_parameters:
  generate:
    guidance_scale: 7.5
    num_inference_steps: 40
    
  img2img:
    strength: 0.8
    guidance_scale: 7.5
    num_inference_steps: 40
    
  inpaint:
    strength: 0.95
    guidance_scale: 7.5
    num_inference_steps: 50
    mask_blur: 8
    masked_area_padding: 32
    
  upscale:
    strength: 0.7
    guidance_scale: 5.0
    num_inference_steps: 30
```

## Performance Tuning

### Optimization Profiles

```yaml
optimization_profiles:
  speed:
    inference_steps: 20
    enable_attention_slicing: true
    enable_vae_slicing: true
    enable_cpu_offload: false
    batch_size: 1
    compile_model: true
    compile_vae: true
    
  balanced:
    inference_steps: 40
    enable_attention_slicing: auto
    enable_vae_slicing: false
    enable_cpu_offload: false
    batch_size: 2
    compile_model: false
    
  quality:
    inference_steps: 80
    enable_attention_slicing: false
    enable_vae_slicing: false
    enable_cpu_offload: false
    batch_size: 1
    compile_model: false
    enable_refinement: true
```

### Hardware-Specific Tuning

```yaml
hardware_profiles:
  rtx_3090:
    vram_buffer: 1000  # Reserve 1GB
    optimal_tile_size: 1024
    max_batch_size: 4
    enable_tf32: true
    enable_cudnn_benchmark: true
    
  rtx_4090:
    vram_buffer: 2000  # Reserve 2GB
    optimal_tile_size: 1536
    max_batch_size: 8
    enable_tf32: true
    enable_flash_attention: true
    
  apple_silicon:
    device: mps
    optimal_tile_size: 768
    max_batch_size: 2
    enable_metal_performance_shaders: true
```

## Environment Variables

### Core Variables

```bash
# Configuration directory
export EXPANDOR_CONFIG_DIR=/custom/config/path

# Cache directory
export EXPANDOR_CACHE_DIR=/large/disk/cache

# Model directory
export EXPANDOR_MODEL_DIR=/models

# Output directory
export EXPANDOR_OUTPUT_DIR=/outputs

# Temp directory
export EXPANDOR_TEMP_DIR=/tmp/expandor
```

### Performance Variables

```bash
# VRAM limit (MB)
export EXPANDOR_VRAM_LIMIT=8000

# CPU threads
export EXPANDOR_CPU_THREADS=8

# Enable optimizations
export EXPANDOR_ENABLE_TF32=1
export EXPANDOR_ENABLE_CUDNN_BENCHMARK=1
export EXPANDOR_COMPILE_MODEL=1
```

### Debug Variables

```bash
# Logging
export EXPANDOR_LOG_LEVEL=DEBUG
export EXPANDOR_LOG_FILE=/var/log/expandor.log

# Debug features
export EXPANDOR_SAVE_STAGES=1
export EXPANDOR_SAVE_METADATA=1
export EXPANDOR_VERBOSE=1
```

## Examples

### Minimal Configuration

```yaml
# ~/.config/expandor/config.yaml
default_model: sdxl
default_quality: balanced
```

### Production Configuration

```yaml
# Production settings with redundancy
models:
  primary:
    model_id: /models/production/sdxl_v1.safetensors
    dtype: float16
    device: cuda:0
    
  backup:
    model_id: stabilityai/stable-diffusion-xl-base-1.0
    dtype: float16
    device: cuda:1

preferences:
  save_metadata: true
  embed_metadata: true
  preserve_original_prompt: true
  retry_on_oom: true
  max_retries: 5

performance:
  max_workers: 8
  tile_size: 1024
  cache_size: 8192
  enable_profiling: true

output:
  format: png
  optimize: false  # Lossless for production
  backup_original: true

monitoring:
  enable_metrics: true
  metrics_endpoint: "http://metrics.internal:9090"
  report_interval: 60
```

### Development Configuration

```yaml
# Development settings with debugging
default_quality: fast
default_strategy: direct

preferences:
  verbose_errors: true
  save_metadata: true
  show_progress: true
  confirm_overwrites: false

performance:
  max_workers: 2
  compile_model: false  # Faster iteration

debug:
  save_stages: true
  stage_format: jpg  # Smaller files
  log_level: DEBUG
  profile_performance: true
  trace_memory: true
```

### Custom Preset Configuration

```yaml
# Custom artistic preset
presets:
  artistic:
    quality:
      inference_steps: 80
      cfg_scale: 12.0
      sampler: "DPM++ 3M SDE"
      
    loras:
      - name: artistic_style
        weight: 0.9
      - name: color_enhance
        weight: 0.5
        
    prompts:
      append: "masterpiece, best quality, highly detailed"
      negative_append: "lowres, bad anatomy, bad hands"
      
    post_processing:
      sharpen: 0.3
      color_boost: 1.1
      contrast: 1.05
```

## Best Practices

1. **Start with defaults** - Only override what you need
2. **Use profiles** - Create profiles for different use cases
3. **Version your configs** - Keep configs in version control
4. **Test incrementally** - Change one setting at a time
5. **Monitor performance** - Use metrics to optimize
6. **Document custom settings** - Add comments to explain why
7. **Backup configurations** - Before major changes
8. **Use environment-specific configs** - Dev vs production

## Troubleshooting

### Configuration Not Loading

```bash
# Check config location
expandor --config-info

# Validate config syntax
expandor --validate-config ~/.config/expandor/config.yaml

# Use verbose mode
expandor -v --config custom.yaml input.jpg -r 4K
```

### Conflicting Settings

```yaml
# Debug configuration conflicts
debug:
  log_config_resolution: true
  show_config_sources: true
  warn_on_override: true
```

### Performance Issues

```yaml
# Profile configuration impact
profiling:
  enable: true
  profile_stages: true
  measure_vram: true
  output_file: "profile_results.json"
```

## See Also

- [CLI_USAGE.md](CLI_USAGE.md) - Command-line interface guide
- [MODELS.md](MODELS.md) - Model-specific configurations
- [API Documentation](API.md) - Python API configuration
- [Examples](../examples/custom_config.py) - Configuration examples