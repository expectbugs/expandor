# Expandor Model Support Guide

## Table of Contents

1. [Overview](#overview)
2. [Supported Models](#supported-models)
3. [Model Requirements](#model-requirements)
4. [Model Configuration](#model-configuration)
5. [Installation](#installation)
6. [Model-Specific Settings](#model-specific-settings)
7. [Performance Considerations](#performance-considerations)
8. [LoRA Compatibility](#lora-compatibility)
9. [Custom Models](#custom-models)
10. [Troubleshooting](#troubleshooting)

## Overview

Expandor supports a wide range of image generation models through its adapter system. Each model family has specific requirements, optimal settings, and performance characteristics.

## Supported Models

### Stable Diffusion XL (SDXL)

**Family**: Stable Diffusion XL  
**Recommended**: ✅ **Best for general use**

```yaml
model_id: stabilityai/stable-diffusion-xl-base-1.0
min_vram: 10GB
optimal_vram: 16GB+
```

**Variants**:
- `stabilityai/stable-diffusion-xl-base-1.0` - Base model
- `stabilityai/stable-diffusion-xl-refiner-1.0` - Refiner model
- `segmind/SSD-1B` - Distilled version (faster, lower quality)
- `segmind/Segmind-Vega` - Optimized variant

**Key Features**:
- Native 1024x1024 generation
- Excellent quality/speed balance
- Wide LoRA support
- Best community support

### Stable Diffusion 1.5 (SD15)

**Family**: Stable Diffusion  
**Recommended**: ✅ **Best for low VRAM**

```yaml
model_id: runwayml/stable-diffusion-v1-5
min_vram: 4GB
optimal_vram: 8GB+
```

**Variants**:
- `runwayml/stable-diffusion-v1-5` - Base model
- `stabilityai/stable-diffusion-v1-4` - Previous version
- `CompVis/stable-diffusion-v1-4` - Original
- `dreamlike-art/dreamlike-diffusion-1.0` - Artistic variant

**Key Features**:
- Native 512x512 generation
- Low VRAM requirements
- Massive LoRA ecosystem
- Fast generation

### Stable Diffusion 2.x (SD2)

**Family**: Stable Diffusion 2  
**Recommended**: ⚠️ **Limited LoRA support**

```yaml
model_id: stabilityai/stable-diffusion-2-1
min_vram: 6GB
optimal_vram: 10GB+
```

**Variants**:
- `stabilityai/stable-diffusion-2-1` - Latest 768x768
- `stabilityai/stable-diffusion-2-1-base` - 512x512
- `stabilityai/stable-diffusion-2` - Original

**Key Features**:
- Native 768x768 generation
- Improved architecture
- Limited community adoption
- Different CLIP model

### Stable Diffusion 3 (SD3)

**Family**: Stable Diffusion 3  
**Recommended**: ⚠️ **Experimental**

```yaml
model_id: stabilityai/stable-diffusion-3-medium
min_vram: 16GB
optimal_vram: 24GB+
```

**Variants**:
- `stabilityai/stable-diffusion-3-medium` - 2B parameter model
- More variants coming soon

**Key Features**:
- Advanced architecture
- Better text rendering
- Higher quality output
- Higher VRAM requirements

### FLUX

**Family**: FLUX  
**Recommended**: ⚠️ **High-end hardware only**

```yaml
model_id: black-forest-labs/FLUX.1-dev
min_vram: 24GB
optimal_vram: 40GB+
```

**Variants**:
- `black-forest-labs/FLUX.1-dev` - Development version
- `black-forest-labs/FLUX.1-schnell` - Fast version

**Key Features**:
- State-of-the-art quality
- Extremely high VRAM usage
- Advanced capabilities
- Slower generation

## Model Requirements

### Hardware Requirements by Model

| Model | Min VRAM | Recommended VRAM | Min RAM | Storage |
|-------|----------|------------------|---------|---------|
| SD 1.5 | 4GB | 8GB | 8GB | 5GB |
| SD 2.x | 6GB | 10GB | 12GB | 6GB |
| SDXL | 10GB | 16GB | 16GB | 10GB |
| SD3 | 16GB | 24GB | 32GB | 15GB |
| FLUX | 24GB | 40GB | 64GB | 25GB |

### Resolution Constraints

```yaml
model_constraints:
  sd15:
    min_size: 256
    max_size: 1024
    optimal_size: 512
    size_multiple: 8
    
  sd2:
    min_size: 384
    max_size: 1536
    optimal_size: 768
    size_multiple: 8
    
  sdxl:
    min_size: 512
    max_size: 2048
    optimal_size: 1024
    size_multiple: 8
    
  flux:
    min_size: 256
    max_size: 2048
    optimal_size: 1024
    size_multiple: 16  # Note: 16 for FLUX
```

### Aspect Ratio Support

```yaml
aspect_ratios:
  sd15:
    min: 0.5   # 1:2
    max: 2.0   # 2:1
    optimal: [1.0, 0.75, 1.33]  # 1:1, 3:4, 4:3
    
  sdxl:
    min: 0.5   # 1:2
    max: 2.0   # 2:1
    optimal: [1.0, 0.75, 1.33, 0.56, 1.78]  # More flexibility
    
  flux:
    min: 0.25  # 1:4
    max: 4.0   # 4:1
    optimal: "any"  # Very flexible
```

## Model Configuration

### Basic Model Setup

```yaml
# ~/.config/expandor/config.yaml
models:
  sdxl:
    model_id: stabilityai/stable-diffusion-xl-base-1.0
    dtype: float16
    device: cuda
    variant: fp16  # Download fp16 variant
    
  sdxl_refiner:
    model_id: stabilityai/stable-diffusion-xl-refiner-1.0
    dtype: float16
    device: cuda
    use_as_refiner: true
    
  sd15:
    model_id: runwayml/stable-diffusion-v1-5
    dtype: float32  # Better compatibility
    device: cuda
```

### Advanced Model Configuration

```yaml
models:
  custom_sdxl:
    model_id: /path/to/local/model.safetensors
    dtype: float16
    device: cuda
    
    # Pipeline parameters
    custom_pipeline_params:
      safety_checker: null
      requires_safety_checker: false
      feature_extractor: null
      watermarker: null
      
    # Scheduler configuration
    scheduler: "DPMSolverMultistepScheduler"
    scheduler_config:
      use_karras_sigmas: true
      algorithm_type: "dpmsolver++"
      
    # Component overrides
    components:
      vae: /path/to/custom/vae.safetensors
      text_encoder: /path/to/custom/text_encoder
      tokenizer: /path/to/custom/tokenizer
```

### Memory Optimization by Model

```yaml
model_optimizations:
  sd15:
    enable_attention_slicing: true
    enable_vae_slicing: false
    enable_cpu_offload: true
    
  sdxl:
    enable_attention_slicing: auto
    enable_vae_slicing: true
    enable_cpu_offload: false
    enable_model_cpu_offload: true
    
  flux:
    enable_attention_slicing: true
    enable_vae_slicing: true
    enable_sequential_cpu_offload: true
    offload_state_dict: true
```

## Installation

### Installing Models

#### From HuggingFace

```bash
# Models will be downloaded automatically on first use
expandor input.jpg -r 4K --model sdxl

# Pre-download models
expandor --download-model sdxl
expandor --download-model sd15
```

#### From Local Files

```bash
# Use local model file
expandor input.jpg -r 4K --model-path /models/my_custom_sdxl.safetensors

# Configure in config.yaml
models:
  my_model:
    model_id: /absolute/path/to/model.safetensors
```

#### From Custom Sources

```python
# Using Python API
from expandor.utils.model_manager import ModelManager

manager = ModelManager()
manager.download_model(
    model_id="custom/model",
    source="https://example.com/model.safetensors",
    verify_hash="sha256:abc123..."
)
```

### Model Storage

Default locations:
```
~/.cache/huggingface/hub/       # HuggingFace cache
~/.cache/expandor/models/       # Expandor cache
/usr/local/share/expandor/models/  # System-wide models
```

Configure custom locations:
```yaml
# ~/.config/expandor/config.yaml
paths:
  model_cache: /large/disk/models
  huggingface_cache: /large/disk/hf_cache
```

## Model-Specific Settings

### SDXL Optimal Settings

```yaml
sdxl_settings:
  # Generation
  inference_steps: 40-60
  guidance_scale: 7.0-8.0
  
  # Resolution
  base_resolution: 1024
  resolutions:
    - [1024, 1024]  # 1:1
    - [1152, 896]   # 9:7
    - [1216, 832]   # 3:2
    - [1344, 768]   # 7:4
    - [1536, 640]   # 12:5
    
  # Refiner
  use_refiner: true
  refiner_start: 0.8
  refiner_strength: 0.3
  
  # Samplers
  recommended_samplers:
    - "DPM++ 2M SDE Karras"
    - "DPM++ 3M SDE Karras"
    - "Euler a"
```

### SD 1.5 Optimal Settings

```yaml
sd15_settings:
  # Generation
  inference_steps: 20-40
  guidance_scale: 7.5
  
  # Resolution
  base_resolution: 512
  resolutions:
    - [512, 512]    # 1:1
    - [512, 768]    # 2:3
    - [768, 512]    # 3:2
    - [640, 640]    # 1:1 alternative
    
  # Performance
  clip_skip: 1
  use_karras_sigmas: true
  
  # Samplers
  recommended_samplers:
    - "DPM++ 2M Karras"
    - "Euler a"
    - "DDIM"
```

### FLUX Optimal Settings

```yaml
flux_settings:
  # Generation
  inference_steps: 50-100
  guidance_scale: 3.5-5.0  # Lower than SD
  
  # Resolution
  base_resolution: 1024
  size_multiple: 16  # Important!
  
  # Memory
  requires_sequential_offload: true
  max_batch_size: 1
  
  # Special parameters
  use_rope: true
  time_shift: 1.0
```

## Performance Considerations

### Speed Comparison

| Model | Steps | Time (RTX 3090) | Time (RTX 4090) |
|-------|-------|------------------|------------------|
| SD 1.5 | 20 | 2-3s | 1-2s |
| SD 1.5 | 50 | 5-7s | 3-4s |
| SDXL | 30 | 8-10s | 4-6s |
| SDXL | 60 | 15-20s | 8-12s |
| FLUX | 50 | 45-60s | 25-35s |

### Memory Usage

```yaml
memory_usage:
  # Approximate VRAM usage for 1024x1024
  sd15:
    fp32: 6GB
    fp16: 4GB
    int8: 2.5GB
    
  sdxl:
    fp32: 16GB
    fp16: 10GB
    int8: 6GB
    
  flux:
    fp32: 40GB+
    fp16: 24GB
    int8: 15GB
```

### Optimization Strategies

```yaml
optimization_by_hardware:
  low_vram_4gb:
    models: ["sd15"]
    dtype: float16
    strategies: ["tiled", "cpu_offload"]
    max_resolution: 768
    
  medium_vram_8gb:
    models: ["sd15", "sd2"]
    dtype: float16
    strategies: ["direct", "tiled"]
    max_resolution: 1024
    
  high_vram_16gb:
    models: ["sd15", "sd2", "sdxl"]
    dtype: float16
    strategies: ["progressive", "direct"]
    max_resolution: 2048
    
  ultra_vram_24gb:
    models: ["all"]
    dtype: float16
    strategies: ["any"]
    max_resolution: 4096
```

## LoRA Compatibility

### LoRA Support by Model

| Model | LoRA Support | Ecosystem | Notes |
|-------|--------------|-----------|-------|
| SD 1.5 | Excellent | Massive | Most LoRAs available |
| SD 2.x | Limited | Small | Different architecture |
| SDXL | Excellent | Growing | High quality LoRAs |
| SD3 | Limited | New | Format differences |
| FLUX | Experimental | Minimal | In development |

### LoRA Configuration

```yaml
lora_compatibility:
  sd15:
    max_loras: 5
    weight_range: [-2.0, 2.0]
    formats: ["safetensors", "ckpt", "pt"]
    
  sdxl:
    max_loras: 10
    weight_range: [-2.0, 2.0]
    formats: ["safetensors"]
    requires_sdxl_loras: true
    
  cross_compatibility:
    sd15_to_sdxl: false
    sdxl_to_sd15: false
    sd2_to_sd15: limited
```

## Custom Models

### Adding Custom Models

```yaml
# ~/.config/expandor/models.yaml
custom_models:
  my_custom_model:
    base_model: "sdxl"  # Inherit settings
    model_id: /path/to/model.safetensors
    
    # Override settings
    constraints:
      optimal_size: 1216
      size_multiple: 8
      
    # Custom components
    components:
      vae: /path/to/vae.safetensors
      
    # Recommended settings
    recommended:
      inference_steps: 45
      guidance_scale: 6.5
      sampler: "DPM++ 2M SDE Karras"
```

### Model Validation

```python
# Validate custom model
from expandor.utils.model_manager import ModelManager

manager = ModelManager()
validation = manager.validate_model("/path/to/model.safetensors")

print(f"Model type: {validation['type']}")
print(f"Compatible: {validation['compatible']}")
print(f"Warnings: {validation['warnings']}")
```

### Model Conversion

```bash
# Convert between formats
expandor-convert \
  --input model.ckpt \
  --output model.safetensors \
  --type sdxl \
  --dtype float16
```

## Troubleshooting

### Common Issues

#### Model Loading Errors

```bash
# Error: Model not found
expandor --download-model sdxl
expandor --verify-models

# Error: Incompatible model format
expandor-convert --input model.ckpt --output model.safetensors
```

#### VRAM Issues

```yaml
# Force model optimization
models:
  sdxl:
    force_optimizations:
      attention_slicing: true
      vae_slicing: true
      cpu_offload: true
```

#### Performance Issues

```bash
# Profile model performance
expandor input.jpg -r 4K --profile-model

# Use optimized variant
expandor input.jpg -r 4K --model sdxl --dtype int8
```

### Model-Specific Errors

#### SDXL Refiner Issues
```yaml
# Disable refiner if causing issues
sdxl_settings:
  use_refiner: false
  
# Or adjust refiner settings
refiner_settings:
  start_step_ratio: 0.8
  strength: 0.2
```

#### FLUX Memory Errors
```yaml
# Force maximum memory optimization
flux_settings:
  force_sequential_offload: true
  offload_state_dict: true
  gradient_checkpointing: true
  chunk_size: 1
```

#### SD2 LoRA Compatibility
```yaml
# Enable compatibility mode
sd2_settings:
  lora_compatibility_mode: true
  use_v_parameterization: true
```

## Best Practices

1. **Start with SD 1.5** for testing and low VRAM
2. **Use SDXL** for production quality
3. **Reserve FLUX** for highest quality needs
4. **Match LoRAs** to correct model version
5. **Test settings** on small images first
6. **Monitor VRAM** usage during processing
7. **Keep models updated** for bug fixes
8. **Use fp16** for optimal performance/quality

## Model Recommendations

### By Use Case

| Use Case | Recommended Model | Settings |
|----------|-------------------|----------|
| Quick previews | SD 1.5 | fast preset |
| Web graphics | SDXL | balanced preset |
| Print quality | SDXL + Refiner | ultra preset |
| Artistic | SDXL + LoRAs | high preset |
| Photorealistic | FLUX | ultra preset |
| Low VRAM | SD 1.5 | tiled strategy |

### By Hardware

| GPU | VRAM | Recommended | Max Resolution |
|-----|------|-------------|----------------|
| GTX 1060 | 6GB | SD 1.5 | 768x768 |
| RTX 3060 | 12GB | SDXL | 1536x1536 |
| RTX 3090 | 24GB | SDXL/FLUX | 2048x2048 |
| RTX 4090 | 24GB | FLUX | 4096x4096 |

## Future Model Support

### Planned Support
- Stable Diffusion 3 (full support)
- DALL-E compatible models
- Midjourney-style models
- Custom training integration

### In Development
- Real-time model switching
- Model merging support
- Automatic model optimization
- Cloud model support

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) - Model configuration details
- [CLI_USAGE.md](CLI_USAGE.md) - Using models via CLI
- [API Documentation](API.md) - Model API reference
- [Examples](../examples/) - Model-specific examples