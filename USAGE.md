# Expandor Usage Guide

This comprehensive guide covers all aspects of using Expandor for image expansion and resolution adaptation.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [ControlNet Usage](#controlnet-usage)
5. [Configuration](#configuration)
6. [Strategies](#strategies)
7. [Troubleshooting](#troubleshooting)

## Installation

### Basic Installation

```bash
# Basic installation with Diffusers support
pip install expandor[diffusers]

# With ControlNet support
pip install expandor[diffusers] opencv-python

# All features
pip install expandor[all]
```

### Initial Setup

```bash
# Run interactive setup wizard
expandor --setup

# Setup ControlNet configuration
expandor --setup-controlnet

# Verify installation
expandor --test
```

## Basic Usage

### CLI Examples

```bash
# Simple upscale to 4K
expandor image.jpg -r 4K

# Specific resolution
expandor image.jpg -r 3840x2160

# Double the size
expandor image.jpg -r 2x

# Change aspect ratio
expandor portrait.jpg -r ultrawide

# With quality preset
expandor image.jpg -r 4K -q ultra

# Save to specific location
expandor image.jpg -r 4K -o output/upscaled.png

# Batch processing
expandor *.jpg --batch output/ -r 4K
```

### Python API

```python
from expandor import Expandor, ExpandorConfig
from expandor.adapters import DiffusersPipelineAdapter

# Initialize adapter
adapter = DiffusersPipelineAdapter(
    model_id="stabilityai/stable-diffusion-xl-base-1.0"
)
expandor = Expandor(adapter)

# Basic expansion
config = ExpandorConfig(
    source_image="photo.jpg",
    target_resolution=(3840, 2160),
    quality_preset="high"
)
result = expandor.expand(config)
result.save("expanded.png")
```

## Advanced Features

### LoRA Support

```bash
# Single LoRA
expandor image.jpg -r 4K --lora style_anime

# Multiple LoRAs with weights
expandor image.jpg -r 4K --lora style_anime:0.7 --lora detail_enhance:0.5
```

### Memory Management

```bash
# Limit VRAM usage
expandor large.jpg -r 8K --vram-limit 8192

# Force CPU offload strategy
expandor huge.jpg -r 8K --strategy cpu_offload

# Use tiled processing
expandor image.jpg -r 8K --strategy tiled
```

### Debugging

```bash
# Verbose output
expandor image.jpg -r 4K --verbose

# Save intermediate stages
expandor image.jpg -r 4K --save-stages

# Custom stage directory
expandor image.jpg -r 4K --save-stages --stage-dir debug/

# Dry run (preview without processing)
expandor image.jpg -r 4K --dry-run
```

## ControlNet Usage

ControlNet allows structure-guided expansion, preserving important features like edges, depth, and composition.

### Setup ControlNet

```bash
# Initial setup (creates controlnet_config.yaml)
expandor --setup-controlnet

# Force recreate configuration
expandor --setup-controlnet --force
```

### Basic ControlNet Expansion

```bash
# Use ControlNet with automatic edge detection
expandor architecture.jpg -r 4K --strategy controlnet_progressive

# With custom prompt
expandor building.jpg -r 4K --strategy controlnet_progressive \
    --prompt "high quality architectural photograph, detailed"
```

### Python API with ControlNet

```python
from expandor import Expandor, ExpandorConfig
from expandor.adapters import DiffusersPipelineAdapter

# Initialize and load ControlNet
adapter = DiffusersPipelineAdapter(
    model_id="stabilityai/stable-diffusion-xl-base-1.0"
)
adapter.load_controlnet("diffusers/controlnet-canny-sdxl-1.0", "canny")

expandor = Expandor(adapter)

# Configure ControlNet expansion
config = ExpandorConfig(
    source_image="architecture.jpg",
    target_resolution=(3840, 2160),
    strategy="controlnet_progressive",
    prompt="high quality architectural photograph",
    negative_prompt="blurry, low quality",
    strategy_params={
        "controlnet_config": {
            "control_type": "canny",
            "controlnet_strength": 0.8,
            "extract_at_each_step": True,
            "canny_low_threshold": 100,
            "canny_high_threshold": 200,
            "canny_dilate": False,
            "canny_l2_gradient": False
        }
    }
)

result = expandor.expand(config)
```

### ControlNet Extractors

```python
from expandor.processors.controlnet_extractors import ControlNetExtractor
from PIL import Image

extractor = ControlNetExtractor()
image = Image.open("photo.jpg")

# Canny edge detection
edges = extractor.extract_canny(
    image,
    low_threshold=100,
    high_threshold=200,
    dilate=True,  # Thicken edges
    l2_gradient=False  # Use L1 gradient
)

# Blur extraction for soft guidance
blurred = extractor.extract_blur(
    image,
    radius=10,
    blur_type="gaussian"  # or "box", "motion"
)

# Depth extraction (requires depth models)
# depth = extractor.extract_depth(image)
```

### ControlNet Configuration

Edit `~/.config/expandor/controlnet_config.yaml`:

```yaml
defaults:
  negative_prompt: ""
  controlnet_strength: 1.0
  strength: 0.8
  num_inference_steps: 50
  guidance_scale: 7.5

extractors:
  canny:
    low_threshold_default: 100
    high_threshold_default: 200
    kernel_size: 3
    dilation_iterations: 1
  blur:
    radius_default: 10
    valid_types: ["gaussian", "box", "motion"]

models:
  sdxl:
    canny: "diffusers/controlnet-canny-sdxl-1.0"
    blur: "diffusers/controlnet-blur-sdxl-1.0"
```

## Configuration

### User Configuration

Create/edit `~/.config/expandor/config.yaml`:

```yaml
# Model settings
models:
  sdxl:
    model_id: stabilityai/stable-diffusion-xl-base-1.0
    dtype: float16
    device: cuda

# LoRA configurations
loras:
  - name: style_anime
    path: /path/to/anime_style.safetensors
    weight: 0.7
    type: style

# Default settings
default_quality: high
default_strategy: auto

# Processing preferences
preferences:
  save_metadata: true
  auto_select_strategy: true
  artifact_detection: true
  seam_repair: true
```

### Environment Variables

```bash
# Override config directory
export EXPANDOR_CONFIG_DIR=/custom/path

# Disable cache
export EXPANDOR_NO_CACHE=1

# Force CPU processing
export CUDA_VISIBLE_DEVICES=""
```

## Strategies

### Strategy Selection

```bash
# Auto selection (default)
expandor image.jpg -r 4K

# Force specific strategy
expandor image.jpg -r 4K --strategy progressive

# Available strategies:
# - auto: Automatic selection
# - direct: Simple upscale (fast)
# - progressive: Multi-step expansion
# - swpo: Sliding window for extreme ratios
# - tiled: Memory-efficient tiled processing
# - cpu_offload: CPU-based processing
# - controlnet_progressive: Structure-guided expansion
```

### Strategy Guidelines

| Strategy | Best For | Speed | Quality | VRAM |
|----------|----------|-------|---------|------|
| Direct | Small upscales (<2x) | Fast | Good | Low |
| Progressive | Large expansions (2-4x) | Medium | Excellent | Medium |
| SWPO | Extreme ratios (>4x) | Slow | Excellent | High |
| Tiled | Limited VRAM | Slow | Good | Low |
| CPU Offload | No GPU | Very Slow | Good | None |
| ControlNet | Structured images | Medium | Excellent | High |

## Troubleshooting

### Common Issues

#### Out of Memory

```bash
# Use memory-efficient strategies
expandor image.jpg -r 8K --strategy tiled

# Limit VRAM
expandor image.jpg -r 8K --vram-limit 6144

# Force CPU processing
expandor image.jpg -r 8K --strategy cpu_offload
```

#### ControlNet Errors

```bash
# Reinstall with ControlNet dependencies
pip install expandor[diffusers] opencv-python

# Recreate configuration
expandor --setup-controlnet --force

# Check configuration
expandor --test
```

#### Visible Seams

```bash
# Use higher quality preset
expandor image.jpg -r ultrawide -q ultra

# Force SWPO for extreme ratios
expandor image.jpg -r ultrawide --strategy swpo

# Enable artifact detection (default)
expandor image.jpg -r 4K --artifact-detection
```

### Debug Information

```bash
# Full debug output
expandor image.jpg -r 4K --verbose --save-stages

# Check system info
expandor --test --verbose

# Dry run to preview
expandor image.jpg -r 4K --dry-run
```

### Performance Tips

1. **Start with lower quality** for testing
2. **Use appropriate strategy** for your expansion ratio
3. **Monitor VRAM usage** with --verbose
4. **Process similar images** in batches
5. **Clear VRAM cache** between large jobs

## Examples

### Portrait to Landscape

```bash
# Convert 768x1344 portrait to 3840x2160 landscape
expandor portrait.jpg -r 4K -q high --strategy progressive
```

### Extreme Ultrawide

```bash
# Create 5120x1440 ultrawide from square image
expandor square.jpg -r ultrawide -q ultra --strategy swpo
```

### Architectural Preservation

```bash
# Preserve building structure during expansion
expandor building.jpg -r 4K --strategy controlnet_progressive \
    --prompt "detailed architectural photograph"
```

### Batch Processing with LoRAs

```bash
# Process folder with style enhancement
expandor images/*.jpg --batch output/ -r 2x \
    --lora detail_enhance:0.5 --lora sharpen:0.3
```

For more examples, see the `examples/` directory in the repository.