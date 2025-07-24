# Expandor CLI Usage Guide

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Command Reference](#command-reference)
5. [Resolution Formats](#resolution-formats)
6. [Quality Presets](#quality-presets)
7. [Strategies](#strategies)
8. [Batch Processing](#batch-processing)
9. [LoRA Management](#lora-management)
10. [Advanced Options](#advanced-options)
11. [Environment Variables](#environment-variables)
12. [Examples](#examples)
13. [Troubleshooting](#troubleshooting)

## Overview

The Expandor CLI provides a powerful command-line interface for image expansion tasks. It supports single file processing, batch operations, and extensive customization through command-line options and configuration files.

## Installation

After installing Expandor, the CLI will be available as:

```bash
expandor --help
```

Or through the Python module:

```bash
python -m expandor.cli --help
```

## Basic Usage

### Simple Expansion

```bash
# Expand to a specific resolution
expandor input.jpg --resolution 3840x2160 --output output.png

# Use resolution preset
expandor input.jpg --resolution 4K

# Use multiplier
expandor input.jpg --resolution 2x
```

### Output Options

```bash
# Specify output file
expandor input.jpg --resolution 4K --output expanded.png

# Specify output directory (keeps original filename)
expandor input.jpg --resolution 4K --output-dir results/

# Specify output format
expandor input.jpg --resolution 4K --output-format webp
```

## Command Reference

### Synopsis

```bash
expandor [OPTIONS] INPUT_PATH
```

### Positional Arguments

- `INPUT_PATH`: Path to input image or pattern for batch processing
  - Single file: `image.png`
  - Wildcard: `*.jpg`
  - Directory: `images/` (processes all images in directory)

### Core Options

#### `--resolution, -r`
Target resolution for expansion.

Formats:
- Explicit: `3840x2160`
- Preset: `HD`, `2K`, `4K`, `5K`, `8K`
- Multiplier: `2x`, `3.5x`
- Percentage: `150%`

```bash
expandor input.jpg -r 3840x2160
expandor input.jpg -r 4K
expandor input.jpg -r 2.5x
```

#### `--quality, -q`
Quality preset for processing.

Options: `fast`, `balanced`, `high`, `ultra`

```bash
expandor input.jpg -r 4K -q ultra
```

#### `--strategy, -s`
Force specific expansion strategy.

Options: `auto`, `direct`, `progressive`, `tiled`, `swpo`, `hybrid`

```bash
expandor input.jpg -r 4K -s progressive
```

### Output Options

#### `--output, -o`
Output file path (single file processing only).

```bash
expandor input.jpg -r 4K -o ~/Pictures/expanded.png
```

#### `--output-dir, -d`
Output directory for processed files.

```bash
expandor *.jpg -r 4K -d results/
```

#### `--output-format`
Output image format.

Options: `png`, `jpg`/`jpeg`, `webp`

```bash
expandor input.jpg -r 4K --output-format webp
```

#### `--output-quality`
JPEG/WebP compression quality (1-100).

```bash
expandor input.jpg -r 4K --output-format jpg --output-quality 95
```

### Model and Pipeline Options

#### `--model, -m`
Select model to use.

Options: `sdxl`, `sd15`, `sd2`, `sd3`, `flux`

```bash
expandor input.jpg -r 4K -m sdxl
```

#### `--adapter`
Pipeline adapter to use.

Options: `auto`, `diffusers`, `comfyui`, `a1111`, `mock`

```bash
expandor input.jpg -r 4K --adapter diffusers
```

#### `--model-path`
Path to custom model (overrides --model).

```bash
expandor input.jpg -r 4K --model-path /models/custom_sdxl.safetensors
```

### LoRA Options

#### `--lora, -l`
Add LoRA to the stack (can be used multiple times).

```bash
expandor input.jpg -r 4K -l style_anime -l detail_enhance
```

#### `--lora-weight`
Set weight for the last specified LoRA.

```bash
expandor input.jpg -r 4K -l style_anime --lora-weight 0.8
```

### Generation Parameters

#### `--prompt, -p`
Positive prompt for generation.

```bash
expandor input.jpg -r 4K -p "beautiful landscape, high quality, detailed"
```

#### `--negative-prompt, -n`
Negative prompt for generation.

```bash
expandor input.jpg -r 4K -n "blurry, low quality, artifacts"
```

#### `--seed`
Seed for reproducible results.

```bash
expandor input.jpg -r 4K --seed 42
```

#### `--steps`
Number of inference steps (overrides quality preset).

```bash
expandor input.jpg -r 4K --steps 80
```

#### `--cfg-scale`
Classifier-free guidance scale.

```bash
expandor input.jpg -r 4K --cfg-scale 7.5
```

### Memory and Performance

#### `--vram-limit`
Maximum VRAM to use in MB.

```bash
expandor input.jpg -r 8K --vram-limit 8000  # 8GB limit
```

#### `--tile-size`
Tile size for tiled processing.

```bash
expandor input.jpg -r 8K --strategy tiled --tile-size 512
```

#### `--cpu-offload`
Enable CPU offload for low VRAM.

```bash
expandor input.jpg -r 4K --cpu-offload
```

### Quality Control

#### `--no-artifact-detection`
Disable automatic artifact detection and repair.

```bash
expandor input.jpg -r 4K --no-artifact-detection
```

#### `--artifact-threshold`
Sensitivity for artifact detection (0.0-1.0).

```bash
expandor input.jpg -r 4K --artifact-threshold 0.05
```

### Debug and Development

#### `--save-stages`
Save intermediate processing stages.

```bash
expandor input.jpg -r 4K --save-stages
```

#### `--stage-dir`
Directory for saving stages.

```bash
expandor input.jpg -r 4K --save-stages --stage-dir debug/
```

#### `--verbose, -v`
Enable verbose output.

```bash
expandor input.jpg -r 4K -v
```

#### `--dry-run`
Preview operations without processing.

```bash
expandor input.jpg -r 4K --dry-run
```

### Configuration

#### `--config, -c`
Path to custom configuration file.

```bash
expandor input.jpg -r 4K -c custom_config.yaml
```

#### `--setup`
Run interactive setup wizard.

```bash
expandor --setup
```

### Information

#### `--help, -h`
Show help message.

```bash
expandor --help
```

#### `--version`
Show version information.

```bash
expandor --version
```

## Resolution Formats

### Explicit Resolution
Specify exact width and height:
```bash
expandor input.jpg -r 3840x2160
expandor input.jpg -r 1920x1080
```

### Resolution Presets
Use common resolution names:
- `HD`: 1920x1080
- `FHD`: 1920x1080 (alias for HD)
- `2K`: 2560x1440
- `QHD`: 2560x1440 (alias for 2K)
- `4K`: 3840x2160
- `UHD`: 3840x2160 (alias for 4K)
- `5K`: 5120x2880
- `8K`: 7680x4320

```bash
expandor input.jpg -r 4K
expandor input.jpg -r QHD
```

### Multipliers
Scale by a factor:
```bash
expandor input.jpg -r 2x      # Double size
expandor input.jpg -r 1.5x    # 1.5x size
expandor input.jpg -r 0.5x    # Half size (downscale)
```

### Percentages
Scale by percentage:
```bash
expandor input.jpg -r 150%    # 150% of original
expandor input.jpg -r 200%    # Double size
```

### Aspect Ratio Preservation
Add `@` to preserve aspect ratio:
```bash
expandor input.jpg -r 3840x@   # Width 3840, height auto
expandor input.jpg -r @x2160   # Height 2160, width auto
```

## Quality Presets

### fast
- Inference steps: 20-30
- Lower denoising strength
- Suitable for: Previews, iteration
- Processing time: Fastest

### balanced (default)
- Inference steps: 40-50
- Balanced settings
- Suitable for: General use
- Processing time: Moderate

### high
- Inference steps: 60-80
- Higher quality settings
- Suitable for: Final outputs
- Processing time: Slower

### ultra
- Inference steps: 80-100+
- Maximum quality settings
- Suitable for: Professional use
- Processing time: Slowest

## Strategies

### auto (default)
Automatically selects the best strategy based on:
- Expansion factor
- Aspect ratio change
- Available VRAM
- Image characteristics

### direct
- Single-step upscaling
- Best for: Small expansions (<2x)
- Fastest processing
- Minimal VRAM usage

### progressive
- Multi-step expansion
- Best for: Large expansions (2x-4x)
- Better context preservation
- Moderate VRAM usage

### tiled
- Processes in tiles
- Best for: Limited VRAM
- Any expansion size
- Memory efficient

### swpo
- Sliding Window Progressive Outpaint
- Best for: Extreme aspect ratios
- Handles >4x expansions
- Higher VRAM usage

### hybrid
- Combines multiple strategies
- Adaptive processing
- Best quality/performance balance
- Auto-selected based on image

## Batch Processing

### Wildcard Patterns
```bash
# Process all JPG files
expandor "*.jpg" -r 4K -d results/

# Process specific pattern
expandor "IMG_*.png" -r 2K -d processed/

# Multiple patterns
expandor "*.jpg" "*.png" -r 4K -d output/
```

### Directory Processing
```bash
# Process all images in directory
expandor images/ -r 4K -d output/

# Recursive processing
expandor images/ -r 4K -d output/ --recursive
```

### Batch Options
```bash
# Skip existing files
expandor *.jpg -r 4K -d output/ --skip-existing

# Overwrite existing files
expandor *.jpg -r 4K -d output/ --overwrite

# Parallel processing
expandor *.jpg -r 4K -d output/ --workers 4
```

## LoRA Management

### Basic LoRA Usage
```bash
# Single LoRA
expandor input.jpg -r 4K -l style_anime

# Multiple LoRAs
expandor input.jpg -r 4K -l style_anime -l detail_enhance
```

### LoRA Weights
```bash
# Set weight for each LoRA
expandor input.jpg -r 4K \
  -l style_anime --lora-weight 0.8 \
  -l detail_enhance --lora-weight 0.6
```

### LoRA Paths
```bash
# Use LoRA by path
expandor input.jpg -r 4K \
  -l /models/loras/custom_style.safetensors
```

### LoRA Configuration
Create `~/.config/expandor/loras.yaml`:
```yaml
loras:
  style_anime:
    path: /models/loras/anime_style_v2.safetensors
    default_weight: 0.7
    type: style
  
  detail_enhance:
    path: /models/loras/detail_tweaker.safetensors
    default_weight: 0.6
    type: detail
```

## Advanced Options

### Custom Pipeline Parameters
```bash
# Override pipeline parameters
expandor input.jpg -r 4K \
  --pipeline-param guidance_scale=8.5 \
  --pipeline-param num_inference_steps=80
```

### Memory Management
```bash
# Clear CUDA cache frequently
expandor input.jpg -r 8K --clear-cache-frequency 5

# Enable sequential CPU offload
expandor input.jpg -r 4K --sequential-offload

# Set attention slicing
expandor input.jpg -r 4K --attention-slicing auto
```

### Advanced Tiling
```bash
# Custom tile configuration
expandor input.jpg -r 8K \
  --strategy tiled \
  --tile-size 768 \
  --tile-overlap 128 \
  --tile-blend-mode gaussian
```

## Environment Variables

### EXPANDOR_CONFIG_DIR
Custom configuration directory:
```bash
export EXPANDOR_CONFIG_DIR=/custom/config/path
expandor input.jpg -r 4K
```

### EXPANDOR_CACHE_DIR
Cache directory for models:
```bash
export EXPANDOR_CACHE_DIR=/large/disk/cache
expandor input.jpg -r 4K
```

### EXPANDOR_LOG_LEVEL
Logging verbosity:
```bash
export EXPANDOR_LOG_LEVEL=DEBUG
expandor input.jpg -r 4K
```

### CUDA_VISIBLE_DEVICES
GPU selection:
```bash
export CUDA_VISIBLE_DEVICES=0,1
expandor input.jpg -r 4K
```

## Examples

### Basic Examples

```bash
# Simple 4K expansion
expandor photo.jpg -r 4K

# High quality with specific output
expandor photo.jpg -r 4K -q high -o photo_4k.png

# Batch processing to directory
expandor *.jpg -r 2560x1440 -d upscaled/
```

### Quality-Focused Examples

```bash
# Ultra quality for print
expandor artwork.png -r 8K -q ultra --save-stages

# Fast preview
expandor concept.jpg -r 2K -q fast --dry-run

# Balanced with custom steps
expandor photo.jpg -r 4K -q balanced --steps 60
```

### Memory-Constrained Examples

```bash
# Low VRAM system (4GB)
expandor large.jpg -r 4K --vram-limit 4000 --strategy tiled

# CPU offload for very limited VRAM
expandor huge.png -r 8K --cpu-offload --tile-size 512

# Aggressive memory management
expandor *.jpg -r 4K --clear-cache-frequency 1 --sequential-offload
```

### Aspect Ratio Changes

```bash
# Convert 16:9 to 21:9 ultrawide
expandor movie_still.jpg -r 5120x2160 -s swpo

# Convert 4:3 to 16:9
expandor old_photo.jpg -r 1920x1080 -s progressive

# Extreme panorama
expandor landscape.jpg -r 10240x2160 -s swpo --save-stages
```

### LoRA Combinations

```bash
# Anime style with detail enhancement
expandor character.jpg -r 4K \
  -l anime_style --lora-weight 0.8 \
  -l detail_enhance --lora-weight 0.6

# Multiple style LoRAs (will auto-resolve conflicts)
expandor portrait.jpg -r 4K \
  -l photorealistic \
  -l face_enhance \
  -l color_boost
```

### Production Pipeline

```bash
# Full production pipeline with metadata
expandor raw_render.png \
  -r 4K \
  -q ultra \
  -l production_quality \
  -p "professional photograph, extreme detail" \
  -n "artifacts, noise, blur" \
  --seed 12345 \
  --save-stages \
  --stage-dir production_stages/ \
  -o final_4k.png

# Batch production with logging
expandor renders/*.png \
  -r 4K \
  -q high \
  -d production_output/ \
  --workers 4 \
  --skip-existing \
  -v > production.log
```

## Troubleshooting

### Common Issues

#### Out of Memory (CUDA OOM)
```bash
# Solution 1: Use tiled strategy
expandor input.jpg -r 4K --strategy tiled --tile-size 512

# Solution 2: Limit VRAM usage
expandor input.jpg -r 4K --vram-limit 6000

# Solution 3: Enable CPU offload
expandor input.jpg -r 4K --cpu-offload
```

#### Slow Processing
```bash
# Use faster quality preset
expandor input.jpg -r 4K -q fast

# Reduce inference steps
expandor input.jpg -r 4K --steps 30

# Use direct strategy for small expansions
expandor input.jpg -r 2K -s direct
```

#### Visible Seams or Artifacts
```bash
# Increase quality
expandor input.jpg -r 4K -q ultra

# Adjust artifact detection
expandor input.jpg -r 4K --artifact-threshold 0.02

# Use SWPO for extreme ratios
expandor input.jpg -r 5120x1440 -s swpo
```

#### Command Not Found
```bash
# Use Python module directly
python -m expandor.cli input.jpg -r 4K

# Check installation
pip show expandor

# Reinstall
pip install --upgrade expandor
```

### Debug Mode

Enable maximum debugging information:
```bash
expandor input.jpg -r 4K \
  --verbose \
  --save-stages \
  --stage-dir debug/ \
  --dry-run
```

Check debug stages:
```bash
ls -la debug/
# stage_001_original.png
# stage_002_preprocessed.png
# stage_003_expanded.png
# stage_004_artifacts_detected.png
# stage_005_refined.png
# metadata.json
```

### Getting Help

```bash
# Show all options
expandor --help

# Show version and system info
expandor --version --verbose

# Run diagnostic
expandor --diagnose

# Check configuration
expandor --config-info
```

## Tips and Best Practices

1. **Start with lower quality** for testing, then increase for final output
2. **Use --dry-run** to preview operations before processing
3. **Save stages** when debugging issues or fine-tuning parameters
4. **Monitor VRAM** usage with --verbose to optimize settings
5. **Use batch processing** for similar images to save time
6. **Configure defaults** in ~/.config/expandor/config.yaml
7. **Test strategies** to find the best for your image types
8. **Keep LoRAs organized** with meaningful names and weights

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) - Detailed configuration guide
- [MODELS.md](MODELS.md) - Supported models and requirements
- [API Documentation](API.md) - Python API reference
- [Examples](../examples/) - Example scripts and use cases