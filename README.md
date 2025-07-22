# Expandor

Universal Image Resolution Adaptation System

## Overview

Expandor is a standalone, model-agnostic image resolution and aspect ratio adaptation system. It provides intelligent strategy selection for expanding images to any target resolution while maintaining maximum quality.

## Features

- **VRAM-Aware Processing**: Automatically adapts to available GPU memory
- **Multiple Expansion Strategies**: Progressive outpainting, SWPO, direct upscaling, and more
- **Zero Quality Compromise**: Aggressive artifact detection and repair
- **Model Agnostic**: Works with any image generation pipeline
- **Fail Loud Philosophy**: Clear error messages, no silent failures

## Installation

```bash
pip install expandor
```

For development:
```bash
git clone https://github.com/yourusername/expandor.git
cd expandor
pip install -e ".[dev]"
```

## Quick Start

```python
from expandor import Expandor, ExpandorConfig
from PIL import Image

# Initialize Expandor
expandor = Expandor()

# Configure expansion
config = ExpandorConfig(
    source_image=Image.open("input.png"),
    target_resolution=(3840, 2160),
    prompt="A beautiful landscape",
    seed=42,
    source_metadata={'model': 'SDXL'},
    quality_preset='high'
)

# Expand image
result = expandor.expand(config)

# Save result
result.image.save("output.png")
print(f"Expanded to {result.size} using {result.strategy_used}")
```

## Development

This project is under active development. See CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE file for details.

## Attribution

Adapted from the [ai-wallpaper](https://github.com/user/ai-wallpaper) project.