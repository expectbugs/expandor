# Expandor: Universal Image Resolution Adaptation System

[![Version](https://img.shields.io/badge/version-0.4.0-blue.svg)](https://github.com/yourusername/expandor)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Expandor is a powerful, model-agnostic image resolution and aspect ratio adaptation system that can expand images to any target resolution while maintaining maximum quality. Originally adapted from the [ai-wallpaper](https://github.com/user/ai-wallpaper) project, Expandor provides a standalone solution for intelligent image expansion.

## 🌟 Key Features

- **Universal Compatibility**: Works with any image generation pipeline (Diffusers, ComfyUI, A1111, custom)
- **Intelligent Strategy Selection**: Automatically chooses the best expansion method based on your hardware and requirements
- **VRAM-Aware Processing**: Adapts to available GPU memory with automatic fallback strategies
- **Extreme Aspect Ratio Support**: Handle expansions up to 8x with specialized algorithms
- **Production-Ready CLI**: Full-featured command-line interface with batch processing
- **Quality Assurance**: Built-in artifact detection and repair systems
- **LoRA Support**: Advanced LoRA stacking with conflict resolution
- **Comprehensive Configuration**: Flexible configuration system with presets and overrides

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Strategies](#strategies)
- [Examples](#examples)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ VRAM for optimal performance
- 16GB+ RAM

### Install from PyPI

```bash
# Basic installation (CPU only)
pip install expandor

# With AI model support (recommended)
pip install expandor[diffusers]

# All features
pip install expandor[all]

# Development
pip install expandor[dev]
```

### Install from Source

```bash
git clone https://github.com/yourusername/expandor
cd expandor
pip install -e .[all]
```

## Quick Start

### 1. Initial Setup

```bash
# Run interactive setup (recommended)
expandor --setup

# Or verify existing setup
expandor --test
```

### 2. Basic Usage

```bash
# Upscale to 4K
expandor photo.jpg -r 4K

# Specific resolution
expandor photo.jpg -r 3840x2160

# Batch processing
expandor *.jpg --batch output/ -r 2x

# With quality preset
expandor photo.jpg -r 4K -q ultra
```

### 3. Advanced Options

```bash
# Limit VRAM usage
expandor large.jpg -r 8K --vram-limit 8192

# Specific model
expandor photo.jpg -r 4K --model sdxl

# Save intermediate stages
expandor photo.jpg -r 4K --save-stages
```

## Troubleshooting

### Common Issues

1. **Import Error**: Install with `pip install expandor[diffusers]`
2. **CUDA Error**: Use `--device cpu` or check GPU drivers
3. **Memory Error**: Use `--vram-limit` or `--strategy cpu_offload`
4. **Config Error**: Run `expandor --setup` to recreate config

### Getting Help

```bash
# Show all options
expandor --help

# Check configuration
expandor --test

# Version info
expandor --version
```

## What's New in v0.4.0

- ✅ Production-ready CLI interface
- ✅ Automatic model management
- ✅ Smart VRAM handling
- ✅ LoRA support
- ✅ Better error messages
- ❌ Breaking: Requires adapter for initialization

See [CHANGELOG.md](CHANGELOG.md) for details.

# With LoRA stacking
expandor input.jpg --resolution 4K --lora style_anime --lora detail_enhance
```

## 📖 Usage

### Command Line Interface

The Expandor CLI provides a comprehensive set of options:

```bash
expandor [OPTIONS] INPUT_PATH
```

**Key Options:**
- `--resolution`: Target resolution (e.g., "3840x2160", "4K", "2x")
- `--quality`: Quality preset (fast/balanced/high/ultra)
- `--strategy`: Force specific strategy (auto/direct/progressive/tiled/swpo)
- `--output`: Output file path
- `--output-dir`: Output directory for batch processing
- `--lora`: Add LoRA (can be used multiple times)
- `--save-stages`: Save intermediate stages for debugging
- `--dry-run`: Preview without processing

Run `expandor --help` for full options.

### Python API

```python
from expandor import Expandor
from expandor.core.config import ExpandorConfig
from expandor.adapters import DiffusersPipelineAdapter

# Basic expansion
expandor = Expandor(
    pipeline_adapter=DiffusersPipelineAdapter(),
    config=ExpandorConfig(target_width=3840, target_height=2160)
)
result = expandor.expand(image)

# With custom pipeline parameters
config = ExpandorConfig(
    target_width=2560,
    target_height=1440,
    custom_pipeline_params={
        'guidance_scale': 7.5,
        'num_inference_steps': 50
    }
)

# Memory-efficient processing
config = ExpandorConfig(
    target_width=7680,
    target_height=4320,
    strategy='tiled',
    tile_size=512,
    enable_memory_efficient=True
)
```

## ⚙️ Configuration

### User Configuration

Create a user configuration file at `~/.config/expandor/config.yaml`:

```yaml
models:
  sdxl:
    model_id: stabilityai/stable-diffusion-xl-base-1.0
    dtype: float16
    device: cuda

loras:
  - name: detail_enhancer
    path: /path/to/loras/detail.safetensors
    weight: 0.7
    type: detail

default_quality: high
default_strategy: auto

preferences:
  save_metadata: true
  auto_select_strategy: true
```

### Quality Presets

- **fast**: Quick processing, suitable for previews
- **balanced**: Good quality/speed trade-off (default)
- **high**: High quality for final outputs
- **ultra**: Maximum quality, no compromises

### Resolution Presets

Built-in resolution presets include:
- HD: 1920x1080
- 2K: 2560x1440
- 4K: 3840x2160
- 5K: 5120x2880
- 8K: 7680x4320

## 🎯 Strategies

Expandor includes multiple expansion strategies:

### Direct Upscale
- Best for: Small to moderate size increases (<2x)
- Fast, single-step processing
- Ideal when aspect ratio remains the same

### Progressive Outpaint
- Best for: Large expansions (2x-4x)
- Multi-step expansion with context preservation
- Handles moderate aspect ratio changes

### SWPO (Sliding Window Progressive Outpaint)
- Best for: Extreme aspect ratio changes (>4x)
- Sliding window approach with overlap
- Perfect for ultrawide conversions

### Tiled Processing
- Best for: Limited VRAM situations
- Processes image in tiles
- Memory-efficient for any size

### Auto Selection
Let Expandor choose the optimal strategy based on:
- Expansion factor
- Aspect ratio change
- Available VRAM
- Quality requirements

## 📚 Examples

Check the `examples/` directory for comprehensive examples:

- `basic_usage.py`: Simple expansion examples
- `batch_processing.py`: Processing multiple images
- `custom_config.py`: Advanced configuration
- `advanced_features.py`: LoRA stacking, metadata, debugging

## 🔧 Pipeline Adapters

### Supported Adapters

- **DiffusersPipelineAdapter**: For Hugging Face Diffusers
- **ComfyUIPipelineAdapter**: For ComfyUI integration (coming soon)
- **A1111PipelineAdapter**: For Automatic1111 WebUI (coming soon)
- **MockPipelineAdapter**: For testing and development

### Creating Custom Adapters

```python
from expandor.adapters import BasePipelineAdapter

class MyCustomAdapter(BasePipelineAdapter):
    def load_pipeline(self, model_name: str, **kwargs):
        # Load your pipeline
        pass
    
    def generate(self, prompt: str, width: int, height: int, **kwargs):
        # Generate image
        pass
    
    def inpaint(self, image, mask, prompt: str, **kwargs):
        # Inpaint image
        pass
```

## 🐛 Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- Use `--strategy tiled` for limited VRAM
- Reduce `--quality` setting
- Enable memory-efficient mode

**Visible Seams**
- Increase quality preset
- Enable artifact detection (default)
- Use SWPO for extreme ratios

**Slow Processing**
- Use `--quality fast` for previews
- Disable `--save-stages` unless debugging
- Use smaller tile sizes

### Debug Mode

Enable comprehensive debugging:

```bash
expandor input.jpg --resolution 4K --verbose --save-stages --stage-dir debug/
```

## 📊 Performance Tips

1. **VRAM Management**
   - Monitor usage with `--verbose`
   - Use tiled strategy for large outputs
   - Clear cache with `CUDA_EMPTY_CACHE=1`

2. **Batch Processing**
   - Use `--output-dir` for multiple files
   - Process in parallel when possible
   - Group by similar resolutions

3. **Quality vs Speed**
   - Start with `balanced` quality
   - Use `fast` for iteration
   - Reserve `ultra` for final outputs

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/expandor.git
cd expandor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Adapted from the [ai-wallpaper](https://github.com/user/ai-wallpaper) project
- Thanks to the Hugging Face Diffusers team
- Community contributors and testers

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/expandor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/expandor/discussions)
- **Documentation**: [Full Docs](https://expandor.readthedocs.io) (coming soon)

---

Made with ❤️ for the AI art community