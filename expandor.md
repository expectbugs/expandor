# Expandor - Universal Image Resolution Adaptation System

## Executive Summary

Expandor is a standalone, model-agnostic image resolution and aspect ratio adaptation system that transforms ANY image to ANY resolution with maximum quality using AI enhancement. Currently at v0.6.0 with Phase 5 (ControlNet) complete, it provides intelligent strategy selection, seamless integration with AI models, structure-guided expansion via ControlNet, and maintains the project's core philosophy of "quality over all" with no silent failures.

## Current Status

### ✅ Completed (Phases 1-3)
- Core extraction from ai-wallpaper with full attribution
- Main Expandor class with **mandatory adapter pattern**
- All expansion strategies (Direct, Progressive, SWPO, Tiled, CPU Offload)
- Quality validation and artifact detection systems
- Comprehensive test suite
- YAML configuration system

### ✅ Phase 4: Production Readiness (Complete)
**Implemented:**
- CLI interface with argparse
- User configuration system (~/.config/expandor/config.yaml)
- DiffusersPipelineAdapter for HuggingFace models
- LoRA support with conflict detection
- Interactive setup wizard
- Strict FAIL LOUD philosophy throughout
- All imports moved to top of files
- Relative imports within package
- Quality thresholds configurable via YAML

**Partial Features (As Designed):**
- ComfyUI/A1111: Documented placeholder adapters (Phase 6)

**Known Limitations:**
- 166 flake8 style warnings (77% reduction from 712)
- ControlNet currently limited to SDXL models only
- Depth extraction requires additional models

### ✅ Phase 5: ControlNet Implementation (Complete)
**Implemented:**
- Full ControlNet generation support for structure-guided expansion
- Canny edge detection for structure preservation
- Blur extraction for soft guidance control
- Depth extraction support (requires depth models)
- ControlNetProgressiveStrategy for guided expansion
- Dynamic pipeline management for VRAM efficiency
- Configuration system via controlnet_config.yaml
- CLI setup command: `expandor --setup-controlnet`
- Comprehensive test suite and examples
- Aligned ExpandorConfig API across codebase

### 📋 Phase 6: Planned
- Complete ComfyUI and A1111 adapter implementations
- Fix remaining code style issues
- Performance benchmarks and optimizations

## Core Philosophy

- **Quality Over All**: No compromises, perfect or fail
- **Fail Loud**: All errors explicit with solutions
- **No Backwards Compatibility**: Clean, forward-looking design
- **Model Agnostic**: Works with any pipeline via adapters
- **Zero Seams**: No visible artifacts, ever
- **Elegance & Completeness**: Sophisticated solutions for complex problems

## Architecture

### Required Usage Pattern
```python
# Expandor REQUIRES an adapter - no exceptions
from expandor.adapters import DiffusersPipelineAdapter

adapter = DiffusersPipelineAdapter(model_id="stabilityai/sdxl")
expandor = Expandor(adapter)  # adapter is mandatory
```

### Key Components
- **Core**: Expandor, strategy selector, VRAM manager, boundary tracker
- **Strategies**: Direct upscale, progressive outpaint, SWPO, tiled, CPU offload
- **Quality**: Artifact detector, quality validator, seam repair
- **Adapters**: Diffusers (working), ComfyUI/A1111 (planned)
- **CLI**: Full interface with setup wizard

## Critical Technical Details

### Import Style
- Use relative imports: `from ..utils import`
- Never absolute: ~~`from expandor.utils import`~~

### Error Handling
- Fail loud with helpful messages
- No silent failures or warnings
- Include solutions in errors

### Quality Thresholds
- Configurable via `quality_thresholds.yaml`
- Presets: ultra, high, balanced, fast

### VRAM Management
- Automatic strategy selection based on available memory
- Fallback chain: Full → Tiled → CPU Offload

## Usage Examples

### CLI (Primary Interface)
```bash
# Simple upscale
expandor photo.jpg -r 4K

# Change aspect ratio
expandor portrait.jpg -r ultrawide -q ultra

# Batch processing
expandor *.jpg --batch output/ -r 2x

# With specific model
expandor image.png --model sdxl --strategy progressive

# Setup ControlNet
expandor --setup-controlnet

# Use ControlNet for structure-preserving expansion
expandor architecture.jpg -r 4K --strategy controlnet_progressive
```

### Python API
```python
from expandor import Expandor
from expandor.adapters import DiffusersPipelineAdapter

# Setup (required)
adapter = DiffusersPipelineAdapter(
    model_id="stabilityai/stable-diffusion-xl-base-1.0"
)
expandor = Expandor(adapter)

# Basic expansion
from expandor.core.config import ExpandorConfig

config = ExpandorConfig(
    source_image="photo.jpg",
    target_resolution=(3840, 2160),
    quality_preset="ultra"
)
result = expandor.expand(config)

# ControlNet expansion (structure-guided)
adapter.load_controlnet("diffusers/controlnet-canny-sdxl-1.0", "canny")

config = ExpandorConfig(
    source_image="architecture.jpg",
    target_resolution=(3840, 2160),
    strategy="controlnet_progressive",
    prompt="high quality architectural photograph",
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

## Implementation Roadmap

### Phase 4: Production Readiness (Current - 2-3 days)
Fix critical issues and complete CLI:
1. Remove duplicate methods and fix imports
2. Eliminate backwards compatibility 
3. Implement strict FAIL LOUD
4. Complete testing scripts
5. Update documentation

### Phase 5: ControlNet Integration (✅ Complete)
- Full ControlNet implementation for structure-guided expansion
- Control extractors (Canny, Depth, Blur)
- Controlled progressive strategies
- Dynamic pipeline management
- Complete configuration system

### Phase 6: Real Adapters (3-4 days)
- Complete ComfyUI adapter
- Complete A1111 adapter
- Model management system

### Phase 7: SD 3.5 & Advanced (3-4 days)
- SD 3.5 integration
- Advanced LoRA management
- Hybrid pipeline strategies

### Phase 8: Polish (1 week)
- Performance optimizations
- Advanced batch processing
- Distribution packaging

## Success Metrics

- **Quality**: Zero visible seams or artifacts
- **Usability**: Simple CLI that anyone can use
- **Reliability**: Clear errors, no surprises
- **Performance**: Efficient VRAM usage
- **Flexibility**: Works with any model
- **Elegance**: Well-architected, maintainable code

## Known Limitations

1. **ControlNet**: Currently limited to SDXL models only
2. **Depth Extraction**: Requires separate depth estimation models
3. **ComfyUI/A1111**: Placeholder adapters for Phase 6
4. **Real-ESRGAN**: Requires separate installation
5. **SD 3.5**: Planned for Phase 7

## Getting Started

```bash
# Install
pip install expandor[diffusers]

# Setup
expandor --setup

# Test
expandor test.jpg -r 4K

# Get help
expandor --help
```

For implementation details, see:
- `CLAUDE.md` - Project context and philosophy
- `CHANGELOG.md` - Version history and breaking changes
- `problems_resolved.md` - Phase 4 issues resolution
- `PHASE4_COMPLETION_SUMMARY.md` - Phase 4 completion details

Remember: **Quality over all, fail loud, no compromises. Elegance and completeness are worth the complexity.**