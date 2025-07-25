# Expandor Changelog

All notable changes to Expandor are documented here.

## [0.5.1] - 2025-07-25

### 🐛 Fixed
- **Progressive Outpaint Strategy**: Fixed hardcoded denoising strength of 0.75 that was causing harsh visible seams
  - Now properly reads `config.denoising_strength` for adaptive strength
  - Now properly reads `config.num_inference_steps` for adaptive steps
  - Now properly reads `config.guidance_scale` for adaptive guidance
  - Seam repair now uses proportional strength (40% of main strength)
- **Real-ESRGAN Integration**: Added Python wrapper for better integration
  - Automatically detects and uses Real-ESRGAN installation
  - Falls back gracefully if not available
  
### ✨ Added
- **Extensive Real-World Testing**: Tested with actual SDXL models and images
  - Ultrawide expansion (5376x768) with progressive outpainting
  - Portrait orientation expansion (768x1344 → 2160x3840) 
  - Direct upscale strategy with Real-ESRGAN wrapper
  - Strategy comparison tests on complex images

### 🔄 Changed
- **Test Suite Updates**: All tests now use new mandatory adapter API
  - Updated all integration tests to use MockPipelineAdapter
  - Fixed pipeline registration to share adapters with orchestrator
  - Removed unsupported config parameters from tests

### 📝 Documentation
- Added comprehensive test update summary (TEST_UPDATE_SUMMARY.md)
- Added fix summary documenting remaining hardcoded values (fix_summary.md)
- Updated examples to demonstrate proper denoising strength usage

### 🚧 Known Issues
- **Other Strategies**: Still have hardcoded values that need fixing:
  - Tiled Expansion: hardcoded strength values (0.2/0.3/0.4)
  - SWPO Strategy: hardcoded blur_radius=50, strength=0.02
  - CPU Offload: hardcoded strength values (0.9/0.3)

## [0.5.0] - 2025-07-24

### 🚨 Breaking Changes
- **Major API Update**: Migrated from LegacyExpandorConfig to new ExpandorConfig
  - Removed pipeline references from config (now managed by adapters)
  - Added flexible target dimension specification
  - Added more generation parameters (negative_prompt, denoising_strength, etc.)
  - Removed auto_refine field (now always enabled for quality)
  - Added helper methods: get_target_resolution(), get_source_image(), etc.
- Removed expandor_wrapper.py (no longer needed)
- Removed all legacy compatibility code

### 🔄 Changed
- Updated all strategies to use new ExpandorConfig API
- Updated pipeline orchestrator to use registry only
- Updated metadata tracker for new config structure
- Fixed circular import in strategies module
- Improved strategy selector to respect user-specified strategies

### 🐛 Fixed
- Config field access throughout codebase now uses proper helper methods
- Strategy mapping now works correctly with user-friendly names
- Import ordering and style issues (reduced warnings from 712 to 166)

## [0.4.0] - 2025-07-24

### 🎯 Focus: Production Readiness

This release prioritizes stability and reliability over new features.

### ✨ Added
- **CLI Interface**: Full command-line interface with argparse
  - Single image processing: `expandor image.jpg -r 4K`
  - Batch processing: `expandor *.jpg --batch output/`
  - Interactive setup: `expandor --setup`
- **User Configuration**: `~/.config/expandor/config.yaml` support
- **DiffusersPipelineAdapter**: Production-ready HuggingFace integration
- **LoRA Support**: Automatic conflict detection and weight adjustment
- **VRAM Management**: `--vram-limit` flag for memory control
- **Progress Bars**: Visual feedback for long operations
- **Quality Presets**: Configurable via `quality_thresholds.yaml`

### 🔄 Changed
- **BREAKING**: Removed all backwards compatibility
  - `Expandor()` now requires a pipeline adapter
  - No more path-based initialization
- **Imports**: All internal imports now use relative paths
- **Error Handling**: Strict FAIL LOUD philosophy
  - Config errors cause immediate failure
  - Helpful error messages with solutions
- **Logging**: Replaced all print() with proper logging

### 🐛 Fixed
- Import statements moved to file tops (glob, shutil, gc)
- All internal imports now use relative paths
- Silent configuration failures now fail loud
- Config validation errors provide detailed solutions
- Fixed example scripts to use correct API

### 📝 Known Limitations
- **ControlNet**: Can load models but not generate (Phase 5)
- **ComfyUI Adapter**: Placeholder only (Phase 5) 
- **A1111 Adapter**: Placeholder only (Phase 5)
- **Real-ESRGAN**: Requires separate installation
- **API Mismatch**: Tests use new API, core uses LegacyExpandorConfig (Phase 5)
- **Code Style**: 712 flake8 warnings remain (no functional impact)

### 💔 Breaking Changes
1. **Initialization**:
   ```python
   # OLD (no longer works):
   expandor = Expandor()
   expandor = Expandor("/path/to/config")
   
   # NEW (required):
   from expandor.adapters import DiffusersPipelineAdapter
   adapter = DiffusersPipelineAdapter(model_id="...")
   expandor = Expandor(adapter)
   ```

2. **Configuration**: Invalid configs now fail immediately instead of skipping

3. **Imports**: Package now uses relative imports internally

### 📦 Dependencies
- Python ≥ 3.8
- torch ≥ 2.0
- diffusers ≥ 0.24.0 (for DiffusersAdapter)
- PIL/Pillow ≥ 9.0
- numpy ≥ 1.20
- tqdm ≥ 4.65

### 🚀 Migration Guide

From v0.3.x to v0.4.0:

1. **Update initialization**:
   ```python
   # Add adapter
   from expandor.adapters import DiffusersPipelineAdapter
   adapter = DiffusersPipelineAdapter(
       model_id="stabilityai/stable-diffusion-xl-base-1.0"
   )
   expandor = Expandor(adapter)
   ```

2. **Fix configuration**:
   ```bash
   # Validate your config
   expandor --test
   
   # Or create fresh config
   expandor --setup
   ```

3. **Update imports** (if using as library):
   - No more `from expandor.module import ...`
   - Use relative imports in custom code

### 🎮 Quick Start

```bash
# Install
pip install expandor[diffusers]

# Setup
expandor --setup

# Use
expandor photo.jpg -r 4K -q ultra
```

---

## Previous Releases

### [0.3.0] - 2024-01-15
- Advanced strategies (SWPO, CPU Offload)
- Quality validation system
- Boundary tracking

### [0.2.0] - 2024-01-08  
- Core implementation
- Basic strategies
- VRAM management

### [0.1.0] - 2024-01-01
- Initial release
- Core extraction from ai-wallpaper