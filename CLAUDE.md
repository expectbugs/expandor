# Expandor Project - Critical Context for Implementation

## ⚠️ READ THIS ENTIRE FILE BEFORE STARTING ⚠️

You are working with Expandor, a complex image resolution adaptation system. This file contains critical context about the project's current state and philosophy.

## What is Expandor?

Expandor is a **standalone, model-agnostic** image resolution and aspect ratio adaptation system that can expand images to any target resolution while maintaining maximum quality. It is adapted from the ai-wallpaper project but designed to work independently with any image generation pipeline.

### Core Philosophy: QUALITY & ELEGANCE
- **NO SILENT FAILURES** - Every error must be loud and explicit
- **NO QUALITY COMPROMISE** - Perfect results or explicit failure
- **ALL OR NOTHING** - Operations either work perfectly or fail completely
- **NO BACKWARDS COMPATIBILITY** - Clean, forward-looking design only
- **ELEGANCE OVER SIMPLICITY** - Sophisticated solutions for complex problems

## Project Origin & Attribution

Expandor is adapted from [ai-wallpaper](https://github.com/user/ai-wallpaper), specifically extracting and generalizing:
- VRAMCalculator → VRAMManager
- ResolutionManager → DimensionCalculator  
- AspectAdjuster → ProgressiveOutpaintStrategy
- SmartArtifactDetector → ArtifactDetector

All original logic and algorithms are preserved with full attribution.

## Hardware Context

Originally developed for:
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **Priority**: Quality over speed/space - no limits on time or storage
- **Goal**: Generate amazing, incredibly detailed, ultimate high quality images

## Current Implementation Status

### ✅ Completed (Phases 1-3)
- Core extraction and infrastructure
- Main Expandor class with adapter pattern
- All expansion strategies (Direct, Progressive, SWPO, Tiled, CPU Offload)
- Quality validation and artifact detection systems
- Comprehensive test suite
- Configuration system with YAML support

### ✅ Phase 4 Status (Complete)
- ✅ CLI interface with argparse
- ✅ User configuration system
- ✅ DiffusersPipelineAdapter (functional)
- ✅ ControlNet: Model loading only (generation in Phase 5)
- ✅ ComfyUI/A1111 adapters: Documented placeholders
- ✅ All critical code quality issues fixed

## Critical Technical Details

### 1. Adapter Pattern (REQUIRED)
```python
# Expandor now REQUIRES a pipeline adapter - no exceptions
from expandor.adapters import DiffusersPipelineAdapter
adapter = DiffusersPipelineAdapter(model_id="stabilityai/sdxl")
expandor = Expandor(adapter)  # adapter is mandatory
```

### 2. Import Style
- Use relative imports within the package: `from ..utils import`
- No absolute imports: ~~`from expandor.utils import`~~
- Consistency is critical for maintainability

### 3. Error Handling
- All errors must fail loud with helpful messages
- Config validation raises ValueError on ANY invalid data
- No print statements - use logging only
- Include solutions in error messages

### 4. Quality Thresholds
- Now configurable via `quality_thresholds.yaml`
- Different presets: ultra, high, balanced, fast
- Thresholds loaded dynamically by quality systems

## Implementation Phases

### Phase 1-3: ✅ Complete
Core functionality implemented and tested

### Phase 4: ✅ Complete
All production readiness tasks completed:
1. ✅ Fixed all import issues (moved to top of files)
2. ✅ Removed backwards compatibility (adapter now mandatory)
3. ✅ Implemented strict FAIL LOUD philosophy
4. ✅ Completed partial implementations (ControlNet model loading)
5. ✅ Added comprehensive testing scripts
6. ✅ Updated all documentation

### Phase 4.5: ✅ Critical API Fix Complete
- ✅ Migrated from LegacyExpandorConfig to new ExpandorConfig API
- ✅ All strategies updated to use new config
- ✅ Removed all legacy code and backwards compatibility
- ✅ Fixed circular imports and reduced style warnings by 77%
- ✅ Version bumped to 0.5.0

### Phase 5: 📋 In Progress
- [ ] Full ControlNet implementation (generation, not just loading)
- [ ] ComfyUI adapter implementation
- [ ] A1111 adapter implementation
- [ ] Advanced features and optimizations

## Known Issues (Resolved)

1. ~~**API Mismatch**: Tests and examples use new ExpandorConfig API while core uses LegacyExpandorConfig~~ ✅ FIXED
2. ~~**Code Style**: 712 flake8 warnings~~ ✅ REDUCED to 166 warnings (77% reduction)
3. **Test Coverage**: Now measurable with unified API

All Phase 4 and Phase 5 critical issues have been resolved.

## Testing Requirements

- Mock adapter tests must pass
- Real CLI usage must work
- Config validation must fail loud
- All imports must be relative
- No duplicate methods or functions
- Code quality tools must pass

## Key Algorithms (Unchanged)

### Progressive Expansion Planning
- Calculate aspect ratio change
- If >8x, reject (too extreme)
- Plan steps with decreasing ratios
- Center original image for better context

### SWPO Window Planning
- Calculate total expansion needed
- Determine window size and overlap
- Plan windows to cover expansion
- Ensure last window reaches exact target

### Artifact Detection
- Check known boundary positions
- Color discontinuity detection
- Gradient spike detection  
- Frequency domain analysis
- Combine all methods with weights

## Remember

This project prioritizes **quality over everything else**. When implementing fixes:
1. Follow the FAIL LOUD philosophy strictly
2. Create elegant, complete solutions - complexity is acceptable for quality
3. Test everything thoroughly
4. Document limitations honestly

The goal is perfect image expansion with zero compromises. Elegance and completeness matter more than simplicity.