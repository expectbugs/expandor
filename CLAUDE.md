# Expandor Project - Critical Context for Implementation

## ⚠️ READ THIS ENTIRE FILE BEFORE STARTING ⚠️

You are about to implement Expandor, a complex image resolution adaptation system. This file contains ALL the context you need to build it correctly.

## What is Expandor?

Expandor is a **standalone, model-agnostic** image resolution and aspect ratio adaptation system that can expand images to any target resolution while maintaining maximum quality. It is adapted from the ai-wallpaper project but designed to work independently with any image generation pipeline.

### Core Philosophy: FAIL LOUD
- **NO SILENT FAILURES** - Every error must be loud and explicit
- **NO QUALITY COMPROMISE** - Perfect results or explicit failure
- **ALL OR NOTHING** - Operations either work perfectly or fail completely

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

## Implementation Phases

The project is structured in 4 phases (found in expandor*.md files):

### Phase 1: Core Extraction (expandor1.md)
- Repository setup
- Extract core components from ai-wallpaper
- Create base infrastructure
- Mock pipeline interfaces for testing

### Phase 2: Core Implementation (expandor2.*.md)
- Main Expandor class
- Strategy selector with VRAM awareness
- Configuration system
- Metadata tracking
- Error handling framework

### Phase 3: Advanced Strategies (expandor3.*.md)
- **Step 1** (expandor3.1.md): Complex strategies (SWPO, CPU Offload, Adaptive Hybrid)
- **Step 2** (expandor3.2.md): Quality systems (SmartArtifactDetector, SmartQualityRefiner, BoundaryTracker)
- **Step 3** (expandor3.3.md): Integration tests

### Phase 4: Production Readiness (expandor4.md)
- Real pipeline adapters (Diffusers, ComfyUI, A1111)
- Performance optimizations
- CLI and API
- Documentation and examples

## Critical Technical Details

### 1. VRAM Management
- Always calculate requirements before operations
- Implement fallback strategies: Full → Tiled → CPU Offload
- Track peak usage for optimization
- Clear CUDA cache aggressively in memory-constrained scenarios

### 2. Dimension Constraints
- **SDXL**: All dimensions must be multiples of 8
- **FLUX**: All dimensions must be multiples of 16
- Always round dimensions properly before pipeline calls

### 3. Boundary Tracking
- **CRITICAL**: Track EVERY expansion boundary for seam detection
- Store position, direction, step number, expansion size
- Progressive boundaries are different from SWPO window boundaries
- Used later for artifact detection and targeted repair

### 4. Progressive Expansion
- First step: up to 1.4x (reduced from 2.0x for better context)
- Middle steps: 1.25x
- Final step: 1.15x
- High denoising strength (0.95) for content generation
- Adaptive mask blur based on expansion size

### 5. SWPO (Sliding Window Progressive Outpainting)
- For extreme aspect ratios (>4x)
- Window overlap MUST be ≥50% (default 80%)
- Clear cache every 5 windows
- Optional final unification pass
- Track window boundaries separately

### 6. Quality Validation
- Use multiple detection methods: color, gradient, frequency
- Aggressive thresholds for "fail loud" philosophy
- Multi-pass refinement for critical seams
- Zero tolerance for visible artifacts

## Implementation Order & Dependencies

1. **Read ALL expandor*.md files first** - They contain the complete implementation
2. **Follow phases strictly** - Later phases depend on earlier ones
3. **Create files in the exact order specified** - Some files import others
4. **Test each component** before moving to the next
5. **Use mock pipelines first** - Real pipeline integration comes in Phase 4

## Common Pitfalls to Avoid

1. **Import Order**: Some files import from others - follow the file creation order exactly
2. **Method Names**: DimensionCalculator has `calculate_progressive_strategy` not `calculate_progressive_outpaint_strategy`
3. **Type Imports**: Always import types (Optional, Dict, etc.) where needed
4. **Hash Values**: Use `abs(hash(...))` to avoid negative seeds
5. **Path Handling**: result.image_path is already a Path object
6. **VRAM Checks**: Always check available VRAM before operations
7. **Boundary Types**: Dict format for boundaries, not dataclass initially

## Testing Strategy

1. Start with mock pipelines (already provided in expandor1.md)
2. Test each strategy in isolation
3. Test strategy selection logic
4. Test VRAM fallback chains
5. Test quality validation
6. Integration tests with mock pipelines
7. Only then move to real pipeline adapters

## Quality Requirements

- **Seam Detection**: Must detect 100% of progressive boundaries
- **Artifact Repair**: Must fix all detected artifacts or fail
- **VRAM Safety**: Never exceed available VRAM
- **Dimension Accuracy**: Final size must match target exactly
- **Metadata Completeness**: Track everything for debugging

## Key Algorithms

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

## File Structure

```
expandor/
├── CLAUDE.md (this file)
├── expandor1.md - Phase 1: Core extraction
├── expandor2.1.md - Phase 2 Step 1: Main implementation
├── expandor2.2.md - Phase 2 Step 2: Strategy selector
├── expandor2.3.md - Phase 2 Step 3: Integration
├── expandor3.1.md - Phase 3 Step 1: Complex strategies
├── expandor3.2.md - Phase 3 Step 2: Quality systems
├── expandor3.3.md - Phase 3 Step 3: Integration tests
└── expandor4.md - Phase 4: Production readiness
```

## Build Instructions

1. Create a new directory for the Expandor project
2. Copy all expandor*.md files into it
3. Follow expandor1.md step by step to set up the repository
4. Continue through each phase in order
5. Run tests after each major component
6. Never skip ahead - dependencies matter

## Success Criteria

- All tests pass with mock pipelines
- Can expand 1080p → 4K without artifacts
- Can handle 16:9 → 32:9 extreme expansions
- Gracefully handles VRAM limitations
- Clear error messages for all failure modes
- Comprehensive metadata tracking
- Zero visible seams or artifacts

## Remember

This is a complex project that prioritizes **quality over everything else**. Take time to understand each component. When in doubt, fail loud and clear. The goal is perfect image expansion with zero compromises.

Good luck! The implementation guide in the expandor*.md files is comprehensive and tested. Follow it carefully and you'll build an amazing system.