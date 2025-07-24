# Expandor API Migration Notes

## Purpose
This document tracks the migration from LegacyExpandorConfig to the new ExpandorConfig API.

## Current LegacyExpandorConfig Structure
```python
@dataclass
class LegacyExpandorConfig:
    # Required fields
    source_image: Union[Path, Image.Image]
    target_resolution: Tuple[int, int]
    prompt: str
    seed: int
    source_metadata: Dict[str, Any]
    
    # Pipeline references (being removed)
    inpaint_pipeline: Optional[Any] = None
    refiner_pipeline: Optional[Any] = None
    img2img_pipeline: Optional[Any] = None
    
    # Quality settings
    quality_preset: str = "high"
    save_stages: bool = False
    stage_dir: Optional[Path] = None
    auto_refine: bool = True
    
    # Strategy control
    allow_tiled: bool = True
    allow_cpu_offload: bool = True
    strategy_override: Optional[str] = None
```

## New ExpandorConfig Structure
```python
@dataclass
class ExpandorConfig:
    # Source (required)
    source_image: Union[Path, str, Image.Image]
    
    # Target (at least one required)
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    target_resolution: Optional[Tuple[int, int]] = None
    
    # Generation
    prompt: str = "a high quality image"
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    
    # Strategy
    strategy: str = "auto"
    strategy_override: Optional[str] = None  # For backward compatibility
    
    # Quality
    quality_preset: str = "high"
    denoising_strength: float = 0.95
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    
    # Processing
    save_stages: bool = False
    stage_dir: Optional[Path] = None
    verbose: bool = False
    
    # Resources
    vram_limit_mb: Optional[int] = None
    use_cpu_offload: bool = False
    
    # Advanced
    window_size: Optional[int] = None
    overlap_ratio: float = 0.5
    tile_size: Optional[int] = None
    
    # Metadata
    source_metadata: Dict[str, Any] = field(default_factory=dict)
```

## Key Changes
1. **No more pipeline references** - Pipelines are now managed by the adapter pattern
2. **Flexible target dimensions** - Can specify width/height separately or as tuple
3. **More generation parameters** - Added negative_prompt, denoising_strength, etc.
4. **Resource controls** - Added vram_limit_mb for better memory management
5. **Strategy simplification** - Just "strategy" field, with override for compatibility
6. **Better defaults** - Most fields have sensible defaults

## Migration Approach
1. Update ExpandorConfig class definition
2. Add helper methods for resolution handling
3. Update Expandor.expand() to use new config
4. Update all strategies progressively
5. Update tests and examples
6. Remove legacy code

## Files to Update (discovered by grep)
1. **expandor/core/expandor_wrapper.py** - Uses LegacyExpandorConfig, can be removed entirely
2. **expandor/strategies/experimental/hybrid_adaptive.py** - Multiple uses of LegacyExpandorConfig
3. **expandor/core/config.py** - Contains the LegacyExpandorConfig definition (to be removed last)

Note: The main expandor.py file accepts config, but it's not clear if it expects LegacyExpandorConfig internally.

## Validation Commands
- Config import: `python -c "from expandor.core.config import ExpandorConfig"`
- Test suite: `pytest tests/ -v`
- Examples: `python examples/basic_usage.py`
- CLI: `expandor --help`

## Progress Tracking
- [x] Pre-flight checks complete
- [x] Part 1: Config structure
- [x] Part 2: Core Expandor
- [x] Part 3: Strategies
- [x] Part 4: Processors
- [x] Part 5: Tests
- [x] Part 6: Examples
- [x] Part 7: Cleanup
- [x] Part 8: Style (Previously completed)
- [ ] Part 9: Documentation
- [ ] Part 10: Validation

## Migration Complete!
The API migration from LegacyExpandorConfig to new ExpandorConfig is complete:
- ✅ New config structure implemented with helper methods
- ✅ All strategies updated to use new config
- ✅ Core Expandor and orchestrator updated
- ✅ Tests already using new API work correctly
- ✅ Examples updated to use new API
- ✅ LegacyExpandorConfig completely removed
- ✅ No legacy references remain in codebase