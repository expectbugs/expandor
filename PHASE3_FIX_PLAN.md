# Phase 3 Fix Plan - Systematic Resolution Strategy

## Overview
This document provides a systematic plan to fix all issues identified in the comprehensive code review.

## Fix Order (by dependency and severity)

### Stage 1: Core Infrastructure Fixes
These must be fixed first as other components depend on them.

#### 1.1 Fix Exception Classes
**Files**: `expandor/core/exceptions.py`
**Changes**:
```python
class StrategyError(ExpandorError):
    """Strategy selection or execution errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, **kwargs):
        self.details = details
        super().__init__(message, **kwargs)
```

#### 1.2 Fix Enum Comparison
**Files**: `expandor/processors/artifact_detector_enhanced.py`
**Changes**:
```python
from enum import IntEnum  # Change import

class ArtifactSeverity(IntEnum):  # Use IntEnum for ordering
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
```

#### 1.3 Standardize Boundary Data Structure
**Decision**: Keep dict format from BoundaryTracker, update consumers
**Files**: 
- `expandor/processors/boundary_analysis.py`
- `expandor/processors/quality_orchestrator.py`

### Stage 2: Interface Standardization

#### 2.1 Fix Quality Validator Return Keys
**File**: `expandor/processors/quality_validator.py`
**Changes**: Either update validator OR update consumers. Recommend updating consumers to match validator output:
```python
# In expandor.py:
result.quality_score = validation_result.get('quality_score', 1.0)
result.seams_detected = validation_result.get('seam_count', 0)
if validation_result.get('issues_found', False):
```

#### 2.2 Fix Config Parameter Names
**File**: `expandor/core/config.py`
**Changes**: Add `force_strategy` as alias for `strategy_override`:
```python
@property
def force_strategy(self):
    return self.strategy_override
```

#### 2.3 Fix Strategy Override
**File**: `expandor/core/strategy_selector.py`
**Changes**: Instantiate strategy from override name:
```python
if hasattr(config, 'strategy_override') and config.strategy_override:
    strategy_map = {
        'direct_upscale': DirectUpscaleStrategy,
        'progressive_outpaint': ProgressiveOutpaintStrategy,
        'swpo': SWPOStrategy,
        'cpu_offload': CPUOffloadStrategy,
        'tiled_expansion': TiledExpansionStrategy,
        'hybrid_adaptive': HybridAdaptiveStrategy
    }
    
    if config.strategy_override in strategy_map:
        strategy_class = strategy_map[config.strategy_override]
        return strategy_class(self.config, self.metrics, self.logger)
```

### Stage 3: Strategy-Specific Fixes

#### 3.1 Fix DirectUpscaleStrategy
**File**: `expandor/strategies/direct_upscale.py`
**Changes**: Make Real-ESRGAN optional with PIL fallback:
```python
def _run_realesrgan(...):
    if not self.realesrgan_path:
        # Fallback to PIL
        return self._run_pil_upscale(input_path, scale)
    # ... existing code
```

#### 3.2 Fix Progressive Strategy
**File**: `expandor/strategies/progressive_outpaint.py`
**Changes**: Handle img2img pipeline option:
```python
def execute(self, config, context=None):
    # Check for img2img override
    use_img2img = context and context.get('config_overrides', {}).get('use_img2img', False)
    if use_img2img and config.img2img_pipeline:
        # Use img2img instead of inpaint
```

#### 3.3 Fix Hybrid Adaptive
**File**: `expandor/strategies/experimental/hybrid_adaptive.py`
**Changes**: Remove unsupported config overrides

### Stage 4: Test Fixes

#### 4.1 Update Test Assertions
**Files**: All test files
**Changes**:
- Use class names for strategy assertions
- Use `strategy_override` instead of `force_strategy`
- Fix expected keys in result validation

#### 4.2 Fix Boundary Test Data
**Files**: Test files creating boundaries
**Changes**: Pass dicts instead of mock BoundaryInfo objects

### Stage 5: Validation & Safety

#### 5.1 Add Null Checks
**All image operations**: Add checks before using images:
```python
if image_path is None:
    raise ValueError("Image path cannot be None")
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")
```

#### 5.2 Add Type Validation
**All public methods**: Add runtime type checking for critical parameters

## Implementation Order

1. **Day 1**: Core Infrastructure (Stages 1.1-1.3)
   - Fix exception classes
   - Fix enum comparison
   - Standardize boundary structure

2. **Day 2**: Interface Standardization (Stage 2)
   - Fix all key mismatches
   - Fix config parameters
   - Fix strategy override

3. **Day 3**: Strategy Fixes (Stage 3)
   - Fix each strategy's specific issues
   - Add fallbacks and safety checks

4. **Day 4**: Test Updates (Stage 4)
   - Update all test assertions
   - Fix test data structures

5. **Day 5**: Validation & Testing (Stage 5)
   - Add comprehensive validation
   - Run full test suite
   - Fix any remaining issues

## Success Criteria

- All 47 integration tests passing
- All unit tests passing
- No type errors or exceptions
- Consistent interfaces throughout
- Clear error messages for all failure modes

## Risk Mitigation

1. **Make changes incrementally** - Test after each fix
2. **Keep backward compatibility** where possible
3. **Document all interface changes**
4. **Add type hints** to prevent future issues
5. **Create integration tests** for each fix

## Estimated Effort

- Total fixes needed: ~50 code changes
- Estimated time: 20-30 hours
- Complexity: Medium-High (due to interdependencies)

## Notes

- Many issues stem from inconsistent interfaces
- Consider adding a validation layer between components
- Type checking (mypy) would catch many of these issues
- Need better integration tests that test actual component interaction