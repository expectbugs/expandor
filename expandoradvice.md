# Expandor Implementation Best Practices

Now that the implementation plans have been fixed and are internally consistent, here are best practices for successful implementation.

## What's Already Fixed in the Plans

The following issues have been resolved in expandor1.md through expandor3.3.md:
- ✅ All imports use absolute paths with `expandor.` prefix
- ✅ Consistent use of `BaseExpansionStrategy` class
- ✅ Lazy loading prevents circular dependencies
- ✅ Single VRAMError definition with proper signature
- ✅ Memory management with gpu_memory_manager context
- ✅ Dict-based boundary tracking (no dataclasses)
- ✅ Missing utility files added to Phase 3.2
- ✅ Correct mock_pipelines.py filename

## Implementation Order

**CRITICAL**: Follow the phases in order. Each phase builds on the previous one:
1. **Phase 1** (expandor1.md): Core extraction and repository setup
2. **Phase 2** (expandor2.*.md): Main implementation and basic strategies  
3. **Phase 3** (expandor3.*.md): Advanced strategies and quality systems
4. **Phase 4** (not yet created): Production readiness

Within each phase, follow the step numbers exactly. Files often import from other files created earlier in the same phase.

### Key Files Created in Each Phase

**Phase 1**: Core components (VRAMManager, DimensionCalculator, base strategies)
**Phase 2**: Main Expandor class, StrategySelector, basic strategies  
**Phase 3.1**: SWPO, CPU offload, adaptive strategies, TiledProcessor
**Phase 3.2**: ArtifactDetector, SmartRefiner, QualityValidator, missing utils
**Phase 3.3**: Integration tests and mock pipelines

## 1. **Create a Build Checklist**
Before starting, create a checklist from the expandor*.md files marking off each step as you complete it. This prevents skipping critical steps or creating files out of order.

## 2. **Implement Comprehensive Logging**
Add debug logging at every critical decision point:
- Strategy selection reasoning
- VRAM calculations and decisions  
- Boundary tracking operations
- Quality validation results
- Pipeline execution parameters

This will be invaluable when debugging issues.

## 3. **Build Incrementally with Validation**
After implementing each major component:
1. Write a simple test script to verify it works
2. Check that all imports resolve correctly
3. Run any provided unit tests
4. Only then proceed to the next component

## 4. **Create Visual Debug Tools**
Consider creating simple visualization utilities:
- Draw boundaries on images to verify tracking
- Visualize mask generation for expansions
- Show VRAM usage over time
- Display seam detection results

## 5. **Mock Pipeline Behavior**
The mock pipelines should simulate realistic behavior:
- Add controlled artifacts at boundaries to test detection
- Simulate VRAM constraints to test fallback strategies
- Include deterministic randomness for reproducible testing

## 6. **Error Message Templates**
Create consistent error message formats that include:
- What operation failed
- Why it failed  
- Current state when it failed
- Suggestions for resolution
- Relevant configuration values

## 7. **Version Control Strategy**
- Commit after each successfully implemented component
- Use descriptive commit messages referencing the phase/step
- Tag major milestones (end of each phase)
- Never commit broken code

## 8. **Testing Pyramid**
1. **Unit tests**: Each component in isolation
2. **Integration tests**: Components working together
3. **System tests**: Full expansion scenarios
4. **Stress tests**: Extreme cases (8K, 32:9 ratios)
5. **Failure tests**: Verify "fail loud" behavior

## 9. **Documentation as You Go**
- Add docstrings to every public method
- Document non-obvious algorithmic choices
- Keep a "decisions log" for why certain approaches were chosen
- Update the README with actual usage examples that work

## 10. **Performance Profiling Points**
Even though quality is prioritized over speed, add timing markers to identify bottlenecks:
- Pipeline execution time
- VRAM transfer operations
- Image manipulation operations
- Artifact detection passes

## 11. **Defensive Programming**
- Validate all inputs at component boundaries
- Never assume pipeline outputs are correct size/format
- Check tensor dtypes match expectations
- Verify image color spaces (RGB vs RGBA)

## 12. **Recovery Strategies**
Implement cleanup for partial failures:
- Delete incomplete output files
- Free VRAM allocations
- Reset pipeline states
- Clear temporary directories

## 13. **Configuration Validation**
Create a configuration validator that checks:
- Target resolution is achievable
- Aspect ratio change is within limits
- Required pipelines are available
- VRAM estimates don't exceed hardware

## 14. **Boundary Case Matrix**
Test these specific challenging cases:
- Square → Ultra-wide (1:1 → 32:9)
- Portrait → Landscape (9:16 → 16:9)
- Tiny → Large (512x512 → 8192x8192)
- Non-standard ratios (1.85:1 → 2.39:1)

## 15. **Integration Patterns**
When connecting components:
- Use dependency injection for pipelines
- Pass loggers explicitly, don't use globals
- Make strategies stateless where possible
- Use context managers for resource cleanup

## 16. **Critical Implementation Points**

1. **Boundary tracking is EVERYTHING** - The Dict-based boundary format must be consistent
2. **VRAM calculation must be pessimistic** - Always use safety margins
3. **Test with mock pipelines first** - mock_pipelines.py provides realistic behavior
4. **Memory management is crucial** - gpu_memory_manager context prevents leaks
5. **Progressive expansion ratios are tuned** - Don't modify without extensive testing

## 17. **Final Success Validation**

Before considering the project complete, ensure:
- Can expand a 1344x768 image to 5376x768 without visible seams
- Can expand a 1920x1080 image to 3840x2160 with perfect quality
- Handles VRAM exhaustion gracefully with clear messages
- All artifact detection methods trigger on test cases
- Metadata tracking captures complete operation history

## Ready for Implementation

The expandor*.md files now contain a complete, internally consistent implementation plan with all architectural issues resolved. The plans have been carefully updated to:
- Eliminate circular dependencies
- Standardize all interfaces
- Include all missing components
- Follow Python best practices

**Phase 4 is still needed** for production deployment (CLI, API, real pipeline adapters), but Phases 1-3 will give you a fully functional system that can be tested with mock pipelines.

Trust the process, follow the plans methodically, and you'll build an exceptional image expansion system. The key is patience and careful attention to each step.