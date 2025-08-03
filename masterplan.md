# Configuration System Refactoring Master Plan

## Executive Summary

This masterplan outlines a comprehensive refactoring of the Expandor configuration system to achieve TRUE "COMPLETE CONFIGURABILITY" with "NO HARDCODED VALUES" while implementing the FAIL LOUD philosophy throughout. The refactoring will be executed in 7 phases, each building upon the previous one.

## Core Principles

1. **NO HARDCODED VALUES** - Every configurable value MUST come from configuration files
2. **FAIL LOUD** - Any missing or invalid configuration MUST cause immediate, clear failure
3. **SINGLE SOURCE OF TRUTH** - Each configuration value has ONE authoritative location
4. **COMPLETE CONFIGURABILITY** - ALL values that could change must be configurable
5. **ELEGANCE OVER SIMPLICITY** - Sophisticated, complete solutions for complex problems

## Current Issues Addressed

### Critical Issues
1. **Hardcoded Fallbacks Everywhere** - All strategies use `value or HARDCODED` patterns
2. **Multiple Config Loading Patterns** - No unified access method
3. **Duplicate Values** - Same parameters in multiple YAML files
4. **Missing FAIL LOUD** - Silent fallbacks throughout codebase
5. **User Config Not Integrated** - ~/.config/expandor/ not implemented
6. **Path Config Missing** - Null paths with no defaults
7. **No Schema Validation** - Config files can be invalid
8. **Inconsistent Access** - Each component loads config differently

## Phase 1: Core Configuration Infrastructure (2-3 days)

### 1.1 Create ConfigurationManager (New Central Component)
```python
# expandor/core/configuration_manager.py
class ConfigurationManager:
    """Singleton configuration manager - the ONLY way to access config"""
    
    _instance = None
    _config_cache = {}
    _user_overrides = {}
    _env_overrides = {}
    
    def __init__(self):
        # Load and validate ALL configuration on startup
        # Cache everything for performance
        # Apply hierarchy: base → user → env → runtime
    
    def get_value(self, key: str, context: Optional[Dict] = None) -> Any:
        """Get config value - FAILS LOUD if not found"""
    
    def get_strategy_config(self, strategy_name: str) -> Dict:
        """Get complete config for a strategy"""
    
    def get_processor_config(self, processor_name: str) -> Dict:
        """Get complete config for a processor"""
```

### 1.2 Configuration Hierarchy Implementation
```
1. System Defaults (immutable, packaged with code)
2. User Configuration (~/.config/expandor/config.yaml)
3. Environment Variables (EXPANDOR_*)
4. Runtime Overrides (CLI args, API calls)
```

### 1.3 Replace ConfigLoader/ConfigResolver
- ConfigurationManager becomes the ONLY config access point
- Remove all direct YAML loading from strategies/processors
- ConfigLoader becomes internal to ConfigurationManager

## Phase 2: Schema Validation System (1-2 days)

### 2.1 Create JSON Schema for Each YAML File
```yaml
# expandor/config/schemas/base_defaults.schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["expandor_defaults"],
  "properties": {
    "expandor_defaults": {
      "type": "object",
      "required": [
        "quality_preset", "strategy", "denoising_strength",
        "guidance_scale", "num_inference_steps", ...
      ],
      "properties": {
        "denoising_strength": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        ...
      }
    }
  }
}
```

### 2.2 Implement Validation on Load
- Use jsonschema library for validation
- FAIL LOUD on any schema violation
- Provide clear error messages with fix instructions

### 2.3 Add Config Migration System
- Version config files
- Provide automatic migration for breaking changes
- Warn about deprecated fields

## Phase 3: Remove ALL Hardcoded Values (3-4 days)

### 3.1 Strategy Updates
Replace all patterns like:
```python
# BEFORE (WRONG):
self.outpaint_strength = config.denoising_strength if hasattr(
    config, 'denoising_strength') else 0.75

# AFTER (CORRECT):
self.outpaint_strength = self.config_manager.get_value(
    'strategies.progressive_outpaint.outpaint_strength',
    context={'config': config}
)
```

### 3.2 Processor Updates
Replace all `.get()` with defaults:
```python
# BEFORE (WRONG):
self.seam_threshold = float(preset_config.get("seam_threshold", 0.25))

# AFTER (CORRECT):
self.seam_threshold = self.config_manager.get_value(
    'processors.artifact_detector.seam_threshold',
    context={'preset': config.quality_preset}
)
```

### 3.3 Create Comprehensive Defaults File
```yaml
# expandor/config/comprehensive_defaults.yaml
# This file contains EVERY configurable value in the system
# Organized by component for clarity

strategies:
  progressive_outpaint:
    outpaint_strength: 0.75
    base_steps: 50
    guidance_scale: 7.5
    first_step_ratio: 1.4
    middle_step_ratio: 1.25
    final_step_ratio: 1.15
    base_mask_blur: 32
    # ... every single value currently hardcoded
    
  cpu_offload:
    min_tile_size: 512
    default_tile_size: 512
    max_tile_size: 1536
    min_overlap: 64
    default_overlap: 128
    # ... etc

processors:
  artifact_detector:
    seam_threshold: 0.25
    color_threshold: 30
    gradient_threshold: 0.25
    # ... etc
```

## Phase 4: Consolidate Configuration Files (2 days)

### 4.1 New Configuration Structure
```
expandor/config/
├── schemas/                      # JSON schemas for validation
│   ├── master.schema.json
│   └── ...
├── defaults/
│   ├── master_defaults.yaml      # ALL default values
│   └── presets/                  # Quality presets only
│       ├── ultra.yaml
│       ├── high.yaml
│       ├── balanced.yaml
│       └── fast.yaml
├── examples/                     # Example user configs
│   ├── minimal.yaml
│   ├── advanced.yaml
│   └── production.yaml
└── config_structure.md          # Documentation
```

### 4.2 Master Defaults File Structure
```yaml
# master_defaults.yaml - SINGLE SOURCE OF TRUTH
version: "1.0"

# Core defaults
core:
  quality_preset: "custom"
  strategy: "auto"
  # ... all core values

# Strategy parameters (replaces strategy_parameters.yaml)
strategies:
  progressive_outpaint:
    # ... all values
  cpu_offload:
    # ... all values
  # ... etc

# Processing parameters (replaces processing_params.yaml)
processing:
  artifact_detection:
    # ... all values
  boundary_analysis:
    # ... all values
  # ... etc

# VRAM management (replaces vram_*.yaml)
vram:
  thresholds:
    tiled_processing: 4000
    minimum: 2000
  strategies:
    conservative:
      # ... settings
    balanced:
      # ... settings

# Model constraints (replaces model_constraints.yaml)
models:
  sdxl:
    min_resolution: 1024
    # ... etc

# Output quality (replaces output_quality.yaml)
output:
  formats:
    png:
      compression: 0
    # ... etc

# Paths (NEW - fixes null path issue)
paths:
  cache_dir: "~/.cache/expandor"
  output_dir: "./output"
  temp_dir: "/tmp/expandor"
```

### 4.3 Quality Presets Become Overlays
```yaml
# presets/ultra.yaml - ONLY overrides, no duplication
version: "1.0"
extends: "master_defaults"

# Only specify what changes from defaults
core:
  num_inference_steps: 50  # Override from 30
  guidance_scale: 8.0      # Override from 7.5

strategies:
  progressive_outpaint:
    base_strength: 0.70    # Override from 0.75
```

## Phase 5: User Configuration Integration (1-2 days)

### 5.1 User Config Search Path
```python
# ConfigurationManager will search in order:
1. $EXPANDOR_CONFIG_PATH (if set)
2. ~/.config/expandor/config.yaml
3. ./expandor.yaml (current directory)
4. /etc/expandor/config.yaml (system-wide)
```

### 5.2 User Config Structure
```yaml
# ~/.config/expandor/config.yaml
version: "1.0"

# User can override ANY value from master_defaults
core:
  quality_preset: "high"  # User's default preset

paths:
  cache_dir: "/mnt/fast-ssd/expandor-cache"
  output_dir: "~/Pictures/AI"

# User's custom preset
custom_presets:
  my_preset:
    extends: "high"
    core:
      num_inference_steps: 40
```

### 5.3 Environment Variable Support
```bash
# Any config value can be overridden via env
EXPANDOR_CORE_QUALITY_PRESET=ultra
EXPANDOR_PATHS_CACHE_DIR=/tmp/cache
EXPANDOR_STRATEGIES_PROGRESSIVE_OUTPAINT_BASE_STRENGTH=0.8
```

## Phase 6: Path Configuration System (1 day)

### 6.1 Smart Path Resolution
```python
class PathResolver:
    """Resolves and validates all file paths"""
    
    def resolve_path(self, path_config: str, create: bool = True) -> Path:
        """
        Resolve path with smart expansion:
        - ~ → home directory
        - $VAR → environment variable
        - relative → relative to CWD
        Create directory if create=True
        """
```

### 6.2 Default Path Structure
```
~/.config/expandor/          # User config
~/.cache/expandor/           # Cache files
  ├── models/                # Downloaded models
  ├── pipelines/             # Cached pipelines
  └── temp/                  # Temporary files
~/Pictures/expandor/         # Default output (customizable)
```

## Phase 7: Testing & Validation (2-3 days)

### 7.1 Configuration Test Suite
```python
# tests/test_configuration_system.py
class TestConfigurationSystem:
    def test_no_hardcoded_values():
        """Scan entire codebase for hardcoded values"""
        
    def test_fail_loud_on_missing():
        """Verify FAIL LOUD behavior"""
        
    def test_schema_validation():
        """Test all schemas validate correctly"""
        
    def test_user_config_override():
        """Test configuration hierarchy"""
```

### 7.2 Migration Guide
- Document for users how to migrate existing configs
- Provide migration tool if needed
- Clear upgrade instructions

### 7.3 Performance Validation
- Ensure config caching works correctly
- Measure startup time impact
- Optimize if needed

## Implementation Order & Timeline

1. **Week 1**: Phases 1-2 (Core Infrastructure + Schema Validation)
2. **Week 2**: Phase 3 (Remove Hardcoded Values)
3. **Week 3**: Phases 4-5 (Consolidation + User Config)
4. **Week 4**: Phases 6-7 (Paths + Testing)

## Success Criteria

1. **Zero Hardcoded Values** - Automated scan finds none
2. **100% FAIL LOUD** - Missing config always fails explicitly
3. **Single Config Access** - All components use ConfigurationManager
4. **User Config Works** - ~/.config/expandor/ fully integrated
5. **All Tests Pass** - Comprehensive test coverage
6. **Performance Maintained** - No significant slowdown
7. **Documentation Complete** - Full user and developer docs

## Risk Mitigation

1. **Backwards Compatibility**: Version configs, provide migration
2. **Performance Impact**: Aggressive caching in ConfigurationManager
3. **Complex Migration**: Phased approach, extensive testing
4. **User Confusion**: Clear documentation, good defaults

## Version Bump

After successful implementation:
- Version 0.7.0 - Major configuration system overhaul
- Update changelog with breaking changes
- Provide migration guide

This masterplan ensures Expandor achieves TRUE "COMPLETE CONFIGURABILITY" while maintaining the FAIL LOUD philosophy and code quality standards.