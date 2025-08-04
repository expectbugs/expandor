# Expandor Configuration System - Comprehensive Map

## Overview

The Expandor configuration system is designed to provide "COMPLETE CONFIGURABILITY" with "NO HARDCODED VALUES". However, the current implementation has multiple layers and paths that don't fully achieve these goals.

## Configuration Files Structure

```
expandor/config/
├── base_defaults.yaml          # Base default values for ExpandorConfig
├── quality_presets.yaml        # Quality level definitions (ultra/high/balanced/fast)
├── quality_thresholds.yaml     # Detection thresholds by quality preset
├── strategy_parameters.yaml    # Strategy-specific parameters
├── vram_strategies.yaml       # VRAM management strategies
├── vram_thresholds.yaml       # VRAM thresholds for strategy selection
├── model_constraints.yaml     # Model-specific constraints
├── processing_params.yaml     # All processing parameters
├── output_quality.yaml        # Output format quality settings
├── resolution_presets.yaml    # Common resolution presets
├── controlnet_config.yaml     # ControlNet configuration
├── pipeline_config.py         # Python-based pipeline configuration
├── lora_manager.py           # LoRA management code
├── user_config.py            # User configuration handler
└── config_resolver.py        # Configuration resolution logic
```

## Configuration Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER INPUT                                 │
│  ExpandorConfig(source_image, target_resolution, quality_preset)    │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ExpandorConfig.__post_init__()                   │
│  1. Track which fields were explicitly set by user                  │
│  2. Convert string paths to Path objects                            │
│  3. Initialize ConfigLoader and ConfigResolver                      │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ConfigResolver.resolve_config()               │
│  Applies configuration hierarchy:                                    │
│  1. Load base_defaults.yaml → Apply to None fields                 │
│  2. If quality_preset != "custom" → Apply preset overrides         │
│  3. Validate all required fields are set                           │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Quality Preset Application                        │
│  Maps preset values to ExpandorConfig fields via hardcoded mapping: │
│  - generation → num_inference_steps, guidance_scale, etc.          │
│  - expansion → denoising_strength, mask_blur_ratio, etc.           │
│  - validation → enable_artifacts_check, thresholds, etc.           │
│  - performance → enable_xformers, batch_size, etc.                 │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ExpandorConfig._validate()                      │
│  Validates configuration completeness and correctness                │
│  FAILS LOUD if critical fields are missing                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Strategy Configuration Loading

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Strategy Initialization                         │
│  Each strategy loads its own parameters independently:              │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Strategy.__init__()                                │
│  1. Create new ConfigLoader instance                                │
│  2. Load strategy_parameters.yaml                                   │
│  3. Extract strategy-specific section                              │
│  4. Validate required parameters exist (FAIL LOUD)                 │
│  ⚠️ ISSUE: Some strategies have hardcoded fallbacks                │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Strategy.execute()                                │
│  ⚠️ ISSUE: Many strategies use:                                     │
│  - config.value if hasattr(config, 'value') else HARDCODED         │
│  - config.value or HARDCODED                                       │
│  This violates "NO HARDCODED VALUES" principle                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Processor Configuration Loading

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Processor Initialization                             │
│  (e.g., EnhancedArtifactDetector, EdgeAnalyzer)                   │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Load quality_thresholds.yaml                            │
│  1. Create ConfigLoader with config directory                       │
│  2. Load quality preset thresholds                                  │
│  3. ⚠️ ISSUE: Uses .get() with hardcoded defaults                  │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Load processing_params.yaml                             │
│  Additional parameters for processing operations                     │
│  ⚠️ ISSUE: Also uses .get() with hardcoded defaults                │
└─────────────────────────────────────────────────────────────────────┘
```

## Configuration Value Sources and Ownership

### 1. Base Defaults (base_defaults.yaml)
- **Purpose**: Provide base values when no preset selected
- **Owner**: ExpandorConfig fields
- **Loaded by**: ConfigResolver._load_base_defaults()

### 2. Quality Presets (quality_presets.yaml)
- **Purpose**: Override base defaults for quality levels
- **Owner**: Quality-specific overrides
- **Loaded by**: ConfigResolver._apply_quality_preset()
- **Presets**: custom, ultra, high, balanced, fast

### 3. Strategy Parameters (strategy_parameters.yaml)
- **Purpose**: Strategy-specific configuration values
- **Owner**: Individual strategy classes
- **Loaded by**: Each strategy in __init__()
- **Strategies**: progressive_outpaint, controlnet_progressive, cpu_offload, swpo, tiled_expansion, hybrid_adaptive, direct_upscale

### 4. Processing Parameters (processing_params.yaml)
- **Purpose**: All image processing parameters
- **Owner**: Various processors and utilities
- **Loaded by**: Individual processors as needed
- **Sections**: artifact_detection, boundary_analysis, edge_analysis, memory_params, etc.

### 5. VRAM Configuration
- **vram_strategies.yaml**: VRAM management profiles
- **vram_thresholds.yaml**: Thresholds for strategy selection
- **Owner**: VRAMManager and StrategySelector
- **Loaded by**: StrategySelector for decision making

### 6. Quality Thresholds (quality_thresholds.yaml)
- **Purpose**: Detection thresholds by quality preset
- **Owner**: Artifact and quality detection systems
- **Loaded by**: EnhancedArtifactDetector and similar

## Configuration Priority Order

1. **User-specified values** (passed to ExpandorConfig)
2. **Quality preset overrides** (if quality_preset != "custom")
3. **Base defaults** (from base_defaults.yaml)
4. **Hardcoded fallbacks** (⚠️ SHOULD NOT EXIST but currently do)

## Key Issues Highlighted

### 1. Multiple Loading Patterns
- ConfigResolver for ExpandorConfig
- Direct ConfigLoader usage in strategies
- Direct YAML loading in processors
- No unified configuration access

### 2. Hardcoded Values
- Fallbacks in strategy execute() methods
- Default values in processor .get() calls
- Hardcoded mappings in ConfigResolver

### 3. Missing Integration
- User configuration (~/.config/expandor/) not integrated
- No environment variable support implemented
- No command-line override mechanism

### 4. Duplicate Values
- Same parameters in multiple files
- No clear ownership model
- Potential for conflicts

### 5. Validation Gaps
- No schema validation
- Incomplete FAIL LOUD implementation
- Missing configuration completeness checks

## Ideal Configuration Flow (Not Currently Implemented)

```
1. Load system defaults (immutable base)
2. Load user configuration (~/.config/expandor/config.yaml)
3. Apply environment variables (EXPANDOR_*)
4. Apply command-line arguments
5. Apply quality preset (if specified)
6. Validate complete configuration
7. FAIL LOUD on any missing required values
8. Provide unified configuration object to all components
```

## Recommendations

1. **Centralize Configuration Access**: Create a single configuration manager that all components use
2. **Remove ALL Hardcoded Fallbacks**: Implement true FAIL LOUD philosophy
3. **Implement Schema Validation**: Use jsonschema or similar for YAML validation
4. **Clear Value Ownership**: Each config value should have one authoritative source
5. **User Configuration Integration**: Implement the documented user config hierarchy
6. **Configuration Caching**: Load once, use everywhere
7. **Remove Duplicate Values**: Consolidate configuration files
8. **Document Dependencies**: Make config file relationships explicit