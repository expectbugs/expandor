# Expandor Configuration System - Issues and Inconsistencies

## Critical Issues Found

### 1. Hardcoded Fallback Values (Violates "NO HARDCODED VALUES" Principle)

#### A. Strategy Implementations
- **ProgressiveOutpaintStrategy** (progressive_outpaint.py:136-141):
  ```python
  self.outpaint_strength = config.denoising_strength if hasattr(
      config, 'denoising_strength') else 0.75
  self.base_steps = config.num_inference_steps if hasattr(
      config, 'num_inference_steps') else 50
  self.guidance_scale = config.guidance_scale if hasattr(
      config, 'guidance_scale') else 7.5
  ```
  ISSUE: Using hardcoded fallback values (0.75, 50, 7.5) instead of failing loud

- **SWPOStrategy** (swpo_strategy.py:201):
  ```python
  window_size = config.window_size or 200
  ```
  ISSUE: Hardcoded fallback value (200) for window_size

- **DirectUpscaleStrategy** (direct_upscale.py:258-270):
  ```python
  return self.tile_config.get("low_vram", 512)  # HARDCODED: 512
  return self.tile_config.get("unlimited", 2048)  # HARDCODED: 2048
  return self.tile_config.get("high_vram", 1024)  # HARDCODED: 1024
  ```
  ISSUE: Hardcoded fallback values for tile sizes throughout

- **CPUOffloadStrategy** (cpu_offload.py:75-79):
  ```python
  self.min_tile_size = tiled_params.get('min_tile_size', 512)  # HARDCODED
  self.default_tile_size = 512  # HARDCODED
  self.max_tile_size = tiled_params.get('max_tile_size', 1536)  # HARDCODED
  self.min_overlap = 64  # HARDCODED
  self.default_overlap = 128  # HARDCODED
  ```
  ISSUE: Multiple hardcoded tile configuration values

- **TiledExpansionStrategy** (tiled_expansion.py:50-52):
  ```python
  self.default_tile_size = 1024  # HARDCODED
  self.overlap = 256  # HARDCODED
  self.blend_width = 256  # HARDCODED
  ```
  ISSUE: Hardcoded processing parameters

- **ControlNetProgressiveStrategy** (controlnet_progressive.py:27,259):
  ```python
  VRAM_PRIORITY = 4  # HARDCODED
  base_strength = kwargs.get("strength", cn_params.get('default_strength', 0.95))  # HARDCODED: 0.95
  ```
  ISSUE: Hardcoded priority and strength fallback

#### B. Enhanced Artifact Detector (artifact_detector_enhanced.py:79-93):
  ```python
  self.seam_threshold = float(preset_config.get("seam_threshold", 0.25))
  self.color_threshold = float(preset_config.get("color_threshold", 30))
  self.gradient_threshold = float(preset_config.get("gradient_threshold", 0.25))
  # ... more fallbacks
  ```
  ISSUE: Using `.get()` with hardcoded fallback values

#### C. Strategy Selector (strategy_selector.py:239-241):
  ```python
  vram_thresholds = self.vram_strategies.get(
      "thresholds", {"tiled_processing": 4000, "minimum": 2000}
  )
  ```
  ISSUE: Hardcoded fallback dictionary

### 2. Configuration Resolution Hierarchy Issues

#### A. Inconsistent Parameter Mapping
The ConfigResolver maps quality preset values to ExpandorConfig fields through a hardcoded mapping dictionary (config_resolver.py:71-110). This mapping:
- Is not configurable
- May miss new fields added to presets
- Creates a tight coupling between preset structure and config fields

#### B. Missing Configuration Validation
- No schema validation for YAML files
- ConfigLoader._validate_schema() only checks for dict type and None values
- No validation that all required fields exist in config files

### 3. Duplicate Configuration Values

Some values appear in multiple config files with potential conflicts:
- `denoising_strength` appears in:
  - base_defaults.yaml (0.7)
  - quality_presets.yaml (varies by preset)
  - strategy_parameters.yaml (various strategy-specific values)
- `overlap_ratio` appears in:
  - base_defaults.yaml (0.7)
  - strategy_parameters.yaml (0.8 for swpo)
  - processing_params.yaml (0.8 in dimension_calculation)

### 4. Inconsistent Loading Patterns

Different components load configuration differently:
- Strategies load from strategy_parameters.yaml directly
- ExpandorConfig uses ConfigResolver with hierarchy
- Some processors load quality_thresholds.yaml directly
- No centralized configuration access pattern

### 5. Missing FAIL LOUD Implementation

Despite the principle, many places handle missing config gracefully:
- ConfigLoader.load_yaml() returns empty dict on failure
- Strategies use `hasattr()` checks with fallbacks
- Processors use `.get()` with defaults

### 6. Configuration File Dependencies

The system has implicit dependencies between config files that aren't documented:
- quality_presets.yaml references values that should match processing_params.yaml
- strategy_parameters.yaml has values that overlap with base_defaults.yaml
- No clear ownership of which file "owns" which configuration value

### 7. User Configuration Override Issues

The user configuration system (~/.config/expandor/config.yaml) is referenced in documentation but:
- Not loaded by ConfigResolver
- No clear mechanism for user overrides
- UserConfig class exists but isn't integrated with main config flow

### 8. Path Configuration Missing

Despite requirements for configurable paths:
- cache_dir, output_dir, temp_dir are null in base_defaults.yaml
- No default path configuration mechanism
- CLI doesn't set these paths consistently

### 9. Strategy-Specific Configuration Loading

Each strategy loads its own parameters from strategy_parameters.yaml:
- Creates multiple ConfigLoader instances
- No caching of loaded configuration
- Potential for inconsistent config if files change during execution

### 10. Quality Preset Application

The quality preset system has issues:
- "custom" preset is defined but treated specially in code
- Preset application only happens if quality_preset != "custom"
- No way to partially override preset values

## Summary

The configuration system has evolved through multiple iterations, leaving:
1. **Hardcoded fallback values throughout the codebase** - Found in ALL strategy implementations
2. **Multiple configuration loading patterns** - ConfigResolver, direct ConfigLoader, and inline loading
3. **Inconsistent validation and error handling** - Mix of FAIL LOUD and silent fallbacks
4. **Duplicate and potentially conflicting values** - Same parameters in multiple YAML files
5. **Missing user configuration integration** - ~/.config/expandor/ not implemented
6. **Widespread violations of the "NO HARDCODED VALUES" principle** - Every major component has hardcoded values

### Hardcoded Values by Category:
- **Tile Sizes**: 512, 768, 1024, 1536, 2048 (pixels)
- **VRAM Thresholds**: 2000, 4000, 6000, 8000 (MB)
- **Processing Steps**: 20, 30, 40, 50, 60, 80, 100
- **Strength Values**: 0.15, 0.25, 0.3, 0.35, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
- **Guidance Scales**: 5.0, 6.0, 6.5, 7.0, 7.5, 8.0
- **Overlap Values**: 64, 128, 256 (pixels)
- **Blur Values**: 32, 100, 150, 200, 300 (pixels)
- **Aspect Ratios**: 1.15, 1.25, 1.4, 1.5, 2.0, 4.0, 8.0

These issues make the configuration system fragile and difficult to maintain, contradicting the stated goal of "COMPLETE CONFIGURABILITY". The system requires a comprehensive refactor to:
1. Remove ALL hardcoded values
2. Implement true FAIL LOUD philosophy
3. Create a single source of truth for each configuration value
4. Provide a unified configuration access pattern