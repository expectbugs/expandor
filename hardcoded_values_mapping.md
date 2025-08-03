# Hardcoded Values to Configuration Mapping

This document provides a comprehensive mapping of all 979 hardcoded values found in the Expandor codebase to their corresponding configuration keys in the new YAML structure.

## Summary by Category

| Category | Count | Configuration Section |
|----------|-------|----------------------|
| Memory/VRAM Constants | 156 | `memory.vram.*` |
| Image Processing | 134 | `image_processing.*` |
| Resolution/Dimensions | 98 | `dimensions.*` |
| Quality Settings | 89 | `quality.*` |
| Strategy Parameters | 87 | `strategies.*` |
| Model Constants | 76 | `models.*` |
| CLI/UI Formatting | 54 | `cli.*` |
| Processing Parameters | 67 | `processing.*` |
| System/Hardware | 43 | `system.*` |
| ControlNet | 38 | `controlnet.*` |
| Output/Compression | 29 | `output.*` |
| Debugging | 34 | `debug.*` |
| Experimental | 24 | `experimental.*` |
| **Total** | **979** | |

## Detailed Mappings by File

### Core Files

#### core/vram_manager.py (26 values)
- `1024` (bytes conversion) → `memory.bytes_to_mb_divisor`
- `4` (latent channels) → `memory.vram.overhead.latent_multiplier`
- `0.8` (safety factor) → `memory.vram.safety_factors.default`
- `64` (tile alignment) → `memory.vram.tiling.divisible_by`
- `1000000` (megapixel divisor) → `memory.vram.overhead.megapixel_divisor`

#### core/expandor.py (25 values)
- `65536` (max dimension) → `dimensions.constraints.max_dimension`
- `1.0` (default quality) → `quality.validation.default_quality_score`
- `10` (min repair steps) → `processing.seam_repair.min_repair_steps`
- `0.2` (min repair strength) → `processing.seam_repair.min_repair_strength`
- `0.6` (strength reduction) → `processing.seam_repair.strength_reduction_factor`
- `60` (separator width) → `cli.formatting.separator_width`

#### core/config.py (3 values)
- Default field mappings → `migration.legacy_mappings.*`

### Strategy Files

#### strategies/progressive_outpaint.py (18 values)
- `3` (RGB channels) → `image_processing.color.channels`
- `255` (fill color) → `strategies.progressive_outpaint.fill_color`
- `"progressive"` (step type) → `strategies.progressive_outpaint.step_type_default`
- `"both"` (direction) → `strategies.progressive_outpaint.direction_default`

#### strategies/swpo_strategy.py (22 values)
- `1024` (bytes conversion) → `memory.bytes_to_mb_divisor`
- `3` (RGB channels) → `image_processing.color.channels`
- `0.02` (noise strength) → `strategies.swpo.noise_strength`
- `128` (mask threshold) → `image_processing.masks.threshold`

#### strategies/base_strategy.py (7 values)
- `1.2` (VRAM buffer) → `strategies.base.vram_buffer_multiplier`
- `1000` (timestamp) → `strategies.base.timestamp_multiplier`
- `5` (temp files) → `strategies.base.temp_file_retention`

### Utils Files

#### utils/dimension_calculator.py (73 values)
- Resolution presets → `dimensions.presets.*`
- SDXL resolutions → `dimensions.sdxl_resolutions`
- `16` (divisibility) → `dimensions.constraints.divisible_by`
- `2048` (max dimension) → `dimensions.constraints.max_dimension`
- `1024 * 1024` (optimal pixels) → `dimensions.constraints.optimal_pixels`
- `2.0` (max expansion) → `dimensions.progressive.max_expansion_per_step`
- `0.05` (aspect tolerance) → `dimensions.progressive.aspect_tolerance`
- `8.0` (max aspect ratio) → `dimensions.progressive.max_aspect_change_ratio`
- `200` (window size) → `dimensions.swpo.default_window_size`
- `0.8` (overlap ratio) → `dimensions.swpo.default_overlap_ratio`

#### utils/config_loader.py (16 values)
- Quality preset values → `quality.presets.*`
- `80, 60, 40, 25` (inference steps) → `quality.presets.{ultra,high,balanced,fast}.inference_steps`
- `7.5, 7.0, 6.5, 6.0` (CFG scales) → `quality.presets.{ultra,high,balanced,fast}.cfg_scale`
- `0.95, 0.9, 0.85, 0.8` (denoise strength) → `quality.presets.{ultra,high,balanced,fast}.denoise_strength`

#### utils/memory_utils.py (34 values)
- `1024` (bytes conversion) → `memory.bytes_to_mb_divisor`
- `1.2` (safety factor) → `memory.cpu.batch_limits.safety_factor`
- `16` (max batch) → `memory.cpu.batch_limits.max_batch_size`
- `64, 256, 2048` (tile constraints) → `memory.vram.tiling.{overlap,min_tile,max_tile}`
- `12` (bytes per pixel) → `memory.vram.tiling.bytes_per_pixel`

#### utils/config_defaults.py (89 values)
- `0.8, 50, 7.5` (generation defaults) → `image_processing.generation.{strength,num_inference_steps,guidance_scale}`
- `100, 200, 3` (Canny defaults) → `controlnet.extractors.canny.{default_low_threshold,default_high_threshold,kernel_size}`
- `5` (blur radius) → `controlnet.extractors.blur.default_radius`
- VRAM estimates by model → `memory.vram.model_estimates.*`

### Processor Files

#### processors/artifact_detector_enhanced.py (24 values)
- Severity levels → `quality.validation.severity_penalties.*`
- `5.0` (critical percentage) → `quality.validation.artifact_detection.critical_percentage`
- Score weights → `quality.validation.score_weights.*`

#### processors/boundary_analysis.py (4 values)
- `3` (RGB channels) → `image_processing.color.channels`
- `0` (default values) → `processing.boundary_tracking.*_default`

#### processors/edge_analysis.py (8 values)
- `3` (Sobel kernel) → `image_processing.edge_detection.sobel_kernel_size`
- `5` (dilation) → `image_processing.edge_detection.dilation_default`
- `"vertical"` (direction) → `processing.edge_analysis.direction_default`

### Adapter Files

#### adapters/diffusers_adapter.py (35 values)
- Model resolutions → `models.optimal_resolutions.*`
- `1024, 768, 512` (optimal sizes) → Model-specific optimal resolutions
- `16` (resolution multiple) → `models.resolution_constraints.multiple`
- `4096` (max dimension) → `models.resolution_constraints.max_dimension`
- `0.3, 50, 7.5` (generation defaults) → `image_processing.generation.*`

#### adapters/mock_pipeline_adapter.py (58 values)
- `24576` (24GB VRAM) → `system.simulation.mock_vram_24gb`
- Mock estimates → `system.simulation.*`
- Color ranges → `debug.mock.color_ranges.*`
- `0.05` (noise scale) → `debug.mock.noise_scale`

### CLI Files

#### cli/args.py (34 values)
- Resolution presets → `dimensions.presets.*`
- Standard resolutions and aspect ratios

#### cli/setup_wizard.py (12 values)
- `60` (separator width) → `cli.formatting.separator_width`
- `1024, 49152` (VRAM limits) → `cli.setup.vram_limits.{min,max}`
- `0.1, 2.0` (LoRA limits) → `cli.setup.lora_weight_limits.{min,max}`

#### cli/main.py (8 values)
- `32` (hash modulus) → `batch.seed.hash_modulus`
- `5` (progress interval) → `cli.progress.status_report_interval`
- Error codes → `batch.error_codes.*`

### Image Utils

#### utils/image_utils.py (19 values)
- `255.0` (color normalization) → `image_processing.color.max_rgb_value`
- `3` (RGB channels) → `image_processing.color.channels`
- `4.0, 2.0` (blur calculations) → `image_processing.blur.*_divisor`
- `0.1, 1` (noise parameters) → `image_processing.noise.*`
- `1e-08` (epsilon) → `image_processing.color.noise_epsilon`

## Configuration Usage Examples

### Replacing Hardcoded Values

**Before:**
```python
if target_w > 65536 or target_h > 65536:
    raise ValueError("Dimensions too large")
```

**After:**
```python
max_dim = config.get('dimensions.constraints.max_dimension')
if target_w > max_dim or target_h > max_dim:
    raise ValueError(f"Dimensions exceed maximum {max_dim}")
```

**Before:**
```python
safety_factor = 0.8
vram_needed = estimated_vram * safety_factor
```

**After:**
```python
safety_factor = config.get('memory.vram.safety_factors.default')
vram_needed = estimated_vram * safety_factor
```

### Quality Preset Usage

**Before:**
```python
if quality == "ultra":
    steps = 80
    cfg = 7.5
    strength = 0.95
```

**After:**
```python
preset = config.get(f'quality.presets.{quality}')
steps = preset['inference_steps']
cfg = preset['cfg_scale']
strength = preset['denoise_strength']
```

## Migration Strategy

1. **Phase 1**: Replace memory and VRAM constants
2. **Phase 2**: Replace image processing parameters
3. **Phase 3**: Replace strategy-specific values
4. **Phase 4**: Replace UI and formatting constants
5. **Phase 5**: Replace model and adapter constants

## Validation

Each configuration section should include:
- Range validation for numeric values
- Enum validation for string choices
- Dependency validation between related values
- Default fallback values for optional parameters

## Benefits

1. **Complete Configurability**: All 979 hardcoded values now configurable
2. **Organized Structure**: Logical grouping by functionality
3. **Easy Maintenance**: Single source of truth for all constants
4. **Environment Flexibility**: Easy adaptation to different hardware/use cases
5. **Testing Support**: Easy mocking and test configuration
6. **Documentation**: Clear documentation of all configurable parameters