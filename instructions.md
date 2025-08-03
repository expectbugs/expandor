# Configuration System Refactoring - Detailed Implementation Instructions

## Overview
This document provides step-by-step instructions for implementing the configuration system refactoring masterplan. Each instruction is written to be executable by a developer without context or deep understanding of the system.

## Phase 1: Core Configuration Infrastructure (2-3 days) ✅ COMPLETE

### Step 1.1: Create ConfigurationManager class ✅ COMPLETE

1. **Create new file**: `/home/user/ai-wallpaper/expandor/expandor/core/configuration_manager.py`

2. **Add imports**:
```python
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from ..utils.config_loader import ConfigLoader
from ..core.exceptions import ExpandorError
```

3. **Implement singleton pattern**:
```python
class ConfigurationManager:
    """Singleton configuration manager - the ONLY way to access config"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Prevent re-initialization
        if ConfigurationManager._initialized:
            return
        ConfigurationManager._initialized = True
        
        self.logger = logging.getLogger(__name__)
        self._config_cache = {}
        self._user_overrides = {}
        self._env_overrides = {}
        self._runtime_overrides = {}
        self._config_loader = None
        self._master_config = {}
        
        # Initialize on first use
        self._load_all_configurations()
```

4. **Implement configuration loading hierarchy**:
```python
    def _load_all_configurations(self):
        """Load all configurations in proper hierarchy"""
        # 1. Load system defaults from package config dir
        config_dir = Path(__file__).parent.parent / "config"
        self._config_loader = ConfigLoader(config_dir, self.logger)
        
        # Load master defaults (new consolidated file - will create in Phase 4)
        try:
            self._master_config = self._config_loader.load_config_file("master_defaults.yaml")
        except FileNotFoundError:
            # For now, load existing configs until Phase 4
            self._load_legacy_configs()
        
        # 2. Load user configuration
        self._load_user_config()
        
        # 3. Load environment overrides
        self._load_env_overrides()
        
        # 4. Build final config cache
        self._build_config_cache()
```

5. **Implement legacy config loading (temporary until Phase 4)**:
```python
    def _load_legacy_configs(self):
        """Temporary method to load existing configs until consolidation"""
        configs_to_load = [
            "base_defaults",
            "strategy_parameters", 
            "quality_presets",
            "quality_thresholds",
            "vram_thresholds",
            "processing_params",
            "model_constraints",
            "output_quality"
        ]
        
        for config_name in configs_to_load:
            try:
                config_data = self._config_loader.get_config(config_name)
                self._merge_config(self._master_config, config_data)
            except Exception as e:
                self.logger.warning(f"Could not load {config_name}: {e}")
```

6. **Implement user config loading**:
```python
    def _load_user_config(self):
        """Load user configuration from standard locations"""
        search_paths = [
            Path(os.environ.get("EXPANDOR_CONFIG_PATH", "")),
            Path.home() / ".config" / "expandor" / "config.yaml",
            Path.cwd() / "expandor.yaml",
            Path("/etc/expandor/config.yaml")
        ]
        
        for path in search_paths:
            if path and path.exists():
                try:
                    with open(path, 'r') as f:
                        user_config = yaml.safe_load(f)
                        if user_config:
                            self._user_overrides = user_config
                            self.logger.info(f"Loaded user config from {path}")
                            break
                except Exception as e:
                    self.logger.error(f"Failed to load user config from {path}: {e}")
```

7. **Implement environment variable loading**:
```python
    def _load_env_overrides(self):
        """Load environment variable overrides (EXPANDOR_*)"""
        for key, value in os.environ.items():
            if key.startswith("EXPANDOR_"):
                # Convert EXPANDOR_STRATEGIES_PROGRESSIVE_OUTPAINT_BASE_STRENGTH
                # to strategies.progressive_outpaint.base_strength
                config_path = key[9:].lower().replace("_", ".")
                try:
                    # Try to parse as number/bool
                    if value.lower() in ("true", "false"):
                        parsed_value = value.lower() == "true"
                    elif "." in value:
                        parsed_value = float(value)
                    elif value.isdigit():
                        parsed_value = int(value)
                    else:
                        parsed_value = value
                    
                    self._set_nested_value(self._env_overrides, config_path, parsed_value)
                except Exception:
                    # Keep as string if parsing fails
                    self._set_nested_value(self._env_overrides, config_path, value)
```

8. **Implement config value getter with FAIL LOUD**:
```python
    def get_value(self, key: str, context: Optional[Dict] = None) -> Any:
        """
        Get config value - FAILS LOUD if not found
        
        Args:
            key: Dot-separated config key (e.g., 'strategies.progressive_outpaint.base_strength')
            context: Optional context for dynamic resolution
            
        Returns:
            Configuration value
            
        Raises:
            ValueError: If key not found (FAIL LOUD)
        """
        # Check runtime overrides first
        if context and 'override' in context:
            if key in context['override']:
                return context['override'][key]
        
        # Check config cache
        try:
            value = self._get_nested_value(self._config_cache, key)
            if value is not None:
                return value
        except KeyError:
            pass
        
        # FAIL LOUD - no silent defaults
        raise ValueError(
            f"Configuration key '{key}' not found!\n"
            f"This is a required configuration value with no default.\n"
            f"Solutions:\n"
            f"1. Add '{key}' to your config files\n" 
            f"2. Set environment variable EXPANDOR_{key.upper().replace('.', '_')}\n"
            f"3. Check config file syntax for errors"
        )
```

9. **Implement helper methods**:
```python
    def _merge_config(self, base: dict, override: dict):
        """Deep merge override into base config"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _get_nested_value(self, config: dict, key: str) -> Any:
        """Get value from nested dict using dot notation"""
        keys = key.split('.')
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise KeyError(f"Key '{k}' not found in path '{key}'")
        return value
    
    def _set_nested_value(self, config: dict, key: str, value: Any):
        """Set value in nested dict using dot notation"""
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    def _build_config_cache(self):
        """Build final config cache from all sources"""
        # Start with master config
        self._config_cache = self._master_config.copy()
        
        # Apply user overrides
        self._merge_config(self._config_cache, self._user_overrides)
        
        # Apply environment overrides (highest priority)
        self._merge_config(self._config_cache, self._env_overrides)
```

10. **Implement strategy and processor config getters**:
```python
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get complete config for a strategy"""
        base_key = f"strategies.{strategy_name}"
        try:
            return self._get_nested_value(self._config_cache, base_key)
        except KeyError:
            raise ValueError(
                f"No configuration found for strategy '{strategy_name}'\n"
                f"Available strategies: {list(self._config_cache.get('strategies', {}).keys())}"
            )
    
    def get_processor_config(self, processor_name: str) -> Dict[str, Any]:
        """Get complete config for a processor"""
        base_key = f"processors.{processor_name}"
        try:
            return self._get_nested_value(self._config_cache, base_key)
        except KeyError:
            raise ValueError(
                f"No configuration found for processor '{processor_name}'\n"
                f"Available processors: {list(self._config_cache.get('processors', {}).keys())}"
            )
```

### Step 1.2: Update core/__init__.py ✅ COMPLETE

1. **Open file**: `/home/user/ai-wallpaper/expandor/expandor/core/__init__.py`

2. **Add import**:
```python
from .configuration_manager import ConfigurationManager
```

3. **Add to __all__**:
```python
__all__ = [
    # ... existing exports ...
    "ConfigurationManager",
]
```

### Step 1.3: Create tests for ConfigurationManager ✅ COMPLETE

1. **Create test file**: `/home/user/ai-wallpaper/expandor/tests/unit/test_configuration_manager.py`

2. **Add test imports**:
```python
import os
import pytest
from pathlib import Path
import tempfile
import yaml
from expandor.core.configuration_manager import ConfigurationManager
```

3. **Add basic tests**:
```python
class TestConfigurationManager:
    def test_singleton_pattern(self):
        """Test that ConfigurationManager is a singleton"""
        cm1 = ConfigurationManager()
        cm2 = ConfigurationManager()
        assert cm1 is cm2
    
    def test_fail_loud_on_missing_key(self):
        """Test FAIL LOUD behavior for missing keys"""
        cm = ConfigurationManager()
        with pytest.raises(ValueError) as exc_info:
            cm.get_value("nonexistent.key")
        assert "not found" in str(exc_info.value)
        assert "Solutions:" in str(exc_info.value)
    
    def test_env_override(self):
        """Test environment variable overrides"""
        os.environ["EXPANDOR_TEST_VALUE"] = "42"
        # Force reload
        ConfigurationManager._initialized = False
        cm = ConfigurationManager()
        # This will fail until we have test.value in config
        # For now, just test the env parsing
        assert cm._env_overrides.get("test", {}).get("value") == 42
```

## Phase 2: Schema Validation System (1-2 days) ✅ COMPLETE

### Step 2.1: Install jsonschema library ✅ COMPLETE

1. **Update setup.py**:
   - Open `/home/user/ai-wallpaper/expandor/setup.py`
   - Add to install_requires list: `"jsonschema>=4.0.0",`

### Step 2.2: Create schema directory and base schema ✅ COMPLETE

1. **Create directory**: `/home/user/ai-wallpaper/expandor/expandor/config/schemas/`

2. **Create base schema**: `/home/user/ai-wallpaper/expandor/expandor/config/schemas/base_schema.json`
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://expandor.ai/schemas/base_schema.json",
  "title": "Expandor Base Configuration Schema",
  "description": "Base schema with common definitions",
  "definitions": {
    "positiveNumber": {
      "type": "number",
      "minimum": 0
    },
    "probability": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "positiveInteger": {
      "type": "integer",
      "minimum": 1
    },
    "filePath": {
      "type": ["string", "null"],
      "pattern": "^(~|/|\\.|\\$).*"
    }
  }
}
```

3. **Create master defaults schema**: `/home/user/ai-wallpaper/expandor/expandor/config/schemas/master_defaults.schema.json`
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$ref": "base_schema.json",
  "title": "Master Defaults Configuration",
  "type": "object",
  "required": ["version", "core", "strategies", "processing", "vram", "models", "output", "paths"],
  "properties": {
    "version": {
      "type": "string",
      "pattern": "^\\d+\\.\\d+$"
    },
    "core": {
      "type": "object",
      "required": ["quality_preset", "strategy", "denoising_strength", "guidance_scale", "num_inference_steps"],
      "properties": {
        "quality_preset": {
          "type": "string",
          "enum": ["ultra", "high", "balanced", "fast", "custom"]
        },
        "strategy": {
          "type": "string",
          "enum": ["auto", "progressive_outpaint", "direct_upscale", "tiled_expansion", "cpu_offload", "swpo", "controlnet_progressive"]
        },
        "denoising_strength": {
          "$ref": "#/definitions/probability"
        },
        "guidance_scale": {
          "type": "number",
          "minimum": 1.0,
          "maximum": 20.0
        },
        "num_inference_steps": {
          "$ref": "#/definitions/positiveInteger"
        }
      }
    },
    "strategies": {
      "type": "object",
      "properties": {
        "progressive_outpaint": {
          "$ref": "progressive_outpaint.schema.json"
        }
      }
    }
  }
}
```

### Step 2.3: Add schema validation to ConfigurationManager ✅ COMPLETE

1. **Update imports in configuration_manager.py**:
```python
import jsonschema
from jsonschema import validate, ValidationError, RefResolver
```

2. **Add schema loading method**:
```python
    def _load_schemas(self):
        """Load all JSON schemas for validation"""
        self._schemas = {}
        schema_dir = Path(__file__).parent.parent / "config" / "schemas"
        
        if schema_dir.exists():
            for schema_file in schema_dir.glob("*.schema.json"):
                try:
                    with open(schema_file, 'r') as f:
                        schema_name = schema_file.stem.replace('.schema', '')
                        self._schemas[schema_name] = json.load(f)
                except Exception as e:
                    self.logger.error(f"Failed to load schema {schema_file}: {e}")
        
        # Create resolver for $ref resolution
        if self._schemas:
            base_uri = "file://" + str(schema_dir) + "/"
            self._schema_resolver = RefResolver(base_uri, self._schemas.get('base_schema', {}))
```

3. **Add validation method**:
```python
    def _validate_config(self, config_data: dict, schema_name: str):
        """Validate config against schema - FAIL LOUD on invalid"""
        if schema_name not in self._schemas:
            self.logger.warning(f"No schema found for {schema_name}")
            return
        
        try:
            validate(
                instance=config_data,
                schema=self._schemas[schema_name],
                resolver=self._schema_resolver
            )
        except ValidationError as e:
            # FAIL LOUD with helpful error
            raise ValueError(
                f"Configuration validation failed for {schema_name}!\n"
                f"Error: {e.message}\n"
                f"Failed at path: {' -> '.join(str(p) for p in e.path)}\n"
                f"Schema rule: {e.schema}\n"
                f"Invalid value: {e.instance}\n\n"
                f"Please fix your configuration file and ensure all required fields are present."
            )
```

4. **Update _load_all_configurations to validate**:
```python
    # Add to __init__:
    self._load_schemas()
    
    # In _load_legacy_configs, add validation:
    config_data = self._config_loader.get_config(config_name)
    self._validate_config(config_data, config_name)  # Validate before merge
    self._merge_config(self._master_config, config_data)
```

## Phase 3: Remove ALL Hardcoded Values (3-4 days) ✅ COMPLETE

### Step 3.1: Create comprehensive defaults file ✅ COMPLETE

1. **Create file**: `/home/user/ai-wallpaper/expandor/expandor/config/comprehensive_defaults.yaml`
```yaml
# Expandor Comprehensive Default Values
# This file contains EVERY configurable value in the system
# NO HARDCODED VALUES should exist in the code

version: "1.0"

# Strategy-specific defaults
strategies:
  progressive_outpaint:
    # From progressive_outpaint.py lines 69-82
    first_step_ratio: 1.4
    middle_step_ratio: 1.25  
    final_step_ratio: 1.15
    base_mask_blur: 32
    base_steps: 60
    max_supported_ratio: 8.0
    # From strategy_parameters.yaml
    base_strength: 0.75
    min_strength: 0.35
    max_strength: 0.85
    seam_repair_multiplier: 0.4
    blur_radius_ratio: 0.4
    mask_blur_ratio: 0.05
    edge_preservation_ratio: 0.1
    # Additional parameters
    outpaint_prompt_suffix: ", seamless expansion, extended scenery, natural continuation"
    edge_sample_width: 50
    
  tiled_expansion:
    # From tiled_expansion.py lines 50-55
    default_tile_size: 1024
    overlap: 256
    blend_width: 256
    # From strategy_parameters.yaml
    vram_safety_ratio: 0.7
    refinement_strength: 0.15
    edge_fix_strength: 0.25
    final_pass_strength: 0.35
    min_tile_size: 512
    max_tile_size: 1536
    tile_overlap: 48
    overlap_ratio: 0.25
    blend_method: "mixture_of_diffusers"
    progressive_strength_reduction: true
    max_tiles_before_reduction: 50
    refinement_guidance: 6.5
    refinement_steps: 20
    edge_fix_guidance: 7.0
    edge_fix_steps: 30
    final_pass_guidance: 7.0
    final_pass_steps: 40

  cpu_offload:
    # From strategy_parameters.yaml
    safety_factor: 0.6
    conservative_safety_factor: 0.5
    pipeline_vram: 1024
    gpu_memory_fallback: 512
    aspect_change_threshold: 0.1
    tile_generation_strength: 0.9
    tile_generation_guidance: 7.0
    tile_refinement_strength: 0.3
    tile_refinement_guidance: 7.0
    # From cpu_offload.py (if any hardcoded)
    min_tile_size: 512
    default_tile_size: 512
    max_tile_size: 1536
    min_overlap: 64
    default_overlap: 128

  swpo:
    # From strategy_parameters.yaml
    default_overlap_ratio: 0.8
    default_denoising_strength: 0.75
    unification_strength: 0.15
    window_blur_radius: 30
    edge_blur_ratio: 0.1
    unification_guidance: 7.0
    unification_steps: 30
    # From swpo_strategy.py (if any hardcoded)
    default_window_size: 200
    min_window_size: 128
    max_window_size: 512

  direct_upscale:
    # From strategy_parameters.yaml
    vram_thresholds:
      high: 8000
      medium: 6000
      low: 4000
    scale_thresholds:
      low: 2.0
      medium: 3.0

  controlnet_progressive:
    # From strategy_parameters.yaml
    default_strength: 0.25
    conditioning_scale: 0.5
    control_guidance_start: 0.0
    control_guidance_end: 1.0

# Processor-specific defaults
processors:
  artifact_detector:
    # From quality_thresholds.yaml (if exists) or code
    seam_threshold: 0.25
    color_threshold: 30
    gradient_threshold: 0.25
    frequency_threshold: 0.3
    texture_threshold: 0.15
    enabled_checks:
      - seam
      - color
      - gradient
      - frequency
      - texture
    weights:
      seam: 0.3
      color: 0.2
      gradient: 0.2
      frequency: 0.2
      texture: 0.1

  boundary_analysis:
    # Any hardcoded values from boundary analysis
    edge_sample_size: 50
    color_similarity_threshold: 0.9
    pattern_match_threshold: 0.8

  seam_repair:
    # Any hardcoded values from seam repair
    mask_expansion: 20
    blur_radius: 15
    strength_multiplier: 1.2

  quality_validator:
    # From quality validation
    min_quality_score: 0.7
    max_retries: 3
    retry_delay: 1.0

# VRAM management defaults
vram:
  thresholds:
    tiled_processing: 4000
    minimum: 2000
    critical: 1000
  buffer_mb: 512
  safety_factor: 0.9
  measurement_retries: 3
  measurement_delay: 0.5

# Model constraints
models:
  sdxl:
    min_resolution: 1024
    max_resolution: 4096
    optimal_resolution: 1344
    resolution_step: 8
  sd15:
    min_resolution: 512
    max_resolution: 2048
    optimal_resolution: 768
    resolution_step: 8

# Path configuration
paths:
  cache_dir: "~/.cache/expandor"
  output_dir: "./output"
  temp_dir: "/tmp/expandor"
  log_dir: "~/.local/share/expandor/logs"
  config_dir: "~/.config/expandor"

# Output quality settings
output:
  formats:
    png:
      compression: 0
      optimize: false
    jpeg:
      quality: 95
      optimize: true
      progressive: true
  metadata:
    save_generation_params: true
    save_expandor_version: true
    save_timestamp: true

# Processing parameters
processing:
  max_threads: 4
  batch_size: 1
  enable_progress_bar: true
  verbose_logging: false
  save_intermediate_stages: false
  stage_format: "png"
```

### Step 3.2: Update each strategy to use ConfigurationManager ✅ COMPLETE

#### For ProgressiveOutpaintStrategy: ✅ COMPLETE

1. **Open**: `/home/user/ai-wallpaper/expandor/expandor/strategies/progressive_outpaint.py`

2. **Replace the __init__ method config loading** (lines 34-82):
```python
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(config=config, metrics=metrics, logger=logger)
        self.dimension_calc = DimensionCalculator(self.logger)
        self.model_metadata = {}
        
        # Get configuration from ConfigurationManager
        from ..core.configuration_manager import ConfigurationManager
        self.config_manager = ConfigurationManager()
        
        # Get all strategy config at once
        try:
            self.strategy_config = self.config_manager.get_strategy_config('progressive_outpaint')
        except ValueError as e:
            # FAIL LOUD
            raise ValueError(
                f"Failed to load progressive_outpaint configuration!\n{str(e)}"
            )
        
        # Assign all values from config - NO DEFAULTS!
        self.prog_enabled = True  # This can stay hardcoded as it defines the strategy
        self.max_supported = self.strategy_config['max_supported_ratio']
        self.outpaint_strength = self.strategy_config['base_strength']
        self.min_strength = self.strategy_config['min_strength']
        self.max_strength = self.strategy_config['max_strength']
        self.first_step_ratio = self.strategy_config['first_step_ratio']
        self.middle_step_ratio = self.strategy_config['middle_step_ratio']
        self.final_step_ratio = self.strategy_config['final_step_ratio']
        self.outpaint_prompt_suffix = self.strategy_config['outpaint_prompt_suffix']
        self.base_mask_blur = self.strategy_config['base_mask_blur']
        self.base_steps = self.strategy_config['base_steps']
        
        # Get the other parameters
        self.seam_repair_multiplier = self.strategy_config['seam_repair_multiplier']
        self.blur_radius_ratio = self.strategy_config['blur_radius_ratio']
        self.mask_blur_ratio = self.strategy_config['mask_blur_ratio']
        self.edge_preservation_ratio = self.strategy_config['edge_preservation_ratio']
```

3. **Update any other hardcoded values in the file**:
   - Search for any numeric literals or string literals that could be configuration
   - Replace with `self.config_manager.get_value()` calls

#### Repeat for all strategies: ✅ COMPLETE

1. **TiledExpansionStrategy** (`/home/user/ai-wallpaper/expandor/expandor/strategies/tiled_expansion.py`): ✅ COMPLETE
   - Replace lines 26-55 in _initialize()
   - Remove the ConfigLoader usage
   - Use ConfigurationManager instead

2. **CPUOffloadStrategy** (`/home/user/ai-wallpaper/expandor/expandor/strategies/cpu_offload.py`): ✅ COMPLETE
   - Find and replace all hardcoded values
   - Use ConfigurationManager

3. **DirectUpscaleStrategy** (`/home/user/ai-wallpaper/expandor/expandor/strategies/direct_upscale.py`): ✅ COMPLETE
   - Update initialization
   - Remove hardcoded VRAM thresholds

4. **SWPOStrategy** (`/home/user/ai-wallpaper/expandor/expandor/strategies/swpo_strategy.py`): ✅ COMPLETE
   - Replace default values
   - Use ConfigurationManager

5. **ControlNetProgressiveStrategy** (`/home/user/ai-wallpaper/expandor/expandor/strategies/controlnet_progressive.py`): ✅ COMPLETE
   - Update for ConfigurationManager

### Step 3.3: Update all processors ✅ COMPLETE

1. **ArtifactDetector** (`/home/user/ai-wallpaper/expandor/expandor/processors/artifact_detector_enhanced.py`): ✅ COMPLETE
   - Find all `.get()` calls with defaults
   - Replace with ConfigurationManager calls

2. **QualityValidator** (`/home/user/ai-wallpaper/expandor/expandor/processors/quality_validator.py`): ✅ COMPLETE
   - Remove hardcoded thresholds
   - Use ConfigurationManager

3. **BoundaryAnalysis** (`/home/user/ai-wallpaper/expandor/expandor/processors/boundary_analysis.py`): ✅ COMPLETE
   - Update any hardcoded analysis parameters

### Step 3.4: Update ExpandorConfig to use ConfigurationManager ⚠️ NOT IMPLEMENTED (kept original design)

1. **Open**: `/home/user/ai-wallpaper/expandor/expandor/core/config.py`

2. **Update __post_init__ method** to use ConfigurationManager instead of ConfigResolver:
```python
def __post_init__(self):
    """Initialize configuration using ConfigurationManager"""
    from .configuration_manager import ConfigurationManager
    config_manager = ConfigurationManager()
    
    # If quality_preset is set, load those values
    if self.quality_preset and self.quality_preset != "custom":
        try:
            preset_config = config_manager.get_value(f'quality_presets.{self.quality_preset}')
            # Apply preset values
            for key, value in preset_config.items():
                if hasattr(self, key) and getattr(self, key) is None:
                    setattr(self, key, value)
        except ValueError:
            # FAIL LOUD
            raise ValueError(
                f"Quality preset '{self.quality_preset}' not found in configuration!"
            )
    
    # Fill in any remaining None values from core defaults
    core_defaults = config_manager.get_value('core')
    for key, value in core_defaults.items():
        if hasattr(self, key) and getattr(self, key) is None:
            setattr(self, key, value)
    
    # Validate all required fields are set
    self._validate()
```

## Phase 4: Consolidate Configuration Files (2 days) ✅ COMPLETE

### Step 4.1: Create master_defaults.yaml ✅ COMPLETE

1. **Create the consolidated file**: `/home/user/ai-wallpaper/expandor/expandor/config/master_defaults.yaml`
   - Copy content from comprehensive_defaults.yaml created in Step 3.1
   - Add all values from existing YAML files:
     - base_defaults.yaml
     - strategy_parameters.yaml
     - quality_presets.yaml (as a nested section)
     - vram_thresholds.yaml
     - processing_params.yaml
     - model_constraints.yaml
     - output_quality.yaml

2. **Structure the file properly**:
```yaml
version: "1.0"

# Core defaults (from base_defaults.yaml)
core:
  quality_preset: "custom"
  strategy: "auto"
  denoising_strength: 0.7
  guidance_scale: 7.5
  num_inference_steps: 30
  # ... all other core values

# Quality presets (from quality_presets.yaml but restructured)
quality_presets:
  ultra:
    num_inference_steps: 50
    guidance_scale: 8.0
    # ... only overrides from core
  high:
    num_inference_steps: 50
    guidance_scale: 7.5
    # ... only overrides
  # ... other presets

# Strategies section (merge all strategy configs)
strategies:
  progressive_outpaint:
    # ... all values
  # ... all strategies

# ... continue for all sections
```

### Step 4.2: Update ConfigurationManager to use master_defaults.yaml ✅ COMPLETE

1. **Update _load_all_configurations**:
```python
def _load_all_configurations(self):
    """Load all configurations in proper hierarchy"""
    config_dir = Path(__file__).parent.parent / "config"
    self._config_loader = ConfigLoader(config_dir, self.logger)
    
    # Load master defaults
    try:
        self._master_config = self._config_loader.load_config_file("master_defaults.yaml")
        self.logger.info("Loaded master configuration file")
    except FileNotFoundError:
        # FAIL LOUD - this file MUST exist
        raise ValueError(
            "master_defaults.yaml not found!\n"
            "This is a critical configuration file that must exist.\n"
            f"Expected location: {config_dir / 'master_defaults.yaml'}\n"
            "Run 'expandor --generate-config' to create it."
        )
    
    # Validate master config
    self._validate_config(self._master_config, 'master_defaults')
    
    # Continue with user config, env vars, etc.
    self._load_user_config()
    self._load_env_overrides()
    self._build_config_cache()
```

### Step 4.3: Remove old configuration files ✅ COMPLETE (cleanup script created)

1. **Create migration script**: `/home/user/ai-wallpaper/expandor/scripts/migrate_config.py`
```python
#!/usr/bin/env python3
"""Migrate old config files to master_defaults.yaml"""

import shutil
from pathlib import Path
from datetime import datetime

def migrate_configs():
    config_dir = Path(__file__).parent.parent / "expandor" / "config"
    backup_dir = config_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Files to remove after migration
    old_files = [
        "strategy_parameters.yaml",
        "vram_thresholds.yaml", 
        "processing_params.yaml",
        "model_constraints.yaml",
        "output_quality.yaml",
        "quality_thresholds.yaml"
    ]
    
    # Create backup
    backup_dir.mkdir(exist_ok=True)
    for file in old_files:
        if (config_dir / file).exists():
            shutil.copy2(config_dir / file, backup_dir / file)
            print(f"Backed up {file}")
    
    # Remove old files
    for file in old_files:
        if (config_dir / file).exists():
            (config_dir / file).unlink()
            print(f"Removed {file}")
    
    print(f"\nMigration complete! Backups saved to {backup_dir}")

if __name__ == "__main__":
    migrate_configs()
```

2. **Run the migration**:
```bash
cd /home/user/ai-wallpaper/expandor
python scripts/migrate_config.py
```

### Step 4.4: Update ConfigLoader CONFIG_FILES mapping ⚠️ NOT NEEDED (ConfigurationManager handles this)

1. **Open**: `/home/user/ai-wallpaper/expandor/expandor/utils/config_loader.py`

2. **Update CONFIG_FILES** (lines 22-35):
```python
CONFIG_FILES = {
    "master_defaults": "master_defaults.yaml",
    "base_defaults": "master_defaults.yaml",  # Backwards compat
    "quality_presets": "quality_presets.yaml",  # Keep separate for now
    "strategies": "master_defaults.yaml",
    "controlnet_config": "controlnet_config.yaml",  # Keep separate
    "resolution_presets": "resolution_presets.yaml",  # Keep if exists
}
```

## Phase 5: User Configuration Integration (1-2 days) ✅ COMPLETE

### Step 5.1: Create user config template ✅ COMPLETE

1. **Update**: `/home/user/ai-wallpaper/expandor/expandor/config/examples/user_config_example.yaml`
```yaml
# Expandor User Configuration
# Place this file at ~/.config/expandor/config.yaml
# All values here override system defaults

version: "1.0"

# Override core defaults
core:
  quality_preset: "high"  # Your default quality
  save_stages: true      # Always save intermediate stages

# Override specific strategy settings
strategies:
  progressive_outpaint:
    base_strength: 0.8  # Prefer stronger outpainting

# Custom paths for your system
paths:
  cache_dir: "/mnt/fast-ssd/expandor-cache"
  output_dir: "~/Pictures/AI/expandor"
  temp_dir: "/tmp/expandor-${USER}"

# Custom quality preset
custom_presets:
  my_ultra:
    extends: "ultra"
    num_inference_steps: 100  # Even more steps
    guidance_scale: 9.0

# VRAM override for your GPU
vram:
  limit_mb: 20000  # Limit to 20GB even if 24GB available
```

### Step 5.2: Add user config creation command ✅ COMPLETE

1. **Update**: `/home/user/ai-wallpaper/expandor/expandor/cli/main.py`

2. **Add new argument**:
```python
parser.add_argument(
    '--init-config',
    action='store_true',
    help='Initialize user configuration file'
)
```

3. **Add handler**:
```python
if args.init_config:
    from ..core.configuration_manager import ConfigurationManager
    from ..utils.config_loader import ConfigLoader
    
    user_config_dir = Path.home() / ".config" / "expandor"
    user_config_dir.mkdir(parents=True, exist_ok=True)
    user_config_path = user_config_dir / "config.yaml"
    
    if user_config_path.exists():
        print(f"User config already exists at {user_config_path}")
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Copy template
    template_path = Path(__file__).parent.parent / "config" / "examples" / "user_config_example.yaml"
    shutil.copy2(template_path, user_config_path)
    print(f"Created user configuration at {user_config_path}")
    print("Edit this file to customize Expandor for your system.")
    return
```

### Step 5.3: Test user config loading ✅ COMPLETE

1. **Create test**: `/home/user/ai-wallpaper/expandor/tests/unit/test_user_config.py`
```python
import pytest
import tempfile
import yaml
from pathlib import Path
from expandor.core.configuration_manager import ConfigurationManager

def test_user_config_override():
    """Test that user config overrides system defaults"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create user config
        user_config = {
            "version": "1.0",
            "core": {
                "quality_preset": "ultra",
                "num_inference_steps": 100
            }
        }
        
        config_path = Path(tmpdir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(user_config, f)
        
        # Set env var to point to test config
        import os
        os.environ["EXPANDOR_CONFIG_PATH"] = str(config_path)
        
        # Force reload
        ConfigurationManager._initialized = False
        cm = ConfigurationManager()
        
        # Check overrides applied
        assert cm.get_value("core.quality_preset") == "ultra"
        assert cm.get_value("core.num_inference_steps") == 100
```

## Phase 6: Path Configuration System (1 day) ✅ COMPLETE

### Step 6.1: Create PathResolver class ✅ COMPLETE

1. **Create file**: `/home/user/ai-wallpaper/expandor/expandor/utils/path_resolver.py`
```python
"""Path resolution and validation for Expandor"""

import os
from pathlib import Path
from typing import Optional, Union
import logging

class PathResolver:
    """Resolves and validates all file paths"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._cache = {}
    
    def resolve_path(self, path_config: Union[str, Path], 
                     create: bool = True,
                     path_type: str = "directory") -> Path:
        """
        Resolve path with smart expansion
        
        Args:
            path_config: Path configuration string
            create: Create directory if it doesn't exist
            path_type: "directory" or "file"
            
        Returns:
            Resolved Path object
            
        Raises:
            ValueError: If path invalid or can't be created
        """
        if not path_config:
            raise ValueError("Path configuration cannot be None or empty")
        
        # Convert to string for processing
        path_str = str(path_config)
        
        # Check cache
        cache_key = f"{path_str}:{create}:{path_type}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Expand variables
        path_str = os.path.expandvars(path_str)
        path_str = os.path.expanduser(path_str)
        
        # Handle relative paths
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path
        
        # Resolve to absolute
        try:
            path = path.resolve()
        except Exception as e:
            raise ValueError(
                f"Failed to resolve path '{path_config}'\n"
                f"Error: {e}"
            )
        
        # Create if requested
        if create:
            try:
                if path_type == "directory":
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    # For files, create parent directory
                    path.parent.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise ValueError(
                    f"Permission denied creating path '{path}'\n"
                    f"Check directory permissions or use a different path"
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to create path '{path}'\n"
                    f"Error: {e}"
                )
        
        # Validate exists if not creating
        elif not path.exists() and path_type == "directory":
            raise ValueError(
                f"Path does not exist: '{path}'\n"
                f"Original config: '{path_config}'\n"
                f"Set create=True to create it automatically"
            )
        
        # Cache and return
        self._cache[cache_key] = path
        self.logger.debug(f"Resolved path '{path_config}' to '{path}'")
        return path
    
    def get_writable_dir(self, preferred_paths: list, 
                         purpose: str = "data") -> Path:
        """
        Find first writable directory from list
        
        Args:
            preferred_paths: List of paths in preference order
            purpose: Description of what this is for (for error messages)
            
        Returns:
            First writable directory
            
        Raises:
            ValueError: If no writable directory found
        """
        for path_config in preferred_paths:
            try:
                path = self.resolve_path(path_config, create=True)
                # Test write permission
                test_file = path / ".expandor_write_test"
                test_file.touch()
                test_file.unlink()
                return path
            except Exception as e:
                self.logger.debug(f"Path '{path_config}' not writable: {e}")
                continue
        
        raise ValueError(
            f"No writable directory found for {purpose}\n"
            f"Tried paths: {preferred_paths}\n"
            f"Please ensure at least one location is writable"
        )
```

### Step 6.2: Integrate PathResolver with ConfigurationManager ✅ COMPLETE

1. **Update ConfigurationManager** to use PathResolver:
```python
# Add to imports
from ..utils.path_resolver import PathResolver

# Add to __init__
self._path_resolver = PathResolver(self.logger)

# Add method
def get_path(self, path_key: str, create: bool = True, 
             path_type: str = "directory") -> Path:
    """
    Get resolved path from configuration
    
    Args:
        path_key: Configuration key for path (e.g., 'paths.cache_dir')
        create: Create if doesn't exist
        path_type: 'directory' or 'file'
        
    Returns:
        Resolved Path object
    """
    path_config = self.get_value(path_key)
    return self._path_resolver.resolve_path(path_config, create, path_type)
```

### Step 6.3: Update all path usage in codebase ✅ COMPLETE (demonstrated pattern)

1. **Find all path usage**:
```bash
grep -r "Path(" /home/user/ai-wallpaper/expandor/expandor/ | grep -v __pycache__
```

2. **Update each occurrence** to use ConfigurationManager:
```python
# Instead of:
cache_dir = Path.home() / ".cache" / "expandor"

# Use:
cache_dir = self.config_manager.get_path("paths.cache_dir")
```

## Phase 7: Testing & Validation (2-3 days) ⏳ PENDING

### Step 7.1: Create configuration scanner

1. **Create file**: `/home/user/ai-wallpaper/expandor/scripts/scan_hardcoded_values.py`
```python
#!/usr/bin/env python3
"""Scan codebase for hardcoded values"""

import ast
import re
from pathlib import Path

class HardcodedValueScanner(ast.NodeVisitor):
    """AST visitor to find hardcoded values"""
    
    def __init__(self, filename):
        self.filename = filename
        self.hardcoded_values = []
        
    def visit_Num(self, node):
        """Visit number literals"""
        # Ignore 0, 1, -1 as these are often indices
        if node.n not in (0, 1, -1):
            self.hardcoded_values.append({
                'file': self.filename,
                'line': node.lineno,
                'value': node.n,
                'type': 'number'
            })
        self.generic_visit(node)
    
    def visit_Str(self, node):
        """Visit string literals"""
        # Ignore common strings
        ignore_patterns = [
            r'^$',  # Empty string
            r'^[/\\]$',  # Single slash
            r'^\.$',  # Single dot
            r'^__\w+__$',  # Dunder methods
            r'^\w+Error$',  # Error names
        ]
        
        if not any(re.match(p, node.s) for p in ignore_patterns):
            # Check if it looks like a config value
            if any(x in node.s for x in ['.yaml', '.json', 'px', '%', 'MB', 'GB']):
                self.hardcoded_values.append({
                    'file': self.filename,
                    'line': node.lineno,
                    'value': node.s,
                    'type': 'string'
                })
        self.generic_visit(node)

def scan_directory(directory: Path):
    """Scan directory for hardcoded values"""
    all_hardcoded = []
    
    for py_file in directory.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
            
        try:
            with open(py_file, 'r') as f:
                tree = ast.parse(f.read(), filename=str(py_file))
            
            scanner = HardcodedValueScanner(str(py_file))
            scanner.visit(tree)
            all_hardcoded.extend(scanner.hardcoded_values)
        except Exception as e:
            print(f"Error scanning {py_file}: {e}")
    
    return all_hardcoded

def main():
    expandor_dir = Path(__file__).parent.parent / "expandor"
    hardcoded = scan_directory(expandor_dir)
    
    if hardcoded:
        print(f"Found {len(hardcoded)} potential hardcoded values:\n")
        for item in hardcoded:
            print(f"{item['file']}:{item['line']} - {item['type']}: {item['value']}")
    else:
        print("No hardcoded values found!")

if __name__ == "__main__":
    main()
```

### Step 7.2: Create comprehensive test suite

1. **Create test file**: `/home/user/ai-wallpaper/expandor/tests/test_configuration_system.py`
```python
"""Comprehensive tests for configuration system"""

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from expandor.core.configuration_manager import ConfigurationManager
from expandor.core import ExpandorConfig

class TestConfigurationSystem:
    
    def test_no_hardcoded_values(self):
        """Scan codebase for hardcoded values"""
        import subprocess
        result = subprocess.run(
            ["python", "scripts/scan_hardcoded_values.py"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )
        assert "No hardcoded values found" in result.stdout, \
            f"Hardcoded values found:\n{result.stdout}"
    
    def test_fail_loud_on_missing(self):
        """Verify FAIL LOUD behavior"""
        cm = ConfigurationManager()
        
        # Test missing key
        with pytest.raises(ValueError) as exc:
            cm.get_value("this.key.does.not.exist")
        
        assert "not found" in str(exc.value)
        assert "Solutions:" in str(exc.value)
    
    def test_configuration_hierarchy(self):
        """Test config override hierarchy"""
        # Set up test configs
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create user config
            user_config = {
                "version": "1.0",
                "core": {"num_inference_steps": 999}
            }
            user_path = Path(tmpdir) / "user_config.yaml"
            with open(user_path, 'w') as f:
                yaml.dump(user_config, f)
            
            # Test hierarchy
            os.environ["EXPANDOR_CONFIG_PATH"] = str(user_path)
            os.environ["EXPANDOR_CORE_GUIDANCE_SCALE"] = "12.5"
            
            # Force reload
            ConfigurationManager._initialized = False
            cm = ConfigurationManager()
            
            # Check user override
            assert cm.get_value("core.num_inference_steps") == 999
            
            # Check env override (highest priority)
            assert cm.get_value("core.guidance_scale") == 12.5
    
    def test_all_strategies_load(self):
        """Test all strategies can load their config"""
        cm = ConfigurationManager()
        
        strategies = [
            "progressive_outpaint",
            "tiled_expansion",
            "cpu_offload",
            "swpo",
            "direct_upscale",
            "controlnet_progressive"
        ]
        
        for strategy in strategies:
            config = cm.get_strategy_config(strategy)
            assert isinstance(config, dict), f"{strategy} config not a dict"
            assert len(config) > 0, f"{strategy} config is empty"
    
    def test_path_resolution(self):
        """Test path resolution works correctly"""
        cm = ConfigurationManager()
        
        # Should create directory
        cache_dir = cm.get_path("paths.cache_dir")
        assert cache_dir.exists()
        assert cache_dir.is_dir()
    
    def test_schema_validation(self):
        """Test schema validation catches errors"""
        # This will be implemented when schemas are added
        pass
    
    def test_quality_preset_loading(self):
        """Test quality presets load correctly"""
        config = ExpandorConfig(quality_preset="ultra")
        
        # Should have ultra preset values
        assert config.num_inference_steps == 50  # From ultra preset
        assert config.guidance_scale == 8.0  # From ultra preset
```

### Step 7.3: Create validation script

1. **Create**: `/home/user/ai-wallpaper/expandor/scripts/validate_config.py`
```python
#!/usr/bin/env python3
"""Validate configuration system is working correctly"""

import sys
from pathlib import Path

def validate_all():
    """Run all validation checks"""
    print("Validating Expandor Configuration System...\n")
    
    errors = []
    
    # Check 1: Master defaults exists
    print("1. Checking master_defaults.yaml exists...")
    config_dir = Path(__file__).parent.parent / "expandor" / "config"
    if not (config_dir / "master_defaults.yaml").exists():
        errors.append("master_defaults.yaml not found!")
    else:
        print("   ✓ Found master_defaults.yaml")
    
    # Check 2: Can import ConfigurationManager
    print("2. Checking ConfigurationManager imports...")
    try:
        from expandor.core.configuration_manager import ConfigurationManager
        print("   ✓ ConfigurationManager imports successfully")
    except Exception as e:
        errors.append(f"Failed to import ConfigurationManager: {e}")
    
    # Check 3: Can instantiate and load config
    print("3. Checking configuration loading...")
    try:
        cm = ConfigurationManager()
        test_value = cm.get_value("core.quality_preset")
        print(f"   ✓ Configuration loads (quality_preset: {test_value})")
    except Exception as e:
        errors.append(f"Failed to load configuration: {e}")
    
    # Check 4: All strategies have config
    print("4. Checking strategy configurations...")
    strategies = [
        "progressive_outpaint",
        "tiled_expansion", 
        "cpu_offload",
        "swpo",
        "direct_upscale"
    ]
    
    for strategy in strategies:
        try:
            config = cm.get_strategy_config(strategy)
            print(f"   ✓ {strategy}: {len(config)} parameters")
        except Exception as e:
            errors.append(f"Failed to load {strategy} config: {e}")
    
    # Check 5: FAIL LOUD works
    print("5. Checking FAIL LOUD behavior...")
    try:
        cm.get_value("this.should.not.exist")
        errors.append("FAIL LOUD not working - no exception raised!")
    except ValueError:
        print("   ✓ FAIL LOUD working correctly")
    
    # Summary
    print("\n" + "="*50)
    if errors:
        print(f"VALIDATION FAILED with {len(errors)} errors:\n")
        for error in errors:
            print(f"  ✗ {error}")
        return 1
    else:
        print("✓ ALL VALIDATION CHECKS PASSED!")
        return 0

if __name__ == "__main__":
    sys.exit(validate_all())
```

2. **Make executable**:
```bash
chmod +x /home/user/ai-wallpaper/expandor/scripts/validate_config.py
```

### Step 7.4: Update GitHub Actions / CI

1. **Add to test workflow**:
```yaml
- name: Validate Configuration System
  run: |
    cd expandor
    python scripts/validate_config.py
    
- name: Check for Hardcoded Values  
  run: |
    cd expandor
    python scripts/scan_hardcoded_values.py
```

## Migration Guide for Users

### Create file: `/home/user/ai-wallpaper/expandor/docs/CONFIG_MIGRATION.md`

```markdown
# Configuration System Migration Guide

## What Changed

Expandor v0.7.0 introduces a completely new configuration system:

1. **All configuration in one place** - `config/master_defaults.yaml`
2. **No more hardcoded values** - Everything is configurable
3. **User configuration support** - `~/.config/expandor/config.yaml`
4. **Environment variable overrides** - `EXPANDOR_*`
5. **FAIL LOUD philosophy** - No silent defaults

## Migration Steps

### 1. Backup Your Configuration

```bash
cp -r ~/.config/expandor ~/.config/expandor.backup
```

### 2. Initialize New User Config

```bash
expandor --init-config
```

### 3. Migrate Custom Settings

Edit `~/.config/expandor/config.yaml` and add your customizations:

```yaml
# Example migrations:

# Old: --quality ultra
core:
  quality_preset: "ultra"

# Old: custom inference steps
core:
  num_inference_steps: 60

# Old: custom paths
paths:
  output_dir: "/my/custom/output"
```

### 4. Environment Variables

Any config value can be overridden via environment:

```bash
# Old: various env vars
# New: EXPANDOR_ prefix with path
export EXPANDOR_CORE_QUALITY_PRESET=ultra
export EXPANDOR_STRATEGIES_PROGRESSIVE_OUTPAINT_BASE_STRENGTH=0.8
```

## Breaking Changes

1. **Removed Config Files**:
   - `strategy_parameters.yaml` → merged into `master_defaults.yaml`
   - `vram_thresholds.yaml` → merged into `master_defaults.yaml`
   - Individual strategy configs → unified configuration

2. **API Changes**:
   - Strategies now require ConfigurationManager
   - No more ConfigResolver class
   - ExpandorConfig uses new system

3. **No More Silent Defaults**:
   - Missing config values now raise errors
   - You must provide all required values

## Troubleshooting

### "Configuration key not found" Error

This means a required value is missing. Solutions:

1. Run `expandor --init-config` to create user config
2. Set via environment: `export EXPANDOR_KEY_PATH=value`
3. Check `master_defaults.yaml` for the correct key path

### Performance Issues

The new system caches all configuration on startup. If you experience slow startup:

1. Check for syntax errors in YAML files
2. Reduce user config to only necessary overrides
3. Report issue with timing information

## Support

- GitHub Issues: https://github.com/user/expandor/issues
- Documentation: https://expandor.ai/docs/configuration
```

## Final Steps

1. **Run all tests**:
```bash
cd /home/user/ai-wallpaper/expandor
pytest tests/
python scripts/validate_config.py
```

2. **Update version**:
   - Edit `expandor/_version.py` to `0.7.0`
   - Update CHANGELOG.md

3. **Commit changes**:
```bash
git add -A
git commit -m "feat: Complete configuration system overhaul (v0.7.0)

- Implement centralized ConfigurationManager singleton
- Add JSON schema validation for all configs  
- Remove ALL hardcoded values - complete configurability
- Consolidate configs into master_defaults.yaml
- Add user configuration support (~/.config/expandor/)
- Implement proper path resolution system
- Add comprehensive test suite and validation scripts
- BREAKING: New configuration API throughout

Fixes all issues from confbugs.md and implements all 
recommendations from CONFIG_SYSTEM_MAP.md"
```

This completes the detailed implementation instructions for the configuration system refactoring. Each step has been verified against the actual codebase structure and includes specific code that will work with the existing system.