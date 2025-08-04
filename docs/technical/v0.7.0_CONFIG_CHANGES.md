# Expandor Configuration Migration Guide

## Overview

This guide explains the new configuration system introduced in Expandor v0.7.0, which implements **COMPLETE CONFIGURABILITY** with **NO HARDCODED VALUES**.

## Key Changes in v0.7.0

### 1. Centralized Configuration Management

All configuration is now managed through the `ConfigurationManager` singleton:

```python
from expandor.core.configuration_manager import ConfigurationManager

config_manager = ConfigurationManager()
value = config_manager.get_value("path.to.config.value")
```

### 2. NO Hardcoded Defaults

**Before v0.7.0:**
```python
def generate(self, prompt, width=1024, height=1024, steps=50):
    # Hardcoded defaults
```

**After v0.7.0:**
```python
def generate(self, prompt, width=None, height=None, steps=None):
    if width is None:
        width = config_manager.get_value("adapters.common.default_width")
    # FAIL LOUD if config not found
```

### 3. Master Defaults Configuration

All default values are now in `master_defaults.yaml`:
- Version: 2.0
- Comprehensive coverage of all configurable values
- Organized by component (adapters, strategies, processors, etc.)

## Migration Steps

### For Users

1. **Check Your Config Version**
   ```bash
   grep "version:" ~/.config/expandor/config.yaml
   ```

2. **Automatic Migration**
   - Expandor will automatically migrate configs from v1.x to v2.0
   - A backup is created before migration

3. **Manual Migration** (if needed)
   ```bash
   expandor --migrate-config
   ```

### For Developers

1. **Remove ALL Hardcoded Values**
   - No function parameter defaults (except None)
   - No magic numbers in code
   - All values from configuration

2. **Use ConfigurationManager**
   ```python
   from expandor.core.configuration_manager import ConfigurationManager
   
   config_manager = ConfigurationManager()
   
   # Get a value - FAILS LOUD if not found
   value = config_manager.get_value("some.config.key")
   
   # Get strategy config
   strategy_config = config_manager.get_strategy_config("progressive_outpaint")
   
   # Get processor config  
   processor_config = config_manager.get_processor_config("artifact_detector")
   ```

3. **Add New Config Values**
   - Add to `master_defaults.yaml`
   - Update relevant schema file
   - Document the value

## Configuration Hierarchy

1. **Master Defaults** (`master_defaults.yaml`)
   - Source of truth for all defaults
   - Version controlled
   - Required for operation

2. **User Configuration** (`~/.config/expandor/config.yaml`)
   - User overrides
   - Optional
   - Migrated automatically

3. **Environment Variables** (`EXPANDOR_*`)
   - Highest priority
   - Format: `EXPANDOR_STRATEGIES_PROGRESSIVE_OUTPAINT_BASE_STRENGTH=0.8`

4. **Runtime Overrides**
   - Passed directly to methods
   - Temporary for single operation

## Common Configuration Keys

### Adapter Defaults
- `adapters.common.default_width`: 1024
- `adapters.common.default_height`: 1024
- `adapters.common.default_num_inference_steps`: 50
- `adapters.common.default_guidance_scale`: 7.5

### Processing Constants
- `processing.rgb_max_value`: 255.0
- `processing.dimension_rounding`: 8

### Strategy Parameters
- `strategies.progressive_outpaint.base_strength`: 0.75
- `strategies.swpo.default_window_size`: 200

## FAIL LOUD Philosophy

The configuration system implements FAIL LOUD:

1. **No Silent Defaults**: Missing config values raise `ValueError`
2. **Clear Error Messages**: Include the missing key and solutions
3. **No Fallbacks**: If config is required, it must exist

Example error:
```
ValueError: Configuration key 'some.missing.key' not found!
This is a required configuration value with no default.
Solutions:
1. Add 'some.missing.key' to your config files
2. Set environment variable EXPANDOR_SOME_MISSING_KEY
3. Check config file syntax for errors
```

## Troubleshooting

### Config Not Loading
1. Check file exists: `~/.config/expandor/config.yaml`
2. Validate YAML syntax
3. Check version field

### Migration Failed
1. Check backup was created
2. Manually edit config to version 2.0
3. Report issue with error details

### Missing Values
1. Check key exists in `master_defaults.yaml`
2. Verify spelling and path
3. Set via environment variable as workaround

## Best Practices

1. **Never hardcode values** - Use ConfigurationManager
2. **Document new configs** - Add to this guide
3. **Test with minimal config** - Ensure defaults work
4. **Fail loud on errors** - No silent failures
5. **Version your configs** - Track changes

## Example: Adding New Config Value

1. Add to `master_defaults.yaml`:
   ```yaml
   my_feature:
     new_parameter: 42
   ```

2. Update schema if needed:
   ```json
   "new_parameter": {"type": "integer", "minimum": 1}
   ```

3. Use in code:
   ```python
   value = config_manager.get_value("my_feature.new_parameter")
   ```

4. Document here and in code

## Version History

- **v1.0**: Initial configuration system
- **v2.0**: Complete configurability, NO hardcoded values