# Configuration Migration Guide

## Overview

Expandor uses a versioned configuration system to ensure smooth upgrades and backwards compatibility. This guide explains how configuration migration works and how to handle version changes.

## Current Configuration Version

**Current Version: 2.0**

All configuration files should include a `version` field at the top level:

```yaml
version: "2.0"
# ... rest of configuration
```

## Automatic Migration

Expandor automatically migrates user configurations when:

1. Loading a user configuration file with an older version
2. The system can safely migrate to the current version
3. Migration rules are defined for the version transition

### Migration Process

1. **Detection**: ConfigurationManager detects version mismatch
2. **Backup**: Original configuration is backed up with timestamp
3. **Migration**: Configuration is migrated to current version
4. **Save**: Migrated configuration replaces the original
5. **Validation**: New configuration is validated against schema

Example backup filename: `config.yaml.backup_20250803_141523`

## Manual Migration

To manually migrate configuration files:

```bash
# Check current configuration version
expandor --check-config

# Migrate configuration file
python -m expandor.scripts.migrate_config /path/to/config.yaml

# Migrate with custom backup location
python -m expandor.scripts.migrate_config /path/to/config.yaml --backup-dir /backups
```

## Version History

### Version 2.0 (Current)
- Complete configuration system refactoring
- NO HARDCODED VALUES principle
- All settings in master_defaults.yaml
- Comprehensive path configuration
- Strategy-specific configurations
- Quality preset system
- VRAM management settings

### Version 1.0 to 2.0 Migration

Key changes:
- Moved all hardcoded values to configuration
- Added comprehensive path settings
- Restructured strategy configurations
- Added VRAM and memory profiles
- Enhanced quality presets

Migration mapping:
```yaml
# Version 1.0
core:
  quality_preset: "high"
  num_inference_steps: 50

# Version 2.0
quality_presets:
  high:
    generation:
      num_inference_steps: 50
```

## Configuration File Locations

### System Configuration
- `expandor/config/master_defaults.yaml` - System defaults (version controlled)

### User Configuration
Search order:
1. `$EXPANDOR_USER_CONFIG_PATH` (if set)
2. `~/.config/expandor/config.yaml`
3. `./expandor.yaml` (current directory)
4. `/etc/expandor/config.yaml`

## Migration Rules

### Adding New Fields
When new fields are added in a version:
- Default values are used if not specified
- User is notified of new available options
- No action required unless customization desired

### Renaming Fields
When fields are renamed:
- Old field values are automatically mapped to new names
- Deprecated fields are removed after migration
- User is notified of changes

### Restructuring
When configuration structure changes:
- Values are moved to new locations
- Nested structures are created as needed
- Original semantics are preserved

## Troubleshooting

### Migration Failures

If automatic migration fails:

1. **Check Error Message**: The error will indicate what went wrong
2. **Review Backup**: Check the backup file created before migration
3. **Manual Fix**: Edit configuration to match current structure
4. **Recreate**: Use `expandor --init-config` to create fresh config

### Version Mismatch Errors

```
ValueError: Configuration migration required but migration script not found!
User config at /home/user/.config/expandor/config.yaml has version 1.0, but system expects 2.0.
Please run: expandor --migrate-config
```

**Solution**: Update Expandor or manually migrate configuration

### Validation Errors

After migration, configuration is validated. Common issues:

1. **Missing Required Fields**: Add missing fields from master_defaults.yaml
2. **Invalid Values**: Check value types and ranges
3. **Unknown Fields**: Remove fields not in current version

## Best Practices

1. **Always Include Version**: Every config file should have a version field
2. **Keep Backups**: Migration creates backups, but keep your own too
3. **Test After Migration**: Verify functionality after migration
4. **Use Latest Template**: For new configs, use `expandor --init-config`

## Custom Presets and Migration

Custom presets are preserved during migration:

```yaml
# Version 1.0
my_preset:
  steps: 100
  
# Version 2.0 (after migration)
custom_presets:
  my_preset:
    generation:
      num_inference_steps: 100
```

## Environment Variables

Environment variables are version-independent and always use current structure:

```bash
# Always uses current structure
export EXPANDOR_VRAM_SAFETY_FACTOR=0.95
export EXPANDOR_QUALITY_PRESETS_ULTRA_GENERATION_NUM_INFERENCE_STEPS=120
```

## Extending Migration

To add custom migration rules:

1. Edit `expandor/scripts/migrate_config.py`
2. Add migration method for version transition
3. Update `migration_map` dictionary
4. Test with various configurations

Example:
```python
def migrate_2_0_to_3_0(self, config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate from version 2.0 to 3.0"""
    config = config.copy()
    config['version'] = '3.0'
    
    # Add migration logic here
    
    return config
```

## Future Compatibility

The configuration system is designed for future extensibility:

- Version field is mandatory
- Unknown fields are preserved during migration
- Backwards compatibility for at least 2 major versions
- Clear migration paths between versions

## Getting Help

If you encounter issues with configuration migration:

1. Check this documentation
2. Review error messages carefully
3. Check backup files
4. Run `expandor --check-config` for validation
5. Report issues at: https://github.com/expandor/expandor/issues