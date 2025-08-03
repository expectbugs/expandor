#!/usr/bin/env python3
"""
Migrate configuration files between versions
Handles version upgrades and format changes
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from collections import defaultdict
import argparse


class ConfigMigrator:
    """Handle configuration migration between versions"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.backup_dir = config_dir / 'backups'
        self.migrations = []
        self.current_version = None
        self.target_version = "2.0"
        
        # Define migration steps
        self.migration_map = {
            ("1.0", "2.0"): self.migrate_1_0_to_2_0,
            ("1.1", "2.0"): self.migrate_1_1_to_2_0,
            ("1.2", "2.0"): self.migrate_1_2_to_2_0,
        }
        
        # Field mappings for different versions
        self.field_mappings = {
            "1.0_to_2.0": {
                # Old field -> New field
                "enable_artifact_detection": "quality.enable_artifacts_check",
                "artifact_threshold": "quality.artifact_detection_threshold",
                "max_repair_attempts": "quality.max_artifact_repair_attempts",
                "enable_cpu_offload": "resources.use_cpu_offload",
                "save_intermediate": "processing.save_stages",
                "compression": "output.formats.png.compression",
                "quality": "output.formats.jpeg.quality",
            }
        }

    def detect_version(self, config: Dict[str, Any]) -> Optional[str]:
        """Detect configuration version"""
        # Check explicit version field
        if 'version' in config:
            return str(config['version'])
        
        # For user configs without version, check if it's a minimal override file
        # that doesn't need migration (just contains overrides compatible with 2.0)
        config_keys = set(config.keys())
        
        # Keys that indicate old structure needing migration
        old_structure_keys = {
            'enable_artifact_detection', 'artifact_threshold', 
            'max_repair_attempts', 'enable_cpu_offload',
            'save_intermediate', 'compression', 'quality'
        }
        
        # Keys that are valid in v2.0 structure
        v2_structure_keys = {
            'quality_presets', 'strategies', 'quality_global',
            'paths', 'vram', 'output', 'processing', 'core'
        }
        
        # If config has any old structure keys, it needs migration
        if config_keys & old_structure_keys:
            return "1.0"
        
        # If config only has v2.0 structure keys, treat as 2.0
        if config_keys & v2_structure_keys:
            return "2.0"
        
        # Empty or minimal configs are compatible with 2.0
        if len(config) == 0 or all(k in ['version'] for k in config_keys):
            return "2.0"
        
        # Try to detect based on structure
        if 'quality_presets' in config and 'strategies' in config:
            # Newer structure
            if 'quality_global' in config:
                return "1.2"
            else:
                return "1.1"
        else:
            # If we can't determine, assume it's an old structure
            return "1.0"
    
    def backup_configs(self) -> Path:
        """Create backup of current configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        
        print(f"Creating backup at: {backup_path}")
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all YAML files
        for yaml_file in self.config_dir.glob("*.yaml"):
            shutil.copy2(yaml_file, backup_path / yaml_file.name)
        
        # Copy schemas directory if exists
        schema_dir = self.config_dir / 'schemas'
        if schema_dir.exists():
            shutil.copytree(schema_dir, backup_path / 'schemas')
        
        return backup_path
    
    def load_config(self, file_path: Path) -> Tuple[Dict[str, Any], str]:
        """Load configuration and detect version"""
        try:
            with open(file_path) as f:
                config = yaml.safe_load(f) or {}
            
            version = self.detect_version(config)
            return config, version
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}, "unknown"
    
    def migrate_1_0_to_2_0(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 1.0 to 2.0"""
        print("Migrating from version 1.0 to 2.0...")
        
        new_config = {
            "version": "2.0",
            "quality_global": {
                "default_preset": "balanced"
            },
            "quality_presets": {},
            "strategies": {},
            "processing": {},
            "output": {
                "formats": {
                    "png": {},
                    "jpeg": {}
                }
            },
            "paths": {},
            "vram": {
                "estimation": {},
                "offloading": {}
            },
            "quality_thresholds": {},
            "system": {}
        }
        
        # Map old fields to new structure
        mapping = self.field_mappings["1.0_to_2.0"]
        
        def set_nested(obj: dict, path: str, value: Any):
            """Set value in nested dictionary"""
            parts = path.split('.')
            current = obj
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        
        # Migrate fields
        for old_key, value in config.items():
            if old_key in mapping:
                new_path = mapping[old_key]
                set_nested(new_config, new_path, value)
            elif old_key != 'version':
                # Try to preserve unknown fields
                new_config[old_key] = value
        
        # Add default values for new required fields
        self.add_required_defaults(new_config)
        
        return new_config
    
    def migrate_1_1_to_2_0(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 1.1 to 2.0"""
        print("Migrating from version 1.1 to 2.0...")
        
        # 1.1 is closer to 2.0, mainly needs version bump and validation
        new_config = config.copy()
        new_config['version'] = "2.0"
        
        # Ensure all required sections exist
        self.add_required_defaults(new_config)
        
        return new_config
    
    def migrate_1_2_to_2_0(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 1.2 to 2.0"""
        print("Migrating from version 1.2 to 2.0...")
        
        # 1.2 to 2.0 is mostly compatible
        new_config = config.copy()
        new_config['version'] = "2.0"
        
        return new_config
    
    def add_required_defaults(self, config: Dict[str, Any]):
        """Add required default values for version 2.0"""
        defaults = {
            "quality_global.default_preset": "balanced",
            "paths.cache_dir": "~/.cache/expandor",
            "paths.output_dir": "./output",
            "paths.temp_dir": "/tmp/expandor",
            "processing.save_stages": False,
            "processing.verbose": False,
            "processing.batch_size": 1,
            "output.formats.png.compression": 0,
            "output.formats.jpeg.quality": 95,
            "vram.estimation.latent_multiplier": 4.5,
            "vram.offloading.enable_sequential": True,
        }
        
        def get_nested(obj: dict, path: str) -> Any:
            """Get value from nested dictionary"""
            parts = path.split('.')
            current = obj
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    return None
                current = current[part]
            return current
        
        def set_nested(obj: dict, path: str, value: Any):
            """Set value in nested dictionary"""
            parts = path.split('.')
            current = obj
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            if parts[-1] not in current:
                current[parts[-1]] = value
        
        # Add defaults for missing fields
        for path, default_value in defaults.items():
            if get_nested(config, path) is None:
                set_nested(config, path, default_value)
    
    def merge_user_customizations(self, old_config: Dict[str, Any], 
                                  new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user customizations from old config into new config"""
        print("Merging user customizations...")
        
        def merge_recursive(base: dict, custom: dict):
            """Recursively merge custom values into base"""
            for key, value in custom.items():
                if key in base:
                    if isinstance(base[key], dict) and isinstance(value, dict):
                        merge_recursive(base[key], value)
                    else:
                        # Preserve user customization
                        base[key] = value
                else:
                    # Add custom field
                    base[key] = value
        
        # Create a copy to avoid modifying new_config
        merged = new_config.copy()
        
        # Extract user customizations (values different from defaults)
        # This is simplified - in practice would compare against known defaults
        user_custom = {}
        for key, value in old_config.items():
            if key not in ['version']:
                user_custom[key] = value
        
        merge_recursive(merged, user_custom)
        
        return merged
    
    def validate_migration(self, config: Dict[str, Any]) -> List[str]:
        """Validate migrated configuration"""
        errors = []
        
        # Check version
        if 'version' not in config:
            errors.append("Missing 'version' field")
        elif config['version'] != self.target_version:
            errors.append(f"Version mismatch: expected {self.target_version}, got {config['version']}")
        
        # Check required top-level fields
        required_fields = [
            'quality_global', 'quality_presets', 'strategies',
            'processing', 'output', 'paths', 'vram'
        ]
        
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        return errors
    
    def save_config(self, config: Dict[str, Any], file_path: Path):
        """Save configuration to file"""
        # Add header comment
        header = f"""# Expandor Configuration v{config.get('version', 'unknown')}
# Generated by migration tool on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 
# This is the master configuration file containing all default values.
# User customizations should be placed in user_config.yaml
#
"""
        
        with open(file_path, 'w') as f:
            f.write(header)
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, width=120)
    
    def migrate(self, dry_run: bool = False) -> bool:
        """Execute migration"""
        print(f"Starting configuration migration in: {self.config_dir}")
        print("=" * 80)
        
        # Load master configuration
        master_path = self.config_dir / 'master_defaults.yaml'
        if not master_path.exists():
            print(f"Error: master_defaults.yaml not found at {master_path}")
            return False
        
        # Load and detect version
        config, version = self.load_config(master_path)
        
        if version == "unknown":
            print("Error: Could not detect configuration version")
            return False
        
        print(f"Detected configuration version: {version}")
        
        if version == self.target_version:
            print(f"Configuration is already at target version {self.target_version}")
            return True
        
        # Find migration path
        migration_key = (version, self.target_version)
        if migration_key not in self.migration_map:
            print(f"Error: No migration path from {version} to {self.target_version}")
            return False
        
        if not dry_run:
            # Create backup
            backup_path = self.backup_configs()
            print(f"Backup created at: {backup_path}")
        
        # Execute migration
        migration_func = self.migration_map[migration_key]
        new_config = migration_func(config)
        
        # Validate migration
        errors = self.validate_migration(new_config)
        if errors:
            print("\nMigration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        if dry_run:
            print("\nDry run complete. Configuration would be migrated successfully.")
            print("\nPreview of changes:")
            # Show a diff or summary of changes
            return True
        
        # Save migrated configuration
        self.save_config(new_config, master_path)
        print(f"\nConfiguration migrated successfully to version {self.target_version}")
        
        # Migrate other config files if needed
        self.migrate_related_configs()
        
        return True
    
    def migrate_related_configs(self):
        """Migrate related configuration files"""
        # This would handle migration of quality_presets.yaml, strategies.yaml, etc.
        # For now, we'll just ensure they're compatible
        pass
    
    def rollback(self, backup_name: str):
        """Rollback to a previous backup"""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            print(f"Error: Backup not found at {backup_path}")
            return False
        
        print(f"Rolling back to backup: {backup_name}")
        
        # Copy files back
        for yaml_file in backup_path.glob("*.yaml"):
            dest = self.config_dir / yaml_file.name
            shutil.copy2(yaml_file, dest)
            print(f"  Restored: {yaml_file.name}")
        
        print("Rollback complete!")
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Migrate Expandor configuration files")
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be migrated without making changes')
    parser.add_argument('--rollback', type=str,
                        help='Rollback to a specific backup (provide backup directory name)')
    parser.add_argument('--list-backups', action='store_true',
                        help='List available backups')
    
    args = parser.parse_args()
    
    # Determine config directory
    script_path = Path(__file__).resolve()
    config_dir = script_path.parent.parent / 'expandor' / 'config'
    
    if not config_dir.exists():
        print(f"Error: Config directory not found at {config_dir}")
        sys.exit(1)
    
    # Create migrator
    migrator = ConfigMigrator(config_dir)
    
    # Handle commands
    if args.list_backups:
        backup_dir = config_dir / 'backups'
        if backup_dir.exists():
            backups = sorted(backup_dir.iterdir())
            if backups:
                print("Available backups:")
                for backup in backups:
                    print(f"  - {backup.name}")
            else:
                print("No backups found")
        else:
            print("No backup directory found")
        sys.exit(0)
    
    if args.rollback:
        success = migrator.rollback(args.rollback)
        sys.exit(0 if success else 1)
    
    # Run migration
    success = migrator.migrate(dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()