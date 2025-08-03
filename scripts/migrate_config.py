#!/usr/bin/env python3
"""
Configuration migration tool for Expandor
Helps users migrate from old configuration formats to new
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
import yaml

from expandor.utils.config_migrator import ConfigMigrator


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Expandor configuration files to latest format"
    )
    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to configuration file to migrate"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output path (default: updates in place with backup)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes"
    )
    parser.add_argument(
        "--format",
        choices=["yaml", "json"],
        default="yaml",
        help="Output format (default: yaml)"
    )
    
    args = parser.parse_args()
    
    if not args.config_file.exists():
        print(f"Error: Configuration file not found: {args.config_file}")
        sys.exit(1)
    
    # Create migrator
    migrator = ConfigMigrator()
    
    # Load and detect version
    print(f"Loading configuration from {args.config_file}...")
    config, version = migrator.load_config(args.config_file)
    
    if version == "unknown":
        print("Warning: Could not detect configuration version")
        print("Assuming version 1.0")
        version = "1.0"
    
    print(f"Detected configuration version: {version}")
    
    if version == migrator.target_version:
        print(f"Configuration is already at target version {migrator.target_version}")
        sys.exit(0)
    
    # Migrate
    print(f"Migrating from version {version} to {migrator.target_version}...")
    
    if args.dry_run:
        print("\nDRY RUN - No changes will be made")
        print("\nMigration would include:")
        
        if version == "1.0":
            print("- Consolidate separate config files into unified structure")
            print("- Update quality preset format")
            print("- Add new required fields with sensible defaults")
            print("- Reorganize configuration hierarchy")
        
        print("\nExample changes:")
        print("- quality_thresholds section would be added")
        print("- VRAM configuration would be expanded")
        print("- Processing parameters would be reorganized")
        
        sys.exit(0)
    
    # Perform migration
    try:
        migrated_config = migrator.migrate_to_v2(config)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Create backup
            backup_path = args.config_file.with_suffix(f"{args.config_file.suffix}.backup")
            print(f"Creating backup at {backup_path}...")
            args.config_file.rename(backup_path)
            output_path = args.config_file
        
        # Save migrated config
        print(f"Saving migrated configuration to {output_path}...")
        
        with open(output_path, 'w') as f:
            if args.format == "yaml":
                yaml.dump(migrated_config, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(migrated_config, f, indent=2)
        
        print(f"\n✅ Migration completed successfully!")
        print(f"Configuration migrated from version {version} to {migrator.target_version}")
        
        if not args.output:
            print(f"Original configuration backed up to: {backup_path}")
        
        # Show summary of changes
        print("\nSummary of changes:")
        print(f"- Added {len(migrated_config) - len(config)} top-level sections")
        print("- Updated configuration structure to v2.0 format")
        print("- Added required fields with default values")
        print("\nPlease review the migrated configuration and adjust values as needed.")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()