#!/usr/bin/env python3
"""
Merge comprehensive hardcoded values configuration into master_defaults.yaml
This preserves existing structure while adding all missing configuration values
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, preserving existing values in base"""
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        elif key not in result:
            result[key] = value
    
    return result


def merge_configurations():
    """Merge comprehensive config into master defaults"""
    # Load existing master defaults
    master_path = Path(__file__).parent.parent / 'expandor' / 'config' / 'master_defaults.yaml'
    with open(master_path) as f:
        master_config = yaml.safe_load(f)
    
    # Load comprehensive hardcoded config
    comprehensive_path = Path(__file__).parent.parent / 'comprehensive_hardcoded_config.yaml'
    with open(comprehensive_path) as f:
        comprehensive_config = yaml.safe_load(f)
    
    # Merge configurations
    merged_config = deep_merge(master_config, comprehensive_config)
    
    # Save backup of original
    backup_path = master_path.with_suffix('.yaml.backup')
    with open(backup_path, 'w') as f:
        yaml.dump(master_config, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Saved backup to {backup_path}")
    
    # Save merged configuration
    with open(master_path, 'w') as f:
        yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Merged configuration saved to {master_path}")
    
    # Report statistics
    original_keys = count_keys(master_config)
    comprehensive_keys = count_keys(comprehensive_config)
    merged_keys = count_keys(merged_config)
    
    print(f"\nMerge Statistics:")
    print(f"  Original keys: {original_keys}")
    print(f"  Comprehensive keys: {comprehensive_keys}")
    print(f"  Merged keys: {merged_keys}")
    print(f"  New keys added: {merged_keys - original_keys}")


def count_keys(config: Dict[str, Any], prefix: str = "") -> int:
    """Count total number of configuration keys"""
    count = 0
    
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        count += 1
        
        if isinstance(value, dict):
            count += count_keys(value, full_key)
    
    return count


if __name__ == "__main__":
    merge_configurations()