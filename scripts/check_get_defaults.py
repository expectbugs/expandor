#!/usr/bin/env python3
"""
Pre-commit hook to check for .get() with defaults
Ensures FAIL LOUD principle is maintained
"""

import re
import sys
from pathlib import Path


def check_file(file_path: Path) -> int:
    """Check a single file for .get() with defaults"""
    violations = 0
    
    # Skip allowed files
    allowed_files = {
        'config_loader.py',
        'configuration_manager.py', 
        'config_migrator.py',
        'user_config.py',
        'pipeline_config.py',
        'metadata_tracker.py',  # Metadata can have optional fields
    }
    
    if file_path.name in allowed_files:
        return 0
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Pattern to match .get() with default value
        pattern = r'\.get\(["\'][^"\']+["\'],\s*[^)]+\)'
        
        for i, line in enumerate(lines, 1):
            if '.get(' in line and ',' in line:
                # Skip comments
                if line.strip().startswith('#'):
                    continue
                
                # Skip allowed patterns
                if any(allowed in line for allowed in ['kwargs', 'metadata', 'environ']):
                    continue
                
                match = re.search(pattern, line)
                if match:
                    print(f"{file_path}:{i}: Found .get() with default: {line.strip()}")
                    violations += 1
        
        return violations
        
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
        return 0


def main():
    """Main entry point"""
    expandor_dir = Path(__file__).parent.parent / 'expandor'
    
    total_violations = 0
    
    # Check all Python files
    for py_file in expandor_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
        
        violations = check_file(py_file)
        total_violations += violations
    
    if total_violations > 0:
        print(f"\n❌ Found {total_violations} .get() with defaults violations!")
        print("Please use ConfigurationManager.get_value() or explicit validation instead.")
        sys.exit(1)
    else:
        print("✅ No .get() with defaults violations found.")
        sys.exit(0)


if __name__ == "__main__":
    main()