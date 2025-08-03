#!/usr/bin/env python3
"""
Fix all .get() calls with defaults to use ConfigurationManager
This enforces the FAIL LOUD principle - no silent defaults
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple


class GetDefaultsFixer(ast.NodeTransformer):
    """AST transformer to replace .get() with defaults"""
    
    def __init__(self):
        self.changes = []
        self.current_file = None
        
    def visit_Call(self, node):
        """Visit function calls looking for .get() with defaults"""
        self.generic_visit(node)
        
        # Check if this is a .get() call
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr == 'get' and
            len(node.args) >= 2):  # Has default value
            
            # Record the change needed
            if hasattr(node, 'lineno'):
                self.changes.append({
                    'line': node.lineno,
                    'type': 'get_with_default',
                    'node': node
                })
        
        return node


def analyze_file(file_path: Path) -> List[Dict]:
    """Analyze a file for .get() with defaults"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        fixer = GetDefaultsFixer()
        fixer.current_file = file_path
        fixer.visit(tree)
        
        return fixer.changes
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return []


def fix_get_defaults_in_file(file_path: Path) -> int:
    """Fix .get() calls in a single file"""
    changes_made = 0
    
    # Skip certain files
    skip_patterns = [
        'config_loader.py',  # Config loading infrastructure
        'configuration_manager.py',  # Core config system
        'config_migrator.py',  # Migration tools
        'user_config.py',  # User config handling
        'pipeline_config.py',  # Pipeline config
        '__pycache__',
        '.pyc'
    ]
    
    if any(pattern in str(file_path) for pattern in skip_patterns):
        return 0
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Pattern to match .get() with default value
        # This regex matches: .get("key", default) or .get('key', default)
        pattern = r'\.get\((["\'][^"\']+["\'])\s*,\s*([^)]+)\)'
        
        new_lines = []
        for i, line in enumerate(lines):
            if '.get(' in line and ',' in line:
                # Check if this is a dictionary get with default
                match = re.search(pattern, line)
                if match:
                    key = match.group(1)
                    default = match.group(2).strip()
                    
                    # Skip if it's already using config manager or is a special case
                    if 'config_manager' in line or 'ConfigurationManager' in line:
                        new_lines.append(line)
                        continue
                    
                    # Determine context from variable name and key
                    if 'detection_result' in line:
                        # Special handling for detection results
                        new_lines.append(line)  # Keep as is for now
                    elif 'kwargs' in line or 'metadata' in line:
                        # Keep kwargs and metadata gets
                        new_lines.append(line)
                    else:
                        # This needs to be fixed
                        print(f"Found .get() with default in {file_path}:{i+1}")
                        print(f"  Key: {key}, Default: {default}")
                        # For now, just mark it
                        new_lines.append(line)
                        changes_made += 1
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        return changes_made
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0


def main():
    """Main entry point"""
    expandor_dir = Path(__file__).parent.parent / 'expandor'
    
    print("Analyzing .get() calls with defaults...")
    print("=" * 80)
    
    total_issues = 0
    files_with_issues = []
    
    # Find all Python files
    for py_file in expandor_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
            
        issues = fix_get_defaults_in_file(py_file)
        if issues > 0:
            total_issues += issues
            files_with_issues.append((py_file, issues))
    
    # Report summary
    print("\n" + "=" * 80)
    print(f"Total .get() with defaults found: {total_issues}")
    print(f"Files affected: {len(files_with_issues)}")
    
    if files_with_issues:
        print("\nFiles with issues:")
        for file_path, count in sorted(files_with_issues, key=lambda x: x[1], reverse=True):
            print(f"  {file_path.relative_to(expandor_dir)}: {count} issues")
    
    # Create fix recommendations
    print("\n" + "=" * 80)
    print("Recommendations:")
    print("1. Replace all .get() with defaults with ConfigurationManager.get_value()")
    print("2. Add missing configuration keys to master_defaults.yaml")
    print("3. Update code to handle KeyError/ValueError for missing configs")
    print("4. Special cases like detection_result.get() may need different handling")


if __name__ == "__main__":
    main()