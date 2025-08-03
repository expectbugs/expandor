#!/usr/bin/env python3
"""
Script to systematically fix .get() patterns in expandor codebase
According to FAIL LOUD principle - replace silent defaults with explicit validation
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add expandor to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_get_patterns(file_path: Path) -> List[Dict]:
    """Analyze .get() patterns in a file and categorize them"""
    patterns = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.splitlines()
    
    # Find all .get( patterns with context
    for i, line in enumerate(lines):
        get_matches = re.finditer(r'(\w+)\.get\(["\']([^"\']+)["\'](?:,\s*([^)]+))?\)', line)
        for match in get_matches:
            var_name = match.group(1)
            key_name = match.group(2)
            default_value = match.group(3)
            
            pattern_info = {
                'file': file_path,
                'line_num': i + 1,
                'line_content': line.strip(),
                'var_name': var_name,
                'key_name': key_name,
                'default_value': default_value,
                'category': categorize_pattern(var_name, key_name, line, lines, i)
            }
            patterns.append(pattern_info)
    
    return patterns

def categorize_pattern(var_name: str, key_name: str, line: str, lines: List[str], line_idx: int) -> str:
    """Categorize the type of .get() pattern"""
    
    # Look at surrounding context
    context_before = lines[max(0, line_idx-3):line_idx]
    context_after = lines[line_idx+1:min(len(lines), line_idx+4)]
    
    # Config-related patterns that should use ConfigurationManager
    config_vars = ['config', 'self.config', 'processor_config', 'strategy_config', 
                   'preset_config', 'validation_config', 'orchestrator_config']
    if any(var_name.endswith(cv) or var_name == cv for cv in config_vars):
        return 'config_lookup'
    
    # Runtime data that should FAIL LOUD
    runtime_vars = ['boundary', 'result', 'validation_result', 'detection_result', 
                   'scores', 'metadata', 'details', 'analysis', 'edge_analysis']
    if any(var_name.endswith(rv) or var_name == rv for rv in runtime_vars):
        return 'runtime_data'
    
    # Optional/informational data that might be okay to keep
    info_vars = ['info', 'available_features', 'results', 'gpu', 'features']
    if any(var_name.endswith(iv) or var_name == iv for iv in info_vars):
        return 'informational'
    
    # Default to requiring review
    return 'needs_review'

def create_fix_for_pattern(pattern: Dict) -> str:
    """Create the appropriate fix for a pattern"""
    category = pattern['category']
    var_name = pattern['var_name']
    key_name = pattern['key_name']
    default_value = pattern['default_value']
    line = pattern['line_content']
    
    if category == 'runtime_data':
        # FAIL LOUD for runtime data
        return f"""            # FAIL LOUD if {key_name} missing from {var_name}
            if "{key_name}" not in {var_name}:
                raise QualityError(
                    f"{var_name} missing required '{key_name}' field",
                    details={{"available_keys": list({var_name}.keys())}}
                )
            # Replace: {var_name}.get("{key_name}", {default_value})
            # With: {var_name}["{key_name}"]"""
    
    elif category == 'config_lookup':
        # Use ConfigurationManager for config lookups
        return f"""            # Use ConfigurationManager instead of .get() for config lookup
            # Replace: {var_name}.get("{key_name}", {default_value})
            # With: self.config_manager.get_value("path.to.{key_name}")
            # NOTE: Config path needs to be determined based on context"""
    
    elif category == 'informational':
        # These might be okay to keep, but add comment
        return f"""            # REVIEWED: {var_name}.get("{key_name}", {default_value}) 
            # This is informational/optional data - .get() may be appropriate here"""
    
    else:
        return f"""            # NEEDS_REVIEW: {var_name}.get("{key_name}", {default_value})
            # Determine if this should FAIL LOUD or use ConfigurationManager"""

def main():
    """Main function to analyze and suggest fixes"""
    expandor_dir = Path(__file__).parent.parent / "expandor"
    
    print("🔍 Analyzing .get() patterns in expandor codebase...")
    print("=" * 60)
    
    all_patterns = []
    
    # Find all Python files
    for py_file in expandor_dir.rglob("*.py"):
        if py_file.name.startswith('.') or 'test' in str(py_file):
            continue
            
        patterns = analyze_get_patterns(py_file)
        all_patterns.extend(patterns)
    
    # Group by category
    by_category = {}
    for pattern in all_patterns:
        cat = pattern['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(pattern)
    
    # Print summary
    print(f"Found {len(all_patterns)} .get() patterns:")
    for category, patterns in by_category.items():
        print(f"  {category}: {len(patterns)} patterns")
    
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS:")
    print("=" * 60)
    
    # Print detailed analysis by category
    for category in ['runtime_data', 'config_lookup', 'needs_review', 'informational']:
        if category not in by_category:
            continue
            
        patterns = by_category[category]
        print(f"\n📋 {category.upper()} ({len(patterns)} patterns):")
        print("-" * 40)
        
        for pattern in patterns:
            print(f"\n📁 {pattern['file'].relative_to(expandor_dir)}")
            print(f"   Line {pattern['line_num']}: {pattern['line_content']}")
            print(f"   Variable: {pattern['var_name']}")
            print(f"   Key: {pattern['key_name']}")
            print(f"   Default: {pattern['default_value']}")
            
            if category == 'runtime_data':
                print("   🚨 PRIORITY: HIGH - Should FAIL LOUD")
                print(f"   Fix: Replace with explicit validation and {pattern['var_name']}[\"{pattern['key_name']}\"]")
            elif category == 'config_lookup':
                print("   ⚙️  PRIORITY: HIGH - Use ConfigurationManager")
                print(f"   Fix: Use self.config_manager.get_value() with proper path")
            elif category == 'needs_review':
                print("   🔍 PRIORITY: MEDIUM - Needs manual review")
            else:
                print("   ℹ️  PRIORITY: LOW - May be appropriate")
    
    print(f"\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Fix 'runtime_data' patterns first (HIGH PRIORITY)")
    print("2. Fix 'config_lookup' patterns (HIGH PRIORITY)") 
    print("3. Review 'needs_review' patterns (MEDIUM PRIORITY)")
    print("4. Consider keeping 'informational' patterns (LOW PRIORITY)")
    print("\nRun the actual fixes with fix_get_defaults.py")

if __name__ == "__main__":
    main()