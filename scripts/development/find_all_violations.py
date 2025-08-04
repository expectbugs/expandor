#!/usr/bin/env python3
"""Find all FAIL LOUD philosophy violations in the codebase"""
import ast
import os
import re
from pathlib import Path
from collections import Counter

class ViolationFinder:
    def __init__(self):
        self.get_violations = []
        self.param_defaults = []
        self.magic_numbers = []
        self.or_patterns = []
        
    def find_all_violations(self, directory):
        """Find all violations in the given directory"""
        for py_file in Path(directory).rglob('*.py'):
            if '__pycache__' in str(py_file) or 'venv' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    lines = content.splitlines()
                
                # Find .get() violations
                for i, line in enumerate(lines):
                    if match := re.search(r'\.get\([\'\"]\w+[\'\"],\s*.+\)', line):
                        # Skip if it's marked as optional
                        if 'optional' not in line.lower() and '# ok' not in line.lower():
                            self.get_violations.append({
                                'file': str(py_file),
                                'line': i + 1,
                                'code': line.strip()
                            })
                    
                    # Find "or" pattern violations
                    if re.search(r'(get_[^(]+\(\)|[a-zA-Z_]+)\s+or\s+[0-9"\']', line):
                        self.or_patterns.append({
                            'file': str(py_file),
                            'line': i + 1,
                            'code': line.strip()
                        })
                
                # Parse AST for function defaults and magic numbers
                try:
                    tree = ast.parse(content)
                    self._find_function_defaults(tree, str(py_file))
                    self._find_magic_numbers(tree, str(py_file), lines)
                except:
                    pass
                    
            except Exception as e:
                print(f"Error processing {py_file}: {e}")
    
    def _find_function_defaults(self, tree, filename):
        """Find function parameter defaults"""
        class DefaultFinder(ast.NodeVisitor):
            def __init__(self, parent):
                self.parent = parent
                self.filename = filename
                
            def visit_FunctionDef(self, node):
                if node.args.defaults:
                    for i, default in enumerate(node.args.defaults):
                        param_idx = len(node.args.args) - len(node.args.defaults) + i
                        param_name = node.args.args[param_idx].arg
                        
                        # Skip if it's None (optional parameter)
                        if isinstance(default, ast.Constant) and default.value is None:
                            continue
                            
                        self.parent.param_defaults.append({
                            'file': self.filename,
                            'line': node.lineno,
                            'function': node.name,
                            'param': param_name,
                            'default': ast.unparse(default) if hasattr(ast, 'unparse') else str(default)
                        })
                self.generic_visit(node)
        
        finder = DefaultFinder(self)
        finder.filename = filename
        finder.visit(tree)
    
    def _find_magic_numbers(self, tree, filename, lines):
        """Find magic numbers in code"""
        class NumberFinder(ast.NodeVisitor):
            def __init__(self, parent):
                self.parent = parent
                self.filename = filename
                self.lines = lines
                
            def visit_Constant(self, node):
                if isinstance(node.value, (int, float)):
                    # Skip common acceptable values
                    if node.value not in (0, 1, -1, 2, None, True, False):
                        # Check if it's part of a log/calculation
                        line = self.lines[node.lineno - 1] if node.lineno <= len(self.lines) else ""
                        # Skip if it's in a string or comment
                        if not re.search(rf'\b{node.value}\b', line):
                            return
                        
                        self.parent.magic_numbers.append({
                            'file': self.filename,
                            'line': node.lineno,
                            'value': node.value,
                            'context': line.strip()
                        })
                self.generic_visit(node)
        
        finder = NumberFinder(self)
        finder.filename = filename
        finder.lines = lines
        finder.visit(tree)
    
    def print_summary(self):
        """Print summary of all violations"""
        print("="*80)
        print("FAIL LOUD PHILOSOPHY VIOLATIONS SUMMARY")
        print("="*80)
        
        print(f"\n1. .get() with defaults: {len(self.get_violations)} violations")
        if self.get_violations:
            # Group by file
            file_counts = Counter(v['file'] for v in self.get_violations)
            print("\nTop files:")
            for file, count in file_counts.most_common(5):
                print(f"  {file}: {count}")
            print("\nExamples:")
            for v in self.get_violations[:3]:
                print(f"  {v['file']}:{v['line']}")
                print(f"    {v['code']}")
        
        print(f"\n2. Function parameter defaults: {len(self.param_defaults)} violations")
        if self.param_defaults:
            # Group by file
            file_counts = Counter(v['file'] for v in self.param_defaults)
            print("\nTop files:")
            for file, count in file_counts.most_common(5):
                print(f"  {file}: {count}")
            print("\nExamples:")
            for v in self.param_defaults[:3]:
                print(f"  {v['file']}:{v['line']} - {v['function']}({v['param']}={v['default']})")
        
        print(f"\n3. Magic numbers: {len(self.magic_numbers)} violations")
        if self.magic_numbers:
            # Count by value
            value_counts = Counter(v['value'] for v in self.magic_numbers)
            print("\nMost common values:")
            for value, count in value_counts.most_common(10):
                print(f"  {value}: {count} occurrences")
        
        print(f"\n4. 'or' pattern fallbacks: {len(self.or_patterns)} violations")
        if self.or_patterns:
            print("\nExamples:")
            for v in self.or_patterns[:3]:
                print(f"  {v['file']}:{v['line']}")
                print(f"    {v['code']}")
        
        print(f"\nTOTAL VIOLATIONS: {len(self.get_violations) + len(self.param_defaults) + len(self.magic_numbers) + len(self.or_patterns)}")

if __name__ == "__main__":
    finder = ViolationFinder()
    finder.find_all_violations('expandor')
    finder.print_summary()