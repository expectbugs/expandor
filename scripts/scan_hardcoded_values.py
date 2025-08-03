#!/usr/bin/env python3
"""
Scan for hardcoded values in Expandor codebase
Implements comprehensive pattern matching to find all hardcoded values
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json
from collections import defaultdict


class HardcodedValueScanner:
    """Comprehensive scanner for hardcoded values"""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.results = defaultdict(list)
        self.stats = defaultdict(int)
        
        # Patterns to detect hardcoded values
        self.patterns = {
            'get_with_default': re.compile(r'\.get\s*\([^,)]+,\s*([^)]+)\)'),
            'or_pattern': re.compile(r'\bor\s+([0-9]+\.?[0-9]*|True|False|"[^"]*"|\'[^\']*\')\b'),
            'direct_assignment': re.compile(r'^\s*(\w+)\s*[:=]\s*([0-9]+\.?[0-9]*)\s*(?:#.*)?$', re.MULTILINE),
            'field_default': re.compile(r'(\w+)\s*:\s*\w+\s*=\s*([0-9]+\.?[0-9]*|True|False|"[^"]*"|\'[^\']*\')'),
            'function_default': re.compile(r'def\s+\w+\([^)]*\w+\s*=\s*([0-9]+\.?[0-9]*|True|False|"[^"]*"|\'[^\']*\')[^)]*\)'),
            'if_comparison': re.compile(r'if\s+[^:]+[<>=]+\s*([0-9]+\.?[0-9]*)\s*:'),
            'range_calls': re.compile(r'range\s*\(\s*([0-9]+)\s*(?:,\s*([0-9]+))?\s*\)'),
            'list_index': re.compile(r'\[\s*([0-9]+)\s*\]'),
            'math_operations': re.compile(r'[+\-*/]\s*([0-9]+\.?[0-9]*)\b'),
            'string_format': re.compile(r'[fF]?["\']\{[^}]*:\.([0-9]+)[^}]*\}'),
        }
        
        # Files to exclude from scanning
        self.exclude_patterns = {
            '__pycache__',
            '.git',
            '.pytest_cache',
            'tests',  # Tests often have acceptable hardcoded values
            'examples',  # Examples can have hardcoded values
        }
        
        # Known acceptable hardcoded values
        self.whitelist = {
            '0', '1', '-1', '2', '0.0', '1.0', '0.5',  # Common mathematical values
            '8',  # SDXL dimension multiple
            '100', '255',  # Common image processing values
            'True', 'False', 'None',  # Boolean/None values
        }

    def should_scan_file(self, file_path: Path) -> bool:
        """Check if file should be scanned"""
        # Check exclude patterns
        for exclude in self.exclude_patterns:
            if exclude in str(file_path):
                return False
        
        # Only scan Python files
        return file_path.suffix == '.py'
    
    def analyze_ast_node(self, node, file_path: Path, source_lines: List[str]):
        """Analyze AST node for hardcoded values"""
        if isinstance(node, ast.Constant):
            # Check numeric constants
            if isinstance(node.value, (int, float)) and str(node.value) not in self.whitelist:
                self.add_finding(
                    file_path, 
                    node.lineno, 
                    f"Numeric constant: {node.value}",
                    "ast_constant",
                    source_lines[node.lineno - 1].strip() if node.lineno <= len(source_lines) else ""
                )
        
        elif isinstance(node, ast.Call):
            # Check for .get() calls with defaults
            if (isinstance(node.func, ast.Attribute) and 
                node.func.attr == 'get' and 
                len(node.args) >= 2):
                default_arg = node.args[1]
                if isinstance(default_arg, ast.Constant):
                    self.add_finding(
                        file_path,
                        node.lineno,
                        f".get() with default: {ast.unparse(default_arg)}",
                        "get_default",
                        source_lines[node.lineno - 1].strip() if node.lineno <= len(source_lines) else ""
                    )
        
        # Recursively analyze child nodes
        for child in ast.iter_child_nodes(node):
            self.analyze_ast_node(child, file_path, source_lines)
    
    def scan_file_patterns(self, file_path: Path, content: str):
        """Scan file content using regex patterns"""
        lines = content.split('\n')
        
        for pattern_name, pattern in self.patterns.items():
            for match in pattern.finditer(content):
                # Get line number
                line_start = content.count('\n', 0, match.start()) + 1
                line_content = lines[line_start - 1].strip() if line_start <= len(lines) else ""
                
                # Skip if in comment
                if '#' in line_content and line_content.index('#') < match.start() - content.rfind('\n', 0, match.start()):
                    continue
                
                # Extract matched value
                matched_value = match.group(1)
                
                # Skip whitelisted values
                if matched_value.strip('"\'') in self.whitelist:
                    continue
                
                self.add_finding(
                    file_path,
                    line_start,
                    f"{pattern_name}: {matched_value}",
                    pattern_name,
                    line_content
                )
    
    def scan_file(self, file_path: Path):
        """Scan a single file for hardcoded values"""
        try:
            content = file_path.read_text()
            lines = content.split('\n')
            
            # AST-based analysis
            try:
                tree = ast.parse(content, filename=str(file_path))
                self.analyze_ast_node(tree, file_path, lines)
            except SyntaxError as e:
                print(f"Syntax error in {file_path}: {e}")
            
            # Pattern-based analysis
            self.scan_file_patterns(file_path, content)
            
            self.stats['files_scanned'] += 1
            
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
            self.stats['scan_errors'] += 1
    
    def add_finding(self, file_path: Path, line_num: int, description: str, 
                    finding_type: str, line_content: str):
        """Add a finding to results"""
        relative_path = file_path.relative_to(self.root_path)
        
        finding = {
            'file': str(relative_path),
            'line': line_num,
            'type': finding_type,
            'description': description,
            'code': line_content
        }
        
        self.results[str(relative_path)].append(finding)
        self.stats['total_findings'] += 1
        self.stats[f'findings_{finding_type}'] += 1
    
    def scan(self):
        """Scan all Python files in the project"""
        print(f"Scanning for hardcoded values in: {self.root_path}")
        print("=" * 80)
        
        # Find all Python files
        python_files = list(self.root_path.rglob("*.py"))
        
        for file_path in python_files:
            if self.should_scan_file(file_path):
                self.scan_file(file_path)
        
        print(f"\nScanned {self.stats['files_scanned']} files")
        print(f"Found {self.stats['total_findings']} potential hardcoded values")
    
    def generate_report(self, output_file: Path = None):
        """Generate detailed report of findings"""
        report_lines = []
        report_lines.append("# Hardcoded Values Scan Report")
        report_lines.append(f"\nTotal findings: {self.stats['total_findings']}")
        report_lines.append(f"Files scanned: {self.stats['files_scanned']}")
        report_lines.append(f"Files with issues: {len(self.results)}")
        
        # Summary by type
        report_lines.append("\n## Summary by Type")
        for key, value in sorted(self.stats.items()):
            if key.startswith('findings_'):
                finding_type = key.replace('findings_', '')
                report_lines.append(f"- {finding_type}: {value}")
        
        # Detailed findings by file
        report_lines.append("\n## Detailed Findings\n")
        
        for file_path in sorted(self.results.keys()):
            findings = self.results[file_path]
            if findings:
                report_lines.append(f"### {file_path}")
                report_lines.append(f"Found {len(findings)} issues:\n")
                
                # Sort by line number
                for finding in sorted(findings, key=lambda x: x['line']):
                    report_lines.append(f"- Line {finding['line']}: {finding['description']}")
                    report_lines.append(f"  ```python")
                    report_lines.append(f"  {finding['code']}")
                    report_lines.append(f"  ```")
                
                report_lines.append("")
        
        # Priority fixes
        report_lines.append("\n## Priority Fixes\n")
        report_lines.append("1. Fix all `.get()` calls with defaults - use ConfigurationManager")
        report_lines.append("2. Remove all `or` patterns with fallback values")
        report_lines.append("3. Move all numeric constants to configuration files")
        report_lines.append("4. Replace direct assignments with config lookups")
        
        report_content = '\n'.join(report_lines)
        
        if output_file:
            output_file.write_text(report_content)
            print(f"\nReport saved to: {output_file}")
        else:
            print("\n" + report_content)
        
        # Also save JSON format for programmatic processing
        json_file = output_file.with_suffix('.json') if output_file else Path('hardcoded_values.json')
        with open(json_file, 'w') as f:
            json.dump({
                'stats': dict(self.stats),
                'findings': dict(self.results)
            }, f, indent=2)
        print(f"JSON report saved to: {json_file}")


def main():
    """Main entry point"""
    # Determine project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent / 'expandor'
    
    if not project_root.exists():
        print(f"Error: Project root not found at {project_root}")
        sys.exit(1)
    
    # Create scanner and run
    scanner = HardcodedValueScanner(project_root)
    scanner.scan()
    
    # Generate report
    report_file = script_path.parent / 'hardcoded_values_report.md'
    scanner.generate_report(report_file)
    
    # Exit with error code if findings exist
    if scanner.stats['total_findings'] > 0:
        print(f"\n⚠️  Found {scanner.stats['total_findings']} hardcoded values that need to be fixed!")
        sys.exit(1)
    else:
        print("\n✅ No hardcoded values found!")
        sys.exit(0)


if __name__ == '__main__':
    main()