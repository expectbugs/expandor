#!/usr/bin/env python3
"""
StrictValidator - Enforce FAIL LOUD compliance
Ensures no silent defaults or fallbacks in the codebase
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple


class StrictValidator:
    """Validates codebase for FAIL LOUD compliance"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.violations: List[Dict] = []
        self.stats = {
            "total_files": 0,
            "files_scanned": 0,
            "violations_found": 0,
            "get_with_defaults": 0,
            "function_defaults": 0,
            "or_patterns": 0,
            "try_except_pass": 0,
            "magic_numbers": 0,
            "print_statements": 0
        }
        
    def scan_file(self, file_path: Path) -> None:
        """Scan a single file for violations"""
        self.stats["files_scanned"] += 1
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return
            
        for line_num, line in enumerate(lines, 1):
            # Skip comments and strings
            if line.strip().startswith('#') or '"""' in line or "'''" in line:
                continue
                
            # Check for .get() with defaults
            if '.get(' in line and ',' in line and not '# Optional' in line:
                if re.search(r'\.get\([^,]+,\s*[^)]+\)', line):
                    self.add_violation(file_path, line_num, "get_with_defaults", 
                                     f".get() with default value: {line.strip()}")
                    
            # Check for function defaults (excluding Optional[...] = None)
            if 'def ' in line and '=' in line:
                if not re.search(r'Optional\[.*\]\s*=\s*None', line):
                    # Check for actual default values
                    if re.search(r'def\s+\w+.*:\s*\w+\s*=\s*[^,)]+', line):
                        self.add_violation(file_path, line_num, "function_defaults",
                                         f"Function with default parameter: {line.strip()}")
                        
            # Check for 'or' pattern fallbacks
            if ' or ' in line and not any(x in line for x in ['# Optional', '# OK', 'logger']):
                if re.search(r'=.*\sor\s', line) or re.search(r'return.*\sor\s', line):
                    self.add_violation(file_path, line_num, "or_patterns",
                                     f"'or' pattern fallback: {line.strip()}")
                    
            # Check for try/except with pass
            if 'except' in line and 'pass' in line:
                self.add_violation(file_path, line_num, "try_except_pass",
                                 f"Silent exception handling: {line.strip()}")
                
            # Check for magic numbers (common ones)
            magic_patterns = [
                (r'\b1024\b', '1024'),
                (r'\b512\b', '512'),
                (r'\b768\b', '768'),
                (r'\b255\b', '255'),
                (r'\b0\.8\b', '0.8'),
                (r'\b7\.5\b', '7.5'),
                (r'\b2048\b', '2048'),
            ]
            for pattern, value in magic_patterns:
                if re.search(pattern, line) and not any(x in line for x in ['#', 'logger', '__']):
                    self.add_violation(file_path, line_num, "magic_numbers",
                                     f"Magic number {value}: {line.strip()}")
                    
            # Check for print statements (should use logger)
            if 'print(' in line and not any(x in line for x in ['# OK', '# Debug']):
                self.add_violation(file_path, line_num, "print_statements",
                                 f"Print statement instead of logger: {line.strip()}")
                
    def add_violation(self, file_path: Path, line_num: int, violation_type: str, message: str):
        """Add a violation to the list"""
        self.violations.append({
            "file": str(file_path.relative_to(self.root_dir)),
            "line": line_num,
            "type": violation_type,
            "message": message
        })
        self.stats["violations_found"] += 1
        self.stats[violation_type] += 1
        
    def scan_directory(self, directory: Path, exclude_dirs: Set[str] = None) -> None:
        """Recursively scan directory for Python files"""
        if exclude_dirs is None:
            exclude_dirs = {'venv', '__pycache__', '.git', 'build', 'dist', '.pytest_cache'}
            
        for item in directory.iterdir():
            if item.is_dir():
                if item.name not in exclude_dirs:
                    self.scan_directory(item, exclude_dirs)
            elif item.is_file() and item.suffix == '.py':
                self.stats["total_files"] += 1
                self.scan_file(item)
                
    def generate_report(self) -> Dict:
        """Generate validation report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "root_directory": str(self.root_dir),
            "statistics": self.stats,
            "violations": self.violations,
            "summary": {
                "compliant": self.stats["violations_found"] == 0,
                "severity": self._calculate_severity()
            }
        }
        
    def _calculate_severity(self) -> str:
        """Calculate overall severity level"""
        if self.stats["violations_found"] == 0:
            return "PASS"
        elif self.stats["violations_found"] < 10:
            return "LOW"
        elif self.stats["violations_found"] < 50:
            return "MEDIUM"
        else:
            return "HIGH"
            
    def print_summary(self, report: Dict) -> None:
        """Print human-readable summary"""
        print("\n" + "="*80)
        print("STRICT VALIDATOR REPORT")
        print("="*80)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Directory: {report['root_directory']}")
        print("\nSTATISTICS:")
        for key, value in report['statistics'].items():
            print(f"  {key}: {value}")
        print(f"\nSEVERITY: {report['summary']['severity']}")
        print(f"COMPLIANT: {'YES' if report['summary']['compliant'] else 'NO'}")
        
        if report['violations']:
            print("\nTOP VIOLATIONS:")
            # Group by type
            by_type = {}
            for v in report['violations']:
                by_type.setdefault(v['type'], []).append(v)
                
            for vtype, violations in by_type.items():
                print(f"\n{vtype.upper()} ({len(violations)} violations):")
                for v in violations[:5]:  # Show first 5
                    print(f"  {v['file']}:{v['line']} - {v['message']}")
                if len(violations) > 5:
                    print(f"  ... and {len(violations) - 5} more")
                    
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Validate codebase for FAIL LOUD compliance")
    parser.add_argument("path", nargs="?", default=".",
                       help="Path to scan (default: current directory)")
    parser.add_argument("--json", action="store_true",
                       help="Output JSON report")
    parser.add_argument("--output", "-o", 
                       help="Save report to file")
    parser.add_argument("--exclude", nargs="+", default=[],
                       help="Additional directories to exclude")
    
    args = parser.parse_args()
    
    # Validate path
    root_path = Path(args.path).resolve()
    if not root_path.exists():
        print(f"Error: Path '{root_path}' does not exist")
        sys.exit(1)
        
    # Create validator
    validator = StrictValidator(root_path)
    
    # Run scan
    print(f"Scanning {root_path}...")
    exclude_dirs = set(['venv', '__pycache__', '.git', 'build', 'dist', '.pytest_cache'])
    if args.exclude:
        exclude_dirs.update(args.exclude)
    
    if root_path.is_file():
        validator.scan_file(root_path)
    else:
        validator.scan_directory(root_path, exclude_dirs)
    
    # Generate report
    report = validator.generate_report()
    
    # Output results
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        validator.print_summary(report)
        
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")
        
    # Exit with appropriate code
    sys.exit(0 if report['summary']['compliant'] else 1)


if __name__ == "__main__":
    main()