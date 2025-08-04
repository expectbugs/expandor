#!/usr/bin/env python3
import os
import re
from pathlib import Path

violations = []
expandor_dir = Path('expandor')

for py_file in expandor_dir.rglob('*.py'):
    if '__pycache__' in str(py_file):
        continue
    
    with open(py_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Match .get() with default value
        if match := re.search(r'\.get\([\'\"]\w+[\'\"],\s*.+\)', line):
            violations.append({
                'file': str(py_file),
                'line': i + 1,
                'code': line.strip()
            })

print(f'Found {len(violations)} .get() violations:')
for v in violations[:20]:  # Show first 20
    print(f'{v["file"]}:{v["line"]} - {v["code"]}')

# Count by file
from collections import Counter
file_counts = Counter(v['file'] for v in violations)
print("\nTop 10 files with most violations:")
for file, count in file_counts.most_common(10):
    print(f"{file}: {count} violations")