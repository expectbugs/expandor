#!/bin/bash
# Comprehensive code quality validation

set -e  # FAIL LOUD

echo "=== Code Quality Validation ==="

# Install tools
pip install -q flake8 mypy black isort pylint

# 1. Check for duplicate methods
echo -e "\n[1/8] Checking for duplicate methods..."
DUPLICATES=$(grep -r "^def " expandor/ --include="*.py" | \
    sort | uniq -d | wc -l)
[ $DUPLICATES -eq 0 ] && echo "✓ No duplicate methods" || \
    echo "✗ Found $DUPLICATES duplicate methods"

# 2. Check for print statements
echo -e "\n[2/8] Checking for print statements..."
PRINTS=$(grep -r "print(" expandor/ --include="*.py" | \
    grep -v example | grep -v test | wc -l)
[ $PRINTS -eq 0 ] && echo "✓ No print statements" || \
    echo "✗ Found $PRINTS print statements"

# 3. Format with black
echo -e "\n[3/8] Running black formatter..."
black expandor tests examples --check || \
    (echo "✗ Code needs formatting. Run: black expandor tests examples" && false)

# 4. Sort imports
echo -e "\n[4/8] Checking import sorting..."
isort expandor tests examples --check-only || \
    (echo "✗ Imports need sorting. Run: isort expandor tests examples" && false)

# 5. Flake8 linting
echo -e "\n[5/8] Running flake8..."
flake8 expandor --max-line-length=120 \
    --extend-ignore=E203,W503 \
    --exclude=__pycache__,*.pyc,.git,.venv

# 6. Check for TODOs/FIXMEs
echo -e "\n[6/8] Checking for TODOs..."
TODOS=$(grep -r "TODO\|FIXME\|XXX" expandor/ --include="*.py" | wc -l)
echo "Found $TODOS TODO/FIXME comments"
[ $TODOS -gt 0 ] && grep -r "TODO\|FIXME\|XXX" expandor/ --include="*.py" | head -5

# 7. MyPy type checking
echo -e "\n[7/8] Running mypy..."
mypy expandor --ignore-missing-imports --no-strict-optional || \
    echo "⚠️  MyPy found issues (may be ok for dynamic code)"

# 8. Complexity check
echo -e "\n[8/8] Checking code complexity..."
flake8 expandor --select=C901 --max-complexity=15 || \
    echo "⚠️  Some functions are complex (consider refactoring)"

echo -e "\n=== Quality Check Complete ==="
echo "Fix any ✗ marks before release"