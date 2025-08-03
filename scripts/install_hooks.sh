#!/bin/bash
# Install pre-commit hooks for Expandor

echo "Installing pre-commit hooks for Expandor..."

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "pre-commit is not installed. Installing..."
    pip install pre-commit
fi

# Install the pre-commit hooks
pre-commit install

echo "✅ Pre-commit hooks installed successfully!"
echo ""
echo "The following checks will run on every commit:"
echo "  - Check for hardcoded values"
echo "  - Check for .get() with defaults"
echo "  - Validate configuration files"
echo "  - Code formatting (black, isort)"
echo "  - General file checks"
echo ""
echo "To run hooks manually: pre-commit run --all-files"