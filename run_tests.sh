#!/bin/bash
# Run all Expandor tests

set -e

# Activate virtual environment if needed
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Create test output directory
mkdir -p test_output

echo "Running Expandor test suite..."
echo "=============================="

# Run unit tests
echo -e "\n📋 Running unit tests..."
pytest tests/unit -v --tb=short

# Run integration tests
echo -e "\n🔗 Running integration tests..."
pytest tests/integration -v --tb=short

# Run all tests with coverage if requested
if [ "$1" == "--coverage" ]; then
    echo -e "\n📊 Running tests with coverage..."
    pytest tests/ --cov=expandor --cov-report=html --cov-report=term
    echo "Coverage report generated in htmlcov/"
fi

echo -e "\n✅ All tests completed!"