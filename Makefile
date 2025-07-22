# Expandor Makefile

.PHONY: test test-unit test-integration lint format clean install dev-install

# Default Python
PYTHON := python3

# Test commands
test:
	@echo "Running all tests..."
	@pytest tests/ -v

test-unit:
	@echo "Running unit tests..."
	@pytest tests/unit -v

test-integration:
	@echo "Running integration tests..."
	@pytest tests/integration -v

test-coverage:
	@echo "Running tests with coverage..."
	@pytest tests/ --cov=expandor --cov-report=html --cov-report=term

# Code quality
lint:
	@echo "Running linter..."
	@flake8 expandor tests --max-line-length=120 --ignore=E501,W503

format:
	@echo "Formatting code..."
	@black expandor tests --line-length=120

# Installation
install:
	@echo "Installing Expandor..."
	@$(PYTHON) -m pip install .

dev-install:
	@echo "Installing Expandor in development mode..."
	@$(PYTHON) -m pip install -e .
	@$(PYTHON) -m pip install -r requirements-dev.txt

# Cleanup
clean:
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@rm -rf build dist *.egg-info
	@rm -rf .pytest_cache .coverage htmlcov
	@rm -rf temp/

# Help
help:
	@echo "Available targets:"
	@echo "  test            - Run all tests"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-coverage   - Run tests with coverage report"
	@echo "  lint            - Run code linter"
	@echo "  format          - Format code with black"
	@echo "  install         - Install Expandor"
	@echo "  dev-install     - Install in development mode"
	@echo "  clean           - Clean up temporary files"
	@echo "  help            - Show this help message"