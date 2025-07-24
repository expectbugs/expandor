"""
pytest configuration for integration tests
"""

import logging
from pathlib import Path

import pytest


def pytest_configure(config):
    """Configure pytest for integration tests"""
    # Add custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "quality: marks quality validation tests")


@pytest.fixture(scope="session")
def integration_test_dir():
    """Create session-wide test directory"""
    test_dir = Path("test_outputs")
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
