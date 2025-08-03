"""Test FAIL LOUD philosophy compliance"""
import pytest
from expandor.adapters import MockPipelineAdapter

def test_invalid_pipeline_type_fails_loud():
    """Ensure invalid pipeline types fail with clear error"""
    adapter = MockPipelineAdapter()
    with pytest.raises(ValueError, match="Unknown pipeline type"):
        adapter.load_pipeline("invalid_type")

def test_missing_config_fails_loud():
    """Ensure missing config values fail immediately"""
    # Test each component that should fail on missing config
    pass  # Implement based on your components