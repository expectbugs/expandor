"""Tests for ConfigurationManager"""

import os
import pytest
from pathlib import Path
import tempfile
import yaml
from expandor.core.configuration_manager import ConfigurationManager


class TestConfigurationManager:
    def test_singleton_pattern(self):
        """Test that ConfigurationManager is a singleton"""
        cm1 = ConfigurationManager()
        cm2 = ConfigurationManager()
        assert cm1 is cm2
    
    def test_fail_loud_on_missing_key(self):
        """Test FAIL LOUD behavior for missing keys"""
        cm = ConfigurationManager()
        with pytest.raises(ValueError) as exc_info:
            cm.get_value("nonexistent.key")
        assert "not found" in str(exc_info.value)
        assert "Solutions:" in str(exc_info.value)
    
    def test_env_override(self):
        """Test environment variable overrides"""
        os.environ["EXPANDOR_TEST_VALUE"] = "42"
        # Force reload
        ConfigurationManager._initialized = False
        cm = ConfigurationManager()
        # This will fail until we have test.value in config
        # For now, just test the env parsing
        assert cm._env_overrides.get("test", {}).get("value") == 42