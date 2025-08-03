import pytest
import tempfile
import yaml
from pathlib import Path
from expandor.core.configuration_manager import ConfigurationManager


def test_user_config_override():
    """Test that user config overrides system defaults"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create user config
        user_config = {
            "version": "1.0",
            "core": {
                "quality_preset": "ultra",
                "num_inference_steps": 100
            }
        }
        
        config_path = Path(tmpdir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(user_config, f)
        
        # Set env var to point to test config
        import os
        os.environ["EXPANDOR_CONFIG_PATH"] = str(config_path)
        
        # Force reload
        ConfigurationManager._initialized = False
        cm = ConfigurationManager()
        
        # Check overrides applied
        assert cm.get_value("core.quality_preset") == "ultra"
        assert cm.get_value("core.num_inference_steps") == 100