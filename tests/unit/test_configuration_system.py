"""
Comprehensive tests for the configuration system
Ensures NO HARDCODED VALUES and FAIL LOUD principles
"""

import pytest
import sys
from pathlib import Path
import tempfile
import yaml
from unittest.mock import patch

from expandor.core.configuration_manager import ConfigurationManager
from expandor.core.exceptions import ExpandorError


class TestConfigurationSystem:
    """Test the configuration system comprehensively"""
    
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset ConfigurationManager singleton before each test"""
        ConfigurationManager._instance = None
        ConfigurationManager._initialized = False
        yield
        ConfigurationManager._instance = None
        ConfigurationManager._initialized = False
    
    def test_fail_loud_on_missing_key(self):
        """Test that missing keys fail loudly"""
        config_manager = ConfigurationManager()
        
        # Should raise ValueError for missing key
        with pytest.raises(ValueError) as exc_info:
            config_manager.get_value("non.existent.key")
        
        assert "Configuration key 'non.existent.key' not found!" in str(exc_info.value)
        assert "Solutions:" in str(exc_info.value)
    
    def test_no_silent_defaults(self):
        """Test that .get() with defaults is not used"""
        config_manager = ConfigurationManager()
        
        # get_value should never return a default silently
        with pytest.raises(ValueError):
            # This should fail, not return a default
            config_manager.get_value("missing.key.with.no.default")
    
    def test_all_required_sections_exist(self):
        """Test that all required configuration sections exist"""
        config_manager = ConfigurationManager()
        
        required_sections = [
            'quality_presets',
            'strategies', 
            'processors',
            'vram',
            'paths',
            'output',
            'memory',
            'adapters'
        ]
        
        for section in required_sections:
            # Should not raise - section should exist
            config = config_manager.get_value(section)
            assert config is not None
            assert isinstance(config, dict)
    
    def test_no_hardcoded_adapter_defaults(self):
        """Test that adapters don't have hardcoded defaults"""
        config_manager = ConfigurationManager()
        
        # Check adapter defaults exist in config
        assert config_manager.get_value("adapters.common.default_width") == 1024
        assert config_manager.get_value("adapters.common.default_height") == 1024
        assert config_manager.get_value("adapters.common.default_num_inference_steps") == 50
        assert config_manager.get_value("adapters.common.default_guidance_scale") == 7.5
    
    def test_processor_configs_loaded(self):
        """Test that all processor configs are properly loaded"""
        config_manager = ConfigurationManager()
        
        processors = [
            'artifact_detector',
            'seam_repair',
            'quality_validator',
            'boundary_analysis',
            'edge_analysis'
        ]
        
        for processor in processors:
            config = config_manager.get_processor_config(processor)
            assert config is not None
            assert isinstance(config, dict)
    
    def test_strategy_configs_loaded(self):
        """Test that all strategy configs are properly loaded"""
        config_manager = ConfigurationManager()
        
        strategies = [
            'progressive_outpaint',
            'swpo',
            'tiled_expansion',
            'cpu_offload',
            'direct_upscale'
        ]
        
        for strategy in strategies:
            config = config_manager.get_strategy_config(strategy)
            assert config is not None
            assert isinstance(config, dict)
    
    def test_configuration_hierarchy(self):
        """Test configuration hierarchy and overrides"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a user config
            user_config = {
                'version': '2.0',
                'quality_presets': {
                    'custom': {
                        'generation': {
                            'num_inference_steps': 100  # Override default
                        }
                    }
                }
            }
            
            user_config_path = Path(tmpdir) / 'config.yaml'
            with open(user_config_path, 'w') as f:
                yaml.dump(user_config, f)
            
            # Test with user config
            with patch('expandor.core.configuration_manager.ConfigurationManager._find_user_config', 
                      return_value=user_config_path):
                config_manager = ConfigurationManager()
                
                # Should get overridden value
                steps = config_manager.get_value('quality_presets.custom.generation.num_inference_steps')
                assert steps == 100
    
    def test_environment_variable_override(self):
        """Test environment variable overrides"""
        with patch.dict('os.environ', {'EXPANDOR_PROCESSING_BATCH_SIZE': '8'}):
            # Reset singleton to pick up env var
            ConfigurationManager._instance = None
            ConfigurationManager._initialized = False
            
            config_manager = ConfigurationManager()
            batch_size = config_manager.get_value('processing.batch_size')
            
            # Environment variable should override
            assert batch_size == 8
    
    def test_version_checking(self):
        """Test configuration version validation"""
        config_manager = ConfigurationManager()
        
        # Master config should have correct version
        master_config = config_manager.configs.get('master', {})
        assert master_config.get('version') == '2.0'
    
    def test_path_resolution(self):
        """Test that paths are properly resolved"""
        config_manager = ConfigurationManager()
        
        # Get path configs
        cache_dir = config_manager.get_value('paths.cache_dir')
        output_dir = config_manager.get_value('paths.output_dir')
        
        assert cache_dir is not None
        assert output_dir is not None
        
        # Should be strings that can be converted to paths
        assert isinstance(cache_dir, str)
        assert isinstance(output_dir, str)
    
    def test_no_none_values_in_config(self):
        """Test that configuration doesn't contain None values"""
        config_manager = ConfigurationManager()
        
        def check_no_none(config, path=""):
            """Recursively check for None values"""
            if isinstance(config, dict):
                for key, value in config.items():
                    if value is None:
                        pytest.fail(f"Found None value at {path}.{key}")
                    check_no_none(value, f"{path}.{key}")
            elif isinstance(config, list):
                for i, value in enumerate(config):
                    if value is None:
                        pytest.fail(f"Found None value at {path}[{i}]")
                    check_no_none(value, f"{path}[{i}]")
        
        # Check master config
        master_config = config_manager.configs.get('master', {})
        check_no_none(master_config, "master")
    
    def test_numeric_values_are_numbers(self):
        """Test that numeric configuration values are proper numbers"""
        config_manager = ConfigurationManager()
        
        # Check some known numeric values
        numeric_checks = [
            ('memory.bytes_to_mb_divisor', int),
            ('memory.vram.safety_factors.default', float),
            ('processing.rgb_channels', int),
            ('adapters.common.default_width', int),
            ('adapters.common.default_guidance_scale', float)
        ]
        
        for path, expected_type in numeric_checks:
            value = config_manager.get_value(path)
            assert isinstance(value, expected_type), f"{path} should be {expected_type}, got {type(value)}"