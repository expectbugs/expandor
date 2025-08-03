"""
Comprehensive integration tests for the configuration system
Tests the complete configuration flow including:
- ConfigurationManager singleton behavior
- Configuration loading and merging
- FAIL LOUD philosophy
- Environment variable overrides
- User configuration
- Configuration validation
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path
import shutil

from expandor.core.configuration_manager import ConfigurationManager
from expandor.utils.path_resolver import PathResolver
from expandor.core.config import ExpandorConfig


class TestConfigurationSystem:
    """Comprehensive tests for the entire configuration system"""
    
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset ConfigurationManager singleton before each test"""
        ConfigurationManager._instance = None
        ConfigurationManager._initialized = False
        # Clear any env vars
        for key in list(os.environ.keys()):
            if key.startswith('EXPANDOR_'):
                del os.environ[key]
        yield
        # Cleanup after test
        ConfigurationManager._instance = None
        ConfigurationManager._initialized = False
    
    def test_complete_configuration_flow(self):
        """Test the complete configuration loading and merging flow"""
        cm = ConfigurationManager()
        
        # Test that master_defaults.yaml is loaded
        assert cm.get_value('version') is not None
        
        # Test nested configuration access
        assert cm.get_value('vram.safety_factor') == 0.9
        assert cm.get_value('output.formats.png.compression') == 0
        
        # Test strategy configurations
        strategies = cm.get_value('strategies')
        assert 'progressive_outpaint' in strategies
        assert 'tiled_expansion' in strategies
    
    def test_fail_loud_philosophy(self):
        """Test FAIL LOUD behavior throughout the system"""
        cm = ConfigurationManager()
        
        # Missing keys should fail loud
        with pytest.raises(ValueError) as exc_info:
            cm.get_value('this.key.does.not.exist')
        
        error_msg = str(exc_info.value)
        assert 'not found' in error_msg
        assert 'Solutions:' in error_msg
        assert 'Add \'this.key.does.not.exist\' to your config files' in error_msg
    
    def test_environment_override_system(self):
        """Test environment variable override functionality"""
        # Set various env overrides
        os.environ['EXPANDOR_VRAM_SAFETY_FACTOR'] = '0.95'
        os.environ['EXPANDOR_OUTPUT_FORMATS_PNG_COMPRESSION'] = '3'
        os.environ['EXPANDOR_STRATEGIES_PROGRESSIVE_OUTPAINT_BASE_STEPS'] = '100'
        
        # Force reload
        ConfigurationManager._initialized = False
        cm = ConfigurationManager()
        
        # Check overrides applied
        assert cm.get_value('vram.safety_factor') == 0.95
        assert cm.get_value('output.formats.png.compression') == 3
        assert cm.get_value('strategies.progressive_outpaint.base_steps') == 100
    
    def test_user_configuration_loading(self):
        """Test user configuration file loading and merging"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a user config file
            user_config = {
                'version': '2.0',
                'quality_presets': {
                    'custom': {
                        'generation': {
                            'denoising_strength': 0.5,
                            'guidance_scale': 10.0,
                            'num_inference_steps': 25
                        }
                    }
                },
                'paths': {
                    'output_dir': '/custom/output/path'
                }
            }
            
            config_path = Path(tmpdir) / 'user_config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(user_config, f)
            
            # Set path to user config
            os.environ['EXPANDOR_USER_CONFIG_PATH'] = str(config_path)
            
            # Force reload
            ConfigurationManager._initialized = False
            cm = ConfigurationManager()
            
            # Check user config is loaded and merged
            assert cm.get_value('quality_presets.custom.generation.denoising_strength') == 0.5
            assert cm.get_value('paths.output_dir') == '/custom/output/path'
            
            # Check system defaults still exist
            assert cm.get_value('vram.safety_factor') == 0.9  # From master_defaults
    
    def test_expandor_config_integration(self):
        """Test ExpandorConfig integration with ConfigurationManager"""
        # Create a minimal config
        config = ExpandorConfig(
            source_image=Path('/tmp/test.png'),
            target_resolution=(1920, 1080),
            quality_preset='balanced'
        )
        
        # Check that defaults are loaded from ConfigurationManager
        assert config.denoising_strength is not None
        assert config.guidance_scale is not None
        assert config.num_inference_steps is not None
        assert config.save_stages is not None
    
    def test_path_resolver_integration(self):
        """Test PathResolver integration with configuration"""
        resolver = PathResolver()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test path expansion
            test_path = resolver.resolve_path(tmpdir, create=True)
            assert test_path.exists()
            assert test_path.is_dir()
            
            # Test validation
            with pytest.raises(ValueError):
                resolver.resolve_path('/nonexistent/path', create=False)
    
    def test_configuration_validation(self):
        """Test configuration validation and type checking"""
        cm = ConfigurationManager()
        
        # Numeric values should have correct types
        safety_factor = cm.get_value('vram.safety_factor')
        assert isinstance(safety_factor, (int, float))
        assert 0 <= safety_factor <= 1
        
        # Boolean values
        save_intermediate = cm.get_value('processing.save_intermediate_stages')
        assert isinstance(save_intermediate, bool)
        
        # String values
        output_format = cm.get_value('quality_presets.ultra.output.format')
        assert isinstance(output_format, str)
    
    def test_missing_configuration_files(self):
        """Test behavior when configuration files are missing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Point to empty directory
            os.environ['EXPANDOR_CONFIG_DIR'] = tmpdir
            
            # Should fail loud when master_defaults.yaml is missing
            ConfigurationManager._initialized = False
            with pytest.raises(ValueError) as exc_info:
                cm = ConfigurationManager()
            
            assert 'master_defaults.yaml' in str(exc_info.value)
    
    def test_configuration_precedence(self):
        """Test configuration precedence: env > user > system"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create user config
            user_config = {
                'version': '2.0',
                'vram': {
                    'safety_factor': 0.85
                }
            }
            
            config_path = Path(tmpdir) / 'user_config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(user_config, f)
            
            # Set both user config and env override
            os.environ['EXPANDOR_USER_CONFIG_PATH'] = str(config_path)
            os.environ['EXPANDOR_VRAM_SAFETY_FACTOR'] = '0.8'
            
            # Force reload
            ConfigurationManager._initialized = False
            cm = ConfigurationManager()
            
            # Env should override user config
            assert cm.get_value('vram.safety_factor') == 0.8
    
    def test_quality_preset_application(self):
        """Test quality preset application in ExpandorConfig"""
        for preset in ['ultra', 'high', 'balanced', 'fast']:
            config = ExpandorConfig(
                source_image=Path('/tmp/test.png'),
                target_resolution=(1920, 1080),
                quality_preset=preset
            )
            
            # Check preset was applied
            assert config.quality_preset == preset
            assert config.denoising_strength is not None
            assert config.guidance_scale is not None
            assert config.num_inference_steps is not None
    
    def test_strategy_configuration_loading(self):
        """Test strategy-specific configuration loading"""
        cm = ConfigurationManager()
        
        # Test progressive outpaint config
        prog_config = cm.get_strategy_config('progressive_outpaint')
        assert 'base_strength' in prog_config
        assert 'first_step_ratio' in prog_config
        assert 'aspect_ratio_thresholds' in prog_config
        
        # Test tiled expansion config  
        tiled_config = cm.get_strategy_config('tiled_expansion')
        assert 'default_tile_size' in tiled_config
        assert 'overlap' in tiled_config
        
        # Test missing strategy should fail loud
        with pytest.raises(ValueError):
            cm.get_strategy_config('nonexistent_strategy')
    
    @pytest.mark.parametrize("config_key,expected_type", [
        ('vram.safety_factor', float),
        ('vram.thresholds.tiled_processing', int),
        ('processing.save_intermediate_stages', bool),
        ('output.formats.png.compression', int),
        ('strategies.progressive_outpaint.outpaint_prompt_suffix', str),
    ])
    def test_configuration_types(self, config_key, expected_type):
        """Test that configuration values have expected types"""
        cm = ConfigurationManager()
        value = cm.get_value(config_key)
        assert isinstance(value, expected_type), f"{config_key} should be {expected_type}, got {type(value)}"
    
    def test_configuration_completeness(self):
        """Test that all required configuration sections exist"""
        cm = ConfigurationManager()
        
        required_sections = [
            'version',
            'quality_presets',
            'strategies', 
            'processing',
            'output',
            'paths',
            'vram',
            'quality_thresholds',
            'models',
        ]
        
        config = cm._config
        for section in required_sections:
            assert section in config, f"Missing required section: {section}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])