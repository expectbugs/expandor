"""
Comprehensive test suite for the configuration system
Tests the NO HARDCODED VALUES and FAIL LOUD principles
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import yaml

from expandor.core.configuration_manager import ConfigurationManager
from expandor.core.config import ExpandorConfig


class TestConfigurationManager:
    """Test the ConfigurationManager singleton"""
    
    def test_singleton_pattern(self):
        """Test that ConfigurationManager is a true singleton"""
        cm1 = ConfigurationManager()
        cm2 = ConfigurationManager()
        assert cm1 is cm2
    
    def test_fail_loud_on_missing_key(self):
        """Test that missing config keys fail loud"""
        cm = ConfigurationManager()
        
        # Should raise ValueError for missing key
        with pytest.raises(ValueError) as exc_info:
            cm.get_value("this.key.does.not.exist")
        
        assert "not found" in str(exc_info.value)
        assert "EXPANDOR_THIS_KEY_DOES_NOT_EXIST" in str(exc_info.value)
    
    def test_master_defaults_loaded(self):
        """Test that master_defaults.yaml is loaded"""
        cm = ConfigurationManager()
        
        # Check some known values from master_defaults.yaml
        assert cm.get_value("version") == "2.0"
        assert cm.get_value("core.default_strategy") == "auto"
        assert cm.get_value("processing.rgb_max_value") == 255.0
    
    def test_adapter_defaults_loaded(self):
        """Test that adapter defaults are properly loaded"""
        cm = ConfigurationManager()
        
        # Check adapter common defaults
        assert cm.get_value("adapters.common.default_width") == 1024
        assert cm.get_value("adapters.common.default_height") == 1024
        assert cm.get_value("adapters.common.default_num_inference_steps") == 50
        assert cm.get_value("adapters.common.default_guidance_scale") == 7.5
        
        # Check adapter-specific defaults
        assert cm.get_value("adapters.a1111.dimension_multiple") == 64
        assert cm.get_value("adapters.comfyui.dimension_multiple") == 8
    
    def test_strategy_config_loaded(self):
        """Test that strategy configurations are loaded"""
        cm = ConfigurationManager()
        
        # Test progressive_outpaint config
        prog_config = cm.get_strategy_config("progressive_outpaint")
        assert prog_config["first_step_ratio"] == 1.4
        assert prog_config["base_strength"] == 0.75
        assert prog_config["max_supported_ratio"] == 8.0
    
    def test_processor_config_loaded(self):
        """Test that processor configurations are loaded"""
        cm = ConfigurationManager()
        
        # Test artifact_detector config
        art_config = cm.get_processor_config("artifact_detector")
        assert "seam_threshold" in art_config
        assert "skip_validation" in art_config
        assert art_config["artifact_mask_dilation"] == 10


class TestExpandorConfig:
    """Test the ExpandorConfig dataclass"""
    
    def test_no_hardcoded_defaults(self):
        """Test that ExpandorConfig has NO hardcoded defaults"""
        # Create config with minimal required fields
        config = ExpandorConfig(
            source_image="test.png",
            target_width=2048
        )
        
        # All optional fields should be None initially
        assert config.num_inference_steps is None
        assert config.guidance_scale is None
        assert config.denoising_strength is None
        assert config.quality_preset is None
    
    def test_config_loads_from_master_defaults(self):
        """Test that config loads defaults from master_defaults.yaml"""
        config = ExpandorConfig(
            source_image="test.png",
            target_width=2048
        )
        
        # After __post_init__, defaults should be loaded
        assert config.strategy == "auto"  # From core.default_strategy
        assert config.output_format == "png"  # From output.default_format
        assert config.batch_size == 1  # From processing.batch_size
    
    def test_quality_preset_application(self):
        """Test that quality presets are properly applied"""
        config = ExpandorConfig(
            source_image="test.png",
            target_width=2048,
            quality_preset="ultra"
        )
        
        # Ultra preset values should be applied
        assert config.num_inference_steps == 50
        assert config.guidance_scale == 8.0
        assert config.enable_artifacts_check == True


class TestAdapterDefaults:
    """Test that adapters use ConfigurationManager for defaults"""
    
    @patch('expandor.adapters.diffusers_adapter.ConfigurationManager')
    def test_diffusers_adapter_no_hardcoded_defaults(self, mock_cm):
        """Test DiffusersPipelineAdapter has no hardcoded defaults"""
        from expandor.adapters.diffusers_adapter import DiffusersPipelineAdapter
        
        # Mock the ConfigurationManager
        mock_instance = Mock()
        mock_cm.return_value = mock_instance
        mock_instance.get_value.side_effect = lambda key: {
            "adapters.common.default_width": 1024,
            "adapters.common.default_height": 1024,
            "adapters.common.default_num_inference_steps": 50,
            "adapters.common.default_guidance_scale": 7.5,
            "adapters.common.default_negative_prompt": None
        }[key]
        
        # Create adapter
        adapter = DiffusersPipelineAdapter()
        
        # Call generate without optional parameters
        with patch.object(adapter, 'base_pipeline') as mock_pipeline:
            mock_pipeline.return_value.images = [Mock()]
            adapter.generate("test prompt")
            
            # Verify ConfigurationManager was called for defaults
            assert mock_instance.get_value.called
            assert mock_instance.get_value.call_count >= 4  # At least 4 defaults


class TestHardcodedValueScanner:
    """Test for scanning hardcoded values in the codebase"""
    
    def test_no_function_defaults_in_adapters(self):
        """Verify no function parameter defaults in adapter generate/inpaint/img2img methods"""
        import ast
        import glob
        
        adapter_files = glob.glob("expandor/adapters/*_adapter.py")
        
        for filepath in adapter_files:
            with open(filepath, 'r') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check specific method names
                    if node.name in ['generate', 'inpaint', 'img2img', 'enhance']:
                        for arg in node.args.args[1:]:  # Skip 'self'
                            if arg.arg in ['width', 'height', 'num_inference_steps', 
                                         'guidance_scale', 'strength', 'scale_factor']:
                                # These should NOT have defaults
                                idx = node.args.args.index(arg) - 1  # Adjust for self
                                if idx < len(node.args.defaults):
                                    default = node.args.defaults[idx]
                                    if default is not None:
                                        # Allow None as default, but not numeric values
                                        if isinstance(default, ast.Constant) and default.value is not None:
                                            pytest.fail(
                                                f"Hardcoded default found in {filepath}:"
                                                f"{node.name}() parameter '{arg.arg}' "
                                                f"has default value {default.value}"
                                            )


class TestFailLoudPhilosophy:
    """Test that the FAIL LOUD philosophy is implemented"""
    
    def test_config_manager_fails_loud(self):
        """Test that ConfigurationManager fails loud on errors"""
        cm = ConfigurationManager()
        
        # Test missing strategy config
        with pytest.raises(ValueError) as exc_info:
            cm.get_strategy_config("nonexistent_strategy")
        assert "No configuration found for strategy" in str(exc_info.value)
        
        # Test missing processor config
        with pytest.raises(ValueError) as exc_info:
            cm.get_processor_config("nonexistent_processor")
        assert "No configuration found for processor" in str(exc_info.value)
    
    def test_schema_validation_fails_loud(self):
        """Test that schema validation fails loud"""
        # This would require mocking the schema validation
        # to test failure cases
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])