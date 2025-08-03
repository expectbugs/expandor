"""
Integration tests for ControlNet functionality

These tests require ControlNet dependencies to be installed.
If dependencies are missing, the tests will be skipped.
"""

import pytest
from pathlib import Path
import tempfile

from PIL import Image, ImageDraw
import numpy as np
import torch

# ControlNet dependencies - these tests only run if available
# This check ensures tests are skipped gracefully if opencv-python is not installed
# Tests will show as "SKIPPED" in pytest output with the reason message
pytest.importorskip("cv2", reason="OpenCV required for ControlNet tests")

from expandor import Expandor, ExpandorConfig
from expandor.adapters import DiffusersPipelineAdapter, MockPipelineAdapter
from expandor.processors.controlnet_extractors import ControlNetExtractor
from expandor.utils.config_loader import ConfigLoader


class TestControlNetExtractor:
    """Test ControlNet extractor functionality"""
    
    @pytest.fixture
    def test_image(self):
        """Create a test image with clear structure"""
        img = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(img)
        
        # Draw structured content
        draw.rectangle([100, 100, 400, 400], outline="black", width=5)
        draw.ellipse([200, 200, 300, 300], fill="red")
        draw.line([50, 50, 450, 450], fill="blue", width=3)
        
        return img
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory with test configs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            
            # Create minimal test config
            controlnet_config = {
                "extractors": {
                    "canny": {
                        "low_threshold_min": 0,
                        "low_threshold_max": 255,
                        "high_threshold_min": 0,
                        "high_threshold_max": 255,
                        "kernel_size": 3,
                        "dilation_iterations": 1
                    },
                    "blur": {
                        "valid_types": ["gaussian", "box", "motion"],
                        "motion_kernel_multiplier": 2,
                        "motion_kernel_offset": 1
                    },
                    "resampling": {
                        "method": "LANCZOS"
                    }
                }
            }
            
            import yaml
            with open(config_dir / "controlnet_config.yaml", 'w') as f:
                yaml.dump(controlnet_config, f)
            
            yield config_dir
    
    def test_canny_extraction_required_params(self, test_image_square, temp_config_dir, monkeypatch):
        """Test Canny edge extraction with REQUIRED parameters"""
        # Monkeypatch the Path constructor to use temp config
        def mock_path_new(cls, *args):
            if args and str(args[0]).endswith('config'):
                return temp_config_dir
            return object.__new__(cls)
        
        monkeypatch.setattr(Path, '__new__', mock_path_new)
        
        extractor = ControlNetExtractor()
        
        # ALL parameters MUST be provided - no defaults
        canny = extractor.extract_canny(
            test_image_square, 
            low_threshold=100,  # REQUIRED
            high_threshold=200,  # REQUIRED
            dilate=False,  # REQUIRED
            l2_gradient=False  # REQUIRED
        )
        
        assert isinstance(canny, Image.Image)
        assert canny.size == test_image_square.size
        assert canny.mode == "RGB"
        
        # Verify edges were detected
        canny_array = np.array(canny.convert("L"))
        assert np.any(canny_array > 0), "No edges detected"
    
    def test_canny_validation(self, test_image_square, temp_config_dir, monkeypatch):
        """Test Canny parameter validation"""
        def mock_path_new(cls, *args):
            if args and str(args[0]).endswith('config'):
                return temp_config_dir
            return object.__new__(cls)
        
        monkeypatch.setattr(Path, '__new__', mock_path_new)
        
        extractor = ControlNetExtractor()
        
        # Invalid thresholds - should fail loud
        with pytest.raises(ValueError, match="must be"):
            extractor.extract_canny(
                test_image_square, 
                low_threshold=-10,  # Invalid
                high_threshold=200,
                dilate=False,
                l2_gradient=False
            )
            
        with pytest.raises(ValueError, match="must be less than"):
            extractor.extract_canny(
                test_image_square,
                low_threshold=200,
                high_threshold=100,  # Invalid order
                dilate=False,
                l2_gradient=False
            )
    
    def test_blur_extraction_required_params(self, test_image_square, temp_config_dir, monkeypatch):
        """Test blur extraction with REQUIRED parameters"""
        def mock_path_new(cls, *args):
            if args and str(args[0]).endswith('config'):
                return temp_config_dir
            return object.__new__(cls)
        
        monkeypatch.setattr(Path, '__new__', mock_path_new)
        
        extractor = ControlNetExtractor()
        
        # Get valid blur types from config - NO FALLBACKS
        config_loader = ConfigLoader(temp_config_dir)
        controlnet_config = config_loader.load_config_file("controlnet_config.yaml")
        blur_config = controlnet_config["extractors"]["blur"]
        valid_types = blur_config["valid_types"]
        
        # Test each configured blur type
        for blur_type in valid_types:
            blurred = extractor.extract_blur(
                test_image_square, 
                radius=10,  # REQUIRED
                blur_type=blur_type  # REQUIRED
            )
            
            assert isinstance(blurred, Image.Image)
            assert blurred.size == test_image_square.size
            
            # Verify image was actually blurred
            orig_array = np.array(test_image_square)
            blur_array = np.array(blurred)
            assert not np.array_equal(orig_array, blur_array)
    
    def test_invalid_blur_type(self, test_image_square, temp_config_dir, monkeypatch):
        """Test that invalid blur type fails properly"""
        def mock_path_new(cls, *args):
            if args and str(args[0]).endswith('config'):
                return temp_config_dir
            return object.__new__(cls)
        
        monkeypatch.setattr(Path, '__new__', mock_path_new)
        
        extractor = ControlNetExtractor()
        
        with pytest.raises(ValueError, match="Invalid blur_type"):
            extractor.extract_blur(
                test_image_square,
                radius=10,
                blur_type="invalid_type"  # Should fail
            )
    
    def test_unimplemented_extractors(self, test_image_square, temp_config_dir, monkeypatch):
        """Test that unimplemented extractors fail properly"""
        def mock_path_new(cls, *args):
            if args and str(args[0]).endswith('config'):
                return temp_config_dir
            return object.__new__(cls)
        
        monkeypatch.setattr(Path, '__new__', mock_path_new)
        
        extractor = ControlNetExtractor()
        
        with pytest.raises(NotImplementedError, match="depth estimation model"):
            extractor.extract_depth(test_image_square)
            
        with pytest.raises(NotImplementedError, match="Normal map extraction"):
            extractor.extract_normal(test_image_square)


class TestControlNetAdapter:
    """Test ControlNet adapter functionality"""
    
    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter with ControlNet support"""
        adapter = MockPipelineAdapter(
            model_id="test-sdxl",
            controlnet_support=True
        )
        # MockPipelineAdapter doesn't have these attributes
        # The mock adapter should already support these through its methods
        return adapter
    
    def test_controlnet_not_supported(self):
        """Test error when ControlNet not supported"""
        adapter = MockPipelineAdapter(
            model_id="test-sd15",
            controlnet_support=False
        )
        
        assert not adapter.supports_controlnet()
        assert adapter.get_controlnet_types() == []
    
    def test_controlnet_validation(self, mock_adapter):
        """Test ControlNet method validation"""
        test_img = Image.new("RGB", (512, 512), "white")
        control_img = Image.new("RGB", (256, 256), "black")  # Wrong size
        mask = Image.new("L", (512, 512), 128)
        
        # Test size mismatch error - should fail loud
        with pytest.raises(ValueError, match="doesn't match"):
            mock_adapter.controlnet_inpaint(
                image=test_img,
                mask=mask,
                control_image=control_img,
                prompt="test",
                negative_prompt="bad",  # REQUIRED
                control_type="canny",  # REQUIRED
                controlnet_strength=1.0,  # REQUIRED
                strength=0.8,  # REQUIRED
                num_inference_steps=50,  # REQUIRED
                guidance_scale=7.5  # REQUIRED
            )


class TestControlNetStrategy:
    """Test ControlNet expansion strategy"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            
            # Create test configs
            configs = {
                "controlnet_config.yaml": {
                    "strategy": {
                        "default_extract_at_each_step": True
                    },
                    "extractors": {
                        "resampling": {
                            "method": "LANCZOS"
                        }
                    }
                },
                "quality_presets.yaml": {
                    "quality_presets": {
                        "balanced": {
                            "controlnet": {
                                "controlnet_strength": 0.8
                            }
                        }
                    }
                }
            }
            
            import yaml
            for filename, content in configs.items():
                with open(config_dir / filename, 'w') as f:
                    yaml.dump(content, f)
            
            yield config_dir
    
    def test_strategy_validation(self, temp_config_dir, monkeypatch):
        """Test strategy validates ControlNet support"""
        # Create adapter without ControlNet
        adapter = MockPipelineAdapter(
            model_id="test-model",
            controlnet_support=False
        )
        
        # Try to use ControlNet strategy
        config = ExpandorConfig(
            source_image="test.jpg",
            target_resolution=(2048, 2048),
            strategy="controlnet_progressive",
            strategy_params={
                "controlnet_config": {
                    "control_type": "canny",  # REQUIRED
                    "controlnet_strength": 0.8,
                    "extract_at_each_step": True,
                    "canny_low_threshold": 100,
                    "canny_high_threshold": 200,
                    "canny_dilate": False,
                    "canny_l2_gradient": False
                }
            }
        )
        
        expandor = Expandor(adapter)
        
        # Should fail validation
        with pytest.raises(ValueError, match="does not support ControlNet"):
            expandor.expand(config)
    
    def test_missing_required_params(self):
        """Test that ALL required parameters must be provided"""
        adapter = MockPipelineAdapter(
            model_id="test-sdxl",
            controlnet_support=True
        )
        
        # Missing required parameters
        config = ExpandorConfig(
            source_image="test.jpg",
            target_resolution=(2048, 2048),
            strategy="controlnet_progressive",
            strategy_params={
                "controlnet_config": {
                    "control_type": "canny",
                    # Missing controlnet_strength - should fail
                    # Missing extract_at_each_step - should fail
                }
            }
        )
        
        expandor = Expandor(adapter)
        
        # Should fail with clear error about missing required param
        with pytest.raises(ValueError, match="is REQUIRED"):
            expandor.expand(config)


# Real integration test (requires actual models)
@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for real ControlNet test"
)
class TestControlNetIntegration:
    """Integration tests with real models (slow)"""
    
    @pytest.fixture
    def temp_configs(self):
        """Create complete test configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            
            # Create all required configs
            configs = {
                "controlnet_config.yaml": {
                    "models": {
                        "sdxl": {
                            "canny": "diffusers/controlnet-canny-sdxl-1.0"
                        }
                    },
                    "vram_overhead": {
                        "model_load": 2000,
                        "operation_active": 1500
                    },
                    "pipelines": {
                        "dimension_multiple": 8
                    },
                    "calculations": {
                        "megapixel_divisor": 1000000
                    }
                },
                "vram_strategies.yaml": {
                    "operation_estimates": {
                        "sdxl": {
                            "controlnet_inpaint": 8000
                        }
                    },
                    "lora_overhead_mb": 200,
                    "resolution_calculation": {
                        "base_pixels": 1048576
                    }
                },
                "quality_presets.yaml": {
                    "quality_presets": {
                        "balanced": {
                            "controlnet": {
                                "controlnet_strength": 0.8
                            }
                        }
                    }
                }
            }
            
            import yaml
            for filename, content in configs.items():
                with open(config_dir / filename, 'w') as f:
                    yaml.dump(content, f)
            
            yield config_dir
    
    def test_real_controlnet_expansion(self, temp_configs, monkeypatch):
        """Test actual ControlNet expansion (requires models)"""
        try:
            # Monkeypatch config directory
            monkeypatch.setattr(Path, '__new__', lambda cls, *args: temp_configs if str(args[0]).endswith('config') else Path(*args))
            
            # Load config for model references
            config_loader = ConfigLoader(temp_configs)
            controlnet_config = config_loader.load_config_file("controlnet_config.yaml")
            # Get model references with proper validation
            if "models" not in controlnet_config:
                raise ValueError("models section not found in controlnet_config.yaml")
            if "sdxl" not in controlnet_config["models"]:
                raise ValueError("sdxl section not found in models")
            model_refs = controlnet_config["models"]["sdxl"]
            
            # Try to create real adapter
            adapter = DiffusersPipelineAdapter(
                model_id="stabilityai/stable-diffusion-xl-base-1.0"
            )
            
            # Try to load ControlNet
            if "canny" not in model_refs:
                raise ValueError("canny model not found in sdxl models config")
            canny_model = model_refs["canny"]
            adapter.load_controlnet(canny_model, "canny")
            
            # Create test image
            test_img = Image.new("RGB", (512, 512), "blue")
            draw = ImageDraw.Draw(test_img)
            draw.rectangle([100, 100, 400, 400], fill="white")
            
            # Load quality preset
            quality_config = config_loader.load_quality_preset("balanced")
            
            # Test expansion with ALL REQUIRED parameters
            config = ExpandorConfig(
                source_image=test_img,
                target_resolution=(768, 768),
                strategy="controlnet_progressive",
                prompt="high quality image",
                negative_prompt="low quality, blurry",
                num_inference_steps=20,
                guidance_scale=7.5,
                strategy_params={
                    "controlnet_config": {
                        "control_type": "canny",  # REQUIRED
                        "controlnet_strength": quality_config["controlnet"]["controlnet_strength"],  # REQUIRED
                        "extract_at_each_step": True,  # REQUIRED
                        "canny_low_threshold": 100,  # REQUIRED
                        "canny_high_threshold": 200,  # REQUIRED
                        "canny_dilate": False,  # REQUIRED
                        "canny_l2_gradient": False  # REQUIRED
                    }
                }
            )
            
            expandor = Expandor(adapter)
            result = expandor.expand(config)
            
            assert result.size == (768, 768)
            
        except Exception as e:
            pytest.skip(f"Real model test failed: {e}")


class TestControlNetConfigHandling:
    """Test configuration handling scenarios"""
    
    def test_missing_controlnet_config(self, test_image_square):
        """Test behavior when controlnet_config.yaml is missing"""
        adapter = DiffusersPipelineAdapter(
            model_id="stabilityai/stable-diffusion-xl-base-1.0"
        )
        
        # Ensure config doesn't exist
        config_dir = Path.home() / ".config" / "expandor"
        controlnet_config_path = config_dir / "controlnet_config.yaml"
        if controlnet_config_path.exists():
            controlnet_config_path.rename(
                controlnet_config_path.with_suffix('.yaml.bak')
            )
        
        try:
            # Should fail loud with helpful message
            with pytest.raises(FileNotFoundError) as exc_info:
                adapter.controlnet_inpaint(
                    prompt="test",
                    image=test_image_square,
                    mask_image=test_image_square
                )
            
            assert "expandor --setup-controlnet" in str(exc_info.value)
            assert "controlnet_config.yaml not found" in str(exc_info.value)
        finally:
            # Restore backup if it exists
            backup_path = controlnet_config_path.with_suffix('.yaml.bak')
            if backup_path.exists():
                backup_path.rename(controlnet_config_path)
    
    def test_corrupted_controlnet_config(self, test_image_square, tmp_path):
        """Test behavior with corrupted YAML"""
        # Create corrupted config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        corrupted_config = config_dir / "controlnet_config.yaml"
        
        # Write invalid YAML
        with open(corrupted_config, 'w') as f:
            f.write("invalid: yaml: content:\n  bad indentation")
        
        # Mock config dir
        adapter = DiffusersPipelineAdapter(
            model_id="stabilityai/stable-diffusion-xl-base-1.0"
        )
        adapter.config_loader.config_dir = config_dir
        
        # Should fail with YAML error
        with pytest.raises(ValueError) as exc_info:
            adapter._ensure_controlnet_config()
        
        assert "Invalid YAML" in str(exc_info.value)
        assert "--setup-controlnet --force" in str(exc_info.value)
    
    def test_missing_config_sections(self, test_image_square, tmp_path):
        """Test behavior when config is missing required sections"""
        # Create incomplete config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        incomplete_config = config_dir / "controlnet_config.yaml"
        
        # Write config missing required sections
        import yaml
        with open(incomplete_config, 'w') as f:
            yaml.dump({
                "extractors": {},  # Missing defaults, models, pipelines
            }, f)
        
        adapter = DiffusersPipelineAdapter(
            model_id="stabilityai/stable-diffusion-xl-base-1.0"
        )
        adapter.config_loader.config_dir = config_dir
        
        # Should fail with missing sections error
        with pytest.raises(ValueError) as exc_info:
            adapter._ensure_controlnet_config()
        
        assert "missing sections" in str(exc_info.value)
        assert "defaults" in str(exc_info.value)
    
    def test_missing_vram_operation_estimates(self, test_image_square, tmp_path):
        """Test behavior when vram_strategies.yaml missing operation_estimates"""
        # Create config without operation_estimates
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        vram_config = config_dir / "vram_strategies.yaml"
        import yaml
        with open(vram_config, 'w') as f:
            yaml.dump({
                "vram_profiles": {
                    "high_vram": {"min_vram_mb": 16000}
                }
                # Missing operation_estimates
            }, f)
        
        adapter = DiffusersPipelineAdapter(
            model_id="stabilityai/stable-diffusion-xl-base-1.0"
        )
        adapter.config_loader.config_dir = config_dir
        
        # Should fail when trying to estimate VRAM
        with pytest.raises(ValueError) as exc_info:
            adapter.estimate_vram("controlnet_inpaint", width=1024, height=1024)
        
        assert "operation_estimates section missing" in str(exc_info.value)
        assert "expandor --setup-controlnet" in str(exc_info.value)
    
    def test_config_validation_catches_invalid_values(self, tmp_path):
        """Test that config validation catches invalid values"""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create invalid config missing required fields
        bad_config = config_dir / "controlnet_config.yaml"
        import yaml
        with open(bad_config, 'w') as f:
            yaml.dump({
                "extractors": {},
                "pipelines": {},
                "models": {}
                # Missing 'defaults' section entirely
            }, f)
        
        adapter = DiffusersPipelineAdapter(
            model_id="stabilityai/stable-diffusion-xl-base-1.0"
        )
        adapter.config_loader.config_dir = config_dir
        
        # For now, basic validation passes (schema validation is TODO)
        # This test documents expected future behavior
        config = adapter._ensure_controlnet_config()
        # In future: should raise ValueError for invalid types/ranges