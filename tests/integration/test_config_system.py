"""
Integration tests for Expandor configuration system
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from expandor.config import LoRAConfig, ModelConfig, UserConfig, UserConfigManager
from expandor.config.lora_manager import LoRAConflictError, LoRAManager
from expandor.config.pipeline_config import PipelineConfigurator
from expandor.core.config import ExpandorConfig
from expandor.utils.config_loader import ConfigLoader


class TestUserConfigIntegration:
    """Test user configuration management"""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create config manager with temp directory"""
        with patch("expandor.config.user_config.get_default_config_path") as mock_path:
            mock_path.return_value = temp_config_dir / "config.yaml"
            return UserConfigManager()

    def test_config_lifecycle(self, config_manager, temp_config_dir):
        """Test full config lifecycle: create, save, load, update"""
        # Create and save initial config
        config = UserConfig(
            models={
                "sdxl": ModelConfig(
                    model_id="stabilityai/stable-diffusion-xl-base-1.0",
                    dtype="float16",
                    device="cuda",
                )
            },
            default_quality="high",
        )

        config_manager.save(config)

        # Verify file was created
        config_file = temp_config_dir / "config.yaml"
        assert config_file.exists()

        # Load config
        loaded_config = config_manager.load()
        assert loaded_config.default_quality == "high"
        assert "sdxl" in loaded_config.models
        assert loaded_config.models["sdxl"].dtype == "float16"

        # Update config
        loaded_config.default_quality = "ultra"
        loaded_config.models["flux"] = ModelConfig(
            model_id="black-forest-labs/FLUX.1-dev", dtype="float16"
        )

        config_manager.save(loaded_config)

        # Reload and verify updates
        updated_config = config_manager.load()
        assert updated_config.default_quality == "ultra"
        assert "flux" in updated_config.models

    def test_invalid_config_handling(self, config_manager, temp_config_dir):
        """Test handling of invalid configuration data"""
        # Write invalid YAML
        config_file = temp_config_dir / "config.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content: [")

        # Should return default config with warning
        config = config_manager.load()
        assert isinstance(config, UserConfig)
        assert config.default_quality == "balanced"  # Default value

    def test_model_config_validation(self):
        """Test ModelConfig validation"""
        # Valid config
        config = ModelConfig(model_id="test/model", dtype="float16", device="cuda")
        assert config.dtype == "float16"

        # Invalid dtype should be corrected
        config = ModelConfig(model_id="test/model", dtype="invalid", device="cuda")
        assert config.dtype == "float32"  # Default fallback

        # Invalid device should be corrected
        config = ModelConfig(model_id="test/model", dtype="float16", device="invalid")
        assert config.device == "cpu"  # Default fallback

    def test_lora_config_integration(self, config_manager):
        """Test LoRA configuration"""
        config = UserConfig(
            loras=[
                LoRAConfig(
                    name="style_anime",
                    path="/models/loras/anime.safetensors",
                    weight=0.8,
                    type="style",
                ),
                LoRAConfig(
                    name="detail_enhance",
                    path="/models/loras/detail.safetensors",
                    weight=0.6,
                    type="detail",
                ),
            ]
        )

        # Save and reload
        config_manager.save(config)
        loaded = config_manager.load()

        assert len(loaded.loras) == 2
        assert loaded.loras[0].name == "style_anime"
        assert loaded.loras[0].weight == 0.8
        assert loaded.loras[1].type == "detail"

    def test_example_config_creation(self, temp_config_dir):
        """Test example config file creation"""
        example_file = temp_config_dir / "example_config.yaml"

        with patch("expandor.config.user_config.get_default_config_path") as mock_path:
            mock_path.return_value = temp_config_dir / "config.yaml"
            manager = UserConfigManager()
            manager.create_example_config(example_file)

        assert example_file.exists()

        # Load and verify example config
        with open(example_file, "r") as f:
            content = f.read()
            assert "# Expandor User Configuration" in content
            assert "models:" in content
            assert "loras:" in content
            assert "quality:" in content


class TestLoRAManagerIntegration:
    """Test LoRA management and conflict resolution"""

    def test_lora_conflict_detection(self):
        """Test LoRA conflict detection"""
        manager = LoRAManager()

        loras = [
            LoRAConfig(name="style1", type="style", weight=0.8),
            LoRAConfig(name="style2", type="style", weight=0.7),  # Conflict!
        ]

        with pytest.raises(LoRAConflictError) as exc_info:
            manager.check_compatibility(loras)

        assert "Multiple style LoRAs" in str(exc_info.value)

    def test_lora_weight_adjustment(self):
        """Test automatic weight adjustment"""
        manager = LoRAManager()

        loras = [
            LoRAConfig(name="style", type="style", weight=0.9),
            LoRAConfig(name="detail", type="detail", weight=0.8),
            LoRAConfig(name="quality", type="quality", weight=0.7),
        ]

        # Total weight = 2.4, should be scaled down
        adjusted = manager.resolve_lora_stack(loras)

        total_weight = sum(lora.weight for lora in adjusted)
        assert total_weight <= 1.5

        # Check relative weights maintained
        assert adjusted[0].weight > adjusted[1].weight > adjusted[2].weight

    def test_lora_type_detection(self):
        """Test automatic LoRA type detection"""
        manager = LoRAManager()

        # Test various naming patterns
        assert manager.detect_lora_type("anime_style_v2").name == "style"
        assert manager.detect_lora_type("detail_tweaker").name == "detail"
        assert manager.detect_lora_type("portrait_enhance").name == "subject"
        assert manager.detect_lora_type("hq_quality").name == "quality"

        # Unknown type should raise error
        with pytest.raises(LoRAConflictError):
            manager.detect_lora_type("unknown_lora_xyz")

    def test_recommended_inference_steps(self):
        """Test inference step recommendations"""
        manager = LoRAManager()

        # Style LoRA needs more steps
        loras = [LoRAConfig(name="style", type="style", weight=0.8)]
        steps = manager.get_recommended_inference_steps(loras, base_steps=40)
        assert steps > 40

        # Detail LoRA needs even more
        loras = [LoRAConfig(name="detail", type="detail", weight=0.8)]
        steps = manager.get_recommended_inference_steps(loras, base_steps=40)
        assert steps >= 50


class TestPipelineConfiguration:
    """Test pipeline configuration integration"""

    @pytest.fixture
    def pipeline_config(self):
        """Create pipeline configurator"""
        return PipelineConfigurator()

    @pytest.fixture
    def user_config(self):
        """Create sample user config"""
        return UserConfig(
            models={
                "sdxl": ModelConfig(
                    model_id="stabilityai/stable-diffusion-xl-base-1.0",
                    dtype="float16",
                    device="cuda",
                )
            },
            default_quality="high",
            preferences={"save_intermediate": True, "auto_optimize_prompts": False},
        )

    def test_expandor_config_creation(self, pipeline_config, user_config):
        """Test creating Expandor config from user config"""
        expandor_config = pipeline_config.create_expandor_config(
            user_config=user_config,
            target_size=(3840, 2160),
            quality="ultra",  # Override default
        )

        assert isinstance(expandor_config, ExpandorConfig)
        assert expandor_config.target_width == 3840
        assert expandor_config.target_height == 2160
        # Should use override quality, not default
        assert expandor_config.quality_preset == "ultra"

    def test_pipeline_kwargs_generation(self, pipeline_config, user_config):
        """Test pipeline kwargs generation"""
        kwargs = pipeline_config.get_pipeline_kwargs(user_config.models["sdxl"])

        assert "torch_dtype" in kwargs
        assert kwargs["torch_dtype"].__name__ == "float16"
        assert "use_safetensors" in kwargs
        assert kwargs["use_safetensors"] is True

    def test_torch_dtype_mapping(self, pipeline_config):
        """Test dtype string to torch dtype conversion"""
        import torch

        assert pipeline_config._get_torch_dtype("float16") == torch.float16
        assert pipeline_config._get_torch_dtype("float32") == torch.float32
        assert pipeline_config._get_torch_dtype("bfloat16") == torch.bfloat16
        assert pipeline_config._get_torch_dtype("invalid") == torch.float32  # Default

    def test_model_validation(self, pipeline_config, temp_config_dir):
        """Test model path/ID validation"""
        # Create fake local model
        local_model = temp_config_dir / "models" / "test_model"
        local_model.mkdir(parents=True)
        (local_model / "model_index.json").touch()

        models = {
            "local": ModelConfig(model_id=str(local_model)),
            "huggingface": ModelConfig(model_id="runwayml/stable-diffusion-v1-5"),
            "invalid": ModelConfig(model_id="/nonexistent/path"),
        }

        results = pipeline_config.validate_models(models)

        assert results["local"]["valid"] is True
        assert results["local"]["type"] == "local"

        assert results["huggingface"]["valid"] is True
        assert results["huggingface"]["type"] == "huggingface"

        assert results["invalid"]["valid"] is False
        assert "error" in results["invalid"]


class TestConfigLoader:
    """Test configuration file loading"""

    @pytest.fixture
    def config_dir(self):
        """Get actual config directory"""
        return Path(__file__).parent.parent.parent / "expandor" / "config"

    @pytest.fixture
    def loader(self, config_dir):
        """Create config loader"""
        return ConfigLoader(config_dir)

    def test_load_all_configs(self, loader):
        """Test loading all configuration files"""
        config = loader.load_all_configs()

        # Check that key sections are loaded
        assert "strategies" in config
        assert "quality_presets" in config
        assert "vram_strategies" in config
        assert "model_constraints" in config

    def test_strategy_config_loading(self, loader):
        """Test strategy configuration loading"""
        config = loader.load_config_file("strategies.yaml")

        assert "strategies" in config
        strategies = config["strategies"]

        # Check key strategies exist
        assert "direct_upscale" in strategies
        assert "progressive_outpaint" in strategies
        assert "swpo" in strategies

        # Validate strategy structure
        prog_strategy = strategies["progressive_outpaint"]
        assert "description" in prog_strategy
        assert "parameters" in prog_strategy
        assert "vram_requirement" in prog_strategy

    def test_quality_preset_loading(self, loader):
        """Test quality preset loading"""
        presets = loader.load_quality_preset("ultra")

        assert "inference_steps" in presets
        assert presets["inference_steps"] >= 60  # Ultra should have high steps

        # Test fallback for unknown preset
        fallback = loader.load_quality_preset("unknown_preset")
        assert fallback == loader.load_quality_preset("balanced")

    def test_vram_strategy_loading(self, loader):
        """Test VRAM strategy configuration"""
        config = loader.load_config_file("vram_strategies.yaml")

        assert "vram_profiles" in config
        profiles = config["vram_profiles"]

        # Check profile structure
        assert "high_vram" in profiles
        assert "medium_vram" in profiles
        assert "low_vram" in profiles

        high_profile = profiles["high_vram"]
        assert "min_vram" in high_profile
        assert "strategies" in high_profile
        assert "preferred_strategy" in high_profile

    def test_model_constraints_loading(self, loader):
        """Test model constraints configuration"""
        config = loader.load_config_file("model_constraints.yaml")

        assert "model_constraints" in config
        constraints = config["model_constraints"]

        # Check model entries
        assert "sdxl" in constraints
        assert "sd15" in constraints

        sdxl = constraints["sdxl"]
        assert "max_resolution" in sdxl
        assert "optimal_resolution" in sdxl
        assert "min_vram" in sdxl


class TestEndToEndConfiguration:
    """Test complete configuration flow"""

    @pytest.fixture
    def temp_home(self, monkeypatch):
        """Create temporary home directory"""
        temp_dir = tempfile.mkdtemp()
        monkeypatch.setenv("HOME", temp_dir)
        monkeypatch.setenv("USERPROFILE", temp_dir)  # Windows
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_full_configuration_flow(self, temp_home):
        """Test complete configuration setup and usage"""
        # Initialize configuration system
        config_manager = UserConfigManager()

        # Create user config
        user_config = UserConfig(
            models={
                "sdxl": ModelConfig(
                    model_id="stabilityai/stable-diffusion-xl-base-1.0",
                    dtype="float16",
                    device="cuda",
                    custom_pipeline_params={"variant": "fp16"},
                )
            },
            loras=[
                LoRAConfig(
                    name="anime_style",
                    path="./loras/anime.safetensors",
                    weight=0.7,
                    type="style",
                )
            ],
            default_quality="high",
            preferences={"save_metadata": True, "auto_select_strategy": True},
        )

        # Save config
        config_manager.save(user_config)

        # Create pipeline configurator
        pipeline_config = PipelineConfigurator(config_manager)

        # Generate expandor config
        expandor_config = pipeline_config.create_expandor_config(
            user_config=user_config, target_size=(3840, 2160), strategy="progressive"
        )

        assert expandor_config.target_width == 3840
        assert expandor_config.strategy == "progressive"
        assert expandor_config.quality_preset == "high"

        # Test adapter creation (mock)
        with patch(
            "expandor.config.pipeline_config.DiffusersPipelineAdapter"
        ) as mock_adapter:
            adapter = pipeline_config.create_adapter(
                adapter_type="diffusers", model_config=user_config.models["sdxl"]
            )

            mock_adapter.assert_called_once()
            call_kwargs = mock_adapter.call_args[1]
            assert call_kwargs["device"] == "cuda"
