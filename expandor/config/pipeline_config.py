"""
Pipeline configuration that integrates user config with Expandor
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config.user_config import ModelConfig, UserConfigManager
from ..core.config import ExpandorConfig
from ..utils.logging_utils import setup_logger


class PipelineConfigurator:
    """Configures pipelines based on user settings"""

    def __init__(self, user_config_path: Optional[Path] = None):
        """
        Initialize pipeline configurator

        Args:
            user_config_path: Path to user config file
        """
        self.user_config_manager = UserConfigManager(user_config_path)
        self.logger = setup_logger(__name__)

    def create_expandor_config(
        self,
        source_image: Path,
        target_resolution: Tuple[int, int],
        prompt: str,
        seed: int,
        quality_preset: Optional[str] = None,
        strategy_override: Optional[str] = None,
        save_stages: bool = False,
        stage_dir: Optional[Path] = None,
        verbose: bool = False,
        **kwargs,
    ) -> ExpandorConfig:
        """
        Create ExpandorConfig with user preferences applied

        Args:
            source_image: Source image path (for metadata)
            target_resolution: Target resolution tuple
            prompt: Generation prompt
            seed: Random seed
            quality_preset: Quality preset name (uses user default if None)
            strategy_override: Force specific strategy
            save_stages: Save intermediate stages
            stage_dir: Directory for stages
            verbose: Verbose output
            **kwargs: Additional config parameters

        Returns:
            Configured ExpandorConfig
        """
        user_config = self.user_config_manager.load()

        # Apply user defaults
        if quality_preset is None:
            quality_preset = user_config.default_quality

        # Build config using new format
        config = ExpandorConfig(
            target_width=target_resolution[0],
            target_height=target_resolution[1],
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt"),
            seed=seed,
            quality_preset=quality_preset,
            strategy=strategy_override or "auto",
            save_stages=save_stages or user_config.save_intermediate_stages,
            stage_dir=stage_dir,
            verbose=verbose or user_config.verbose_logging,
            vram_limit_mb=kwargs.get("vram_limit_mb", user_config.max_vram_usage_mb),
            enable_artifacts_check=kwargs.get("artifact_detection_level", "aggressive")
            != "disabled",
            artifact_detection_threshold=(
                0.1 if user_config.auto_artifact_detection else 1.0
            ),
        )

        # Store source image path in metadata for reference
        if hasattr(config, "custom_pipeline_params"):
            config.custom_pipeline_params["source_image_path"] = str(source_image)

        return config

    def get_pipeline_kwargs(self, model_name: str) -> Dict[str, Any]:
        """
        Get pipeline initialization kwargs for a model

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of pipeline kwargs
        """
        user_config = self.user_config_manager.load()
        model_config = user_config.models.get(model_name)

        if not model_config or not model_config.enabled:
            raise ValueError(f"Model {model_name} not configured or disabled")

        kwargs = {
            "model_id": model_config.model_id,
            "device": model_config.device,
            "torch_dtype": self._get_torch_dtype(model_config.dtype),
            "use_safetensors": True,
        }

        # Add custom pipeline parameters
        if model_config.custom_pipeline_params:
            kwargs.update(model_config.custom_pipeline_params)

        return kwargs

    def create_adapter(self, model_name: str, adapter_type: str = "auto") -> Any:
        """
        Create a pipeline adapter for a model

        Args:
            model_name: Name of the model from user config
            adapter_type: Type of adapter (diffusers, comfyui, a1111, mock, auto)
                         If "auto", will determine from model configuration

        Returns:
            Pipeline adapter instance
        """
        # Load user config to get model details
        user_config = self.user_config_manager.load()
        
        # Get model configuration
        if model_name not in user_config.models:
            available_models = list(user_config.models.keys())
            raise ValueError(
                f"Model '{model_name}' not found in configuration.\n"
                f"Available models: {available_models}\n"
                f"Run 'expandor --setup' to configure models."
            )
        
        model_config = user_config.models[model_name]
        if not model_config.enabled:
            raise ValueError(
                f"Model '{model_name}' is disabled in configuration.\n"
                f"Enable it in your config file or run 'expandor --setup'."
            )
        
        # Determine adapter type if auto
        if adapter_type == "auto":
            # For now, always use diffusers for HuggingFace models
            if model_config.model_id:
                adapter_type = "diffusers"
            elif model_config.path and model_config.path.endswith(".safetensors"):
                # Could be ComfyUI or A1111 format
                adapter_type = "diffusers"  # Default to diffusers
            else:
                adapter_type = "diffusers"
        
        # Create adapter with model configuration
        kwargs = {
            "model_id": model_config.model_id or model_config.path,
            "variant": model_config.variant,
            "torch_dtype": self._get_torch_dtype(model_config.dtype),
            "device": model_config.device,
            "cache_dir": model_config.cache_dir,
        }
        
        if adapter_type == "diffusers":
            from ..adapters import DiffusersPipelineAdapter
            return DiffusersPipelineAdapter(**kwargs)
        elif adapter_type == "comfyui":
            from ..adapters import ComfyUIPipelineAdapter
            return ComfyUIPipelineAdapter(**kwargs)
        elif adapter_type == "a1111":
            from ..adapters import A1111PipelineAdapter
            return A1111PipelineAdapter(**kwargs)
        elif adapter_type == "mock":
            from ..adapters import MockPipelineAdapter
            return MockPipelineAdapter(**kwargs)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

    def _get_torch_dtype(self, dtype_str: str):
        """Convert string dtype to torch dtype"""
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        return dtype_map.get(dtype_str, torch.float32)

    def validate_models(
        self, models: Dict[str, ModelConfig]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate model configurations

        Args:
            models: Dictionary of model configurations

        Returns:
            Validation results for each model
        """
        results = {}

        for name, config in models.items():
            result = {"valid": True, "errors": [], "warnings": []}

            # Check if model path/id exists
            if config.model_id.startswith("/") or config.model_id.startswith("."):
                # Local path
                model_path = Path(config.model_id)
                if not model_path.exists():
                    result["valid"] = False
                    result["errors"].append(
                        f"Model path does not exist: {config.model_id}"
                    )
                result["type"] = "local"
            else:
                # HuggingFace model ID
                result["type"] = "huggingface"
                # Could add HF model validation here

            # Validate device
            if config.device not in ["cuda", "cpu", "mps"]:
                result["warnings"].append(f"Unusual device: {config.device}")

            # Validate dtype
            if config.dtype not in ["float32", "float16", "bfloat16"]:
                result["warnings"].append(f"Unusual dtype: {config.dtype}")

            results[name] = result

        return results
