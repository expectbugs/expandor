"""
User configuration management for Expandor
Handles ~/.config/expandor/config.yaml
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model"""

    path: Optional[str] = None
    model_id: Optional[str] = None  # For HuggingFace models
    variant: Optional[str] = None
    dtype: str = "fp16"
    device: str = "cuda"
    cache_dir: Optional[str] = None
    enabled: bool = True

    def __post_init__(self):
        """Validate configuration"""
        if not self.path and not self.model_id:
            raise ValueError("ModelConfig must have either 'path' or 'model_id'")

        valid_dtypes = ["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"]
        if self.dtype not in valid_dtypes:
            raise ValueError(
                f"Invalid dtype: {self.dtype}. Must be one of {valid_dtypes}"
            )

        valid_devices = ["cuda", "cpu", "mps"]
        if self.device not in valid_devices:
            raise ValueError(
                f"Invalid device: {
                    self.device}. Must be one of {valid_devices}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class LoRAConfig:
    """Configuration for LoRA models"""

    name: str
    path: str
    weight: float = 1.0
    auto_apply_keywords: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class UserConfig:
    """Complete user configuration"""

    # Model configurations
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    loras: List[LoRAConfig] = field(default_factory=list)

    # Default settings
    default_quality: str = "balanced"
    default_device: str = "cuda"
    default_dtype: str = "fp16"
    cache_directory: Optional[str] = None

    # User preferences
    auto_artifact_detection: bool = True
    use_controlnet: bool = True
    save_intermediate_stages: bool = False
    verbose_logging: bool = False

    # Performance settings
    max_vram_usage_mb: Optional[int] = None
    allow_cpu_offload: bool = True
    allow_tiled_processing: bool = True
    clear_cache_frequency: int = 5

    # Output settings
    output_format: str = "png"
    output_compression: int = 0
    default_output_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization"""
        data = asdict(self)
        # Convert model configs to dicts
        data["models"] = {k: v.to_dict() for k, v in self.models.items()}
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserConfig":
        """Create from dictionary with STRICT validation - FAIL LOUD on errors"""
        config_data = data.copy()

        # Process models - FAIL on ANY invalid config
        if "models" in config_data:
            models = {}
            for key, model_data in config_data["models"].items():
                if not isinstance(model_data, dict):
                    raise TypeError(
                        f"Model '{key}' configuration must be a dictionary, "
                        f"got {type(model_data).__name__}"
                    )

                try:
                    models[key] = ModelConfig(**model_data)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Invalid model configuration '{key}':\n"
                        f"  Error: {str(e)}\n"
                        f"  Config: {model_data}\n"
                        f"  Required fields: path OR model_id\n"
                        f"  Valid dtypes: fp32, fp16, bf16\n"
                        f"  Valid devices: cuda, cpu, mps"
                    ) from e

            config_data["models"] = models

        # Process LoRAs - FAIL on ANY invalid config
        if "loras" in config_data:
            loras = []
            for i, lora_data in enumerate(config_data["loras"]):
                if not isinstance(lora_data, dict):
                    raise TypeError(
                        f"LoRA at index {i} must be a dictionary, "
                        f"got {type(lora_data).__name__}"
                    )

                lora_name = lora_data.get("name", f"lora_{i}")

                try:
                    loras.append(LoRAConfig(**lora_data))
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Invalid LoRA configuration '{lora_name}':\n"
                        f"  Error: {str(e)}\n"
                        f"  Config: {lora_data}\n"
                        f"  Required fields: name, path\n"
                        f"  Optional: weight, auto_apply_keywords, enabled"
                    ) from e

            config_data["loras"] = loras

        # Validate all other fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        invalid_fields = set(config_data.keys()) - valid_fields

        if invalid_fields:
            raise ValueError(
                f"Unknown configuration fields: {', '.join(invalid_fields)}\n"
                f"Valid fields: {', '.join(sorted(valid_fields))}"
            )

        # Create with validated data only
        filtered_data = {k: v for k, v in config_data.items() if k in valid_fields}

        try:
            return cls(**filtered_data)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Configuration validation failed:\n"
                f"  Error: {str(e)}\n"
                f"  Check your config file for correct types and values"
            ) from e


def get_default_config_path() -> Path:
    """Get platform-appropriate config path"""
    import os
    import platform

    system = platform.system()

    if system == "Windows":
        # Windows: Use AppData/Roaming
        base = Path.home() / "AppData" / "Roaming"
    elif system == "Darwin":  # macOS
        # macOS: Use Library/Application Support
        base = Path.home() / "Library" / "Application Support"
    else:  # Linux and others
        # Respect XDG_CONFIG_HOME if set
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            base = Path(xdg_config)
        else:
            base = Path.home() / ".config"

    return base / "expandor" / "config.yaml"


class UserConfigManager:
    """Manages user configuration file"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize user config manager

        Args:
            config_path: Custom config path (defaults to platform-specific location)
        """
        self.config_path = (
            Path(config_path) if config_path else get_default_config_path()
        )
        self.logger = setup_logger(__name__)
        self._ensure_config_dir()

    def _ensure_config_dir(self):
        """Ensure config directory exists"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> UserConfig:
        """
        Load user configuration from file

        Returns:
            UserConfig object
        """
        if not self.config_path.exists():
            self.logger.info(
                f"No user config found at {self.config_path}, using defaults"
            )
            return self._get_default_config()

        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f) or {}

            config = UserConfig.from_dict(data)
            self.logger.info(f"Loaded user config from {self.config_path}")
            return config

        except Exception as e:
            self.logger.error(f"Failed to load user config: {e}")
            self.logger.warning("Using default configuration")
            return self._get_default_config()

    def save(self, config: UserConfig):
        """
        Save user configuration to file

        Args:
            config: UserConfig object to save
        """
        try:
            # Convert to dict and clean up
            data = config.to_dict()

            # Write with nice formatting
            with open(self.config_path, "w") as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    width=120,
                )

            self.logger.info(f"Saved user config to {self.config_path}")

        except Exception as e:
            self.logger.error(f"Failed to save user config: {e}")
            raise

    def update(self, updates: Dict[str, Any]):
        """
        Update specific configuration values

        Args:
            updates: Dictionary of updates to apply
        """
        config = self.load()

        # Apply updates
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif key.startswith("models."):
                # Handle nested model updates
                model_key = key.split(".", 1)[1]
                if model_key in config.models:
                    for attr, val in value.items():
                        setattr(config.models[model_key], attr, val)

        self.save(config)

    def _get_default_config(self) -> UserConfig:
        """Get default configuration"""
        return UserConfig(
            models={
                "sdxl": ModelConfig(
                    model_id="stabilityai/stable-diffusion-xl-base-1.0",
                    variant="fp16",
                    dtype="fp16",
                    device="cuda",
                ),
                "sdxl_refiner": ModelConfig(
                    model_id="stabilityai/stable-diffusion-xl-refiner-1.0",
                    variant="fp16",
                    dtype="fp16",
                    device="cuda",
                    enabled=True,  # Enabled but only used for high/ultra quality
                ),
                "sd3": ModelConfig(
                    model_id="stabilityai/stable-diffusion-3-medium",
                    dtype="fp16",
                    device="cuda",
                    enabled=True,  # Requires HF token
                ),
                "flux": ModelConfig(
                    model_id="black-forest-labs/FLUX.1-schnell",
                    dtype="fp16",
                    device="cuda",
                    enabled=True,  # Large model, needs 16GB+ VRAM
                ),
                "realesrgan": ModelConfig(
                    path=None,  # Will be downloaded automatically
                    dtype="fp32",
                    device="cuda",
                ),
                "controlnet_canny": ModelConfig(
                    model_id="lllyasviel/sd-controlnet-canny",
                    dtype="fp16",
                    device="cuda",
                    enabled=True,  # For structure preservation
                ),
                "controlnet_tile": ModelConfig(
                    model_id="lllyasviel/control_v11f1e_sd15_tile",
                    dtype="fp16",
                    device="cuda",
                    enabled=True,  # For seamless upscaling
                ),
            }
        )

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """
        Get configuration for a specific model

        Args:
            model_name: Name of the model

        Returns:
            ModelConfig or None if not found
        """
        config = self.load()
        return config.models.get(model_name)

    def add_lora(self, lora: LoRAConfig):
        """
        Add a LoRA configuration

        Args:
            lora: LoRAConfig to add
        """
        config = self.load()

        # Check if already exists
        for existing in config.loras:
            if existing.name == lora.name:
                self.logger.warning(
                    f"LoRA {
                        lora.name} already exists, updating"
                )
                config.loras.remove(existing)
                break

        config.loras.append(lora)
        self.save(config)

    def get_applicable_loras(self, prompt: str) -> List[LoRAConfig]:
        """
        Get LoRAs that should be applied based on prompt keywords

        Args:
            prompt: The generation prompt

        Returns:
            List of applicable LoRAConfig objects
        """
        config = self.load()
        applicable = []

        prompt_lower = prompt.lower()

        for lora in config.loras:
            if not lora.enabled:
                continue

            # Check auto-apply keywords
            for keyword in lora.auto_apply_keywords:
                if keyword.lower() in prompt_lower:
                    applicable.append(lora)
                    self.logger.info(
                        f"Auto-applying LoRA {lora.name} due to keyword '{keyword}'"
                    )
                    break

        return applicable

    def create_example_config(self):
        """Create an example configuration file with comments"""
        example_content = """# Expandor User Configuration
# Location: ~/.config/expandor/config.yaml

# Model configurations
models:
  sdxl:
    model_id: "stabilityai/stable-diffusion-xl-base-1.0"
    variant: "fp16"
    dtype: "fp16"
    device: "cuda"
    cache_dir: null  # Uses HuggingFace default cache
    enabled: true

  sdxl_refiner:
    model_id: "stabilityai/stable-diffusion-xl-refiner-1.0"
    variant: "fp16"
    dtype: "fp16"
    device: "cuda"
    enabled: true  # Optional: improves quality but slower
    # Note: Only loads when quality preset is 'high' or 'ultra'

  sd3:
    model_id: "stabilityai/stable-diffusion-3-medium"
    dtype: "fp16"
    device: "cuda"
    enabled: true  # Requires HuggingFace token for access
    # Set HF_TOKEN environment variable or use 'huggingface-cli login'
    # Alternative: Use gated model access on HuggingFace

  flux:
    model_id: "black-forest-labs/FLUX.1-schnell"
    dtype: "fp16"
    device: "cuda"
    enabled: true  # Fast high-quality model
    # Note: Large model, requires ~16GB VRAM
    # For lower VRAM, use FLUX.1-dev with CPU offload

  realesrgan:
    # Path to local model file (optional)
    # path: "/path/to/RealESRGAN_x4plus.pth"
    dtype: "fp32"
    device: "cuda"

  controlnet_canny:
    model_id: "lllyasviel/sd-controlnet-canny"
    dtype: "fp16"
    device: "cuda"
    enabled: true  # Edge detection for structure preservation
    # Automatically used when ControlNet is enabled in preferences
    # Great for maintaining shapes and outlines during expansion

  controlnet_tile:
    model_id: "lllyasviel/control_v11f1e_sd15_tile"
    dtype: "fp16"
    device: "cuda"
    enabled: true  # Tile mode for seamless upscaling
    # Excellent for maintaining texture quality in expansions
    # Automatically selected for upscaling tasks

# LoRA configurations
loras:
  # Example LoRA configuration
  # - name: "detailed_background"
  #   path: "/path/to/detailed_bg_v2.safetensors"
  #   weight: 0.8
  #   auto_apply_keywords:
  #     - "detailed background"
  #     - "intricate scenery"
  #   enabled: true

# Default settings
default_quality: "balanced"  # Options: fast, balanced, high, ultra
default_device: "cuda"
default_dtype: "fp16"
cache_directory: null  # Uses system default

# User preferences
auto_artifact_detection: true
use_controlnet: true
save_intermediate_stages: false
verbose_logging: false

# Performance settings
max_vram_usage_mb: null  # Auto-detect available VRAM
allow_cpu_offload: true
allow_tiled_processing: true
clear_cache_frequency: 5  # Clear CUDA cache every N operations

# Output settings
output_format: "png"
output_compression: 0  # PNG compression level (0-9)
default_output_dir: null  # Current directory by default
"""

        example_path = self.config_path.parent / "config.example.yaml"
        with open(example_path, "w") as f:
            f.write(example_content)

        self.logger.info(f"Created example config at {example_path}")
