"""
Configuration loader for Expandor
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Use importlib.resources for Python 3.7+
try:
    from importlib import resources

    HAS_IMPORTLIB_RESOURCES = True
except ImportError:
    # Python < 3.7
    HAS_IMPORTLIB_RESOURCES = False

# Configuration file mapping
CONFIG_FILES = {
    "base_defaults": "base_defaults.yaml",
    "strategies": "strategies.yaml",
    "quality_presets": "quality_presets.yaml",
    "vram_strategies": "vram_strategies.yaml",
    "model_constraints": "model_constraints.yaml",
    "processing_params": "processing_params.yaml",
    "output_quality": "output_quality.yaml",
    "strategy_parameters": "strategy_parameters.yaml",
    "vram_thresholds": "vram_thresholds.yaml",
    "quality_thresholds": "quality_thresholds.yaml",
    "resolution_presets": "resolution_presets.yaml",
    "controlnet_config": "controlnet_config.yaml",
}


class ConfigLoader:
    """
    Loads and manages configuration from YAML files
    """

    def __init__(self, config_dir: Path,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize config loader

        Args:
            config_dir: Directory containing config YAML files
            logger: Logger instance
        """
        self.config_dir = Path(config_dir)
        self.logger = logger or logging.getLogger(__name__)

        if not self.config_dir.exists():
            self.logger.warning(
                f"Config directory not found: {
                    self.config_dir}"
            )

    def load_all_configs(self) -> Dict[str, Any]:
        """
        Load all configuration files and merge into single config

        Returns:
            Merged configuration dictionary
        """
        # Since v0.7.0, all config is in master_defaults.yaml
        master_config_path = self.config_dir / "master_defaults.yaml"
        if not master_config_path.exists():
            raise FileNotFoundError(
                f"Master configuration file not found: {master_config_path}\n"
                "This file is required for Expandor to function properly.\n"
                "Run 'expandor --setup' to create valid configuration."
            )

        try:
            with open(master_config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            self.logger.debug(f"Loaded master config from {master_config_path}")
        except Exception as e:
            raise ValueError(
                f"Failed to load master configuration from {master_config_path}: {e}\n"
                "This is a critical error - the master_defaults.yaml file must be valid YAML."
            )

        # Also load controlnet_config.yaml if it exists (it's separate)
        controlnet_path = self.config_dir / "controlnet_config.yaml"
        if controlnet_path.exists():
            try:
                with open(controlnet_path, "r") as f:
                    controlnet_config = yaml.safe_load(f)
                    if controlnet_config:
                        config["controlnet_config"] = controlnet_config
                        self.logger.debug(f"Loaded controlnet config from {controlnet_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load controlnet_config.yaml: {e}")

        return config

    def load_config_file(self, filename: str) -> Dict[str, Any]:
        """
        Load a specific configuration file

        Args:
            filename: Name of config file (with or without .yaml extension)

        Returns:
            Configuration dictionary from file
        """
        if not filename.endswith(".yaml"):
            filename += ".yaml"

        file_path = self.config_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        try:
            with open(file_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"Failed to load {filename}: {e}")
            raise

    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get configuration by name using CONFIG_FILES mapping

        Args:
            config_name: Configuration name (e.g., 'strategy_parameters')

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If config name not recognized
            FileNotFoundError: If config file doesn't exist
        """
        if config_name not in CONFIG_FILES:
            raise ValueError(
                f"Unknown configuration: {config_name}\n"
                f"Valid configurations: {list(CONFIG_FILES.keys())}"
            )

        filename = CONFIG_FILES[config_name]
        config_data = self.load_config_file(filename)

        # Validate the config
        self._validate_schema(config_name, config_data)

        return config_data

    def save_config_file(self, filename: str, config: Dict[str, Any],
                         user_config: bool = False) -> Path:
        """
        Save configuration to YAML file

        Args:
            filename: Config filename (e.g., 'controlnet_config.yaml')
            config: Configuration dictionary to save
            user_config: If True, save to user config dir (~/.config/expandor)
                        If False, save to package config dir (default)

        Returns:
            Path to saved config file

        Raises:
            PermissionError: If unable to write to config directory
            ValueError: If config validation fails
        """
        # Validate config structure first
        if not isinstance(config, dict):
            raise ValueError(
                f"Config must be a dictionary, got {type(config)}\n"
                "Ensure you're passing a valid configuration structure."
            )

        if not filename.endswith(".yaml"):
            filename += ".yaml"

        # Determine save location
        if user_config:
            config_dir = Path.home() / ".config" / "expandor"
            config_dir.mkdir(parents=True, exist_ok=True)
        else:
            config_dir = self.config_dir

        file_path = config_dir / filename

        try:
            # Save with helpful header when creating user configs
            with open(file_path, 'w') as f:
                if user_config:
                    f.write("# Expandor Configuration File\n")
                    f.write(
                        f"# Generated by Expandor v{
                            self._get_version()}\n")
                    f.write(f"# Created: {datetime.now().isoformat()}\n")
                    f.write("# This file can be edited manually\n\n")
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            self.logger.info(f"Saved config to {filename}")
            return file_path

        except PermissionError as e:
            raise PermissionError(
                f"Unable to save config to {file_path}\n"
                f"Error: {str(e)}\n"
                "Solutions:\n"
                "1. Check directory permissions\n"
                "2. Run with appropriate privileges\n"
                "3. Use --config-dir to specify writable location"
            )
        except Exception as e:
            self.logger.error(f"Failed to save {filename}: {e}")
            raise

    def get_config_path(self, filename: str) -> Path:
        """
        Get config file path with fallback handling

        Args:
            filename: Config filename

        Returns:
            Path to config file
        """
        if HAS_IMPORTLIB_RESOURCES:
            try:
                # For Python 3.9+
                if hasattr(resources, "files"):
                    config_files = resources.files(
                        "expandor").joinpath("config")
                    return Path(str(config_files.joinpath(filename)))
                else:
                    # For Python 3.7-3.8
                    with resources.path("expandor.config", filename) as path:
                        return Path(path)
            except Exception as e:
                self.logger.debug(
                    f"Could not load config via importlib.resources: {e}")
                # Fall back to file system path
        # Fallback to relative path
        return Path(__file__).parent.parent / "config" / filename

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load YAML file with graceful error handling

        Args:
            filename: YAML filename

        Returns:
            Configuration dict or empty dict if file missing
        """
        if not filename.endswith(".yaml"):
            filename += ".yaml"
            
        # First try config_dir directly
        file_path = self.config_dir / filename
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                self.logger.error(f"Failed to load {filename}: {e}")
                raise
        
        # Then try get_config_path for fallback locations
        try:
            file_path = self.get_config_path(filename)
            if file_path.exists():
                with open(file_path, "r") as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.warning(f"Could not load {filename}: {e}")
        
        return {}

    def load_quality_preset(self, preset_name: str) -> Dict[str, Any]:
        """
        Load a quality preset configuration

        Args:
            preset_name: Name of quality preset

        Returns:
            Quality preset configuration
        """
        # Try loading from YAML first
        presets = self.load_yaml("quality_presets.yaml")
        if presets and "quality_presets" in presets:
            if preset_name in presets["quality_presets"]:
                return presets["quality_presets"][preset_name]

        # Fallback to default presets
        default_presets = {
            "ultra": {
                "inference_steps": 80,
                "cfg_scale": 7.5,
                "denoise_strength": 0.95,
                "blur_radius": 300,
            },
            "high": {
                "inference_steps": 60,
                "cfg_scale": 7.0,
                "denoise_strength": 0.9,
                "blur_radius": 200,
            },
            "balanced": {
                "inference_steps": 40,
                "cfg_scale": 6.5,
                "denoise_strength": 0.85,
                "blur_radius": 150,
            },
            "fast": {
                "inference_steps": 25,
                "cfg_scale": 6.0,
                "denoise_strength": 0.8,
                "blur_radius": 100,
            },
        }

        if preset_name in default_presets:
            return default_presets[preset_name]

        self.logger.warning(
            f"Unknown quality preset: {preset_name}, using 'balanced'")
        return default_presets["balanced"]

    def validate_config(
            self, config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> None:
        """
        Validate configuration against schema

        Args:
            config: Configuration to validate
            schema: Optional schema dict. If None, performs basic validation

        Raises:
            ValueError: If validation fails with detailed error message
        """
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")

        if schema:
            # TODO: Implement schema validation (jsonschema or similar)
            # For now, just check required keys exist
            required_keys = schema.get('required', [])
            missing_keys = [k for k in required_keys if k not in config]
            if missing_keys:
                raise ValueError(
                    f"Missing required configuration keys: {missing_keys}\n"
                    f"Please ensure your config includes all required fields."
                )

    def _get_version(self) -> str:
        """Get expandor version for config headers"""
        try:
            import expandor
            return expandor.__version__
        except (OSError, IOError, yaml.YAMLError):
            return "unknown"

    def _validate_schema(self, config_name: str, config_data: dict) -> bool:
        """Validate config against schema - FAIL LOUD on invalid data"""
        # For now, just ensure it's a dict
        if not isinstance(config_data, dict):
            raise TypeError(
                f"Config {config_name} must be a dictionary, "
                f"got {type(config_data).__name__}"
            )

        # Check for None values (common error)
        none_keys = [k for k, v in config_data.items() if v is None]
        if none_keys:
            raise ValueError(
                f"Config {config_name} has None values for keys: {none_keys}\n"
                f"Please provide valid values in the config file"
            )

        return True
