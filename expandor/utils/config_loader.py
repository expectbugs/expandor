"""
Configuration loader for Expandor
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Try importing pkg_resources with fallback
try:
    import pkg_resources
    HAS_PKG_RESOURCES = True
except ImportError:
    HAS_PKG_RESOURCES = False

class ConfigLoader:
    """
    Loads and manages configuration from YAML files
    """
    
    def __init__(self, config_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize config loader
        
        Args:
            config_dir: Directory containing config YAML files
            logger: Logger instance
        """
        self.config_dir = Path(config_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        if not self.config_dir.exists():
            self.logger.warning(f"Config directory not found: {self.config_dir}")
    
    def load_all_configs(self) -> Dict[str, Any]:
        """
        Load all configuration files and merge into single config
        
        Returns:
            Merged configuration dictionary
        """
        config = {}
        
        # List of config files to load in order
        config_files = [
            "strategies.yaml",
            "quality_presets.yaml",
            "vram_strategies.yaml",
            "model_constraints.yaml",
            "artifact_detection.yaml"
        ]
        
        for config_file in config_files:
            file_path = self.config_dir / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        file_config = yaml.safe_load(f)
                        if file_config:
                            # Extract top-level key from filename
                            key = config_file.replace('.yaml', '').replace('_', ' ').title().replace(' ', '_').lower()
                            
                            # Special handling for certain files
                            if config_file == "quality_presets.yaml":
                                config["quality_presets"] = file_config.get("quality_presets", file_config)
                            elif config_file == "vram_strategies.yaml":
                                config["vram_strategies"] = file_config
                            elif config_file == "model_constraints.yaml":
                                config["model_constraints"] = file_config
                            elif config_file == "artifact_detection.yaml":
                                config["quality_validation"] = file_config
                            else:
                                config.update(file_config)
                                
                    self.logger.debug(f"Loaded config from {config_file}")
                except Exception as e:
                    self.logger.error(f"Failed to load {config_file}: {e}")
                    raise
            else:
                self.logger.warning(f"Config file not found: {file_path}")
        
        return config
    
    def load_config_file(self, filename: str) -> Dict[str, Any]:
        """
        Load a specific configuration file
        
        Args:
            filename: Name of config file (with or without .yaml extension)
            
        Returns:
            Configuration dictionary from file
        """
        if not filename.endswith('.yaml'):
            filename += '.yaml'
        
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"Failed to load {filename}: {e}")
            raise
    
    def save_config_file(self, filename: str, config: Dict[str, Any]):
        """
        Save configuration to file
        
        Args:
            filename: Name of config file
            config: Configuration dictionary to save
        """
        if not filename.endswith('.yaml'):
            filename += '.yaml'
        
        file_path = self.config_dir / filename
        
        try:
            with open(file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            self.logger.info(f"Saved config to {filename}")
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
        if HAS_PKG_RESOURCES:
            try:
                return Path(pkg_resources.resource_filename('expandor', f'config/{filename}'))
            except Exception:
                pass
        # Fallback to relative path
        return Path(__file__).parent.parent / 'config' / filename
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load YAML file with graceful error handling
        
        Args:
            filename: YAML filename
            
        Returns:
            Configuration dict or empty dict if file missing
        """
        try:
            file_path = self.get_config_path(filename)
            if file_path.exists():
                with open(file_path, 'r') as f:
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
        presets = self.load_yaml('quality_presets.yaml')
        if presets and 'quality_presets' in presets:
            if preset_name in presets['quality_presets']:
                return presets['quality_presets'][preset_name]
        
        # Fallback to default presets
        default_presets = {
            'ultra': {
                'inference_steps': 80,
                'cfg_scale': 7.5,
                'denoise_strength': 0.95,
                'blur_radius': 300
            },
            'high': {
                'inference_steps': 60,
                'cfg_scale': 7.0,
                'denoise_strength': 0.9,
                'blur_radius': 200
            },
            'balanced': {
                'inference_steps': 40,
                'cfg_scale': 6.5,
                'denoise_strength': 0.85,
                'blur_radius': 150
            },
            'fast': {
                'inference_steps': 25,
                'cfg_scale': 6.0,
                'denoise_strength': 0.8,
                'blur_radius': 100
            }
        }
        
        if preset_name in default_presets:
            return default_presets[preset_name]
        
        self.logger.warning(f"Unknown quality preset: {preset_name}, using 'balanced'")
        return default_presets['balanced']