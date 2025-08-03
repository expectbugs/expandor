"""
Configuration resolver for Expandor
Implements the configuration hierarchy: Base defaults → Quality preset → User overrides
"""

from typing import Any, Dict, Optional

from ..core.exceptions import ExpandorError
from ..utils.config_loader import ConfigLoader


class ConfigResolver:
    """Resolves configuration values with proper hierarchy"""
    
    def __init__(self, config_loader: ConfigLoader):
        """Initialize with a config loader instance"""
        self.config_loader = config_loader
        self.base_defaults = self._load_base_defaults()
    
    def _load_base_defaults(self) -> Dict[str, Any]:
        """Load base defaults from base_defaults.yaml"""
        defaults = self.config_loader.load_yaml("base_defaults.yaml")
        if not defaults or "expandor_defaults" not in defaults:
            raise ExpandorError(
                "FATAL: base_defaults.yaml missing or invalid!\n"
                "This file MUST exist with 'expandor_defaults' section.\n"
                "The configuration system is broken - cannot continue.\n"
                "Run 'expandor --setup' to restore configuration files."
            )
        return defaults["expandor_defaults"]
    
    def resolve_config(self, config: 'ExpandorConfig') -> None:
        """
        Apply configuration hierarchy:
        1. Base defaults
        2. Quality preset overrides (if specified)
        3. User values (already set)
        """
        # Step 1: Apply base defaults for any None values
        for field_name, default_value in self.base_defaults.items():
            # Skip complex nested structures for now
            if isinstance(default_value, dict):
                continue
                
            if hasattr(config, field_name):
                current_value = getattr(config, field_name)
                # Only set if the field is None AND not explicitly set by user
                if current_value is None and not getattr(config, f"_{field_name}_user_set", False):
                    setattr(config, field_name, default_value)
        
        # Step 2: Apply quality preset if specified
        if config.quality_preset and config.quality_preset != "custom":
            self._apply_quality_preset(config)
        
        # Step 3: Validate all required fields are set
        self._validate_required_fields(config)
    
    def _apply_quality_preset(self, config: 'ExpandorConfig') -> None:
        """Apply quality preset overrides"""
        try:
            preset = self.config_loader.load_quality_preset(config.quality_preset)
        except Exception as e:
            raise ExpandorError(
                f"FATAL: Failed to load quality preset '{config.quality_preset}': {e}\n"
                f"Quality preset system is broken - cannot continue.\n"
                f"Check that quality_presets.yaml contains the '{config.quality_preset}' preset."
            )
        
        # Map preset values to config fields
        # This mapping defines how preset sections/keys map to ExpandorConfig fields
        mappings = {
            # Generation section
            ("generation", "num_inference_steps"): "num_inference_steps",
            ("generation", "guidance_scale"): "guidance_scale",
            ("generation", "scheduler"): "scheduler",
            ("generation", "use_karras_sigmas"): "use_karras_sigmas",
            
            # Expansion section
            ("expansion", "denoising_strength"): "denoising_strength",
            ("expansion", "mask_blur_ratio"): "mask_blur_ratio",
            ("expansion", "overlap_ratio"): "overlap_ratio",
            ("expansion", "edge_blend_pixels"): "edge_blend_pixels",
            
            # Validation section
            ("validation", "enable_artifact_detection"): "enable_artifacts_check",
            ("validation", "artifact_threshold"): "artifact_detection_threshold",
            ("validation", "enable_seam_detection"): "enable_seam_detection",
            ("validation", "seam_threshold"): "seam_threshold",
            ("validation", "max_repair_attempts"): "max_artifact_repair_attempts",
            
            # Performance section
            ("performance", "enable_xformers"): "enable_xformers",
            ("performance", "enable_cpu_offload"): "use_cpu_offload",
            ("performance", "tile_processing"): "tile_processing",
            ("performance", "batch_size"): "batch_size",
            
            # Models section
            ("models", "prefer_refiner"): "prefer_refiner",
            ("models", "refiner_strength"): "refiner_strength",
            ("models", "controlnet_strength"): "controlnet_strength",
            
            # Output section
            ("output", "format"): "output_format",
            ("output", "compression"): "compression_level",
            ("output", "save_intermediate"): "save_intermediate",
            
            # Resolution section
            ("resolution", "upscale_method"): "upscale_method",
            ("resolution", "upscale_denoise"): "upscale_denoise",
        }
        
        # Apply mappings
        for (section, key), field_name in mappings.items():
            if section in preset and key in preset[section]:
                # Only override if user didn't explicitly set this field
                if not getattr(config, f"_{field_name}_user_set", False):
                    setattr(config, field_name, preset[section][key])
    
    def _validate_required_fields(self, config: 'ExpandorConfig') -> None:
        """Validate that all required fields are set"""
        # These fields MUST be set after resolution
        required_fields = [
            ("strategy", "Expansion strategy"),
            ("quality_preset", "Quality preset"),
            ("denoising_strength", "Denoising strength"),
            ("guidance_scale", "Guidance scale"),
            ("num_inference_steps", "Number of inference steps"),
            ("enable_artifacts_check", "Artifact checking"),
            ("save_stages", "Stage saving"),
            ("verbose", "Verbose mode"),
            ("use_cpu_offload", "CPU offload"),
            ("overlap_ratio", "Overlap ratio"),
        ]
        
        missing_fields = []
        for field_name, description in required_fields:
            if not hasattr(config, field_name) or getattr(config, field_name) is None:
                missing_fields.append(f"  - {field_name} ({description})")
        
        if missing_fields:
            raise ExpandorError(
                "FATAL: Configuration system failure - required fields not set!\n"
                f"Missing fields:\n" + "\n".join(missing_fields) + "\n\n"
                "This indicates a critical failure in the configuration system.\n"
                "Possible causes:\n"
                "1. Missing or corrupted base_defaults.yaml\n"
                "2. Invalid quality preset configuration\n"
                "3. Configuration loading system failure\n\n"
                "Run 'expandor --setup' to restore default configuration files."
            )