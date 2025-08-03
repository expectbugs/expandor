"""
Configuration system for Expandor
ALL defaults MUST come from config files - NO HARDCODED VALUES
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from PIL import Image


@dataclass
class ExpandorConfig:
    """
    Configuration for Expandor operations
    ALL defaults MUST come from config files - NO HARDCODED VALUES
    """

    # Source (required)
    source_image: Union[Path, str, Image.Image]

    # Target (at least one required)
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    target_resolution: Optional[Tuple[int, int]] = None

    # Generation - ALL fields optional, defaults loaded from config
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None

    # Strategy
    strategy: Optional[str] = None
    strategy_override: Optional[str] = None

    # Quality - NO HARDCODED DEFAULTS
    quality_preset: Optional[str] = None
    denoising_strength: Optional[float] = None
    guidance_scale: Optional[float] = None
    num_inference_steps: Optional[int] = None
    scheduler: Optional[str] = None
    use_karras_sigmas: Optional[bool] = None
    
    # Quality control
    enable_artifacts_check: Optional[bool] = None
    artifact_detection_threshold: Optional[float] = None
    max_artifact_repair_attempts: Optional[int] = None
    enable_seam_detection: Optional[bool] = None
    seam_threshold: Optional[float] = None

    # Processing
    save_stages: Optional[bool] = None
    stage_dir: Optional[Path] = None
    verbose: Optional[bool] = None
    batch_size: Optional[int] = None

    # Resources
    vram_limit_mb: Optional[int] = None
    use_cpu_offload: Optional[bool] = None
    enable_xformers: Optional[bool] = None
    gradient_checkpointing: Optional[bool] = None

    # Advanced
    window_size: Optional[int] = None
    overlap_ratio: Optional[float] = None
    tile_size: Optional[int] = None
    tile_overlap: Optional[int] = None
    mask_blur_ratio: Optional[float] = None
    edge_blend_pixels: Optional[int] = None

    # Model settings
    controlnet_strength: Optional[float] = None
    refiner_strength: Optional[float] = None
    prefer_refiner: Optional[bool] = None

    # Output settings
    output_format: Optional[str] = None
    compression_level: Optional[int] = None
    save_intermediate: Optional[bool] = None

    # Upscaling
    upscale_method: Optional[str] = None
    upscale_denoise: Optional[float] = None

    # Validation thresholds
    texture_corruption_threshold: Optional[float] = None
    high_freq_ratio: Optional[float] = None
    vram_usage_limit: Optional[float] = None
    max_generation_time: Optional[int] = None

    # File paths
    cache_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    temp_dir: Optional[Path] = None

    # Metadata
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Load defaults and apply presets"""
        # Track which fields were explicitly set by user
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            setattr(self, f"_{field_info.name}_user_set", value is not None)
        
        # Convert string paths to Path objects before config resolution
        if isinstance(self.source_image, str):
            self.source_image = Path(self.source_image)

        if self.stage_dir and not isinstance(self.stage_dir, Path):
            self.stage_dir = Path(self.stage_dir)

        # Use ConfigurationManager for all configuration
        from ..core.configuration_manager import ConfigurationManager
        
        try:
            config_manager = ConfigurationManager()
            
            # Map of YAML field names to dataclass field names
            field_mapping = {
                'enable_artifact_detection': 'enable_artifacts_check',
                'artifact_threshold': 'artifact_detection_threshold',
                'max_repair_attempts': 'max_artifact_repair_attempts',
                'enable_cpu_offload': 'use_cpu_offload',
                'save_intermediate': 'save_stages'
            }
            
            # Apply quality preset if specified
            if self.quality_preset:
                preset_path = f"quality_presets.{self.quality_preset}"
                preset_config = config_manager.get_value(preset_path)
                
                # Apply preset values to unset fields
                
                for category_name, category_values in preset_config.items():
                    if isinstance(category_values, dict):
                        for yaml_field_name, value in category_values.items():
                            # Map YAML field name to dataclass field name
                            field_name = field_mapping.get(yaml_field_name, yaml_field_name)
                            
                            # Check if field exists in dataclass
                            if hasattr(self, field_name):
                                # Only apply if user didn't explicitly set it
                                if getattr(self, field_name) is None:
                                    setattr(self, field_name, value)
            
            # Apply defaults for any remaining None values
            default_preset = config_manager.get_value("quality_global.default_preset")
            if default_preset and default_preset != self.quality_preset:
                default_config = config_manager.get_value(f"quality_presets.{default_preset}")
                for category_name, category_values in default_config.items():
                    if isinstance(category_values, dict):
                        for yaml_field_name, value in category_values.items():
                            # Map YAML field name to dataclass field name
                            field_name = field_mapping.get(yaml_field_name, yaml_field_name)
                            if hasattr(self, field_name) and getattr(self, field_name) is None:
                                setattr(self, field_name, value)
            
            # Apply any remaining global defaults
            # Strategy - get from config
            if self.strategy is None:
                self.strategy = config_manager.get_value("core.default_strategy")
            
            # Paths
            if self.cache_dir is None:
                self.cache_dir = config_manager.get_value("paths.cache_dir")
            if self.output_dir is None:
                self.output_dir = config_manager.get_value("paths.output_dir")
            if self.temp_dir is None:
                self.temp_dir = config_manager.get_value("paths.temp_dir")
                
            # Output settings
            if self.output_format is None:
                # Get default format from config - NO HARDCODED VALUES
                self.output_format = config_manager.get_value("output.default_format")
            if self.compression_level is None:
                # Get compression level for the format
                format_config = config_manager.get_value(f"output.formats.{self.output_format}")
                if isinstance(format_config, dict) and "compression" in format_config:
                    self.compression_level = format_config["compression"]
                else:
                    # FAIL LOUD if compression not configured
                    raise ValueError(
                        f"Compression level not configured for format '{self.output_format}'\n"
                        f"Please add 'output.formats.{self.output_format}.compression' to configuration"
                    )
                    
            # Processing settings
            if self.save_intermediate is None:
                self.save_intermediate = config_manager.get_value("processing.save_intermediate_stages")
            if self.verbose is None:
                self.verbose = config_manager.get_value("processing.verbose_logging")
            if self.batch_size is None:
                self.batch_size = config_manager.get_value("processing.batch_size")
                
        except Exception as e:
            raise ValueError(
                f"FATAL: Failed to load configuration: {e}\n"
                "This is a critical error - configuration system is broken.\n"
                "Please ensure master_defaults.yaml exists and is valid."
            )

        # Convert any string paths that came from config
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir) if self.cache_dir else None
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir) if self.output_dir else None
        if isinstance(self.temp_dir, str):
            self.temp_dir = Path(self.temp_dir) if self.temp_dir else None

        # Validate we have target dimensions
        if not any([self.target_width,
                    self.target_height,
                    self.target_resolution]):
            raise ValueError(
                "Must specify target dimensions via target_width/target_height or target_resolution"
            )

        # If target_resolution is provided, extract width/height
        if self.target_resolution and not (
                self.target_width and self.target_height):
            self.target_width, self.target_height = self.target_resolution
        
        # Ensure target_resolution is set from width/height if not provided
        if not self.target_resolution and self.target_width and self.target_height:
            self.target_resolution = (self.target_width, self.target_height)

        # Now run validation
        self._validate()

    def _validate(self) -> None:
        """Validate configuration completeness and correctness"""
        # Validate source image
        if isinstance(self.source_image, Path):
            if not self.source_image.exists():
                raise FileNotFoundError(
                    f"Source image not found: {self.source_image}")
        elif not isinstance(self.source_image, Image.Image):
            raise TypeError(
                f"source_image must be Path, str, or PIL.Image, got {type(self.source_image)}")

        # Validate dimensions
        target_w = self.target_width or (
            self.target_resolution[0] if self.target_resolution else None)
        target_h = self.target_height or (
            self.target_resolution[1] if self.target_resolution else None)

        if not (target_w and target_h):
            raise ValueError("Target dimensions not properly specified")

        if target_w <= 0 or target_h <= 0:
            raise ValueError(f"Invalid target dimensions: {target_w}x{target_h}")

        # Validate strategy - get valid list from config
        from ..core.configuration_manager import ConfigurationManager
        config_manager = ConfigurationManager()
        
        # Get available strategies from config
        strategies_config = config_manager.get_value("strategies")
        valid_strategies = list(strategies_config.keys()) + ["auto"]
        
        strategy_to_check = self.strategy_override or self.strategy
        if strategy_to_check and strategy_to_check not in valid_strategies:
            raise ValueError(
                f"Invalid strategy: {strategy_to_check}. "
                f"Must be one of {valid_strategies}"
            )

        # Validate quality preset - get valid list from config
        presets_config = config_manager.get_value("quality_presets")
        valid_presets = list(presets_config.keys()) + ["custom"]
        
        if self.quality_preset and self.quality_preset not in valid_presets:
            raise ValueError(
                f"Invalid quality preset: {self.quality_preset}. "
                f"Must be one of {valid_presets}"
            )

        # Validate numeric ranges - these should all be set by config resolver
        if self.denoising_strength is None:
            raise ValueError("FATAL: denoising_strength not set - configuration system failure")
        if not 0.0 <= self.denoising_strength <= 1.0:
            raise ValueError(
                f"denoising_strength must be between 0 and 1, got {self.denoising_strength}")

        if self.guidance_scale is None:
            raise ValueError("FATAL: guidance_scale not set - configuration system failure")
        if self.guidance_scale < 0:
            raise ValueError(
                f"guidance_scale must be non-negative, got {self.guidance_scale}")

        if self.num_inference_steps is None:
            raise ValueError("FATAL: num_inference_steps not set - configuration system failure")
        if self.num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be at least 1, got {self.num_inference_steps}")

        if self.overlap_ratio is None:
            raise ValueError("FATAL: overlap_ratio not set - configuration system failure")
        if self.overlap_ratio < 0 or self.overlap_ratio >= 1:
            raise ValueError(
                f"overlap_ratio must be between 0 and 1, got {self.overlap_ratio}")
        
        # Validate all critical fields are set
        critical_fields = [
            "strategy", "quality_preset", "enable_artifacts_check",
            "save_stages", "verbose", "use_cpu_offload"
        ]
        for field_name in critical_fields:
            if getattr(self, field_name) is None:
                raise ValueError(
                    f"FATAL: {field_name} not set - configuration system failure.\n"
                    f"This indicates the configuration files are missing or invalid."
                )

    def get_target_resolution(self) -> Tuple[int, int]:
        """Get target resolution as tuple"""
        if self.target_resolution:
            return self.target_resolution
        if self.target_width and self.target_height:
            return (self.target_width, self.target_height)
        raise ValueError("Target dimensions not specified")

    def get_source_image(self) -> Image.Image:
        """Get source image as PIL Image"""
        if isinstance(self.source_image, Image.Image):
            return self.source_image
        elif isinstance(self.source_image, (Path, str)):
            return Image.open(self.source_image)
        else:
            raise TypeError(
                f"Cannot convert {
                    type(
                        self.source_image)} to PIL Image")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, Image.Image):
                result[key] = f"<PIL.Image mode={
                    value.mode} size={
                    value.size}>"
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpandorConfig":
        """Create config from dictionary"""
        # Handle special conversions
        if "stage_dir" in data and data["stage_dir"] is not None:
            data["stage_dir"] = Path(data["stage_dir"])
        if "source_image" in data and isinstance(data["source_image"], str):
            # Try to load as path
            source_path = Path(data["source_image"])
            if source_path.exists():
                data["source_image"] = source_path

        return cls(**data)

    @property
    def effective_strategy(self) -> str:
        """Get the effective strategy (considering override)"""
        return self.strategy_override or self.strategy

    @property
    def strategy_map(self) -> Dict[str, str]:
        """Map user-friendly names to internal strategy names"""
        return {
            "direct": "direct_upscale",
            "progressive": "progressive_outpaint",
            "tiled": "tiled_expansion",
            "swpo": "swpo",
            "hybrid": "hybrid_adaptive",
            "cpu_offload": "cpu_offload",
            "auto": "auto",
        }

    @property
    def internal_strategy(self) -> str:
        """Get the internal strategy name for the registry"""
        effective = self.effective_strategy
        return self.strategy_map.get(effective, effective)
