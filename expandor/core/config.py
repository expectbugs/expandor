"""
Configuration system for Expandor
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image


@dataclass
class ExpandorConfig:
    """Configuration for Expandor operations - New API"""

    # Source (required)
    source_image: Union[Path, str, Image.Image]
    
    # Target (at least one required)
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    target_resolution: Optional[Tuple[int, int]] = None
    
    # Generation
    prompt: str = "a high quality image"
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    
    # Strategy
    strategy: str = "auto"
    strategy_override: Optional[str] = None
    
    # Quality
    quality_preset: str = "high"
    denoising_strength: float = 0.95
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    
    # Processing
    save_stages: bool = False
    stage_dir: Optional[Path] = None
    verbose: bool = False
    
    # Resources
    vram_limit_mb: Optional[int] = None
    use_cpu_offload: bool = False
    
    # Advanced
    window_size: Optional[int] = None
    overlap_ratio: float = 0.5
    tile_size: Optional[int] = None
    
    # Metadata
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization"""
        # Convert string paths to Path objects
        if isinstance(self.source_image, str):
            self.source_image = Path(self.source_image)
        
        if self.stage_dir and not isinstance(self.stage_dir, Path):
            self.stage_dir = Path(self.stage_dir)
        
        # Validate we have target dimensions
        if not any([self.target_width, self.target_height, self.target_resolution]):
            raise ValueError(
                "Must specify target dimensions via target_width/target_height or target_resolution"
            )
        
        # If target_resolution is provided, extract width/height
        if self.target_resolution and not (self.target_width and self.target_height):
            self.target_width, self.target_height = self.target_resolution
        
        # Validate strategy
        valid_strategies = ["auto", "direct", "progressive", "tiled", "swpo", "hybrid", "cpu_offload"]
        strategy_to_check = self.strategy_override or self.strategy
        if strategy_to_check not in valid_strategies:
            raise ValueError(
                f"Invalid strategy: {strategy_to_check}. "
                f"Must be one of {valid_strategies}"
            )
        
        # Validate quality preset
        valid_presets = ["fast", "balanced", "high", "ultra"]
        if self.quality_preset not in valid_presets:
            raise ValueError(
                f"Invalid quality preset: {self.quality_preset}. "
                f"Must be one of {valid_presets}"
            )

    def validate(self) -> None:
        """Validate configuration completeness and correctness"""
        # Validate source image
        if isinstance(self.source_image, Path):
            if not self.source_image.exists():
                raise FileNotFoundError(f"Source image not found: {self.source_image}")
        elif not isinstance(self.source_image, Image.Image):
            raise TypeError(
                f"source_image must be Path, str, or PIL.Image, got {type(self.source_image)}"
            )
        
        # Validate dimensions
        target_w = self.target_width or (self.target_resolution[0] if self.target_resolution else None)
        target_h = self.target_height or (self.target_resolution[1] if self.target_resolution else None)
        
        if not (target_w and target_h):
            raise ValueError("Target dimensions not properly specified")
        
        if target_w <= 0 or target_h <= 0:
            raise ValueError(f"Invalid target dimensions: {target_w}x{target_h}")
        
        # Validate numeric ranges
        if not 0.0 <= self.denoising_strength <= 1.0:
            raise ValueError(f"denoising_strength must be between 0 and 1, got {self.denoising_strength}")
        
        if self.guidance_scale < 0:
            raise ValueError(f"guidance_scale must be non-negative, got {self.guidance_scale}")
        
        if self.num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be at least 1, got {self.num_inference_steps}")
        
        if self.overlap_ratio < 0 or self.overlap_ratio >= 1:
            raise ValueError(f"overlap_ratio must be between 0 and 1, got {self.overlap_ratio}")

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
            raise TypeError(f"Cannot convert {type(self.source_image)} to PIL Image")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, Image.Image):
                result[key] = f"<PIL.Image mode={value.mode} size={value.size}>"
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

