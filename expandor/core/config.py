"""
Configuration classes for Expandor
"""

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path
from PIL import Image

@dataclass
class ExpandorConfig:
    """Comprehensive configuration for expansion operation"""
    # Core inputs
    source_image: Union[Path, Image.Image]
    target_resolution: Tuple[int, int]
    prompt: str
    seed: int
    
    # Source information
    source_metadata: Dict[str, Any]
    generation_metadata: Optional[Dict] = None
    
    # Pipeline access
    inpaint_pipeline: Optional[Any] = None
    refiner_pipeline: Optional[Any] = None
    img2img_pipeline: Optional[Any] = None
    
    # Quality & strategy
    quality_preset: str = "ultra"
    strategy_override: Optional[str] = None
    
    # VRAM management
    vram_limit_mb: Optional[float] = None
    allow_cpu_offload: bool = True
    allow_tiled: bool = True
    
    # Progressive/SWPO parameters
    window_size: int = 200
    overlap_ratio: float = 0.8
    denoising_strength: float = 0.95
    min_strength: float = 0.20
    max_strength: float = 0.95
    
    # Refinement parameters
    refinement_passes: Optional[int] = None
    artifact_detection_level: str = "aggressive"
    
    # Tracking and debugging
    save_stages: bool = False
    stage_dir: Optional[Path] = None
    stage_save_callback: Optional[Callable] = None
    verbose: bool = False
    
    # Additional strategy-specific parameters
    swpo_enabled: bool = True
    progressive_stages: Optional[List[Dict[str, Any]]] = None
    tile_size: Optional[int] = None
    auto_refine: bool = True
    edge_blur_radius: int = 20
    
    # Strategy feature flags
    allow_swpo: bool = True
    allow_progressive: bool = True
    
    @property
    def force_strategy(self) -> Optional[str]:
        """Alias for strategy_override for backward compatibility"""
        return self.strategy_override