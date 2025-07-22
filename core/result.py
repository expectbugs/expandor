"""
Result dataclasses for Expandor operations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

@dataclass
class StageResult:
    """Result from a single processing stage"""
    name: str
    method: str
    input_size: Tuple[int, int]
    output_size: Tuple[int, int]
    duration_seconds: float
    vram_used_mb: float = 0.0
    artifacts_detected: int = 0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "method": self.method,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "duration_seconds": self.duration_seconds,
            "vram_used_mb": self.vram_used_mb,
            "artifacts_detected": self.artifacts_detected,
            "metadata": self.metadata or {}
        }

@dataclass
class ExpandorResult:
    """Comprehensive result from expansion operation"""
    # Core results
    image_path: Path
    size: Tuple[int, int]
    success: bool = True
    
    # Stage tracking
    stages: List[StageResult] = field(default_factory=list)
    boundaries: List[Dict] = field(default_factory=list)
    
    # Quality metrics
    seams_detected: int = 0
    artifacts_fixed: int = 0
    refinement_passes: int = 0
    quality_score: float = 1.0
    
    # Resource usage
    vram_peak_mb: float = 0.0
    total_duration_seconds: float = 0.0
    strategy_used: str = ""
    fallback_count: int = 0
    
    # Full metadata (includes generation_metadata updates)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Error information (if success=False)
    error: Optional[Exception] = None
    error_stage: Optional[str] = None
    
    # Optional image object for direct access
    image: Optional[Any] = None  # PIL.Image.Image
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "image_path": str(self.image_path),
            "size": self.size,
            "success": self.success,
            "stages": [stage.to_dict() for stage in self.stages],
            "boundaries": self.boundaries,
            "seams_detected": self.seams_detected,
            "artifacts_fixed": self.artifacts_fixed,
            "refinement_passes": self.refinement_passes,
            "quality_score": self.quality_score,
            "vram_peak_mb": self.vram_peak_mb,
            "total_duration_seconds": self.total_duration_seconds,
            "strategy_used": self.strategy_used,
            "fallback_count": self.fallback_count,
            "metadata": self.metadata,
            "error": str(self.error) if self.error else None,
            "error_stage": self.error_stage
        }