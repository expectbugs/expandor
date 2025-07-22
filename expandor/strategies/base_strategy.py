"""
Base Strategy Class for all expansion strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

class BaseExpansionStrategy(ABC):
    """Base class for all expansion strategies"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.boundary_tracker = []
        self.model_metadata = {
            'progressive_boundaries': [],
            'progressive_boundaries_vertical': [],
            'seam_details': [],
            'used_progressive': False
        }
        
    @abstractmethod
    def execute(self, config) -> Dict[str, Any]:
        """
        Execute the expansion strategy
        
        Args:
            config: ExpandorConfig instance
            
        Returns:
            Dictionary with results including image, path, stages, boundaries
        """
        pass
    
    def validate_inputs(self, config):
        """Validate configuration inputs"""
        if config.target_resolution[0] <= 0 or config.target_resolution[1] <= 0:
            raise ValueError(f"Invalid target resolution: {config.target_resolution}")
        
        if not config.prompt:
            raise ValueError("Prompt cannot be empty")
        
        if config.seed < 0:
            raise ValueError(f"Invalid seed: {config.seed}")
    
    def track_boundary(self, position: int, direction: str, step: int):
        """Track expansion boundary for seam detection"""
        self.boundary_tracker.append({
            'position': position,
            'direction': direction,
            'step': step
        })