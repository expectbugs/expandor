"""
Expansion strategies module
"""

import importlib
from typing import Type, Optional
import logging

from .base_strategy import BaseExpansionStrategy

# Strategy registry mapping names to module paths
STRATEGY_REGISTRY = {
    "direct_upscale": "direct_upscale.DirectUpscaleStrategy",
    "progressive_outpaint": "progressive_outpaint.ProgressiveOutpaintStrategy", 
    "tiled_expansion": "tiled_expansion.TiledExpansionStrategy",
    "swpo": "swpo_strategy.SWPOStrategy",
    "cpu_offload": "cpu_offload.CPUOffloadStrategy",
    "hybrid_adaptive": "experimental.hybrid_adaptive.HybridAdaptiveStrategy"
}

def get_strategy_class(strategy_name: str) -> Type[BaseExpansionStrategy]:
    """
    Dynamically load a strategy class by name
    
    Args:
        strategy_name: Name of strategy from STRATEGY_REGISTRY
        
    Returns:
        Strategy class
        
    Raises:
        ValueError: If strategy not found
        ImportError: If strategy module cannot be loaded
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available strategies: {list(STRATEGY_REGISTRY.keys())}"
        )
    
    # Parse module and class name
    module_path, class_name = STRATEGY_REGISTRY[strategy_name].rsplit('.', 1)
    full_module = f"expandor.strategies.{module_path}"
    
    try:
        # Import module
        module = importlib.import_module(full_module)
        
        # Get class
        strategy_class = getattr(module, class_name)
        
        # Validate it's a proper strategy
        if not issubclass(strategy_class, BaseExpansionStrategy):
            raise TypeError(
                f"{class_name} is not a subclass of BaseExpansionStrategy"
            )
        
        return strategy_class
        
    except ImportError as e:
        raise ImportError(
            f"Failed to import strategy {strategy_name} from {full_module}: {str(e)}"
        )
    except AttributeError:
        raise ImportError(
            f"Strategy class {class_name} not found in {full_module}"
        )

# Convenience imports
from .base_strategy import BaseExpansionStrategy
from .progressive_outpaint import ProgressiveOutpaintStrategy
from .direct_upscale import DirectUpscaleStrategy
# from .tiled_expansion import TiledExpansionStrategy  # Not implemented yet

__all__ = [
    'BaseExpansionStrategy',
    'ProgressiveOutpaintStrategy',
    'DirectUpscaleStrategy',
    # 'TiledExpansionStrategy',  # Not implemented yet
    'STRATEGY_REGISTRY',
    'get_strategy_class'
]