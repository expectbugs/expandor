"""
Expandor pipeline adapters module
"""

from .base_adapter import BasePipelineAdapter
from .mock_pipeline_adapter import MockPipelineAdapter

# Conditional imports for optional dependencies
try:
    from .diffusers_adapter import DiffusersPipelineAdapter

    __all__ = ["BasePipelineAdapter", "MockPipelineAdapter", "DiffusersPipelineAdapter"]
except ImportError:
    __all__ = ["BasePipelineAdapter", "MockPipelineAdapter"]

from .a1111_adapter import A1111PipelineAdapter

# Placeholder adapters (always available)
from .comfyui_adapter import ComfyUIPipelineAdapter

# Add placeholder adapters to exports
__all__.extend(["ComfyUIPipelineAdapter", "A1111PipelineAdapter"])
