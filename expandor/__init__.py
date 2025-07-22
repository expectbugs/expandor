"""
Expandor - Universal Image Resolution Adaptation System
"""

from .core.expandor import Expandor
from .core.config import ExpandorConfig
from .core.exceptions import ExpandorError, VRAMError, StrategyError, QualityError
from .core.result import ExpandorResult, StageResult

__version__ = "0.1.0"
__all__ = [
    "Expandor",
    "ExpandorConfig",
    "ExpandorResult",
    "StageResult",
    "ExpandorError",
    "VRAMError",
    "StrategyError",
    "QualityError"
]