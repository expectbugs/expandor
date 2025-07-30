"""
Expandor - Universal Image Resolution Adaptation System
"""

from .core.config import ExpandorConfig
from .core.exceptions import ExpandorError, QualityError, StrategyError, VRAMError
from .core.expandor import Expandor
from .core.result import ExpandorResult, StageResult

__version__ = "0.6.1"
__all__ = [
    "Expandor",
    "ExpandorConfig",
    "ExpandorResult",
    "StageResult",
    "ExpandorError",
    "VRAMError",
    "StrategyError",
    "QualityError",
]
