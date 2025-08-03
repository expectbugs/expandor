"""
Expandor - Universal Image Resolution Adaptation System
"""

from ._version import __version__
from .core.config import ExpandorConfig
from .core.exceptions import (ExpandorError, QualityError, StrategyError,
                              VRAMError)
from .core.expandor import Expandor
from .core.result import ExpandorResult, StageResult

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
