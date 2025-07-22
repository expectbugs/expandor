"""
Expandor - Universal Image Resolution Adaptation System
"""

from .core.expandor import Expandor
from .core.config import ExpandorConfig
from .core.exceptions import ExpandorError

__version__ = "0.1.0"
__all__ = ["Expandor", "ExpandorConfig", "ExpandorError"]