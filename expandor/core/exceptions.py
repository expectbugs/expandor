"""
Custom exceptions for Expandor
"""

from typing import Optional, Any

class ExpandorError(Exception):
    """Base exception for Expandor"""
    def __init__(self, message: str, stage: Optional[str] = None, 
                 config: Optional[Any] = None, partial_result: Optional[Any] = None):
        self.message = message
        self.stage = stage
        self.config = config
        self.partial_result = partial_result
        super().__init__(message)

class VRAMError(ExpandorError):
    """VRAM-related errors."""
    def __init__(self, operation: str, required_mb: float, 
                available_mb: float, message: str = ""):
        self.operation = operation
        self.required_mb = required_mb
        self.available_mb = available_mb
        base_msg = f"Insufficient VRAM for {operation}: need {required_mb:.1f}MB, have {available_mb:.1f}MB"
        if message:
            base_msg = f"{base_msg}. {message}"
        super().__init__(base_msg)

class StrategyError(ExpandorError):
    """Strategy selection or execution errors"""
    pass

class QualityError(ExpandorError):
    """Quality validation errors"""
    pass