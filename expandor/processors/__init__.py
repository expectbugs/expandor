"""
Processors module for artifact detection and quality enhancement
"""

# Core processors - always available
from .artifact_detector_enhanced import EnhancedArtifactDetector
from .quality_validator import QualityValidator
from .seam_repair import SeamRepairProcessor
from .tiled_processor import TiledProcessor

# ControlNet extractor is COMPLETELY OPTIONAL
# This demonstrates clean separation - core processors work without it
try:
    from .controlnet_extractors import ControlNetExtractor
    HAS_CONTROLNET_EXTRACTOR = True
except ImportError:
    # This is NOT an error - ControlNet is optional
    # Core Expandor functionality is unaffected
    HAS_CONTROLNET_EXTRACTOR = False
    ControlNetExtractor = None

__all__ = [
    "EnhancedArtifactDetector",
    "QualityValidator",
    "SeamRepairProcessor",
    "TiledProcessor",
]

# Only add to exports if available
if HAS_CONTROLNET_EXTRACTOR:
    __all__.append('ControlNetExtractor')
