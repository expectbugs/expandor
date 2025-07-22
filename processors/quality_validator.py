"""
Quality validation and artifact detection
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np

from .artifact_removal import ArtifactDetector

class QualityValidator:
    """
    Validates image quality and detects artifacts
    
    This class implements zero-tolerance artifact detection
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize quality validator
        
        Args:
            config: Global configuration with quality settings
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.artifact_detector = ArtifactDetector(self.logger)
        
        # Load detection settings
        self.detection_config = config.get('quality_validation', {}).get('artifact_detection', {})
    
    def validate(self, 
                 image_path: Path,
                 metadata: Dict[str, Any],
                 detection_level: str = "aggressive") -> Dict[str, Any]:
        """
        Validate image quality and detect artifacts
        
        Args:
            image_path: Path to image to validate
            metadata: Image metadata including boundaries
            detection_level: Detection sensitivity level
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Validating quality with {detection_level} detection")
        
        # Get detection parameters for level
        level_config = self.detection_config.get(detection_level, self.detection_config.get('aggressive', {}))
        
        # Run artifact detection
        detection_result = self.artifact_detector.quick_analysis(image_path, metadata)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(detection_result, level_config)
        
        # Determine if issues were found based on detection level
        issues_found = detection_result['needs_multipass']
        
        # Get seam count from detection
        seam_count = detection_result.get('seam_count', 0)
        
        # Log results
        if issues_found:
            self.logger.warning(
                f"Quality issues detected: {seam_count} seams, "
                f"severity: {detection_result['severity']}, "
                f"quality score: {quality_score:.2f}"
            )
        else:
            self.logger.info(f"Quality validation passed. Score: {quality_score:.2f}")
        
        return {
            "issues_found": issues_found,
            "seam_count": seam_count,
            "quality_score": quality_score,
            "severity": detection_result.get('severity', 'none'),
            "mask": detection_result.get('mask'),
            "details": detection_result
        }
    
    def _calculate_quality_score(self, detection_result: Dict[str, Any], level_config: Dict[str, Any]) -> float:
        """
        Calculate quality score from detection results
        
        Score ranges from 0.0 (worst) to 1.0 (perfect)
        """
        # Start with perfect score
        score = 1.0
        
        # Deduct for seams
        seam_count = detection_result.get('seam_count', 0)
        if seam_count > 0:
            # Each seam reduces score
            seam_penalty = 0.1 * seam_count
            score -= min(seam_penalty, 0.5)  # Cap at 0.5 reduction
        
        # Deduct for severity
        severity = detection_result.get('severity', 'none')
        severity_penalties = {
            'critical': 0.3,
            'high': 0.2,
            'medium': 0.1,
            'low': 0.05,
            'none': 0.0
        }
        score -= severity_penalties.get(severity, 0.0)
        
        # Deduct for mask coverage if available
        if detection_result.get('mask') is not None:
            mask = detection_result['mask']
            # Calculate percentage of image affected
            coverage = np.mean(mask)
            score -= min(coverage * 0.3, 0.3)  # Cap at 0.3 reduction
        
        # Ensure score stays in valid range
        return max(0.0, min(1.0, score))