"""
Quality validation and artifact detection
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .artifact_removal import ArtifactDetector


class QualityValidator:
    """
    Validates image quality and detects artifacts

    This class implements zero-tolerance artifact detection
    """

    def __init__(self, config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None):
        """
        Initialize quality validator

        Args:
            config: Global configuration with quality settings
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.artifact_detector = ArtifactDetector(self.logger)

        # Get configuration from ConfigurationManager
        from ..core.configuration_manager import ConfigurationManager
        self.config_manager = ConfigurationManager()
        
        # Get processor config
        try:
            self.processor_config = self.config_manager.get_processor_config('quality_validator')
        except ValueError as e:
            # FAIL LOUD
            raise ValueError(
                f"Failed to load quality_validator configuration!\n{str(e)}"
            )

        # Load detection settings - this is still from the passed config
        # as it contains runtime detection settings
        self.detection_config = config.get("quality_validation", {}).get(
            "artifact_detection", {}
        )

    def validate(
        self,
        image_path: Path,
        metadata: Dict[str, Any],
        detection_level: str = "aggressive",
    ) -> Dict[str, Any]:
        """
        Validate image quality and detect artifacts

        Args:
            image_path: Path to image to validate
            metadata: Image metadata including boundaries
            detection_level: Detection sensitivity level

        Returns:
            Dictionary with validation results
        """
        self.logger.info(
            f"Validating quality with {detection_level} detection")

        # Get detection parameters for level
        level_config = self.detection_config.get(
            detection_level, self.detection_config.get("aggressive", {})
        )

        # Run artifact detection
        detection_result = self.artifact_detector.quick_analysis(
            image_path, metadata)

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            detection_result, level_config)

        # Determine if issues were found based on detection level
        issues_found = detection_result["needs_multipass"]

        # Get seam count from detection - FAIL LOUD if missing
        if "seam_count" not in detection_result:
            raise QualityError(
                "Detection result missing required 'seam_count' field",
                details={"available_keys": list(detection_result.keys())}
            )
        seam_count = detection_result["seam_count"]

        # Log results
        if issues_found:
            self.logger.warning(
                f"Quality issues detected: {seam_count} seams, "
                f"severity: {detection_result['severity']}, "
                f"quality score: {quality_score:.2f}"
            )
        else:
            self.logger.info(
                f"Quality validation passed. Score: {
                    quality_score:.2f}"
            )

        return {
            "issues_found": issues_found,
            "seam_count": seam_count,
            "quality_score": quality_score,
            "severity": detection_result["severity"] if "severity" in detection_result else "none",
            "mask": detection_result.get("mask"),  # mask is optional
            "details": detection_result,
        }

    def _calculate_quality_score(
        self, detection_result: Dict[str, Any], level_config: Dict[str, Any]
    ) -> float:
        """
        Calculate quality score from detection results

        Score ranges from 0.0 (worst) to 1.0 (perfect)
        """
        # Start with perfect score
        score = self.processor_config['initial_score']

        # Deduct for seams - FAIL LOUD if missing
        if "seam_count" not in detection_result:
            raise QualityError(
                "Detection result missing required 'seam_count' field for quality calculation",
                details={"available_keys": list(detection_result.keys())}
            )
        seam_count = detection_result["seam_count"]
        if seam_count > 0:
            # Each seam reduces score
            seam_penalty = self.processor_config['seam_penalty_per_seam'] * seam_count
            score -= min(seam_penalty, self.processor_config['seam_penalty_max_reduction'])  # Cap at max reduction

        # Deduct for severity - FAIL LOUD if missing
        if "severity" not in detection_result:
            raise QualityError(
                "Detection result missing required 'severity' field for quality calculation",
                details={"available_keys": list(detection_result.keys())}
            )
        severity = detection_result["severity"]
        severity_penalties = self.processor_config['severity_penalties']
        score -= severity_penalties.get(severity, 0.0)

        # Deduct for mask coverage if available (mask is optional)
        if "mask" in detection_result and detection_result["mask"] is not None:
            mask = detection_result["mask"]
            # Calculate percentage of image affected
            coverage = np.mean(mask)
            score -= min(coverage * self.processor_config['mask_coverage_multiplier'], 
                        self.processor_config['mask_coverage_max_reduction'])  # Cap at max reduction

        # Ensure score stays in valid range
        return max(self.processor_config['score_min'], 
                  min(self.processor_config['score_max'], score))
