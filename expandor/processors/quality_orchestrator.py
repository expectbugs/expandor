"""
Quality refinement orchestrator that coordinates all quality systems.
Integrates artifact detection, boundary analysis, and smart refinement.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from ..core.boundary_tracker import BoundaryTracker
from ..core.config import ExpandorConfig
from ..core.exceptions import QualityError
from .artifact_detector_enhanced import (ArtifactSeverity,
                                         EnhancedArtifactDetector)
from .boundary_analysis import BoundaryAnalyzer
from .refinement.smart_refiner import SmartRefiner


class QualityOrchestrator:
    """
    Orchestrates all quality systems for comprehensive quality assurance.

    Follows FAIL LOUD philosophy - any quality issue is addressed or fails.
    """

    def __init__(self, config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None):
        """Initialize quality orchestrator."""
        self.logger = logger or logging.getLogger(__name__)
        self.config = config

        # Get configuration from ConfigurationManager
        from ..core.configuration_manager import ConfigurationManager
        self.config_manager = ConfigurationManager()
        
        # Get processor config
        try:
            self.processor_config = self.config_manager.get_processor_config('quality_orchestrator')
        except ValueError as e:
            # FAIL LOUD
            raise ValueError(
                f"Failed to load quality_orchestrator configuration!\n{str(e)}"
            )

        # Initialize components
        self.artifact_detector = EnhancedArtifactDetector(logger)
        self.boundary_analyzer = BoundaryAnalyzer(logger)
        self.smart_refiner = SmartRefiner(logger=logger)

        # Quality thresholds from processor config (FAIL LOUD - no defaults)
        self.quality_threshold = self.processor_config['quality_threshold']
        self.max_refinement_passes = self.processor_config['max_refinement_passes']
        self.auto_fix_threshold = self.processor_config['auto_fix_threshold']

    def validate_and_refine(
        self,
        image_path: Path,
        boundary_tracker: BoundaryTracker,
        pipeline_registry: Dict[str, Any],
        config: ExpandorConfig,
    ) -> Dict[str, Any]:
        """
        Validate image quality and refine if needed.

        Args:
            image_path: Path to image to validate
            boundary_tracker: Tracker with boundary information
            pipeline_registry: Available pipelines for refinement
            config: Expansion configuration

        Returns:
            Validation and refinement results

        Raises:
            QualityError: If quality cannot be achieved
        """
        start_time = time.time()

        # Load image
        image = Image.open(image_path)

        # Get boundaries for analysis
        boundaries = self._convert_boundaries(
            boundary_tracker.get_all_boundaries())

        self.logger.info(f"Starting quality validation for {image_path}")

        # Initialize results
        results = {
            "success": False,
            "quality_score": self.processor_config['initial_quality_score'],
            "initial_image": str(image_path),
            "final_image": str(image_path),
            "refinement_passes": 0,
            "artifacts_detected": False,
            "artifacts_fixed": 0,
            "duration": 0.0,
        }

        # Step 1: Comprehensive artifact detection
        try:
            detection_result = self.artifact_detector.detect_artifacts_comprehensive(
                image, boundaries, config.quality_preset)
            results["detection_result"] = detection_result
            results["artifacts_detected"] = detection_result.has_artifacts

        except QualityError as e:
            # Critical issues - fail loud
            self.logger.error(f"Critical quality issues detected: {e}")
            raise

        # Step 2: Boundary analysis
        boundary_analysis = self.boundary_analyzer.analyze_boundaries(
            boundary_tracker, image
        )
        results["boundary_analysis"] = boundary_analysis

        # Step 3: Calculate overall quality score
        quality_score = self._calculate_quality_score(
            detection_result, boundary_analysis
        )
        results["quality_score"] = quality_score

        self.logger.info(
            f"Initial quality score: {quality_score:.3f} "
            f"(threshold: {self.quality_threshold})"
        )

        # Step 4: Refine if needed
        refinement_results = None
        final_image_path = image_path
        refinement_passes = 0

        # Auto-refine is now always enabled for quality
        if quality_score < self.quality_threshold:
            # Find appropriate pipeline
            pipeline = self._select_refinement_pipeline(pipeline_registry)

            if not pipeline:
                self.logger.warning(
                    "Quality below threshold but no pipelines available for refinement"
                )
            else:
                # Attempt refinement
                self.logger.info("Starting smart refinement process")

                # Prepare artifacts for refiner
                artifacts = detection_result.edge_artifacts

                # Refine with multiple passes if needed
                current_image = image
                current_score = quality_score

                for pass_num in range(self.max_refinement_passes):
                    if current_score >= self.quality_threshold:
                        break

                    self.logger.info(
                        f"Refinement pass {pass_num + 1}/{self.max_refinement_passes}"
                    )

                    refinement_result = self.smart_refiner.refine_image(
                        image=current_image,
                        artifacts=artifacts,
                        pipeline=pipeline,
                        prompt=config.prompt,
                        boundaries=boundaries,
                        save_stages=config.save_stages,
                        stage_dir=config.stage_dir,
                    )

                    if refinement_result.success:
                        refinement_passes += 1
                        current_image = Image.open(
                            refinement_result.image_path)

                        # Re-validate after refinement
                        detection_result = (
                            self.artifact_detector.detect_artifacts_comprehensive(
                                current_image, boundaries, config.quality_preset))
                        boundary_analysis = self.boundary_analyzer.analyze_boundaries(
                            boundary_tracker, current_image)
                        current_score = self._calculate_quality_score(
                            detection_result, boundary_analysis
                        )

                        self.logger.info(
                            f"Quality after pass {
                                pass_num +
                                1}: {
                                current_score:.3f}"
                        )

                        results["artifacts_fixed"] += refinement_result.regions_refined
                        final_image_path = refinement_result.image_path
                    else:
                        self.logger.warning(
                            f"Refinement pass {
                                pass_num + 1} failed"
                        )
                        break

                results["refinement_passes"] = refinement_passes
                results["quality_score"] = current_score
                quality_score = current_score

        # Final validation
        if quality_score < self.quality_threshold:
            # Check if we should auto-fix or fail
            if quality_score >= self.auto_fix_threshold:
                self.logger.warning(
                    f"Quality {quality_score:.3f} below ideal but above auto-fix "
                    f"threshold {self.auto_fix_threshold}"
                )
                results["success"] = True
            else:
                # FAIL LOUD
                raise QualityError(
                    f"Unable to achieve required quality: {
                        quality_score:.3f} < {
                        self.quality_threshold}",
                    severity="high",
                    details={
                        "final_score": quality_score,
                        "artifacts": detection_result.has_artifacts,
                        "severity": detection_result.severity.value,
                        "boundary_issues": len(boundary_analysis["issues"]),
                        "refinement_passes": refinement_passes,
                    },
                )
        else:
            results["success"] = True

        # Update final results
        results["final_image"] = str(final_image_path)
        results["duration"] = time.time() - start_time

        # Add recommendations
        results["recommendations"] = self._merge_recommendations(
            detection_result.recommendations, boundary_analysis["recommendations"])

        self.logger.info(
            f"Quality validation complete: score={quality_score:.3f}, "
            f"passes={refinement_passes}, duration={results['duration']:.2f}s"
        )

        return results

    def _convert_boundaries(
            self, boundary_infos: List[Any]) -> List[Dict[str, Any]]:
        """Convert BoundaryInfo objects or dicts to standardized format."""
        boundaries = []
        for info in boundary_infos:
            if isinstance(info, dict):
                # Already in dict format from BoundaryTracker
                boundaries.append(info)
            elif hasattr(info, "position") and hasattr(info, "direction"):
                # Convert object to dict
                position = info.position
                # Position might be a tuple (x1,y1,x2,y2) or a single value
                if isinstance(position, (list, tuple)) and len(position) >= 2:
                    boundary_value = (
                        position[0] if info.direction == "vertical" else position[1])
                else:
                    boundary_value = position

                boundaries.append(
                    {
                        "position": boundary_value,
                        "direction": info.direction,
                        "step": getattr(info, "step", self.processor_config['default_step']),
                        "strength": getattr(info, "denoising_strength", self.processor_config['default_denoising_strength']),
                    }
                )
        return boundaries

    def _calculate_quality_score(
        self, detection_result: Any, boundary_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall quality score from all analyses."""
        # Start with boundary analysis score (already 0-1)
        score = boundary_analysis.get("quality_score", self.processor_config['default_quality_score'])

        # Apply artifact severity penalty
        severity_penalties = self.processor_config['severity_penalties']
        severity_penalty = {
            ArtifactSeverity.NONE: severity_penalties['none'],
            ArtifactSeverity.LOW: severity_penalties['low'],
            ArtifactSeverity.MEDIUM: severity_penalties['medium'],
            ArtifactSeverity.HIGH: severity_penalties['high'],
            ArtifactSeverity.CRITICAL: severity_penalties['critical'],
        }
        score -= severity_penalty.get(detection_result.severity, self.processor_config['default_severity_penalty'])

        # Apply detection score penalties (already normalized 0-1)
        for method, method_score in detection_result.detection_scores.items():
            score -= method_score * self.processor_config['method_score_penalty_multiplier']  # Each method can deduct up to 10%

        # Ensure score is in valid range
        return max(self.processor_config['quality_score_min'], 
                  min(self.processor_config['quality_score_max'], score))

    def _select_refinement_pipeline(
        self, pipeline_registry: Dict[str, Any]
    ) -> Optional[Any]:
        """Select best pipeline for refinement."""
        # Prefer inpaint for targeted fixes
        if "inpaint" in pipeline_registry and pipeline_registry["inpaint"]:
            return pipeline_registry["inpaint"]

        # Fallback to img2img
        if "img2img" in pipeline_registry and pipeline_registry["img2img"]:
            return pipeline_registry["img2img"]

        # Check for any available pipeline
        for name, pipeline in pipeline_registry.items():
            if pipeline is not None:
                return pipeline

        return None

    def _merge_recommendations(self, *recommendation_lists) -> List[str]:
        """Merge and deduplicate recommendations."""
        seen = set()
        merged = []

        for rec_list in recommendation_lists:
            for rec in rec_list:
                if rec not in seen:
                    seen.add(rec)
                    merged.append(rec)

        return merged
