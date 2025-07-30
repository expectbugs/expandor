"""
Enhanced artifact detection with multi-method analysis.
Extends the existing ArtifactDetector with additional capabilities.
"""

import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..core.exceptions import QualityError
from .artifact_removal import ArtifactDetector
from .edge_analysis import EdgeAnalyzer, EdgeInfo


class ArtifactSeverity(IntEnum):
    """Severity levels for detected artifacts"""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DetectionResult:
    """Result from artifact detection"""

    has_artifacts: bool
    severity: ArtifactSeverity
    artifact_mask: Optional[np.ndarray]
    seam_locations: List[Dict[str, Any]]
    detection_scores: Dict[str, float]
    total_artifact_pixels: int
    artifact_percentage: float
    edge_artifacts: List[EdgeInfo]  # From EdgeAnalyzer
    recommendations: List[str]


class EnhancedArtifactDetector(ArtifactDetector):
    """
    Enhanced artifact detection with multiple methods.
    Zero tolerance for visible seams or boundaries.

    Integrates with existing EdgeAnalyzer for comprehensive detection.
    """

    def __init__(
        self, logger: Optional[logging.Logger] = None, quality_preset: str = "balanced"
    ):
        """Initialize with configurable thresholds from config file"""
        super().__init__(logger)
        self.edge_analyzer = EdgeAnalyzer(logger)
        self.quality_preset = quality_preset

        # Load thresholds from config
        try:
            from ..utils.config_loader import ConfigLoader

            loader = ConfigLoader()
            config = loader.load_config("quality_thresholds.yaml")

            if not config or "quality_thresholds" not in config:
                raise ValueError("Invalid quality_thresholds.yaml")

            preset_config = config["quality_thresholds"].get(
                quality_preset, config["quality_thresholds"]["balanced"]
            )

            # Apply thresholds with validation
            self.skip_validation = preset_config.get("skip_validation", False)
            self.seam_threshold = float(preset_config.get("seam_threshold", 0.25))
            self.color_threshold = float(preset_config.get("color_threshold", 30))
            self.gradient_threshold = float(
                preset_config.get("gradient_threshold", 0.25)
            )
            self.frequency_threshold = float(
                preset_config.get("frequency_threshold", 0.35)
            )
            self.min_quality_score = float(preset_config.get("min_quality_score", 0.75))
            self.edge_sensitivity = float(preset_config.get("edge_sensitivity", 0.80))
            
            # Load processing params
            proc_config = loader.load_config("processing_params.yaml")
            if proc_config and 'artifact_detection' in proc_config:
                self.gradient_deviation_allowed = float(
                    proc_config['artifact_detection'].get('gradient_deviation_allowed', 0.1)
                )
            else:
                raise ValueError("Missing processing_params.yaml or artifact_detection config")

            self.logger.info(
                f"Initialized artifact detector with '{quality_preset}' preset: "
                f"seam={
                    self.seam_threshold}, color={
                    self.color_threshold}"
            )

        except Exception as e:
            # FAIL LOUD but with helpful fallback
            self.logger.error(f"Failed to load quality thresholds: {e}")
            self.logger.warning("Using default thresholds")

            # Fail loud - configuration is required
            raise QualityError(
                f"Failed to load quality thresholds configuration: {e}",
                details={
                    "error": str(e),
                    "solution": "Ensure quality_thresholds.yaml exists and is valid"
                }
            )

    def detect_artifacts_comprehensive(
        self,
        image: Image.Image,
        boundaries: List[Dict[str, Any]],
        quality_preset: str = "ultra",
    ) -> DetectionResult:
        """
        Comprehensive artifact detection using all available methods.

        Args:
            image: Image to analyze
            boundaries: Known expansion boundaries
            quality_preset: Quality level affecting thresholds

        Returns:
            Comprehensive detection result

        Raises:
            QualityError: If detection fails
        """
        # Skip validation if configured
        if self.skip_validation:
            return DetectionResult(
                has_artifacts=False,
                severity=ArtifactSeverity.NONE,
                artifact_mask=None,
                seam_locations=[],
                detection_scores={"skipped": 1.0},
                total_artifact_pixels=0,
                artifact_percentage=0.0,
                edge_artifacts=[],
                recommendations=["Validation skipped due to 'fast' preset"],
            )

        if image.mode != "RGB":
            image = image.convert("RGB")

        try:
            # Use EdgeAnalyzer for initial analysis
            edge_analysis = self.edge_analyzer.analyze_image(image, boundaries)

            # Get artifacts from edge analysis
            seam_artifacts = edge_analysis.get("seam_artifacts", [])
            general_artifacts = edge_analysis.get("general_artifacts", [])

            # Run additional detection methods
            detection_scores = {}

            # 1. Color discontinuity at boundaries
            color_score = self._analyze_color_discontinuities(image, boundaries)
            detection_scores["color"] = color_score

            # 2. Gradient analysis
            gradient_score = self._analyze_gradients(image, boundaries)
            detection_scores["gradient"] = gradient_score

            # 3. Frequency domain analysis
            frequency_score = self._analyze_frequency_domain(image)
            detection_scores["frequency"] = frequency_score

            # 4. Pattern detection
            pattern_score = self._detect_repetitive_patterns(image)
            detection_scores["pattern"] = pattern_score

            # Calculate overall severity
            severity = self._calculate_severity(
                detection_scores,
                len(seam_artifacts),
                len(general_artifacts),
                quality_preset,
            )

            # Create artifact mask
            artifact_mask = self._create_comprehensive_mask(
                image.size, seam_artifacts + general_artifacts, boundaries
            )

            # Calculate statistics
            # Count pixels with artifacts (mask values > 0)
            total_artifact_pixels = (
                np.count_nonzero(artifact_mask) if artifact_mask is not None else 0
            )
            total_image_pixels = image.width * image.height
            artifact_percentage = (total_artifact_pixels / total_image_pixels) * 100

            # Prepare seam locations
            seam_locations = []
            for artifact in seam_artifacts:
                seam_locations.append(
                    {
                        "position": artifact.position,
                        "direction": (
                            "vertical"
                            if artifact.orientation == np.pi / 2
                            else "horizontal"
                        ),
                        "strength": artifact.strength,
                        "confidence": artifact.confidence,
                    }
                )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                severity, detection_scores, quality_preset
            )

            result = DetectionResult(
                has_artifacts=severity != ArtifactSeverity.NONE,
                severity=severity,
                artifact_mask=artifact_mask,
                seam_locations=seam_locations,
                detection_scores=detection_scores,
                total_artifact_pixels=int(total_artifact_pixels),
                artifact_percentage=artifact_percentage,
                edge_artifacts=seam_artifacts + general_artifacts,
                recommendations=recommendations,
            )

            # FAIL LOUD if critical AND significant area affected
            if severity == ArtifactSeverity.CRITICAL and artifact_percentage > 5.0:
                self.logger.error(
                    f"CRITICAL artifacts detected: {
                        artifact_percentage:.1f}% of image"
                )
                raise QualityError(
                    f"Critical quality issues detected - {len(seam_artifacts)} seams, "
                    f"{artifact_percentage:.1f}% artifacts",
                    severity="critical",
                    details=result.__dict__,
                )

            return result

        except QualityError:
            raise
        except Exception as e:
            raise QualityError(
                f"Artifact detection failed: {str(e)}", severity="unknown"
            ) from e

    def _analyze_color_discontinuities(
        self, image: Image.Image, boundaries: List[Dict[str, Any]]
    ) -> float:
        """Analyze color discontinuities at boundaries."""
        if not boundaries:
            return 0.0

        img_array = np.array(image)
        discontinuity_scores = []

        for boundary in boundaries:
            pos = boundary.get("position", 0)
            direction = boundary.get("direction", "vertical")

            if direction == "vertical" and 0 < pos < image.width - 1:
                # Check vertical boundary
                left_colors = img_array[:, max(0, pos - 5) : pos].mean(axis=(0, 1))
                right_colors = img_array[:, pos : min(image.width, pos + 5)].mean(
                    axis=(0, 1)
                )
                diff = np.linalg.norm(left_colors - right_colors)
                discontinuity_scores.append(diff / 255.0)

            elif direction == "horizontal" and 0 < pos < image.height - 1:
                # Check horizontal boundary
                top_colors = img_array[max(0, pos - 5) : pos, :].mean(axis=(0, 1))
                bottom_colors = img_array[pos : min(image.height, pos + 5), :].mean(
                    axis=(0, 1)
                )
                diff = np.linalg.norm(top_colors - bottom_colors)
                discontinuity_scores.append(diff / 255.0)

        return max(discontinuity_scores) if discontinuity_scores else 0.0

    def _analyze_gradients(
        self, image: Image.Image, boundaries: List[Dict[str, Any]]
    ) -> float:
        """Analyze gradient discontinuities."""
        # Convert to grayscale for gradient analysis
        gray = image.convert("L")
        gray_array = np.array(gray, dtype=np.float32)

        # Calculate gradients using numpy (cv2 fallback handled by
        # EdgeAnalyzer)
        gy, gx = np.gradient(gray_array)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)

        # Check gradients at boundaries
        boundary_scores = []

        for boundary in boundaries:
            pos = boundary.get("position", 0)
            direction = boundary.get("direction", "vertical")

            if direction == "vertical" and 0 < pos < image.width:
                # Check gradient spike at vertical boundary
                boundary_gradient = gradient_magnitude[
                    :, max(0, pos - 2) : min(image.width, pos + 2)
                ]
                max_gradient = boundary_gradient.max()
                boundary_scores.append(max_gradient / 255.0)

            elif direction == "horizontal" and 0 < pos < image.height:
                # Check gradient spike at horizontal boundary
                boundary_gradient = gradient_magnitude[
                    max(0, pos - 2) : min(image.height, pos + 2), :
                ]
                max_gradient = boundary_gradient.max()
                boundary_scores.append(max_gradient / 255.0)

        return max(boundary_scores) if boundary_scores else 0.0

    def _analyze_frequency_domain(self, image: Image.Image) -> float:
        """Analyze frequency domain for artifacts."""
        # Convert to grayscale
        gray = image.convert("L")
        gray_array = np.array(gray, dtype=np.float32)

        # Apply FFT
        fft = np.fft.fft2(gray_array)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shift) + 1)

        # Look for suspicious patterns in frequency domain
        # High frequency artifacts often indicate processing issues
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2

        # Check high frequency content
        high_freq_region = magnitude_spectrum[
            max(0, center_h - h // 4) : min(h, center_h + h // 4),
            max(0, center_w - w // 4) : min(w, center_w + w // 4),
        ]

        # Normalize and calculate score
        high_freq_score = np.std(high_freq_region) / (
            np.mean(magnitude_spectrum) + 1e-6
        )

        return min(1.0, high_freq_score)

    def _detect_repetitive_patterns(self, image: Image.Image) -> float:
        """Detect repetitive patterns that might indicate tiling artifacts."""
        img_array = np.array(image)

        # Use autocorrelation to detect patterns
        pattern_scores = []

        # Check for gradient-like behavior first
        # In a gradient, the difference should be proportional to distance
        is_gradient = True
        # Use gradient deviation threshold (should be loaded in __init__)
        gradient_threshold = getattr(self, 'gradient_deviation_allowed', 0.1)

        # Check for repeating patterns at different scales
        for shift in [64, 128, 256]:
            if shift < min(image.width, image.height) // 2:
                # Horizontal shift correlation
                if image.width > shift * 2:
                    left = img_array[:, : image.width - shift]
                    right = img_array[:, shift:]
                    # Use mean absolute difference instead of correlation for
                    # efficiency
                    diff = np.mean(np.abs(left - right))

                    # Check if this looks like a gradient
                    # For a gradient, diff should be proportional to shift
                    expected_diff = (shift / image.width) * 255
                    if (
                        expected_diff > 0
                        and abs(diff - expected_diff) / expected_diff
                        > gradient_threshold
                    ):
                        is_gradient = False

                    pattern_scores.append(1.0 - diff / 255.0)

                # Vertical shift correlation
                if image.height > shift * 2:
                    top = img_array[: image.height - shift, :]
                    bottom = img_array[shift:, :]
                    diff = np.mean(np.abs(top - bottom))

                    # Check gradient behavior
                    expected_diff = (shift / image.height) * 255
                    if (
                        expected_diff > 0
                        and abs(diff - expected_diff) / expected_diff
                        > gradient_threshold
                    ):
                        is_gradient = False

                    pattern_scores.append(1.0 - diff / 255.0)

        # If it looks like a gradient, return low score
        if is_gradient:
            return 0.0

        return max(pattern_scores) if pattern_scores else 0.0

    def _calculate_severity(
        self,
        scores: Dict[str, float],
        seam_count: int,
        artifact_count: int,
        quality_preset: str,
    ) -> ArtifactSeverity:
        """Calculate overall severity based on all factors."""
        # Adjust thresholds based on quality preset
        multiplier = {
            "ultra": 0.5,  # Most strict
            "high": 0.7,
            "balanced": 1.0,
            "fast": 1.5,  # Most lenient
        }.get(quality_preset, 1.0)

        # Calculate weighted score
        weighted_score = (
            scores.get("color", 0) * 0.3
            + scores.get("gradient", 0) * 0.3
            + scores.get("frequency", 0) * 0.2
            + scores.get("pattern", 0) * 0.2
        )

        # Add penalty for seam count
        seam_penalty = min(1.0, seam_count * 0.2)
        total_score = weighted_score + seam_penalty

        # Apply quality preset multiplier
        adjusted_score = total_score / multiplier

        # Determine severity
        if adjusted_score < 0.1:
            return ArtifactSeverity.NONE
        elif adjusted_score < 0.3:
            return ArtifactSeverity.LOW
        elif adjusted_score < 0.5:
            return ArtifactSeverity.MEDIUM
        elif adjusted_score < 0.7:
            return ArtifactSeverity.HIGH
        else:
            return ArtifactSeverity.CRITICAL

    def _create_comprehensive_mask(
        self,
        image_size: Tuple[int, int],
        artifacts: List[Any],
        boundaries: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Create comprehensive mask of all detected artifacts."""
        mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

        # Add artifacts from edge analysis
        if hasattr(self.edge_analyzer, "create_edge_mask"):
            edge_mask = self.edge_analyzer.create_edge_mask(
                artifacts, image_size, dilation=10
            )
            mask = np.maximum(mask, edge_mask)

        # Add boundary regions
        for boundary in boundaries:
            pos = boundary.get("position", 0)
            direction = boundary.get("direction", "vertical")

            if direction == "vertical" and 0 < pos < image_size[0]:
                # Mark vertical boundary region
                mask[:, max(0, pos - 5) : min(image_size[0], pos + 5)] = 255
            elif direction == "horizontal" and 0 < pos < image_size[1]:
                # Mark horizontal boundary region
                mask[max(0, pos - 5) : min(image_size[1], pos + 5), :] = 255

        return mask

    def _generate_recommendations(
        self, severity: ArtifactSeverity, scores: Dict[str, float], quality_preset: str
    ) -> List[str]:
        """Generate actionable recommendations based on detection results."""
        recommendations = []

        if severity == ArtifactSeverity.NONE:
            recommendations.append(
                "No significant artifacts detected. Quality is excellent."
            )
            return recommendations

        # Color discontinuities
        if scores.get("color", 0) > 0.5:
            recommendations.append(
                "Strong color discontinuities detected. Consider using higher "
                "denoising strength (0.9+) or additional refinement passes."
            )

        # Gradient issues
        if scores.get("gradient", 0) > 0.5:
            recommendations.append(
                "Sharp gradient transitions found. Enable gradient smoothing "
                "or use SWPO strategy for smoother expansions."
            )

        # Frequency artifacts
        if scores.get("frequency", 0) > 0.5:
            recommendations.append(
                "High-frequency artifacts detected. Consider using a lower "
                "guidance scale or additional blur in transition zones."
            )

        # Pattern repetition
        if scores.get("pattern", 0) > 0.5:
            recommendations.append(
                "Repetitive patterns detected. This may indicate tiling issues. "
                "Consider using larger tile sizes or better overlap blending."
            )

        # Quality preset specific
        if quality_preset == "fast" and severity >= ArtifactSeverity.MEDIUM:
            recommendations.append(
                "Consider using 'balanced' or 'high' quality preset for better results."
            )

        # Severity based
        if severity >= ArtifactSeverity.HIGH:
            recommendations.append(
                "High severity artifacts require immediate attention. "
                "Enable auto_refine or apply manual quality refinement."
            )

        return recommendations
