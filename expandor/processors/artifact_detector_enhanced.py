"""
Enhanced artifact detection with multi-method analysis.
Extends the existing ArtifactDetector with additional capabilities.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from PIL import Image
import logging
from pathlib import Path

from .artifact_removal import ArtifactDetector
from .edge_analysis import EdgeAnalyzer, EdgeInfo
from ..core.exceptions import QualityError


class ArtifactSeverity(Enum):
    """Severity levels for detected artifacts"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


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
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize enhanced detector."""
        super().__init__(logger)
        self.edge_analyzer = EdgeAnalyzer(logger)
        
        # Enhanced thresholds for FAIL LOUD approach
        self.seam_threshold = 0.2  # Lower = more aggressive
        self.color_threshold = 20  # Lower = more sensitive
        self.gradient_threshold = 0.15
        self.frequency_threshold = 0.3
        
    def detect_artifacts_comprehensive(self, 
                                     image: Image.Image,
                                     boundaries: List[Dict[str, Any]],
                                     quality_preset: str = "ultra") -> DetectionResult:
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
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        try:
            # Use EdgeAnalyzer for initial analysis
            edge_analysis = self.edge_analyzer.analyze_image(image, boundaries)
            
            # Get artifacts from edge analysis
            seam_artifacts = edge_analysis.get('seam_artifacts', [])
            general_artifacts = edge_analysis.get('general_artifacts', [])
            
            # Run additional detection methods
            detection_scores = {}
            
            # 1. Color discontinuity at boundaries
            color_score = self._analyze_color_discontinuities(image, boundaries)
            detection_scores['color'] = color_score
            
            # 2. Gradient analysis
            gradient_score = self._analyze_gradients(image, boundaries)
            detection_scores['gradient'] = gradient_score
            
            # 3. Frequency domain analysis
            frequency_score = self._analyze_frequency_domain(image)
            detection_scores['frequency'] = frequency_score
            
            # 4. Pattern detection
            pattern_score = self._detect_repetitive_patterns(image)
            detection_scores['pattern'] = pattern_score
            
            # Calculate overall severity
            severity = self._calculate_severity(
                detection_scores,
                len(seam_artifacts),
                len(general_artifacts),
                quality_preset
            )
            
            # Create artifact mask
            artifact_mask = self._create_comprehensive_mask(
                image.size,
                seam_artifacts + general_artifacts,
                boundaries
            )
            
            # Calculate statistics
            total_pixels = artifact_mask.sum() if artifact_mask is not None else 0
            total_image_pixels = image.width * image.height
            artifact_percentage = (total_pixels / total_image_pixels) * 100
            
            # Prepare seam locations
            seam_locations = []
            for artifact in seam_artifacts:
                seam_locations.append({
                    'position': artifact.position,
                    'direction': 'vertical' if artifact.orientation == np.pi/2 else 'horizontal',
                    'strength': artifact.strength,
                    'confidence': artifact.confidence
                })
            
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
                total_artifact_pixels=int(total_pixels),
                artifact_percentage=artifact_percentage,
                edge_artifacts=seam_artifacts + general_artifacts,
                recommendations=recommendations
            )
            
            # FAIL LOUD if critical
            if severity == ArtifactSeverity.CRITICAL:
                self.logger.error(f"CRITICAL artifacts detected: {artifact_percentage:.1f}% of image")
                raise QualityError(
                    f"Critical quality issues detected - {len(seam_artifacts)} seams, "
                    f"{artifact_percentage:.1f}% artifacts",
                    severity="critical",
                    details=result.__dict__
                )
            
            return result
            
        except QualityError:
            raise
        except Exception as e:
            raise QualityError(
                f"Artifact detection failed: {str(e)}",
                severity="unknown"
            ) from e
    
    def _analyze_color_discontinuities(self, image: Image.Image,
                                     boundaries: List[Dict[str, Any]]) -> float:
        """Analyze color discontinuities at boundaries."""
        if not boundaries:
            return 0.0
        
        img_array = np.array(image)
        discontinuity_scores = []
        
        for boundary in boundaries:
            pos = boundary.get('position', 0)
            direction = boundary.get('direction', 'vertical')
            
            if direction == 'vertical' and 0 < pos < image.width - 1:
                # Check vertical boundary
                left_colors = img_array[:, max(0, pos-5):pos].mean(axis=(0, 1))
                right_colors = img_array[:, pos:min(image.width, pos+5)].mean(axis=(0, 1))
                diff = np.linalg.norm(left_colors - right_colors)
                discontinuity_scores.append(diff / 255.0)
                
            elif direction == 'horizontal' and 0 < pos < image.height - 1:
                # Check horizontal boundary
                top_colors = img_array[max(0, pos-5):pos, :].mean(axis=(0, 1))
                bottom_colors = img_array[pos:min(image.height, pos+5), :].mean(axis=(0, 1))
                diff = np.linalg.norm(top_colors - bottom_colors)
                discontinuity_scores.append(diff / 255.0)
        
        return max(discontinuity_scores) if discontinuity_scores else 0.0
    
    def _analyze_gradients(self, image: Image.Image,
                          boundaries: List[Dict[str, Any]]) -> float:
        """Analyze gradient discontinuities."""
        # Convert to grayscale for gradient analysis
        gray = image.convert('L')
        gray_array = np.array(gray, dtype=np.float32)
        
        # Calculate gradients using numpy (cv2 fallback handled by EdgeAnalyzer)
        gy, gx = np.gradient(gray_array)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # Check gradients at boundaries
        boundary_scores = []
        
        for boundary in boundaries:
            pos = boundary.get('position', 0)
            direction = boundary.get('direction', 'vertical')
            
            if direction == 'vertical' and 0 < pos < image.width:
                # Check gradient spike at vertical boundary
                boundary_gradient = gradient_magnitude[:, max(0, pos-2):min(image.width, pos+2)]
                max_gradient = boundary_gradient.max()
                boundary_scores.append(max_gradient / 255.0)
                
            elif direction == 'horizontal' and 0 < pos < image.height:
                # Check gradient spike at horizontal boundary
                boundary_gradient = gradient_magnitude[max(0, pos-2):min(image.height, pos+2), :]
                max_gradient = boundary_gradient.max()
                boundary_scores.append(max_gradient / 255.0)
        
        return max(boundary_scores) if boundary_scores else 0.0
    
    def _analyze_frequency_domain(self, image: Image.Image) -> float:
        """Analyze frequency domain for artifacts."""
        # Convert to grayscale
        gray = image.convert('L')
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
            max(0, center_h - h//4):min(h, center_h + h//4),
            max(0, center_w - w//4):min(w, center_w + w//4)
        ]
        
        # Normalize and calculate score
        high_freq_score = np.std(high_freq_region) / (np.mean(magnitude_spectrum) + 1e-6)
        
        return min(1.0, high_freq_score)
    
    def _detect_repetitive_patterns(self, image: Image.Image) -> float:
        """Detect repetitive patterns that might indicate tiling artifacts."""
        img_array = np.array(image)
        
        # Use autocorrelation to detect patterns
        pattern_scores = []
        
        # Check for repeating patterns at different scales
        for shift in [64, 128, 256]:
            if shift < min(image.width, image.height) // 2:
                # Horizontal shift correlation
                if image.width > shift * 2:
                    left = img_array[:, :image.width-shift]
                    right = img_array[:, shift:]
                    # Use mean absolute difference instead of correlation for efficiency
                    diff = np.mean(np.abs(left - right))
                    pattern_scores.append(1.0 - diff / 255.0)
                
                # Vertical shift correlation
                if image.height > shift * 2:
                    top = img_array[:image.height-shift, :]
                    bottom = img_array[shift:, :]
                    diff = np.mean(np.abs(top - bottom))
                    pattern_scores.append(1.0 - diff / 255.0)
        
        return max(pattern_scores) if pattern_scores else 0.0
    
    def _calculate_severity(self, scores: Dict[str, float],
                          seam_count: int,
                          artifact_count: int,
                          quality_preset: str) -> ArtifactSeverity:
        """Calculate overall severity based on all factors."""
        # Adjust thresholds based on quality preset
        multiplier = {
            'ultra': 0.5,    # Most strict
            'high': 0.7,
            'balanced': 1.0,
            'fast': 1.5      # Most lenient
        }.get(quality_preset, 1.0)
        
        # Calculate weighted score
        weighted_score = (
            scores.get('color', 0) * 0.3 +
            scores.get('gradient', 0) * 0.3 +
            scores.get('frequency', 0) * 0.2 +
            scores.get('pattern', 0) * 0.2
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
    
    def _create_comprehensive_mask(self, image_size: Tuple[int, int],
                                 artifacts: List[Any],
                                 boundaries: List[Dict[str, Any]]) -> np.ndarray:
        """Create comprehensive mask of all detected artifacts."""
        mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        
        # Add artifacts from edge analysis
        if hasattr(self.edge_analyzer, 'create_edge_mask'):
            edge_mask = self.edge_analyzer.create_edge_mask(
                artifacts, image_size, dilation=10
            )
            mask = np.maximum(mask, edge_mask)
        
        # Add boundary regions
        for boundary in boundaries:
            pos = boundary.get('position', 0)
            direction = boundary.get('direction', 'vertical')
            
            if direction == 'vertical' and 0 < pos < image_size[0]:
                # Mark vertical boundary region
                mask[:, max(0, pos-5):min(image_size[0], pos+5)] = 255
            elif direction == 'horizontal' and 0 < pos < image_size[1]:
                # Mark horizontal boundary region
                mask[max(0, pos-5):min(image_size[1], pos+5), :] = 255
        
        return mask
    
    def _generate_recommendations(self, severity: ArtifactSeverity,
                                scores: Dict[str, float],
                                quality_preset: str) -> List[str]:
        """Generate actionable recommendations based on detection results."""
        recommendations = []
        
        if severity == ArtifactSeverity.NONE:
            recommendations.append("No significant artifacts detected. Quality is excellent.")
            return recommendations
        
        # Color discontinuities
        if scores.get('color', 0) > 0.5:
            recommendations.append(
                "Strong color discontinuities detected. Consider using higher "
                "denoising strength (0.9+) or additional refinement passes."
            )
        
        # Gradient issues
        if scores.get('gradient', 0) > 0.5:
            recommendations.append(
                "Sharp gradient transitions found. Enable gradient smoothing "
                "or use SWPO strategy for smoother expansions."
            )
        
        # Frequency artifacts
        if scores.get('frequency', 0) > 0.5:
            recommendations.append(
                "High-frequency artifacts detected. Consider using a lower "
                "guidance scale or additional blur in transition zones."
            )
        
        # Pattern repetition
        if scores.get('pattern', 0) > 0.5:
            recommendations.append(
                "Repetitive patterns detected. This may indicate tiling issues. "
                "Consider using larger tile sizes or better overlap blending."
            )
        
        # Quality preset specific
        if quality_preset == 'fast' and severity >= ArtifactSeverity.MEDIUM:
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