# Expandor Phase 3 Step 2: Quality Systems - FINAL Complete Implementation Guide

## Overview

This is the comprehensive final implementation guide for Quality Systems in Expandor Phase 3. This document provides complete implementations for enhanced artifact detection, boundary analysis, and quality orchestration that properly integrate with existing components.

## Prerequisites

```bash
# 1. Verify required components exist
python -c "from expandor.processors.edge_analysis import EdgeAnalyzer"
python -c "from expandor.processors.artifact_removal import ArtifactDetector"
python -c "from expandor.processors.refinement.smart_refiner import SmartRefiner"
python -c "from expandor.core.boundary_tracker import BoundaryTracker"

# 2. Verify utilities are available
python -c "from expandor.utils.image_utils import create_gradient_mask"
python -c "from expandor.processors.tiled_processor import TiledProcessor"
```

## 1. Enhanced Artifact Detection System

Since we already have `ArtifactDetector` and `EdgeAnalyzer`, we'll create an enhanced version that extends the existing functionality.

### 1.1 Enhanced Artifact Detector

**FILE**: `expandor/processors/artifact_detector_enhanced.py`

```python
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


## 2. Boundary Analysis System

### 2.1 Boundary Analysis Utilities

**FILE**: `expandor/processors/boundary_analysis.py`

```python
"""
Boundary analysis utilities for enhanced quality control.
Works with the existing BoundaryTracker.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw
import logging

from ..core.boundary_tracker import BoundaryTracker, BoundaryInfo
from ..core.exceptions import QualityError


class BoundaryAnalyzer:
    """
    Analyzes boundaries tracked by BoundaryTracker for quality issues.
    
    Provides detailed analysis of expansion boundaries to identify
    potential problem areas.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize boundary analyzer."""
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_boundaries(self, 
                         boundary_tracker: BoundaryTracker,
                         image: Image.Image) -> Dict[str, Any]:
        """
        Analyze all tracked boundaries for potential issues.
        
        Args:
            boundary_tracker: BoundaryTracker with recorded boundaries
            image: Final image to analyze
            
        Returns:
            Analysis results with issue locations and severity
        """
        boundaries = boundary_tracker.get_all_boundaries()
        critical_boundaries = boundary_tracker.get_critical_boundaries()
        
        analysis = {
            'total_boundaries': len(boundaries),
            'critical_boundaries': len(critical_boundaries),
            'issues': [],
            'severity_map': None,
            'recommendations': [],
            'quality_score': 1.0  # Start with perfect score
        }
        
        # Analyze each boundary
        for boundary in boundaries:
            issue = self._analyze_single_boundary(boundary, image)
            if issue:
                analysis['issues'].append(issue)
                # Deduct from quality score
                if issue['severity'] == 'critical':
                    analysis['quality_score'] -= 0.2
                elif issue['severity'] == 'high':
                    analysis['quality_score'] -= 0.1
                elif issue['severity'] == 'medium':
                    analysis['quality_score'] -= 0.05
        
        # Create severity map
        analysis['severity_map'] = self._create_severity_map(
            boundaries, analysis['issues'], image.size
        )
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(
            analysis['issues'], critical_boundaries
        )
        
        # Ensure quality score doesn't go negative
        analysis['quality_score'] = max(0.0, analysis['quality_score'])
        
        # Log summary
        self.logger.info(
            f"Boundary analysis: {len(boundaries)} boundaries, "
            f"{len(analysis['issues'])} issues found, "
            f"quality score: {analysis['quality_score']:.2f}"
        )
        
        return analysis
    
    def _analyze_single_boundary(self, 
                               boundary: BoundaryInfo,
                               image: Image.Image) -> Optional[Dict[str, Any]]:
        """Analyze a single boundary for issues."""
        x1, y1, x2, y2 = boundary.position
        
        # Extract boundary region with margin
        margin = 10
        region_bounds = (
            max(0, x1 - margin),
            max(0, y1 - margin),
            min(image.width, x2 + margin),
            min(image.height, y2 + margin)
        )
        
        try:
            region = image.crop(region_bounds)
            region_array = np.array(region)
            
            # Analyze based on boundary direction
            if boundary.direction == 'vertical':
                issue = self._analyze_vertical_boundary(
                    region_array, boundary, margin
                )
            else:
                issue = self._analyze_horizontal_boundary(
                    region_array, boundary, margin
                )
            
            return issue
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze boundary: {e}")
            return None
    
    def _analyze_vertical_boundary(self, region_array: np.ndarray,
                                 boundary: BoundaryInfo,
                                 margin: int) -> Optional[Dict[str, Any]]:
        """Analyze vertical boundary for seams."""
        # Check vertical seam
        center_x = margin
        if center_x < region_array.shape[1] - margin:
            left_strip = region_array[:, center_x-5:center_x]
            right_strip = region_array[:, center_x:center_x+5]
            
            # Color difference
            color_diff = np.mean(np.abs(
                left_strip.mean(axis=1) - right_strip.mean(axis=1)
            ))
            
            # Texture difference (using std)
            texture_diff = abs(
                np.std(left_strip) - np.std(right_strip)
            )
            
            # Determine severity
            if color_diff > 40 or texture_diff > 30:
                severity = 'high'
            elif color_diff > 20 or texture_diff > 15:
                severity = 'medium'
            elif color_diff > 10 or texture_diff > 7:
                severity = 'low'
            else:
                return None
            
            return {
                'boundary': boundary,
                'type': 'color_discontinuity',
                'severity': severity,
                'metrics': {
                    'color_difference': float(color_diff),
                    'texture_difference': float(texture_diff)
                },
                'position': boundary.position,
                'direction': 'vertical'
            }
        
        return None
    
    def _analyze_horizontal_boundary(self, region_array: np.ndarray,
                                   boundary: BoundaryInfo,
                                   margin: int) -> Optional[Dict[str, Any]]:
        """Analyze horizontal boundary for seams."""
        # Check horizontal seam
        center_y = margin
        if center_y < region_array.shape[0] - margin:
            top_strip = region_array[center_y-5:center_y, :]
            bottom_strip = region_array[center_y:center_y+5, :]
            
            # Color difference
            color_diff = np.mean(np.abs(
                top_strip.mean(axis=0) - bottom_strip.mean(axis=0)
            ))
            
            # Texture difference
            texture_diff = abs(
                np.std(top_strip) - np.std(bottom_strip)
            )
            
            # Determine severity
            if color_diff > 40 or texture_diff > 30:
                severity = 'high'
            elif color_diff > 20 or texture_diff > 15:
                severity = 'medium'
            elif color_diff > 10 or texture_diff > 7:
                severity = 'low'
            else:
                return None
            
            return {
                'boundary': boundary,
                'type': 'color_discontinuity',
                'severity': severity,
                'metrics': {
                    'color_difference': float(color_diff),
                    'texture_difference': float(texture_diff)
                },
                'position': boundary.position,
                'direction': 'horizontal'
            }
        
        return None
    
    def _create_severity_map(self, 
                           boundaries: List[BoundaryInfo],
                           issues: List[Dict[str, Any]],
                           image_size: Tuple[int, int]) -> np.ndarray:
        """Create a severity heatmap of boundary issues."""
        severity_map = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        
        # Add severity for each issue
        for issue in issues:
            boundary = issue['boundary']
            x1, y1, x2, y2 = boundary.position
            
            severity_value = {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.9,
                'critical': 1.0
            }.get(issue['severity'], 0.5)
            
            # Mark issue region with gradient falloff
            if issue['direction'] == 'vertical':
                # Vertical boundary
                for x in range(max(0, x1-10), min(image_size[0], x2+10)):
                    distance = min(abs(x - x1), abs(x - x2))
                    falloff = max(0, 1 - distance / 10)
                    severity_map[y1:y2, x] = np.maximum(
                        severity_map[y1:y2, x],
                        severity_value * falloff
                    )
            else:
                # Horizontal boundary
                for y in range(max(0, y1-10), min(image_size[1], y2+10)):
                    distance = min(abs(y - y1), abs(y - y2))
                    falloff = max(0, 1 - distance / 10)
                    severity_map[y, x1:x2] = np.maximum(
                        severity_map[y, x1:x2],
                        severity_value * falloff
                    )
        
        return severity_map
    
    def _generate_recommendations(self, issues: List[Dict[str, Any]],
                                critical_boundaries: List[BoundaryInfo]) -> List[str]:
        """Generate recommendations based on issues found."""
        recommendations = []
        
        if not issues:
            recommendations.append("No boundary issues detected. Excellent quality!")
            return recommendations
        
        # Count issue types and severities
        issue_types = {}
        severities = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for issue in issues:
            issue_type = issue['type']
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            severities[issue['severity']] = severities.get(issue['severity'], 0) + 1
        
        # Generate recommendations based on findings
        if issue_types.get('color_discontinuity', 0) > 0:
            recommendations.append(
                f"Found {issue_types['color_discontinuity']} color discontinuities. "
                "Consider using higher denoising strength or additional blur at boundaries."
            )
        
        if severities['high'] > 0 or severities['critical'] > 0:
            recommendations.append(
                f"Found {severities['high'] + severities['critical']} high-severity issues. "
                "Enable auto_refine or use SWPO strategy for smoother transitions."
            )
        
        if len(critical_boundaries) > 0:
            recommendations.append(
                f"{len(critical_boundaries)} boundaries marked as critical. "
                "These require special attention during refinement."
            )
        
        if len(issues) > 5:
            recommendations.append(
                "Multiple boundary issues detected. Consider using a different "
                "expansion strategy or increasing quality settings."
            )
        
        # Add quality preset recommendation
        avg_color_diff = np.mean([
            issue['metrics']['color_difference'] 
            for issue in issues 
            if 'color_difference' in issue.get('metrics', {})
        ])
        
        if avg_color_diff > 25:
            recommendations.append(
                "High average color difference at boundaries. "
                "Use 'ultra' quality preset for best results."
            )
        
        return recommendations
    
    def visualize_severity_map(self, severity_map: np.ndarray,
                             save_path: Optional[Path] = None) -> Image.Image:
        """Create a visual representation of the severity map."""
        # Normalize to 0-255 range
        normalized = (severity_map * 255).astype(np.uint8)
        
        # Create RGB image with color coding
        height, width = severity_map.shape
        rgb_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Color code: Green (good) -> Yellow (medium) -> Red (bad)
        for y in range(height):
            for x in range(width):
                severity = severity_map[y, x]
                if severity < 0.3:
                    # Green
                    rgb_map[y, x] = [0, 255, 0]
                elif severity < 0.6:
                    # Yellow
                    rgb_map[y, x] = [255, 255, 0]
                else:
                    # Red
                    rgb_map[y, x] = [255, 0, 0]
        
        # Apply the severity as alpha
        result = Image.fromarray(rgb_map)
        
        if save_path:
            result.save(save_path)
        
        return result


## 3. Quality Orchestration System

### 3.1 Quality Refinement Orchestrator

**FILE**: `expandor/processors/quality_orchestrator.py`

```python
"""
Quality refinement orchestrator that coordinates all quality systems.
Integrates artifact detection, boundary analysis, and smart refinement.
"""

import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from PIL import Image
import numpy as np

from .artifact_detector_enhanced import EnhancedArtifactDetector, ArtifactSeverity
from .boundary_analysis import BoundaryAnalyzer
from .refinement.smart_refiner import SmartRefiner, RefinementResult
from .edge_analysis import EdgeInfo
from ..core.boundary_tracker import BoundaryTracker
from ..core.exceptions import QualityError
from ..core.config import ExpandorConfig


class QualityOrchestrator:
    """
    Orchestrates all quality systems for comprehensive quality assurance.
    
    Follows FAIL LOUD philosophy - any quality issue is addressed or fails.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None):
        """Initialize quality orchestrator."""
        self.logger = logger or logging.getLogger(__name__)
        self.config = config
        
        # Initialize components
        self.artifact_detector = EnhancedArtifactDetector(logger)
        self.boundary_analyzer = BoundaryAnalyzer(logger)
        self.smart_refiner = SmartRefiner(logger=logger)
        
        # Quality thresholds from config
        quality_config = config.get('quality_validation', {})
        self.quality_threshold = quality_config.get('quality_threshold', 0.85)
        self.max_refinement_passes = quality_config.get('max_refinement_passes', 3)
        self.auto_fix_threshold = quality_config.get('auto_fix_threshold', 0.7)
        
    def validate_and_refine(self,
                          image_path: Path,
                          boundary_tracker: BoundaryTracker,
                          pipeline_registry: Dict[str, Any],
                          config: ExpandorConfig) -> Dict[str, Any]:
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
        boundaries = self._convert_boundaries(boundary_tracker.get_all_boundaries())
        
        self.logger.info(f"Starting quality validation for {image_path}")
        
        # Initialize results
        results = {
            'success': False,
            'quality_score': 0.0,
            'initial_image': str(image_path),
            'final_image': str(image_path),
            'refinement_passes': 0,
            'artifacts_detected': False,
            'artifacts_fixed': 0,
            'duration': 0.0
        }
        
        # Step 1: Comprehensive artifact detection
        try:
            detection_result = self.artifact_detector.detect_artifacts_comprehensive(
                image, boundaries, config.quality_preset
            )
            results['detection_result'] = detection_result
            results['artifacts_detected'] = detection_result.has_artifacts
            
        except QualityError as e:
            # Critical issues - fail loud
            self.logger.error(f"Critical quality issues detected: {e}")
            raise
        
        # Step 2: Boundary analysis
        boundary_analysis = self.boundary_analyzer.analyze_boundaries(
            boundary_tracker, image
        )
        results['boundary_analysis'] = boundary_analysis
        
        # Step 3: Calculate overall quality score
        quality_score = self._calculate_quality_score(
            detection_result, boundary_analysis
        )
        results['quality_score'] = quality_score
        
        self.logger.info(
            f"Initial quality score: {quality_score:.3f} "
            f"(threshold: {self.quality_threshold})"
        )
        
        # Step 4: Refine if needed
        refinement_results = None
        final_image_path = image_path
        refinement_passes = 0
        
        if quality_score < self.quality_threshold and config.auto_refine:
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
                    
                    self.logger.info(f"Refinement pass {pass_num + 1}/{self.max_refinement_passes}")
                    
                    refinement_result = self.smart_refiner.refine_image(
                        image=current_image,
                        artifacts=artifacts,
                        pipeline=pipeline,
                        prompt=config.prompt,
                        boundaries=boundaries,
                        save_stages=config.save_stages,
                        stage_dir=config.stage_dir
                    )
                    
                    if refinement_result.success:
                        refinement_passes += 1
                        current_image = Image.open(refinement_result.image_path)
                        
                        # Re-validate after refinement
                        detection_result = self.artifact_detector.detect_artifacts_comprehensive(
                            current_image, boundaries, config.quality_preset
                        )
                        boundary_analysis = self.boundary_analyzer.analyze_boundaries(
                            boundary_tracker, current_image
                        )
                        current_score = self._calculate_quality_score(
                            detection_result, boundary_analysis
                        )
                        
                        self.logger.info(f"Quality after pass {pass_num + 1}: {current_score:.3f}")
                        
                        results['artifacts_fixed'] += refinement_result.regions_refined
                        final_image_path = refinement_result.image_path
                    else:
                        self.logger.warning(f"Refinement pass {pass_num + 1} failed")
                        break
                
                results['refinement_passes'] = refinement_passes
                results['quality_score'] = current_score
                quality_score = current_score
        
        # Final validation
        if quality_score < self.quality_threshold:
            # Check if we should auto-fix or fail
            if quality_score >= self.auto_fix_threshold:
                self.logger.warning(
                    f"Quality {quality_score:.3f} below ideal but above auto-fix "
                    f"threshold {self.auto_fix_threshold}"
                )
                results['success'] = True
            else:
                # FAIL LOUD
                raise QualityError(
                    f"Unable to achieve required quality: {quality_score:.3f} < {self.quality_threshold}",
                    severity="high",
                    details={
                        'final_score': quality_score,
                        'artifacts': detection_result.has_artifacts,
                        'severity': detection_result.severity.value,
                        'boundary_issues': len(boundary_analysis['issues']),
                        'refinement_passes': refinement_passes
                    }
                )
        else:
            results['success'] = True
        
        # Update final results
        results['final_image'] = str(final_image_path)
        results['duration'] = time.time() - start_time
        
        # Add recommendations
        results['recommendations'] = self._merge_recommendations(
            detection_result.recommendations,
            boundary_analysis['recommendations']
        )
        
        self.logger.info(
            f"Quality validation complete: score={quality_score:.3f}, "
            f"passes={refinement_passes}, duration={results['duration']:.2f}s"
        )
        
        return results
    
    def _convert_boundaries(self, boundary_infos: List[Any]) -> List[Dict[str, Any]]:
        """Convert BoundaryInfo objects to dicts for compatibility."""
        boundaries = []
        for info in boundary_infos:
            if hasattr(info, 'position') and hasattr(info, 'direction'):
                boundaries.append({
                    'position': info.position[0] if info.direction == 'vertical' else info.position[1],
                    'direction': info.direction,
                    'step': getattr(info, 'step', 0),
                    'strength': getattr(info, 'denoising_strength', 0.9)
                })
        return boundaries
    
    def _calculate_quality_score(self, 
                               detection_result: Any,
                               boundary_analysis: Dict[str, Any]) -> float:
        """Calculate overall quality score from all analyses."""
        # Start with boundary analysis score (already 0-1)
        score = boundary_analysis.get('quality_score', 1.0)
        
        # Apply artifact severity penalty
        severity_penalty = {
            ArtifactSeverity.NONE: 0.0,
            ArtifactSeverity.LOW: 0.05,
            ArtifactSeverity.MEDIUM: 0.15,
            ArtifactSeverity.HIGH: 0.30,
            ArtifactSeverity.CRITICAL: 0.50
        }
        score -= severity_penalty.get(detection_result.severity, 0.2)
        
        # Apply detection score penalties (already normalized 0-1)
        for method, method_score in detection_result.detection_scores.items():
            score -= method_score * 0.1  # Each method can deduct up to 10%
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, score))
    
    def _select_refinement_pipeline(self, pipeline_registry: Dict[str, Any]) -> Optional[Any]:
        """Select best pipeline for refinement."""
        # Prefer inpaint for targeted fixes
        if 'inpaint' in pipeline_registry and pipeline_registry['inpaint']:
            return pipeline_registry['inpaint']
        
        # Fallback to img2img
        if 'img2img' in pipeline_registry and pipeline_registry['img2img']:
            return pipeline_registry['img2img']
        
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


## 4. Integration with Expandor

### 4.1 Update Expandor Main Class

Add this method to the main `Expandor` class in `expandor/expandor.py`:

```python
def validate_quality(self, result: ExpandorResult, config: ExpandorConfig) -> Dict[str, Any]:
    """
    Validate and potentially refine the expansion result.
    
    Args:
        result: The expansion result to validate
        config: The configuration used for expansion
        
    Returns:
        Quality validation results
        
    Raises:
        QualityError: If quality requirements cannot be met
    """
    # Initialize quality orchestrator
    quality_config = {
        'quality_validation': {
            'quality_threshold': 0.85 if config.quality_preset == 'ultra' else 0.7,
            'max_refinement_passes': 3,
            'auto_fix_threshold': 0.6
        }
    }
    
    orchestrator = QualityOrchestrator(quality_config, self.logger)
    
    # Prepare pipeline registry
    pipeline_registry = {
        'inpaint': config.inpaint_pipeline,
        'img2img': config.img2img_pipeline
    }
    
    # Validate and refine
    validation_results = orchestrator.validate_and_refine(
        image_path=result.image_path,
        boundary_tracker=self.boundary_tracker,
        pipeline_registry=pipeline_registry,
        config=config
    )
    
    # Update result if refined
    if validation_results['final_image'] != str(result.image_path):
        result.image_path = Path(validation_results['final_image'])
        result.metadata['quality_refined'] = True
        result.metadata['refinement_passes'] = validation_results['refinement_passes']
        result.metadata['final_quality_score'] = validation_results['quality_score']
    
    return validation_results
```

## 5. Unit Tests for Quality Systems

### 5.1 Test Enhanced Artifact Detector

**FILE**: `tests/unit/processors/test_artifact_detector_enhanced.py`

```python
"""
Unit tests for enhanced artifact detection
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from expandor.processors.artifact_detector_enhanced import (
    EnhancedArtifactDetector, ArtifactSeverity
)
from expandor.core.exceptions import QualityError


class TestEnhancedArtifactDetector:
    """Test enhanced artifact detection"""
    
    @pytest.fixture
    def detector(self):
        return EnhancedArtifactDetector()
    
    @pytest.fixture
    def test_image_with_seam(self):
        """Create test image with visible seam"""
        img = Image.new('RGB', (512, 512), color=(100, 100, 100))
        draw = ImageDraw.Draw(img)
        
        # Left half different color
        draw.rectangle([0, 0, 256, 512], fill=(50, 50, 50))
        # Right half different color  
        draw.rectangle([256, 0, 512, 512], fill=(150, 150, 150))
        
        return img
    
    def test_detects_color_discontinuity(self, detector, test_image_with_seam):
        """Test detection of color discontinuity at boundary"""
        boundaries = [{
            'position': 256,
            'direction': 'vertical',
            'step': 0
        }]
        
        result = detector.detect_artifacts_comprehensive(
            test_image_with_seam,
            boundaries,
            'ultra'
        )
        
        assert result.has_artifacts
        assert result.severity in [ArtifactSeverity.HIGH, ArtifactSeverity.CRITICAL]
        assert result.detection_scores['color'] > 0.5
    
    def test_clean_image_passes(self, detector):
        """Test clean image has no artifacts"""
        # Create smooth gradient image
        img = Image.new('RGB', (512, 512))
        pixels = img.load()
        
        for x in range(512):
            for y in range(512):
                # Smooth gradient
                r = int((x / 512) * 255)
                g = int((y / 512) * 255)
                b = 128
                pixels[x, y] = (r, g, b)
        
        result = detector.detect_artifacts_comprehensive(img, [], 'ultra')
        
        assert not result.has_artifacts
        assert result.severity == ArtifactSeverity.NONE
    
    def test_recommendations_generated(self, detector, test_image_with_seam):
        """Test that recommendations are generated"""
        boundaries = [{
            'position': 256,
            'direction': 'vertical',
            'step': 0
        }]
        
        result = detector.detect_artifacts_comprehensive(
            test_image_with_seam,
            boundaries,
            'balanced'
        )
        
        assert len(result.recommendations) > 0
        assert any('discontinuit' in r.lower() for r in result.recommendations)
    
    def test_critical_fails_loud(self, detector):
        """Test critical artifacts raise QualityError"""
        # Create image with extreme artifacts
        img = Image.new('RGB', (512, 512))
        draw = ImageDraw.Draw(img)
        
        # Create checkerboard pattern (extreme artifacts)
        for x in range(0, 512, 32):
            for y in range(0, 512, 32):
                if (x // 32 + y // 32) % 2 == 0:
                    draw.rectangle([x, y, x+32, y+32], fill=(255, 255, 255))
                else:
                    draw.rectangle([x, y, x+32, y+32], fill=(0, 0, 0))
        
        boundaries = [{'position': i * 32, 'direction': 'vertical', 'step': i} 
                     for i in range(16)]
        
        with pytest.raises(QualityError) as exc_info:
            detector.detect_artifacts_comprehensive(img, boundaries, 'ultra')
        
        assert 'Critical' in str(exc_info.value)


### 5.2 Test Boundary Analyzer

**FILE**: `tests/unit/processors/test_boundary_analysis.py`

```python
"""
Unit tests for boundary analysis
"""

import pytest
from PIL import Image, ImageDraw
import numpy as np

from expandor.processors.boundary_analysis import BoundaryAnalyzer
from expandor.core.boundary_tracker import BoundaryTracker


class TestBoundaryAnalyzer:
    """Test boundary analysis"""
    
    @pytest.fixture
    def analyzer(self):
        return BoundaryAnalyzer()
    
    @pytest.fixture
    def test_image_with_seam(self):
        """Create test image with visible seam"""
        img = Image.new('RGB', (512, 512), color=(100, 100, 100))
        draw = ImageDraw.Draw(img)
        
        # Create visible vertical seam
        draw.rectangle([0, 0, 256, 512], fill=(50, 50, 50))
        draw.rectangle([256, 0, 512, 512], fill=(150, 150, 150))
        
        return img
    
    def test_boundary_analysis(self, analyzer, test_image_with_seam):
        """Test boundary analyzer finds issues"""
        tracker = BoundaryTracker()
        tracker.add_boundary(
            position=256,
            direction='vertical',
            step=0,
            expansion_size=256,
            source_size=(256, 512),
            target_size=(512, 512),
            method='test'
        )
        
        analysis = analyzer.analyze_boundaries(tracker, test_image_with_seam)
        
        assert analysis['total_boundaries'] == 1
        assert len(analysis['issues']) > 0
        assert len(analysis['recommendations']) > 0
        assert analysis['quality_score'] < 1.0
    
    def test_clean_boundaries(self, analyzer):
        """Test clean image has good quality score"""
        # Create smooth gradient
        img = Image.new('RGB', (512, 512))
        pixels = img.load()
        
        for x in range(512):
            for y in range(512):
                r = int((x / 512) * 255)
                g = int((y / 512) * 255)
                b = 128
                pixels[x, y] = (r, g, b)
        
        tracker = BoundaryTracker()
        tracker.add_boundary(
            position=256,
            direction='vertical',
            step=0,
            expansion_size=256,
            source_size=(256, 512),
            target_size=(512, 512),
            method='test'
        )
        
        analysis = analyzer.analyze_boundaries(tracker, img)
        
        assert analysis['quality_score'] >= 0.9
        assert len(analysis['issues']) == 0


### 5.3 Test Quality Orchestrator

**FILE**: `tests/unit/processors/test_quality_orchestrator.py`

```python
"""
Unit tests for quality orchestration
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw

from expandor.processors.quality_orchestrator import QualityOrchestrator
from expandor.core.boundary_tracker import BoundaryTracker
from expandor.core.config import ExpandorConfig
from expandor.core.exceptions import QualityError


class TestQualityOrchestrator:
    """Test quality orchestration"""
    
    @pytest.fixture
    def test_image_with_seam(self):
        """Create test image with seam and save it"""
        img = Image.new('RGB', (512, 512), color=(100, 100, 100))
        draw = ImageDraw.Draw(img)
        
        # Create visible seam
        draw.rectangle([0, 0, 256, 512], fill=(50, 50, 50))
        draw.rectangle([256, 0, 512, 512], fill=(150, 150, 150))
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            return Path(tmp.name), img
    
    def test_quality_validation_fail_loud(self, test_image_with_seam):
        """Test quality validation fails loud on bad quality"""
        image_path, _ = test_image_with_seam
        
        # Create boundary tracker with seam location
        tracker = BoundaryTracker()
        tracker.add_boundary(
            position=256,
            direction='vertical',
            step=0,
            expansion_size=256,
            source_size=(256, 512),
            target_size=(512, 512),
            method='test'
        )
        
        # Create config
        config = ExpandorConfig(
            source_image=Image.new('RGB', (256, 512)),
            target_resolution=(512, 512),
            prompt="Test",
            seed=42,
            source_metadata={},
            quality_preset='ultra',
            auto_refine=False  # Disable refinement for this test
        )
        
        # Create orchestrator
        orchestrator = QualityOrchestrator(
            {'quality_validation': {'quality_threshold': 0.9}}, 
            None
        )
        
        # Should fail loud due to quality issues
        with pytest.raises(QualityError) as exc_info:
            orchestrator.validate_and_refine(
                image_path,
                tracker,
                {},  # No pipelines
                config
            )
        
        assert "Unable to achieve required quality" in str(exc_info.value)
        
        # Cleanup
        image_path.unlink()
    
    def test_auto_fix_threshold(self, test_image_with_seam):
        """Test auto-fix threshold allows marginal quality"""
        image_path, _ = test_image_with_seam
        
        tracker = BoundaryTracker()
        config = ExpandorConfig(
            source_image=Image.new('RGB', (256, 512)),
            target_resolution=(512, 512),
            prompt="Test",
            seed=42,
            source_metadata={},
            quality_preset='balanced',
            auto_refine=False
        )
        
        # Create orchestrator with lower auto-fix threshold
        orchestrator = QualityOrchestrator(
            {
                'quality_validation': {
                    'quality_threshold': 0.9,
                    'auto_fix_threshold': 0.5
                }
            }, 
            None
        )
        
        # Should succeed with warning
        result = orchestrator.validate_and_refine(
            image_path,
            tracker,
            {},
            config
        )
        
        assert result['success']
        assert result['quality_score'] < 0.9  # Below ideal
        assert result['quality_score'] >= 0.5  # Above auto-fix
        
        # Cleanup
        image_path.unlink()


## Summary

This final comprehensive implementation guide for Phase 3 Step 2 includes:

1. **Enhanced Artifact Detector** that extends the existing ArtifactDetector with:
   - Multi-method detection (color, gradient, frequency, pattern)
   - Severity classification
   - Comprehensive recommendations
   - FAIL LOUD on critical issues

2. **Boundary Analyzer** that works with existing BoundaryTracker to:
   - Analyze each boundary for quality issues
   - Create severity heatmaps
   - Generate actionable recommendations
   - Calculate quality scores

3. **Quality Orchestrator** that coordinates all systems:
   - Validates image quality comprehensively
   - Manages multi-pass refinement
   - Enforces quality thresholds
   - Provides detailed results and recommendations

4. **Integration guidance** for connecting with main Expandor class

5. **Comprehensive unit tests** for all quality systems

All implementations follow the FAIL LOUD philosophy, properly integrate with existing components, and provide production-ready quality enforcement.