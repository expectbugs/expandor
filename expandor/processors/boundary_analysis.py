"""
Boundary analysis utilities for enhanced quality control.
Works with the existing BoundaryTracker.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
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