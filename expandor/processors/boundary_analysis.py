"""
Boundary analysis utilities for enhanced quality control.
Works with the existing BoundaryTracker.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..core.boundary_tracker import BoundaryTracker


class BoundaryAnalyzer:
    """
    Analyzes boundaries tracked by BoundaryTracker for quality issues.

    Provides detailed analysis of expansion boundaries to identify
    potential problem areas.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize boundary analyzer."""
        self.logger = logger or logging.getLogger(__name__)
        
        # Get configuration from ConfigurationManager
        from ..core.configuration_manager import ConfigurationManager
        self.config_manager = ConfigurationManager()
        
        # Get processor config
        try:
            self.processor_config = self.config_manager.get_processor_config('boundary_analysis')
        except ValueError as e:
            # FAIL LOUD
            raise ValueError(
                f"Failed to load boundary_analysis configuration!\n{str(e)}"
            )

    def analyze_boundaries(
        self, boundary_tracker: BoundaryTracker, image: Image.Image
    ) -> Dict[str, Any]:
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
            "total_boundaries": len(boundaries),
            "critical_boundaries": len(critical_boundaries),
            "issues": [],
            "severity_map": None,
            "recommendations": [],
            "quality_score": self.processor_config['initial_quality_score'],  # Start with perfect score
        }

        # Analyze each boundary
        for boundary in boundaries:
            issue = self._analyze_single_boundary(boundary, image)
            if issue:
                analysis["issues"].append(issue)
                # Deduct from quality score
                if issue["severity"] == "critical":
                    analysis["quality_score"] -= self.processor_config['critical_penalty']
                elif issue["severity"] == "high":
                    analysis["quality_score"] -= self.processor_config['high_penalty']
                elif issue["severity"] == "medium":
                    analysis["quality_score"] -= self.processor_config['medium_penalty']

        # Create severity map
        analysis["severity_map"] = self._create_severity_map(
            boundaries, analysis["issues"], image.size
        )

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(
            analysis["issues"], critical_boundaries
        )

        # Ensure quality score doesn't go negative
        analysis["quality_score"] = max(self.processor_config['min_quality_score'], analysis["quality_score"])

        # Log summary
        self.logger.info(
            f"Boundary analysis: {len(boundaries)} boundaries, "
            f"{len(analysis['issues'])} issues found, "
            f"quality score: {analysis['quality_score']:.2f}"
        )

        return analysis

    def _analyze_single_boundary(
        self, boundary: Any, image: Image.Image
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single boundary for issues."""
        # Handle both dict and BoundaryInfo object
        if isinstance(boundary, dict):
            position = boundary["position"]
            direction = boundary["direction"]
        else:
            position = boundary.position
            direction = boundary.direction

        # Create boundary region based on direction
        margin = self.processor_config['boundary_margin']
        if direction == "vertical":
            # Vertical boundary at x=position
            region_bounds = (
                max(0, position - margin),
                0,
                min(image.width, position + margin),
                image.height,
            )
        else:
            # Horizontal boundary at y=position
            region_bounds = (
                0,
                max(0, position - margin),
                image.width,
                min(image.height, position + margin),
            )

        try:
            region = image.crop(region_bounds)
            region_array = np.array(region)

            # Analyze based on boundary direction
            if direction == "vertical":
                issue = self._analyze_vertical_boundary(
                    region_array, boundary, margin, position
                )
            else:
                issue = self._analyze_horizontal_boundary(
                    region_array, boundary, margin, position
                )

            return issue

        except Exception as e:
            self.logger.warning(f"Failed to analyze boundary: {e}")
            return None

    def _analyze_vertical_boundary(
        self, region_array: np.ndarray, boundary: Any, margin: int, position: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze vertical boundary for seams."""
        # Check vertical seam
        center_x = margin
        if center_x <= region_array.shape[1] - margin:
            strip_width = self.processor_config['strip_width']
            left_strip = region_array[:, center_x - strip_width: center_x]
            right_strip = region_array[:, center_x: center_x + strip_width]

            # Color difference
            left_mean = np.mean(
                left_strip, axis=(
                    0, 1))  # Mean color of left strip
            right_mean = np.mean(
                right_strip, axis=(
                    0, 1))  # Mean color of right strip
            color_diff = np.linalg.norm(left_mean - right_mean)

            # Texture difference (using std)
            texture_diff = abs(np.std(left_strip) - np.std(right_strip))

            # Determine severity
            color_thresholds = self.processor_config['color_diff_thresholds']
            texture_thresholds = self.processor_config['texture_diff_thresholds']
            
            if color_diff > color_thresholds['high'] or texture_diff > texture_thresholds['high']:
                severity = "high"
            elif color_diff > color_thresholds['medium'] or texture_diff > texture_thresholds['medium']:
                severity = "medium"
            elif color_diff > color_thresholds['low'] or texture_diff > texture_thresholds['low']:
                severity = "low"
            else:
                return None

            return {
                "boundary": boundary,
                "type": "color_discontinuity",
                "severity": severity,
                "metrics": {
                    "color_difference": float(color_diff),
                    "texture_difference": float(texture_diff),
                },
                "position": position,
                "direction": "vertical",
            }

        return None

    def _analyze_horizontal_boundary(
        self, region_array: np.ndarray, boundary: Any, margin: int, position: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze horizontal boundary for seams."""
        # Check horizontal seam
        center_y = margin
        if center_y <= region_array.shape[0] - margin:
            strip_width = self.processor_config['strip_width']
            top_strip = region_array[center_y - strip_width:center_y, :]
            bottom_strip = region_array[center_y:center_y + strip_width, :]

            # Color difference
            # Mean color of top strip
            top_mean = np.mean(top_strip, axis=(0, 1))
            bottom_mean = np.mean(
                bottom_strip, axis=(0, 1)
            )  # Mean color of bottom strip
            color_diff = np.linalg.norm(top_mean - bottom_mean)

            # Texture difference
            texture_diff = abs(np.std(top_strip) - np.std(bottom_strip))

            # Determine severity
            color_thresholds = self.processor_config['color_diff_thresholds']
            texture_thresholds = self.processor_config['texture_diff_thresholds']
            
            if color_diff > color_thresholds['high'] or texture_diff > texture_thresholds['high']:
                severity = "high"
            elif color_diff > color_thresholds['medium'] or texture_diff > texture_thresholds['medium']:
                severity = "medium"
            elif color_diff > color_thresholds['low'] or texture_diff > texture_thresholds['low']:
                severity = "low"
            else:
                return None

            return {
                "boundary": boundary,
                "type": "color_discontinuity",
                "severity": severity,
                "metrics": {
                    "color_difference": float(color_diff),
                    "texture_difference": float(texture_diff),
                },
                "position": position,
                "direction": "horizontal",
            }

        return None

    def _create_severity_map(
        self,
        boundaries: List[Any],
        issues: List[Dict[str, Any]],
        image_size: Tuple[int, int],
    ) -> np.ndarray:
        """Create a severity heatmap of boundary issues."""
        severity_map = np.zeros(
            (image_size[1], image_size[0]), dtype=np.float32)

        # Add severity for each issue
        for issue in issues:
            position = issue["position"]
            direction = issue["direction"]

            severity_values = self.processor_config['severity_values']
            severity_value = severity_values.get(
                issue["severity"], 
                severity_values['default']
            )

            # Mark issue region with gradient falloff
            if direction == "vertical":
                # Vertical boundary at x=position
                falloff_dist = self.processor_config['gradient_falloff_distance']
                for x in range(
                    max(0, position - falloff_dist), min(image_size[0], position + falloff_dist)
                ):
                    distance = abs(x - position)
                    falloff = max(0, 1 - distance / falloff_dist)
                    severity_map[:, x] = np.maximum(
                        severity_map[:, x], severity_value * falloff
                    )
            else:
                # Horizontal boundary at y=position
                falloff_dist = self.processor_config['gradient_falloff_distance']
                for y in range(
                    max(0, position - falloff_dist), min(image_size[1], position + falloff_dist)
                ):
                    distance = abs(y - position)
                    falloff = max(0, 1 - distance / falloff_dist)
                    severity_map[y, :] = np.maximum(
                        severity_map[y, :], severity_value * falloff
                    )

        return severity_map

    def _generate_recommendations(
        self, issues: List[Dict[str, Any]], critical_boundaries: List[Any]
    ) -> List[str]:
        """Generate recommendations based on issues found."""
        recommendations = []

        if not issues:
            recommendations.append(
                "No boundary issues detected. Excellent quality!")
            return recommendations

        # Count issue types and severities
        issue_types = {}
        severities = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for issue in issues:
            issue_type = issue["type"]
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            severities[issue["severity"]] = severities.get(
                issue["severity"], 0) + 1

        # Generate recommendations based on findings
        if issue_types.get("color_discontinuity", 0) > 0:
            recommendations.append(
                f"Found {issue_types['color_discontinuity']} color discontinuities. "
                "Consider using higher denoising strength or additional blur at boundaries."
            )

        if severities["high"] > 0 or severities["critical"] > 0:
            recommendations.append(
                f"Found {severities['high'] + severities['critical']} high-severity issues. "
                "Enable auto_refine or use SWPO strategy for smoother transitions."
            )

        if len(critical_boundaries) > 0:
            recommendations.append(
                f"{len(critical_boundaries)} boundaries marked as critical. "
                "These require special attention during refinement."
            )

        if len(issues) > self.processor_config['max_issues_before_strategy_recommendation']:
            recommendations.append(
                "Multiple boundary issues detected. Consider using a different "
                "expansion strategy or increasing quality settings.")

        # Add quality preset recommendation
        avg_color_diff = np.mean(
            [
                issue["metrics"]["color_difference"]
                for issue in issues
                if "color_difference" in issue.get("metrics", {})
            ]
        )

        if avg_color_diff > self.processor_config['avg_color_diff_ultra_threshold']:
            recommendations.append(
                "High average color difference at boundaries. "
                "Use 'ultra' quality preset for best results."
            )

        return recommendations

    def visualize_severity_map(
        self, severity_map: np.ndarray, save_path: Optional[Path] = None
    ) -> Image.Image:
        """Create a visual representation of the severity map."""
        # Normalize to 0-255 range
        normalized = (severity_map * self.processor_config['rgb_max_value']).astype(np.uint8)

        # Create RGB image with color coding
        height, width = severity_map.shape
        rgb_map = np.zeros((height, width, 3), dtype=np.uint8)

        # Color code: Green (good) -> Yellow (medium) -> Red (bad)
        for y in range(height):
            for x in range(width):
                severity = severity_map[y, x]
                color_thresholds = self.processor_config['severity_color_thresholds']
                colors = self.processor_config['severity_colors']
                
                if severity < color_thresholds['green_max']:
                    # Green
                    rgb_map[y, x] = colors['green']
                elif severity < color_thresholds['yellow_max']:
                    # Yellow
                    rgb_map[y, x] = colors['yellow']
                else:
                    # Red
                    rgb_map[y, x] = colors['red']

        # Apply the severity as alpha
        result = Image.fromarray(rgb_map)

        if save_path:
            result.save(save_path)

        return result
