"""
Edge analysis for artifact detection
Provides edge detection, analysis, and artifact identification.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger(__name__)


@dataclass
class EdgeInfo:
    """Information about detected edges"""

    position: Tuple[int, int, int, int]  # x1, y1, x2, y2
    strength: float  # Edge strength (0-1)
    orientation: float  # Edge angle in radians
    edge_type: str  # 'seam', 'natural', 'artifact'
    confidence: float  # Detection confidence (0-1)


class EdgeAnalyzer:
    """
    Analyzes edges and boundaries for artifact detection.

    Features:
    - Multiple edge detection algorithms
    - Seam detection at known boundaries
    - Natural vs artificial edge classification
    - Artifact severity assessment
    """

    def __init__(self,
                 logger: Optional[logging.Logger] = None,
                 config: Optional[Dict[str,
                                       Any]] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Get configuration from ConfigurationManager
        from ..core.configuration_manager import ConfigurationManager
        self.config_manager = ConfigurationManager()
        
        # Get processor config
        try:
            self.processor_config = self.config_manager.get_processor_config('edge_analysis')
        except ValueError as e:
            # FAIL LOUD
            raise ValueError(
                f"Failed to load edge_analysis configuration!\n{str(e)}"
            )

        # Store config for later use (keep compatibility)
        self.config = self.processor_config

        # Detection thresholds from config
        self.edge_threshold = self.processor_config['edge_detection_sensitivity']
        self.seam_threshold = self.processor_config['seam_detection_sensitivity']
        self.artifact_threshold = self.processor_config['artifact_detection_sensitivity']
        self.edge_threshold_hough = self.processor_config['edge_threshold_hough']

    def analyze_image(
        self, image: Image.Image, boundaries: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive edge analysis of an image.

        Args:
            image: Image to analyze
            boundaries: Known expansion boundaries

        Returns:
            Analysis results including edges, artifacts, quality score
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy
        img_array = np.array(image)

        # Detect edges using multiple methods
        edges = self._detect_edges_multi(img_array)

        # Analyze known boundaries
        seam_artifacts = []
        if boundaries:
            seam_artifacts = self._analyze_boundaries(
                img_array, boundaries, edges)

        # Find other artifacts
        general_artifacts = self._detect_artifacts(img_array, edges)

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            seam_artifacts, general_artifacts, img_array.shape
        )

        return {
            "edges": edges,
            "seam_artifacts": seam_artifacts,
            "general_artifacts": general_artifacts,
            "quality_score": quality_score,
            "has_issues": len(seam_artifacts) > 0 or len(general_artifacts) > 0,
        }

    def _detect_edges_multi(self, img_array: np.ndarray) -> np.ndarray:
        """
        Detect edges using multiple algorithms and combine results.

        Args:
            img_array: RGB image array

        Returns:
            Combined edge map
        """
        # Convert to grayscale
        gray = np.mean(img_array, axis=2).astype(np.uint8)

        if HAS_CV2:
            # Use OpenCV for better edge detection
            # Canny edge detection
            edges_canny = cv2.Canny(gray, self.processor_config['canny_low_threshold'], self.processor_config['canny_high_threshold'])

            # Sobel edge detection
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
            edges_sobel = (
                edges_sobel /
                edges_sobel.max() *
                255).astype(
                np.uint8)

            # Combine
            edges = np.maximum(edges_canny, edges_sobel)
        else:
            # Fallback to PIL-based edge detection
            img_pil = Image.fromarray(gray)
            edges_pil = img_pil.filter(ImageFilter.FIND_EDGES)
            edges = np.array(edges_pil)

        return edges

    def _analyze_boundaries(
        self, img_array: np.ndarray, boundaries: List[Dict], edges: np.ndarray
    ) -> List[EdgeInfo]:
        """
        Analyze known expansion boundaries for seams.

        Args:
            img_array: Original image
            boundaries: List of boundary positions
            edges: Detected edges

        Returns:
            List of seam artifacts found
        """
        seam_artifacts = []
        height, width = img_array.shape[:2]

        for boundary in boundaries:
            pos = boundary.get("position", 0)
            direction = boundary.get("direction", "vertical")

            if direction == "vertical" and 0 < pos < width:
                # Check vertical seam
                seam_region = edges[:, max(0, pos - self.processor_config['boundary_check_width']): min(width, pos + self.processor_config['boundary_check_width'])]
                seam_strength = np.mean(seam_region) / self.processor_config['edge_normalization_divisor']

                if seam_strength > self.seam_threshold:
                    artifact = EdgeInfo(
                        position=(pos - self.processor_config['boundary_position_offset'], 0, pos + self.processor_config['boundary_position_offset'], height),
                        strength=seam_strength,
                        orientation=self.processor_config['vertical_orientation'],  # Vertical
                        edge_type="seam",
                        confidence=min(
                            seam_strength / self.seam_threshold, 1.0),
                    )
                    seam_artifacts.append(artifact)

            elif direction == "horizontal" and 0 < pos < height:
                # Check horizontal seam
                seam_region = edges[max(0, pos - self.processor_config['boundary_check_width']):min(height, pos + self.processor_config['boundary_check_width']), :]
                seam_strength = np.mean(seam_region) / self.processor_config['edge_normalization_divisor']

                if seam_strength > self.seam_threshold:
                    artifact = EdgeInfo(
                        position=(0, pos - self.processor_config['boundary_position_offset'], width, pos + self.processor_config['boundary_position_offset']),
                        strength=seam_strength,
                        orientation=self.processor_config['horizontal_orientation'],  # Horizontal
                        edge_type="seam",
                        confidence=min(
                            seam_strength / self.seam_threshold, 1.0),
                    )
                    seam_artifacts.append(artifact)

        return seam_artifacts

    def _detect_artifacts(
        self, img_array: np.ndarray, edges: np.ndarray
    ) -> List[EdgeInfo]:
        """
        Detect general artifacts like unnatural edges.

        Args:
            img_array: Original image
            edges: Edge map

        Returns:
            List of detected artifacts
        """
        artifacts = []
        height, width = img_array.shape[:2]

        # Look for suspiciously straight edges
        if HAS_CV2:
            # Use Hough transform to find lines
            lines = cv2.HoughLinesP(
                edges,
                self.processor_config['hough_rho'],
                np.pi / self.processor_config['hough_theta_divisor'],
                threshold=self.edge_threshold_hough,
                minLineLength=min(width, height) // self.processor_config['min_line_length_divisor'],
                maxLineGap=self.processor_config['max_line_gap'],
            )

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]

                    # Calculate line properties
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    angle = np.arctan2(y2 - y1, x2 - x1)

                    # Check if line is suspiciously straight and long
                    if length > min(width, height) * self.processor_config['suspicious_line_length_ratio']:
                        # Check if it's near perfect horizontal/vertical
                        angle_deg = abs(angle * self.processor_config['hough_theta_divisor'] / np.pi)
                        if angle_deg < self.processor_config['angle_tolerance_degrees'] or angle_deg > self.processor_config['straight_angle_threshold'] or abs(
                                angle_deg - self.processor_config['right_angle']) < self.processor_config['angle_tolerance_degrees']:
                            artifact = EdgeInfo(
                                position=(
                                    x1,
                                    y1,
                                    x2,
                                    y2),
                                strength=self.processor_config['artifact_edge_strength'],
                                orientation=angle,
                                edge_type="artifact",
                                confidence=self.processor_config['artifact_confidence'],
                            )
                            artifacts.append(artifact)

        # Look for repetitive patterns (could indicate tiling artifacts)
        # This is a simplified check
        box_size = self.processor_config['pattern_box_size']
        for y in range(0, height - box_size, self.processor_config['pattern_step_size']):
            for x in range(0, width - box_size, self.processor_config['pattern_step_size']):
                region = edges[y:y +
                               self.processor_config['pattern_box_size'], x: x +
                               self.processor_config['pattern_box_size']]
                if np.std(region) > self.processor_config['pattern_variance_threshold']:  # High variance indicates pattern
                    artifact = EdgeInfo(
                        position=(x, y, x + box_size, y + box_size),
                        strength=self.processor_config['pattern_edge_strength'],
                        orientation=0,
                        edge_type="pattern",
                        confidence=self.processor_config['pattern_confidence'],
                    )
                    artifacts.append(artifact)

        return artifacts

    def _calculate_quality_score(
        self,
        seam_artifacts: List[EdgeInfo],
        general_artifacts: List[EdgeInfo],
        image_shape: Tuple[int, ...],
    ) -> float:
        """
        Calculate overall quality score based on artifacts.

        Args:
            seam_artifacts: Detected seam artifacts
            general_artifacts: Other artifacts
            image_shape: Image dimensions

        Returns:
            Quality score (0-1, 1 is perfect)
        """
        if not seam_artifacts and not general_artifacts:
            return self.processor_config['perfect_quality_score']

        # Calculate impact of each artifact type
        seam_impact = 0.0
        for artifact in seam_artifacts:
            # Seams are more severe
            seam_impact += artifact.strength * artifact.confidence * self.processor_config['seam_impact_multiplier']

        artifact_impact = 0.0
        for artifact in general_artifacts:
            artifact_impact += artifact.strength * artifact.confidence * self.processor_config['artifact_impact_multiplier']

        # Normalize by image size (larger images can have more artifacts)
        total_pixels = image_shape[0] * image_shape[1]
        size_factor = min(self.processor_config['quality_score_max'], self.processor_config['normalization_pixels'] / total_pixels)  # Normalize to 1MP

        total_impact = (seam_impact + artifact_impact) * size_factor

        # Convert to quality score
        quality_score = max(self.processor_config['quality_score_min'], self.processor_config['quality_score_max'] - total_impact)

        return quality_score

    def detect_color_discontinuity(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        position: int,
        direction: str = "vertical",
        threshold: float = None,
    ) -> bool:
        """
        Detect color discontinuity between two image regions.

        Args:
            img1: First image region
            img2: Second image region
            position: Boundary position
            direction: 'vertical' or 'horizontal'
            threshold: Color difference threshold

        Returns:
            True if discontinuity detected
        """
        if direction == "vertical":
            # Use default threshold if not provided
            if threshold is None:
                threshold = self.processor_config['default_color_threshold']
            
            # Sample colors on both sides of boundary
            sample_width = self.processor_config['color_sample_width']
            if position > sample_width and position < img1.shape[1] - sample_width:
                left_colors = img1[:, position - sample_width: position]
                right_colors = img2[:, position: position + sample_width]

                # Calculate mean colors
                left_mean = np.mean(left_colors, axis=(0, 1))
                right_mean = np.mean(right_colors, axis=(0, 1))

                # Color difference
                diff = np.linalg.norm(left_mean - right_mean)

                return diff > threshold
        else:
            # Use default threshold if not provided
            if threshold is None:
                threshold = self.processor_config['default_color_threshold']
            
            # Horizontal boundary
            sample_width = self.processor_config['color_sample_width']
            if position > sample_width and position < img1.shape[0] - sample_width:
                top_colors = img1[position - sample_width:position, :]
                bottom_colors = img2[position:position + sample_width, :]

                top_mean = np.mean(top_colors, axis=(0, 1))
                bottom_mean = np.mean(bottom_colors, axis=(0, 1))

                diff = np.linalg.norm(top_mean - bottom_mean)

                return diff > threshold

        return False

    def create_edge_mask(
        self, edges: List[EdgeInfo], image_size: Tuple[int, int], dilation: int = 5
    ) -> np.ndarray:
        """
        Create mask from detected edges.

        Args:
            edges: List of edge information
            image_size: (width, height)
            dilation: Pixels to dilate mask

        Returns:
            Binary mask array
        """
        mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

        for edge in edges:
            x1, y1, x2, y2 = edge.position

            # Draw edge on mask
            if edge.edge_type == "seam":
                # Seams get thicker mask
                thickness = dilation * self.processor_config['seam_thickness_multiplier']
            else:
                thickness = dilation

            # Simple rectangle for now (could use line drawing)
            mask[
                max(0, y1 - thickness): min(image_size[1], y2 + thickness),
                max(0, x1 - thickness): min(image_size[0], x2 + thickness),
            ] = 255

        return mask
