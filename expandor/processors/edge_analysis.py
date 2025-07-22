"""
Edge analysis for artifact detection
Provides edge detection, analysis, and artifact identification.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image, ImageFilter
import logging
from dataclasses import dataclass

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
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Detection thresholds
        self.edge_threshold = 0.1
        self.seam_threshold = 0.3
        self.artifact_threshold = 0.5
        
    def analyze_image(self, image: Image.Image, 
                     boundaries: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Comprehensive edge analysis of an image.
        
        Args:
            image: Image to analyze
            boundaries: Known expansion boundaries
            
        Returns:
            Analysis results including edges, artifacts, quality score
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy
        img_array = np.array(image)
        
        # Detect edges using multiple methods
        edges = self._detect_edges_multi(img_array)
        
        # Analyze known boundaries
        seam_artifacts = []
        if boundaries:
            seam_artifacts = self._analyze_boundaries(img_array, boundaries, edges)
        
        # Find other artifacts
        general_artifacts = self._detect_artifacts(img_array, edges)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            seam_artifacts, general_artifacts, img_array.shape
        )
        
        return {
            'edges': edges,
            'seam_artifacts': seam_artifacts,
            'general_artifacts': general_artifacts,
            'quality_score': quality_score,
            'has_issues': len(seam_artifacts) > 0 or len(general_artifacts) > 0
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
            edges_canny = cv2.Canny(gray, 50, 150)
            
            # Sobel edge detection
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
            edges_sobel = (edges_sobel / edges_sobel.max() * 255).astype(np.uint8)
            
            # Combine
            edges = np.maximum(edges_canny, edges_sobel)
        else:
            # Fallback to PIL-based edge detection
            img_pil = Image.fromarray(gray)
            edges_pil = img_pil.filter(ImageFilter.FIND_EDGES)
            edges = np.array(edges_pil)
        
        return edges
    
    def _analyze_boundaries(self, img_array: np.ndarray, 
                          boundaries: List[Dict],
                          edges: np.ndarray) -> List[EdgeInfo]:
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
            pos = boundary.get('position', 0)
            direction = boundary.get('direction', 'vertical')
            
            if direction == 'vertical' and 0 < pos < width:
                # Check vertical seam
                seam_region = edges[:, max(0, pos-5):min(width, pos+5)]
                seam_strength = np.mean(seam_region) / 255.0
                
                if seam_strength > self.seam_threshold:
                    artifact = EdgeInfo(
                        position=(pos-2, 0, pos+2, height),
                        strength=seam_strength,
                        orientation=np.pi/2,  # Vertical
                        edge_type='seam',
                        confidence=min(seam_strength / self.seam_threshold, 1.0)
                    )
                    seam_artifacts.append(artifact)
                    
            elif direction == 'horizontal' and 0 < pos < height:
                # Check horizontal seam
                seam_region = edges[max(0, pos-5):min(height, pos+5), :]
                seam_strength = np.mean(seam_region) / 255.0
                
                if seam_strength > self.seam_threshold:
                    artifact = EdgeInfo(
                        position=(0, pos-2, width, pos+2),
                        strength=seam_strength,
                        orientation=0,  # Horizontal
                        edge_type='seam',
                        confidence=min(seam_strength / self.seam_threshold, 1.0)
                    )
                    seam_artifacts.append(artifact)
        
        return seam_artifacts
    
    def _detect_artifacts(self, img_array: np.ndarray, 
                         edges: np.ndarray) -> List[EdgeInfo]:
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
                edges, 1, np.pi/180, threshold=100,
                minLineLength=min(width, height) // 4,
                maxLineGap=10
            )
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate line properties
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = np.arctan2(y2-y1, x2-x1)
                    
                    # Check if line is suspiciously straight and long
                    if length > min(width, height) * 0.3:
                        # Check if it's near perfect horizontal/vertical
                        angle_deg = abs(angle * 180 / np.pi)
                        if angle_deg < 5 or angle_deg > 175 or abs(angle_deg - 90) < 5:
                            artifact = EdgeInfo(
                                position=(x1, y1, x2, y2),
                                strength=0.8,
                                orientation=angle,
                                edge_type='artifact',
                                confidence=0.7
                            )
                            artifacts.append(artifact)
        
        # Look for repetitive patterns (could indicate tiling artifacts)
        # This is a simplified check
        for y in range(0, height - 64, 32):
            for x in range(0, width - 64, 32):
                region = edges[y:y+64, x:x+64]
                if np.std(region) > 100:  # High variance indicates pattern
                    artifact = EdgeInfo(
                        position=(x, y, x+64, y+64),
                        strength=0.5,
                        orientation=0,
                        edge_type='pattern',
                        confidence=0.5
                    )
                    artifacts.append(artifact)
        
        return artifacts
    
    def _calculate_quality_score(self, seam_artifacts: List[EdgeInfo],
                               general_artifacts: List[EdgeInfo],
                               image_shape: Tuple[int, ...]) -> float:
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
            return 1.0
        
        # Calculate impact of each artifact type
        seam_impact = 0.0
        for artifact in seam_artifacts:
            # Seams are more severe
            seam_impact += artifact.strength * artifact.confidence * 0.2
        
        artifact_impact = 0.0
        for artifact in general_artifacts:
            artifact_impact += artifact.strength * artifact.confidence * 0.1
        
        # Normalize by image size (larger images can have more artifacts)
        total_pixels = image_shape[0] * image_shape[1]
        size_factor = min(1.0, 1000000 / total_pixels)  # Normalize to 1MP
        
        total_impact = (seam_impact + artifact_impact) * size_factor
        
        # Convert to quality score
        quality_score = max(0.0, 1.0 - total_impact)
        
        return quality_score
    
    def detect_color_discontinuity(self, img1: np.ndarray, img2: np.ndarray,
                                 position: int, direction: str = 'vertical',
                                 threshold: float = 30.0) -> bool:
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
        if direction == 'vertical':
            # Sample colors on both sides of boundary
            if position > 5 and position < img1.shape[1] - 5:
                left_colors = img1[:, position-5:position]
                right_colors = img2[:, position:position+5]
                
                # Calculate mean colors
                left_mean = np.mean(left_colors, axis=(0, 1))
                right_mean = np.mean(right_colors, axis=(0, 1))
                
                # Color difference
                diff = np.linalg.norm(left_mean - right_mean)
                
                return diff > threshold
        else:
            # Horizontal boundary
            if position > 5 and position < img1.shape[0] - 5:
                top_colors = img1[position-5:position, :]
                bottom_colors = img2[position:position+5, :]
                
                top_mean = np.mean(top_colors, axis=(0, 1))
                bottom_mean = np.mean(bottom_colors, axis=(0, 1))
                
                diff = np.linalg.norm(top_mean - bottom_mean)
                
                return diff > threshold
        
        return False
    
    def create_edge_mask(self, edges: List[EdgeInfo], 
                        image_size: Tuple[int, int],
                        dilation: int = 5) -> np.ndarray:
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
            if edge.edge_type == 'seam':
                # Seams get thicker mask
                thickness = dilation * 2
            else:
                thickness = dilation
            
            # Simple rectangle for now (could use line drawing)
            mask[max(0, y1-thickness):min(image_size[1], y2+thickness),
                 max(0, x1-thickness):min(image_size[0], x2+thickness)] = 255
        
        return mask