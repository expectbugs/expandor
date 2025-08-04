"""
Artifact Detection and Removal
Adapted from ai-wallpaper SmartArtifactDetector
Original: ai_wallpaper/processing/smart_detector.py
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image


class ArtifactDetector:
    """Aggressive detection for perfect seamless images"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Get configuration from ConfigurationManager
        from ..core.configuration_manager import ConfigurationManager
        self.config_manager = ConfigurationManager()
        
        # Get processor config
        try:
            self.processor_config = self.config_manager.get_processor_config('artifact_removal')
        except ValueError as e:
            # FAIL LOUD
            raise ValueError(
                f"Failed to load artifact_removal configuration!\n{str(e)}"
            )

    def quick_analysis(self, image_path: Path, metadata: Dict) -> Dict:
        """
        Aggressive seam detection with multiple methods.
        Adapted from lines 22-115 of smart_detector.py
        """
        # Load and prepare for detection
        image = Image.open(image_path)
        original_size = image.size

        # Work at higher resolution for better detection (up to 4K)
        scale = min(self.processor_config['scale_min'], self.processor_config['max_detection_resolution'] / max(image.size))
        if scale < 1.0:
            detect_size = (int(image.width * scale), int(image.height * scale))
            detect_image = image.resize(detect_size, Image.Resampling.LANCZOS)
        else:
            detect_image = image
            detect_size = image.size

        img_array = np.array(detect_image)
        h, w = img_array.shape[:2]

        issues_found = False
        problem_mask = None
        severity = "none"
        seam_count = 0

        # 1. Progressive Boundary Detection (lines 49-71)
        # These metadata fields are OPTIONAL - empty lists are valid when no boundaries exist
        boundaries = metadata.get("progressive_boundaries", [])  # OPTIONAL: may be empty
        boundaries_v = metadata.get("progressive_boundaries_vertical", [])  # OPTIONAL: may be empty
        seam_details = metadata.get("seam_details", [])  # OPTIONAL: may be empty

        if boundaries or boundaries_v:
            self.logger.info(
                f"Detecting seams at {len(boundaries)} H + {len(boundaries_v)} V boundaries"
            )

            # Use multiple detection methods
            seam_mask = self._detect_all_seams(
                img_array, boundaries, boundaries_v, seam_details, scale
            )

            if seam_mask is not None:
                problem_mask = seam_mask
                issues_found = True
                severity = "critical"  # All progressive seams are critical
                seam_count = len(boundaries) + len(boundaries_v)
                self.logger.info("CRITICAL: Found progressive expansion seams")

        # 2. Tile boundary detection (lines 73-85)
        if metadata.get("used_tiled", False):
            tile_boundaries = metadata.get("tile_boundaries", [])
            if len(tile_boundaries) > 0:
                tile_mask = self._detect_tile_artifacts(
                    img_array, tile_boundaries, scale
                )
                if tile_mask is not None:
                    if problem_mask is None:
                        problem_mask = tile_mask
                    else:
                        problem_mask = np.maximum(problem_mask, tile_mask)
                    issues_found = True
                    if severity == "none":
                        severity = "high"
                    self.logger.info("Found tile boundary artifacts")

        # 3. General discontinuity detection (lines 87-94)
        if not issues_found:
            discontinuity_mask = self._detect_discontinuities(img_array)
            if discontinuity_mask is not None:
                problem_mask = discontinuity_mask
                issues_found = True
                severity = "medium"
                self.logger.info("Found general discontinuities")

        # Scale mask back to original size
        if issues_found and scale < 1.0:
            mask_pil = Image.fromarray((problem_mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(original_size, Image.Resampling.LANCZOS)
            problem_mask = np.array(mask_pil) / 255.0

        return {
            "needs_multipass": issues_found,
            "mask": problem_mask,
            "severity": severity,
            "seam_count": seam_count,
        }

    def _detect_all_seams(
        self, img_array, boundaries_h, boundaries_v, seam_details, scale
    ):
        """
        Detect seams using multiple methods.
        Implements methods from smart_detector.py
        """
        h, w = img_array.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.float32)

        # Method 1: Color discontinuity detection
        color_mask = self._detect_color_discontinuities(
            img_array, boundaries_h, boundaries_v, scale
        )
        if color_mask is not None:
            combined_mask = np.maximum(combined_mask, color_mask * self.processor_config['color_detection_weight'])

        # Method 2: Gradient detection
        gradient_mask = self._detect_gradient_discontinuities(
            img_array, boundaries_h, boundaries_v, scale
        )
        if gradient_mask is not None:
            combined_mask = np.maximum(combined_mask, gradient_mask * self.processor_config['gradient_detection_weight'])

        # Method 3: Frequency domain detection
        frequency_mask = self._detect_frequency_artifacts(
            img_array, boundaries_h, boundaries_v, scale
        )
        if frequency_mask is not None:
            combined_mask = np.maximum(combined_mask, frequency_mask * self.processor_config['frequency_detection_weight'])

        # Return combined mask if any issues found
        return combined_mask if np.max(combined_mask) > self.processor_config['combined_mask_threshold'] else None

    def _detect_color_discontinuities(
        self, img_array, boundaries_h, boundaries_v, scale
    ):
        """Detect color discontinuities at boundaries"""
        h, w = img_array.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        # Check horizontal boundaries
        for boundary in boundaries_h:
            x = int(boundary * scale)
            if self.processor_config['boundary_margin'] < x < w - self.processor_config['boundary_margin']:
                # Compare color statistics on both sides
                left_region = img_array[:, max(0, x - self.processor_config['color_sample_region_size']): x]
                right_region = img_array[:, x: min(w, x + self.processor_config['color_sample_region_size'])]

                # Calculate color difference
                left_mean = np.mean(left_region, axis=(0, 1))
                right_mean = np.mean(right_region, axis=(0, 1))
                color_diff = np.linalg.norm(left_mean - right_mean)

                if color_diff > self.processor_config['color_difference_threshold']:  # Threshold for significant difference
                    # Mark seam area
                    mask[:, max(0, x - self.processor_config['boundary_mask_width']): min(w, x + self.processor_config['boundary_mask_width'])
                         ] = min(1.0, color_diff / self.processor_config['color_diff_norm_divisor'])

        # Similar for vertical boundaries
        for boundary in boundaries_v:
            y = int(boundary * scale)
            if self.processor_config['boundary_margin'] < y < h - self.processor_config['boundary_margin']:
                top_region = img_array[max(0, y - self.processor_config['color_sample_region_size']):y, :]
                bottom_region = img_array[y:min(h, y + self.processor_config['color_sample_region_size']), :]

                top_mean = np.mean(top_region, axis=(0, 1))
                bottom_mean = np.mean(bottom_region, axis=(0, 1))
                color_diff = np.linalg.norm(top_mean - bottom_mean)

                if color_diff > self.processor_config['color_difference_threshold']:
                    mask[max(0, y - self.processor_config['boundary_mask_width']):min(h, y + self.processor_config['boundary_mask_width']),
                         :] = min(1.0, color_diff / self.processor_config['color_diff_norm_divisor'])

        return mask if np.max(mask) > 0 else None

    def _detect_gradient_discontinuities(
        self, img_array, boundaries_h, boundaries_v, scale
    ):
        """Detect gradient discontinuities using Sobel filters"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.float32)

        # Check for abnormal gradients at boundaries
        for boundary in boundaries_h:
            x = int(boundary * scale)
            if self.processor_config['gradient_mask_width'] < x < w - self.processor_config['gradient_mask_width']:
                # Look for gradient spikes at boundary
                boundary_grad = grad_mag[:, max(0, x - self.processor_config['gradient_boundary_check_width']): min(w, x + self.processor_config['gradient_boundary_check_width'])]
                if np.mean(boundary_grad) > np.mean(grad_mag) * self.processor_config['gradient_spike_multiplier']:
                    mask[:, max(0, x - self.processor_config['gradient_mask_width']): min(w, x + self.processor_config['gradient_mask_width'])] = self.processor_config['gradient_mask_value']

        for boundary in boundaries_v:
            y = int(boundary * scale)
            if self.processor_config['gradient_mask_width'] < y < h - self.processor_config['gradient_mask_width']:
                boundary_grad = grad_mag[max(0, y - self.processor_config['gradient_boundary_check_width']):min(h, y + self.processor_config['gradient_boundary_check_width']), :]
                if np.mean(boundary_grad) > np.mean(grad_mag) * self.processor_config['gradient_spike_multiplier']:
                    mask[max(0, y - self.processor_config['gradient_mask_width']):min(h, y + self.processor_config['gradient_mask_width']), :] = self.processor_config['gradient_mask_value']

        return mask if np.max(mask) > 0 else None

    def _detect_frequency_artifacts(
            self,
            img_array,
            boundaries_h,
            boundaries_v,
            scale):
        """Detect artifacts in frequency domain"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + self.processor_config['fft_epsilon'])

        # Look for periodic patterns that indicate seams
        # This is a simplified version - real implementation would be more
        # sophisticated
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.float32)

        # Check for regular patterns at boundary locations
        for boundary in boundaries_h:
            x = int(boundary * scale)
            if 0 < x < w:
                # Simple heuristic: check for vertical lines in frequency
                # domain
                freq_x = int(
                    w / 2 + (x - w / 2) * self.processor_config['frequency_location_scale']
                )  # Approximate frequency location
                if (
                    np.mean(magnitude_spectrum[:, freq_x])
                    > np.mean(magnitude_spectrum) * self.processor_config['magnitude_spectrum_multiplier']
                ):
                    mask[:, max(0, x - self.processor_config['frequency_mask_width']): min(w, x + self.processor_config['frequency_mask_width'])] = self.processor_config['frequency_mask_value']

        return mask if np.max(mask) > 0 else None

    def _detect_tile_artifacts(self, img_array, tile_boundaries, scale):
        """Detect artifacts at tile boundaries"""
        # Similar to seam detection but for tile grid patterns
        h, w = img_array.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        for tx, ty in tile_boundaries:
            # Scale tile coordinates
            tx_scaled = int(tx * scale)
            ty_scaled = int(ty * scale)

            # Mark tile boundaries
            if 0 < tx_scaled < w:
                mask[:, max(0, tx_scaled - self.processor_config['tile_boundary_width']): min(w, tx_scaled + self.processor_config['tile_boundary_width'])] = self.processor_config['tile_mask_value']
            if 0 < ty_scaled < h:
                mask[max(0, ty_scaled - self.processor_config['tile_boundary_width']):min(h, ty_scaled + self.processor_config['tile_boundary_width']), :] = self.processor_config['tile_mask_value']

        return mask if np.max(mask) > 0 else None

    def _detect_discontinuities(self, img_array):
        """General discontinuity detection as fallback"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, self.processor_config['canny_threshold_low'], self.processor_config['canny_threshold_high'])

        # Look for long straight lines (potential seams)
        lines = cv2.HoughLinesP(
            edges, self.processor_config['hough_rho'], np.pi / self.processor_config['hough_theta_divisor'], self.processor_config['hough_threshold'], 
            minLineLength=self.processor_config['hough_min_line_length'], maxLineGap=self.processor_config['hough_max_line_gap']
        )

        if lines is not None:
            h, w = gray.shape
            mask = np.zeros((h, w), dtype=np.float32)

            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Draw line on mask
                cv2.line(mask, (x1, y1), (x2, y2), self.processor_config['hough_line_mask_value'], thickness=self.processor_config['hough_line_thickness'])

            return mask if np.max(mask) > 0 else None

        return None
