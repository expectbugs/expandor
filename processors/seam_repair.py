"""
Seam repair processor for artifact removal
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import time

class SeamRepairProcessor:
    """
    Repairs detected seams and artifacts using available pipelines
    
    Implements targeted inpainting to fix quality issues
    """
    
    def __init__(self, pipelines: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize seam repair processor
        
        Args:
            pipelines: Dictionary of available pipelines (inpaint, refiner, etc)
            logger: Logger instance
        """
        self.pipelines = pipelines
        self.logger = logger or logging.getLogger(__name__)
        
        # Determine available repair methods
        self.has_inpaint = 'inpaint' in pipelines
        self.has_refiner = 'refiner' in pipelines
        self.has_img2img = 'img2img' in pipelines
        
        if not any([self.has_inpaint, self.has_refiner, self.has_img2img]):
            self.logger.warning("No repair pipelines available - repair functionality limited")
    
    def repair_seams(self,
                     image_path: Path,
                     artifact_mask: Optional[np.ndarray],
                     prompt: str,
                     metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair detected seams and artifacts
        
        Args:
            image_path: Path to image with artifacts
            artifact_mask: Mask indicating artifact locations (0-1 values)
            prompt: Original generation prompt
            metadata: Image metadata with boundary information
            
        Returns:
            Dictionary with repaired image path and updated metadata
        """
        start_time = time.time()
        
        # Load image
        image = Image.open(image_path)
        original_size = image.size
        
        if artifact_mask is None:
            self.logger.warning("No artifact mask provided - cannot repair")
            return {
                "image_path": image_path,
                "metadata": metadata,
                "seams_repaired": 0,
                "repair_duration": 0
            }
        
        # Count seams to repair
        seam_regions = self._identify_seam_regions(artifact_mask)
        self.logger.info(f"Found {len(seam_regions)} seam regions to repair")
        
        # Select repair method based on available pipelines
        if self.has_inpaint:
            repaired_image = self._repair_with_inpaint(image, artifact_mask, prompt, metadata)
        elif self.has_refiner:
            repaired_image = self._repair_with_refiner(image, artifact_mask, prompt, metadata)
        elif self.has_img2img:
            repaired_image = self._repair_with_img2img(image, artifact_mask, prompt, metadata)
        else:
            # Fallback to basic filtering
            repaired_image = self._repair_with_filtering(image, artifact_mask)
        
        # Save repaired image
        timestamp = int(time.time() * 1000)
        repair_path = image_path.parent / f"repaired_{timestamp}.png"
        repaired_image.save(repair_path, "PNG", compress_level=0)
        
        # Update metadata
        updated_metadata = metadata.copy()
        updated_metadata['repair_applied'] = True
        updated_metadata['repair_method'] = self._get_repair_method()
        updated_metadata['seams_repaired'] = len(seam_regions)
        
        duration = time.time() - start_time
        self.logger.info(f"Repair completed in {duration:.1f}s")
        
        return {
            "image_path": repair_path,
            "metadata": updated_metadata,
            "seams_repaired": len(seam_regions),
            "repair_duration": duration
        }
    
    def _repair_with_inpaint(self, 
                            image: Image.Image,
                            artifact_mask: np.ndarray,
                            prompt: str,
                            metadata: Dict[str, Any]) -> Image.Image:
        """Repair using inpainting pipeline"""
        self.logger.info("Repairing with inpainting pipeline")
        
        # Convert mask to PIL image
        # Expand mask slightly for better coverage
        expanded_mask = self._expand_mask(artifact_mask, pixels=10)
        mask_image = Image.fromarray((expanded_mask * 255).astype(np.uint8))
        
        # Apply slight blur to mask edges
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=5))
        
        # Enhance prompt for repair
        repair_prompt = prompt + ", seamless, continuous, smooth transitions"
        
        # Run inpainting
        pipeline = self.pipelines['inpaint']
        result = pipeline(
            prompt=repair_prompt,
            image=image,
            mask_image=mask_image,
            strength=0.8,  # High strength for seam repair
            guidance_scale=7.5,
            num_inference_steps=50,
            width=image.width,
            height=image.height
        )
        
        return result.images[0]
    
    def _repair_with_refiner(self,
                            image: Image.Image,
                            artifact_mask: np.ndarray,
                            prompt: str,
                            metadata: Dict[str, Any]) -> Image.Image:
        """Repair using refiner pipeline"""
        self.logger.info("Repairing with refiner pipeline")
        
        # For refiner, we process the whole image but with targeted strength
        pipeline = self.pipelines['refiner']
        
        # Create strength map from mask
        # Higher strength where artifacts are detected
        base_strength = 0.2
        artifact_strength = 0.5
        
        # Process with refiner
        result = pipeline(
            prompt=prompt + ", high quality, refined details",
            image=image,
            strength=artifact_strength,  # Use higher strength
            guidance_scale=7.5,
            num_inference_steps=30
        )
        
        # Blend based on mask
        original_array = np.array(image)
        refined_array = np.array(result.images[0])
        
        # Expand mask for smoother blending
        blend_mask = self._expand_mask(artifact_mask, pixels=20)
        blend_mask = self._smooth_mask(blend_mask)
        
        # Blend images
        for c in range(3):
            original_array[:, :, c] = (
                original_array[:, :, c] * (1 - blend_mask) +
                refined_array[:, :, c] * blend_mask
            ).astype(np.uint8)
        
        return Image.fromarray(original_array)
    
    def _repair_with_img2img(self,
                            image: Image.Image,
                            artifact_mask: np.ndarray,
                            prompt: str,
                            metadata: Dict[str, Any]) -> Image.Image:
        """Repair using img2img pipeline"""
        self.logger.info("Repairing with img2img pipeline")
        
        # Similar to refiner but with img2img
        pipeline = self.pipelines['img2img']
        
        result = pipeline(
            prompt=prompt + ", seamless, high quality",
            image=image,
            strength=0.4,
            guidance_scale=7.5,
            num_inference_steps=40
        )
        
        # Blend based on mask
        original_array = np.array(image)
        processed_array = np.array(result.images[0])
        
        blend_mask = self._expand_mask(artifact_mask, pixels=15)
        blend_mask = self._smooth_mask(blend_mask)
        
        for c in range(3):
            original_array[:, :, c] = (
                original_array[:, :, c] * (1 - blend_mask) +
                processed_array[:, :, c] * blend_mask
            ).astype(np.uint8)
        
        return Image.fromarray(original_array)
    
    def _repair_with_filtering(self, image: Image.Image, artifact_mask: np.ndarray) -> Image.Image:
        """Basic repair using image filtering (fallback)"""
        self.logger.warning("Using basic filtering for repair (no pipelines available)")
        
        # Convert to array
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Identify artifact regions
        mask_binary = (artifact_mask > 0.5).astype(np.uint8)
        
        # Apply median filter to artifact regions
        from scipy.ndimage import median_filter
        
        for c in range(3):
            channel = img_array[:, :, c]
            filtered = median_filter(channel, size=5)
            
            # Blend filtered version in artifact regions
            img_array[:, :, c] = np.where(
                mask_binary,
                filtered,
                channel
            )
        
        # Apply slight gaussian blur to smoothen
        result = Image.fromarray(img_array)
        
        # Create smooth blend mask
        blend_mask = Image.fromarray((artifact_mask * 255).astype(np.uint8))
        blend_mask = blend_mask.filter(ImageFilter.GaussianBlur(radius=10))
        
        # Blend with slightly blurred version
        blurred = result.filter(ImageFilter.GaussianBlur(radius=2))
        result = Image.composite(blurred, result, blend_mask)
        
        return result
    
    def _identify_seam_regions(self, mask: np.ndarray) -> list:
        """Identify distinct seam regions from mask"""
        from scipy import ndimage
        
        # Threshold mask
        binary_mask = (mask > 0.1).astype(np.uint8)
        
        # Label connected components
        labeled, num_features = ndimage.label(binary_mask)
        
        regions = []
        for i in range(1, num_features + 1):
            region_mask = (labeled == i)
            regions.append({
                'id': i,
                'pixels': np.sum(region_mask),
                'bounds': self._get_region_bounds(region_mask)
            })
        
        return regions
    
    def _get_region_bounds(self, mask: np.ndarray) -> tuple:
        """Get bounding box of a mask region"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return (rmin, cmin, rmax, cmax)
    
    def _expand_mask(self, mask: np.ndarray, pixels: int = 10) -> np.ndarray:
        """Expand mask by specified pixels"""
        from scipy.ndimage import binary_dilation
        
        # Create structuring element
        struct = np.ones((pixels*2+1, pixels*2+1))
        
        # Dilate mask
        expanded = binary_dilation(mask > 0.1, structure=struct)
        
        return expanded.astype(np.float32)
    
    def _smooth_mask(self, mask: np.ndarray, sigma: float = 5.0) -> np.ndarray:
        """Smooth mask with gaussian blur"""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(mask, sigma=sigma)
    
    def _get_repair_method(self) -> str:
        """Get the repair method used"""
        if self.has_inpaint:
            return "inpainting"
        elif self.has_refiner:
            return "refinement"
        elif self.has_img2img:
            return "img2img"
        else:
            return "filtering"