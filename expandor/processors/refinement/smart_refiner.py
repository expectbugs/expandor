"""
Smart refinement for targeted quality improvement
Intelligently refines specific areas based on artifact detection.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ...core.exceptions import QualityError
from ..edge_analysis import EdgeAnalyzer, EdgeInfo

logger = logging.getLogger(__name__)


@dataclass
class RefinementRegion:
    """Region requiring refinement"""

    bounds: Tuple[int, int, int, int]  # x1, y1, x2, y2
    severity: float  # 0-1, higher needs more refinement
    artifact_type: str  # 'seam', 'blur', 'noise', 'pattern'
    refinement_strength: float  # Suggested denoising strength
    mask_blur: int  # Blur radius for mask


@dataclass
class RefinementResult:
    """Result of refinement operation"""

    success: bool
    image_path: Path
    regions_refined: int
    quality_improvement: float
    iterations: int
    refinement_map: Optional[np.ndarray] = None


class SmartRefiner:
    """
    Intelligently refines images based on detected quality issues.

    Features:
    - Targeted refinement of problem areas
    - Multi-pass iterative improvement
    - Adaptive strength based on severity
    - Quality validation after each pass
    """

    def __init__(
        self,
        max_iterations: int = 3,
        quality_threshold: float = 0.85,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize smart refiner.

        Args:
            max_iterations: Maximum refinement passes
            quality_threshold: Target quality score
            logger: Logger instance
            config: Configuration dictionary
        """
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.logger = logger or logging.getLogger(__name__)
        self.edge_analyzer = EdgeAnalyzer(logger=self.logger)
        
        if config is None:
            # Load from processing_params.yaml
            try:
                from ...utils.config_loader import ConfigLoader
                loader = ConfigLoader()
                proc_config = loader.load_config("processing_params.yaml")
                if proc_config and 'smart_refiner' in proc_config:
                    config = proc_config['smart_refiner']
                else:
                    raise ValueError("Missing smart_refiner config in processing_params.yaml")
            except Exception as e:
                # Fail loud - config required
                raise ValueError(f"Failed to load smart refiner configuration: {e}")

        # Refinement parameters from config
        self.base_strength = config.get('base_strength', 0.4)
        self.strength_multiplier = 1.5  # Increase per severity level
        self.min_region_size = config.get('min_region_size', 32)  # Minimum size for refinement region
        
        # Blur radii from config
        self.large_boundary_blur = config.get('large_boundary_blur', 32)
        self.medium_boundary_blur = config.get('medium_boundary_blur', 16)
        self.small_boundary_blur = config.get('small_boundary_blur', 24)
        
        # Other parameters from config
        self.refinement_steps = config.get('refinement_steps', 30)
        self.refinement_guidance = config.get('refinement_guidance', 7.5)

    def refine_image(
        self,
        image: Image.Image,
        artifacts: List[EdgeInfo],
        pipeline: Any,
        prompt: str,
        boundaries: Optional[List[Dict]] = None,
        save_stages: bool = False,
        stage_dir: Optional[Path] = None,
    ) -> RefinementResult:
        """
        Refine image to fix detected artifacts.

        Args:
            image: Image to refine
            artifacts: Detected artifacts to fix
            pipeline: Inpainting pipeline to use
            prompt: Prompt for generation
            boundaries: Known expansion boundaries
            save_stages: Save intermediate results
            stage_dir: Directory for stages

        Returns:
            RefinementResult with refined image
        """
        if not artifacts:
            # No refinement needed
            result_path = Path("temp") / f"refined_{int(time.time())}.png"
            image.save(result_path)

            return RefinementResult(
                success=True,
                image_path=result_path,
                regions_refined=0,
                quality_improvement=0.0,
                iterations=0,
            )

        # Group artifacts into refinement regions
        regions = self._create_refinement_regions(artifacts, image.size)
        self.logger.info(
            f"Created {
                len(regions)} refinement regions from {
                len(artifacts)} artifacts"
        )

        # Iterative refinement
        current_image = image.copy()
        iterations = 0
        total_improvement = 0.0

        for iteration in range(self.max_iterations):
            self.logger.info(
                f"Refinement iteration {iteration + 1}/{self.max_iterations}"
            )

            # Refine each region
            improved = False
            for region in regions:
                try:
                    refined = self._refine_region(
                        current_image, region, pipeline, prompt
                    )
                    if refined is not None:
                        current_image = refined
                        improved = True

                        # Save stage if requested
                        if save_stages and stage_dir:
                            stage_path = (
                                stage_dir
                                / f"refine_iter{iteration}_region{
                                    regions.index(region)}.png"
                            )
                            current_image.save(stage_path)

                except Exception as e:
                    self.logger.error(f"Failed to refine region: {e}")
                    continue

            iterations += 1

            if not improved:
                self.logger.info("No improvements made in this iteration")
                break

            # Check quality after this iteration
            quality_check = self.edge_analyzer.analyze_image(current_image, boundaries)
            current_quality = quality_check["quality_score"]

            self.logger.info(
                f"Quality after iteration {
                    iteration +
                    1}: {
                    current_quality:.3f}"
            )

            if current_quality >= self.quality_threshold:
                self.logger.info(
                    f"Quality threshold reached: {
                        current_quality:.3f} >= {
                        self.quality_threshold}"
                )
                break

            # Update regions for next iteration based on remaining issues
            if quality_check["has_issues"]:
                new_artifacts = (
                    quality_check["seam_artifacts"] + quality_check["general_artifacts"]
                )
                if new_artifacts:
                    regions = self._create_refinement_regions(
                        new_artifacts, current_image.size
                    )
                else:
                    break
            else:
                break

        # Calculate improvement
        original_quality = self.edge_analyzer.analyze_image(image, boundaries)[
            "quality_score"
        ]
        final_quality = self.edge_analyzer.analyze_image(current_image, boundaries)[
            "quality_score"
        ]
        total_improvement = final_quality - original_quality

        # Save final result
        result_path = Path("temp") / f"refined_final_{int(time.time())}.png"
        result_path.parent.mkdir(exist_ok=True)
        current_image.save(result_path)

        # Create refinement map
        refinement_map = self._create_refinement_map(regions, image.size)

        return RefinementResult(
            success=True,
            image_path=result_path,
            regions_refined=len(regions),
            quality_improvement=total_improvement,
            iterations=iterations,
            refinement_map=refinement_map,
        )

    def _create_refinement_regions(
        self, artifacts: List[EdgeInfo], image_size: Tuple[int, int]
    ) -> List[RefinementRegion]:
        """
        Group artifacts into refinement regions.

        Args:
            artifacts: List of detected artifacts
            image_size: (width, height) of image

        Returns:
            List of refinement regions
        """
        regions = []

        # Group nearby artifacts
        processed = set()

        for i, artifact in enumerate(artifacts):
            if i in processed:
                continue

            # Start new region
            x1, y1, x2, y2 = artifact.position
            severity = artifact.strength
            artifact_type = artifact.edge_type

            # Expand region to minimum size
            width = x2 - x1
            height = y2 - y1

            if width < self.min_region_size:
                expand = (self.min_region_size - width) // 2
                x1 = max(0, x1 - expand)
                x2 = min(image_size[0], x2 + expand)

            if height < self.min_region_size:
                expand = (self.min_region_size - height) // 2
                y1 = max(0, y1 - expand)
                y2 = min(image_size[1], y2 + expand)

            # Check for nearby artifacts to merge
            for j, other in enumerate(artifacts):
                if j <= i or j in processed:
                    continue

                ox1, oy1, ox2, oy2 = other.position

                # Check overlap or proximity
                if x1 <= ox2 and x2 >= ox1 and y1 <= oy2 and y2 >= oy1:
                    # Merge regions
                    x1 = min(x1, ox1)
                    y1 = min(y1, oy1)
                    x2 = max(x2, ox2)
                    y2 = max(y2, oy2)
                    severity = max(severity, other.strength)
                    processed.add(j)

            # Calculate refinement parameters
            if artifact_type == "seam":
                strength = min(
                    0.9, self.base_strength * self.strength_multiplier * severity
                )
                blur = getattr(self, 'large_boundary_blur', 32)
            elif artifact_type == "pattern":
                strength = min(0.7, self.base_strength * severity)
                blur = getattr(self, 'medium_boundary_blur', 16)
            else:
                strength = min(0.6, self.base_strength * severity)
                blur = getattr(self, 'small_boundary_blur', 24)

            region = RefinementRegion(
                bounds=(x1, y1, x2, y2),
                severity=severity,
                artifact_type=artifact_type,
                refinement_strength=strength,
                mask_blur=blur,
            )
            regions.append(region)
            processed.add(i)

        return regions

    def _refine_region(
        self, image: Image.Image, region: RefinementRegion, pipeline: Any, prompt: str
    ) -> Optional[Image.Image]:
        """
        Refine a specific region of the image.

        Args:
            image: Current image
            region: Region to refine
            pipeline: Inpainting pipeline
            prompt: Generation prompt

        Returns:
            Refined image or None if failed
        """
        x1, y1, x2, y2 = region.bounds
        width = x2 - x1
        height = y2 - y1

        # Create mask for region
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)

        # Draw region with feathered edges
        if region.mask_blur > 0:
            # Create feathered mask
            inner_margin = region.mask_blur // 2
            draw.rectangle(
                [
                    x1 + inner_margin,
                    y1 + inner_margin,
                    x2 - inner_margin,
                    y2 - inner_margin,
                ],
                fill=255,
            )
            mask = mask.filter(ImageFilter.GaussianBlur(radius=region.mask_blur))
        else:
            draw.rectangle([x1, y1, x2, y2], fill=255)

        # Add context to prompt based on artifact type
        if region.artifact_type == "seam":
            context_prompt = (
                f"{prompt}, seamless blending, smooth transition, no visible seams"
            )
        elif region.artifact_type == "pattern":
            context_prompt = (
                f"{prompt}, natural texture, varied patterns, no repetition"
            )
        else:
            context_prompt = f"{prompt}, high quality, refined details"

        try:
            # Run inpainting
            result = pipeline(
                prompt=context_prompt,
                image=image,
                mask_image=mask,
                strength=region.refinement_strength,
                num_inference_steps=self.refinement_steps,  # Fewer steps for refinement
                guidance_scale=self.refinement_guidance,
            )

            if hasattr(result, "images") and result.images:
                return result.images[0]
            else:
                self.logger.error("Pipeline returned no images")
                return None

        except Exception as e:
            self.logger.error(f"Refinement failed: {e}")
            return None

    def _create_refinement_map(
        self, regions: List[RefinementRegion], image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create visualization of refinement regions.

        Args:
            regions: List of refinement regions
            image_size: (width, height)

        Returns:
            Refinement intensity map
        """
        refinement_map = np.zeros((image_size[1], image_size[0]), dtype=np.float32)

        for region in regions:
            x1, y1, x2, y2 = region.bounds
            refinement_map[y1:y2, x1:x2] = region.severity

        return refinement_map

    def create_quality_report(
        self,
        original: Image.Image,
        refined: Image.Image,
        boundaries: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Create detailed quality comparison report.

        Args:
            original: Original image
            refined: Refined image
            boundaries: Known boundaries

        Returns:
            Quality report dictionary
        """
        # Analyze both images
        original_analysis = self.edge_analyzer.analyze_image(original, boundaries)
        refined_analysis = self.edge_analyzer.analyze_image(refined, boundaries)

        # Calculate improvements
        quality_improvement = (
            refined_analysis["quality_score"] - original_analysis["quality_score"]
        )
        artifacts_removed = len(original_analysis["seam_artifacts"]) - len(
            refined_analysis["seam_artifacts"]
        )

        report = {
            "original_quality": original_analysis["quality_score"],
            "refined_quality": refined_analysis["quality_score"],
            "quality_improvement": quality_improvement,
            "improvement_percentage": (
                quality_improvement / original_analysis["quality_score"]
            )
            * 100,
            "original_artifacts": {
                "seams": len(original_analysis["seam_artifacts"]),
                "general": len(original_analysis["general_artifacts"]),
            },
            "refined_artifacts": {
                "seams": len(refined_analysis["seam_artifacts"]),
                "general": len(refined_analysis["general_artifacts"]),
            },
            "artifacts_removed": artifacts_removed,
            "success": refined_analysis["quality_score"] >= self.quality_threshold,
        }

        return report
