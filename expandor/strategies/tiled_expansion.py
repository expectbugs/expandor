"""
Tiled Expansion Strategy
Process large images in tiles when VRAM is limited
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..core.config import ExpandorConfig
from ..core.exceptions import StrategyError
from .base_strategy import BaseExpansionStrategy


class TiledExpansionStrategy(BaseExpansionStrategy):
    """
    Tiled processing for VRAM-limited situations

    Processes image in overlapping tiles to handle any size
    without quality compromise - just takes longer
    """

    def _initialize(self):
        """Initialize tiled processing settings"""
        # Default tile settings
        self.default_tile_size = 1024
        self.overlap = 256
        self.blend_width = 256  # Increased for smoother tile blending
        self.min_tile_size = 512
        self.max_tile_size = 2048

    def validate_requirements(self):
        """Validate at least one pipeline is available"""
        if not any(
            [self.inpaint_pipeline, self.refiner_pipeline, self.img2img_pipeline]
        ):
            raise StrategyError(
                "TiledExpansionStrategy requires at least one pipeline "
                "(inpaint, refiner, or img2img)"
            )

    def execute(
        self, config, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute tiled expansion strategy"""
        self._context = context or {}
        start_time = time.time()

        # Validate requirements
        self.validate_requirements()

        # Load source image
        if isinstance(config.source_image, Path):
            source_image = self.validate_image_path(config.source_image)
        else:
            source_image = config.source_image

        source_w, source_h = source_image.size
        target_w, target_h = config.get_target_resolution()

        self.logger.info(
            f"Tiled expansion: {source_w}x{source_h} -> {target_w}x{target_h}"
        )

        # Determine optimal tile size based on VRAM
        tile_size = self._determine_tile_size(target_w, target_h)
        self.logger.info(
            f"Using tile size: {tile_size}x{tile_size} with {
                self.overlap}px overlap"
        )

        # First, upscale to target size using simple resize
        # This gives us the base to refine
        stage_start = time.time()

        base_image = source_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        base_path = self.save_temp_image(base_image, "tiled_base")

        self.record_stage(
            name="initial_resize",
            method="lanczos",
            input_size=(source_w, source_h),
            output_size=(target_w, target_h),
            start_time=stage_start,
        )

        # Calculate tiles
        tiles = self._calculate_tiles(target_w, target_h, tile_size)
        self.logger.info(f"Processing {len(tiles)} tiles")

        # Track tile boundaries
        if self.boundary_tracker:
            # Add tile boundaries
            for x1, y1, x2, y2 in tiles:
                # Add vertical boundaries
                if x1 > 0:
                    self.boundary_tracker.add_boundary(
                        position=x1,
                        direction="horizontal",
                        step=1,
                        expansion_size=0,
                        source_size=(target_w, target_h),
                        target_size=(target_w, target_h),
                        method="tiled",
                        metadata={"boundary_type": "tile_edge"},
                    )
                # Add horizontal boundaries
                if y1 > 0:
                    self.boundary_tracker.add_boundary(
                        position=y1,
                        direction="vertical",
                        step=1,
                        expansion_size=0,
                        source_size=(target_w, target_h),
                        target_size=(target_w, target_h),
                        method="tiled",
                        metadata={"boundary_type": "tile_edge"},
                    )

        # Process each tile
        processed_tiles = []

        for i, (x1, y1, x2, y2) in enumerate(tiles):
            tile_num = i + 1
            self.logger.info(f"Processing tile {tile_num}/{len(tiles)}")

            tile_start = time.time()

            # Extract tile from base image
            tile = base_image.crop((x1, y1, x2, y2))
            tile_w = x2 - x1
            tile_h = y2 - y1

            # Process tile based on available pipelines
            if self.refiner_pipeline:
                processed_tile = self._refine_tile(tile, config.prompt)
            elif self.img2img_pipeline:
                processed_tile = self._img2img_tile(tile, config.prompt)
            elif self.inpaint_pipeline:
                # For inpaint, we need to create a mask
                processed_tile = self._inpaint_tile(tile, config.prompt)
            else:
                # Should not happen due to validation
                raise StrategyError("No suitable pipeline for tile processing")

            # Store processed tile with coordinates
            processed_tiles.append(
                {"image": processed_tile, "coords": (x1, y1, x2, y2)}
            )

            self.record_stage(
                name=f"tile_{tile_num}",
                method="tiled_process",
                input_size=(tile_w, tile_h),
                output_size=(tile_w, tile_h),
                start_time=tile_start,
                metadata={"tile_index": i, "tile_coords": (x1, y1, x2, y2)},
            )

        # Blend tiles back together
        self.logger.info("Blending tiles...")
        blend_start = time.time()

        final_image = self._blend_tiles(
            processed_tiles, (target_w, target_h), self.overlap, self.blend_width
        )

        final_path = self.save_temp_image(final_image, "tiled_final")

        self.record_stage(
            name="tile_blending",
            method="weighted_blend",
            input_size=(target_w, target_h),
            output_size=(target_w, target_h),
            start_time=blend_start,
            metadata={"tiles_blended": len(tiles)},
        )

        # Get boundaries for metadata
        boundaries = []
        if self.boundary_tracker:
            boundaries = self.boundary_tracker.get_all_boundaries()

        return {
            "image": final_image,
            "image_path": final_path,
            "size": (target_w, target_h),
            "stages": self.stage_results,
            "boundaries": boundaries,
            "metadata": {
                "strategy": "tiled_expansion",
                "tile_size": tile_size,
                "tile_count": len(tiles),
                "overlap": self.overlap,
            },
        }

    def _determine_tile_size(self, width: int, height: int) -> int:
        """Determine optimal tile size based on VRAM"""
        # Check available VRAM
        available_vram = self.vram_manager.get_available_vram()

        if not available_vram:
            # CPU mode - use minimum tile size
            return self.min_tile_size

        # Calculate VRAM needed for different tile sizes
        for tile_size in [self.max_tile_size, 1536, 1024, 768, self.min_tile_size]:
            vram_req = self.vram_manager.calculate_generation_vram(tile_size, tile_size)
            if vram_req <= available_vram * 0.7:  # 70% safety
                return tile_size

        # If even minimum doesn't fit, return it anyway (will fail loudly)
        return self.min_tile_size

    def _calculate_tiles(
        self, width: int, height: int, tile_size: int
    ) -> List[Tuple[int, int, int, int]]:
        """Calculate tile positions with overlap"""
        tiles = []

        # Calculate step size (tile size minus overlap)
        step = tile_size - self.overlap

        # Ensure we don't have tiny edge tiles
        min_edge_size = tile_size // 2

        # Generate tiles
        y = 0
        while y < height:
            # Adjust last row if needed
            if y + tile_size > height and height - y < min_edge_size:
                y = max(0, height - tile_size)

            x = 0
            while x < width:
                # Adjust last column if needed
                if x + tile_size > width and width - x < min_edge_size:
                    x = max(0, width - tile_size)

                # Calculate tile bounds
                x1 = x
                y1 = y
                x2 = min(x + tile_size, width)
                y2 = min(y + tile_size, height)

                tiles.append((x1, y1, x2, y2))

                # Move to next column
                if x + tile_size >= width:
                    break
                x += step

            # Move to next row
            if y + tile_size >= height:
                break
            y += step

        return tiles

    def _refine_tile(self, tile: Image.Image, prompt: str) -> Image.Image:
        """Refine a tile using refiner pipeline"""
        try:
            result = self.refiner_pipeline(
                prompt=prompt,
                image=tile,
                strength=0.2,  # Very light refinement to preserve details
                num_inference_steps=20,
                guidance_scale=6.5,  # Lower guidance for better preservation
            ).images[0]
            return result
        except Exception as e:
            self.logger.warning(f"Tile refinement failed: {e}, using original")
            return tile

    def _img2img_tile(self, tile: Image.Image, prompt: str) -> Image.Image:
        """Process tile using img2img pipeline"""
        try:
            result = self.img2img_pipeline(
                prompt=prompt,
                image=tile,
                strength=0.3,  # Lower strength to preserve tile coherence
                num_inference_steps=30,
                guidance_scale=7.0,
            ).images[0]
            return result
        except Exception as e:
            self.logger.warning(f"Tile img2img failed: {e}, using original")
            return tile

    def _inpaint_tile(self, tile: Image.Image, prompt: str) -> Image.Image:
        """Process tile using inpaint pipeline (with full mask)"""
        # Create a mask that processes the entire tile
        mask = Image.new("L", tile.size, 255)  # Full white = process all

        try:
            result = self.inpaint_pipeline(
                prompt=prompt,
                image=tile,
                mask_image=mask,
                strength=0.4,  # Much lower for tile processing
                num_inference_steps=40,
                guidance_scale=7.0,
            ).images[0]
            return result
        except Exception as e:
            self.logger.warning(f"Tile inpainting failed: {e}, using original")
            return tile

    def _blend_tiles(
        self,
        tiles: List[Dict[str, Any]],
        canvas_size: Tuple[int, int],
        overlap: int,
        blend_width: int,
    ) -> Image.Image:
        """Blend tiles together with smooth transitions"""
        canvas = Image.new("RGB", canvas_size)
        weight_map = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.float32)

        # Process each tile
        for tile_info in tiles:
            tile_img = tile_info["image"]
            x1, y1, x2, y2 = tile_info["coords"]

            # Create weight mask for this tile
            tile_w = x2 - x1
            tile_h = y2 - y1
            tile_weight = np.ones((tile_h, tile_w), dtype=np.float32)

            # Apply gradients at edges if not at canvas boundary
            blend_size = min(blend_width, overlap // 2)

            if blend_size > 0:
                # Left edge
                if x1 > 0:
                    for i in range(blend_size):
                        weight = i / blend_size
                        tile_weight[:, i] *= weight

                # Right edge
                if x2 < canvas_size[0]:
                    for i in range(blend_size):
                        weight = i / blend_size
                        tile_weight[:, -(i + 1)] *= weight

                # Top edge
                if y1 > 0:
                    for i in range(blend_size):
                        weight = i / blend_size
                        tile_weight[i, :] *= weight

                # Bottom edge
                if y2 < canvas_size[1]:
                    for i in range(blend_size):
                        weight = i / blend_size
                        tile_weight[-(i + 1), :] *= weight

            # Convert tile to array
            tile_array = np.array(tile_img).astype(np.float32)

            # Get canvas region
            canvas_array = np.array(canvas).astype(np.float32)
            region = canvas_array[y1:y2, x1:x2]
            region_weight = weight_map[y1:y2, x1:x2]

            # Weighted blend
            total_weight = region_weight + tile_weight
            total_weight = np.maximum(total_weight, 1e-8)  # Avoid division by zero

            blended = (
                region * region_weight[:, :, np.newaxis]
                + tile_array * tile_weight[:, :, np.newaxis]
            ) / total_weight[:, :, np.newaxis]

            # Update canvas
            canvas.paste(Image.fromarray(blended.astype(np.uint8)), (x1, y1))
            weight_map[y1:y2, x1:x2] = total_weight

        return canvas
