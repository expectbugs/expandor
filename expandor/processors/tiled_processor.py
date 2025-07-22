"""
Tiled processing for memory-efficient image operations
Handles large images by processing in overlapping tiles.
"""

import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import time

from ..utils.image_utils import gaussian_weights, blend_images

logger = logging.getLogger(__name__)


@dataclass
class TileInfo:
    """Information about a single tile"""
    index: int
    position: Tuple[int, int, int, int]  # x1, y1, x2, y2 in full image
    content_region: Tuple[int, int, int, int]  # Region without overlap
    overlap_left: int
    overlap_top: int
    overlap_right: int
    overlap_bottom: int


class TiledProcessor:
    """
    Processes large images in tiles for memory efficiency.
    
    Features:
    - Configurable tile size and overlap
    - Smart blending at tile boundaries
    - Progress tracking
    - Memory-aware tile sizing
    """
    
    def __init__(self, 
                 default_tile_size: int = 512,
                 default_overlap: int = 64,
                 blend_mode: str = 'gaussian',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize tiled processor.
        
        Args:
            default_tile_size: Default size for tiles
            default_overlap: Default overlap between tiles
            blend_mode: How to blend tiles ('gaussian', 'linear', 'none')
            logger: Logger instance
        """
        self.default_tile_size = default_tile_size
        self.default_overlap = default_overlap
        self.blend_mode = blend_mode
        self.logger = logger or logging.getLogger(__name__)
        
        # Processing statistics
        self.tiles_processed = 0
        self.total_tiles = 0
        self.processing_times = []
        
    def calculate_tiles(self, 
                       image_size: Tuple[int, int],
                       tile_size: Optional[int] = None,
                       overlap: Optional[int] = None,
                       min_tile_size: int = 256) -> List[TileInfo]:
        """
        Calculate tile layout for an image.
        
        Args:
            image_size: (width, height) of full image
            tile_size: Size of each tile (uses default if None)
            overlap: Overlap between tiles (uses default if None)
            min_tile_size: Minimum allowed tile size
            
        Returns:
            List of TileInfo objects
        """
        tile_size = tile_size or self.default_tile_size
        overlap = overlap or self.default_overlap
        width, height = image_size
        
        # Adjust tile size if image is small
        if width < tile_size or height < tile_size:
            tile_size = max(min_tile_size, min(width, height))
            self.logger.info(f"Adjusted tile size to {tile_size} for small image")
        
        tiles = []
        tile_idx = 0
        
        # Calculate effective step size
        step = tile_size - overlap
        
        # Generate tiles
        y = 0
        while y < height:
            # Adjust last row to reach edge
            if y + tile_size > height and y > 0:
                y = height - tile_size
            
            x = 0
            while x < width:
                # Adjust last column to reach edge
                if x + tile_size > width and x > 0:
                    x = width - tile_size
                
                # Calculate tile bounds
                x1, y1 = x, y
                x2 = min(x + tile_size, width)
                y2 = min(y + tile_size, height)
                
                # Calculate overlap amounts
                overlap_left = overlap if x > 0 else 0
                overlap_top = overlap if y > 0 else 0
                overlap_right = overlap if x2 < width else 0
                overlap_bottom = overlap if y2 < height else 0
                
                # Content region (without overlap)
                content_x1 = x1 + overlap_left
                content_y1 = y1 + overlap_top
                content_x2 = x2 - overlap_right
                content_y2 = y2 - overlap_bottom
                
                tile = TileInfo(
                    index=tile_idx,
                    position=(x1, y1, x2, y2),
                    content_region=(content_x1, content_y1, content_x2, content_y2),
                    overlap_left=overlap_left,
                    overlap_top=overlap_top,
                    overlap_right=overlap_right,
                    overlap_bottom=overlap_bottom
                )
                tiles.append(tile)
                tile_idx += 1
                
                # Move to next column
                if x + tile_size >= width:
                    break
                x += step
            
            # Move to next row
            if y + tile_size >= height:
                break
            y += step
        
        self.logger.info(f"Created {len(tiles)} tiles for {width}x{height} image")
        return tiles
    
    def process_image(self,
                     image: Image.Image,
                     process_func: Callable[[Image.Image, TileInfo], Image.Image],
                     tile_size: Optional[int] = None,
                     overlap: Optional[int] = None,
                     save_tiles: bool = False,
                     tile_dir: Optional[Path] = None,
                     **process_kwargs) -> Image.Image:
        """
        Process image in tiles using provided function.
        
        Args:
            image: Input image
            process_func: Function to process each tile
            tile_size: Tile size (uses default if None)
            overlap: Overlap size (uses default if None)
            save_tiles: Save individual tiles for debugging
            tile_dir: Directory to save tiles
            **process_kwargs: Additional arguments for process_func
            
        Returns:
            Processed full image
        """
        # Calculate tiles
        tiles = self.calculate_tiles(image.size, tile_size, overlap)
        self.total_tiles = len(tiles)
        self.tiles_processed = 0
        
        # Create output canvas
        output = Image.new(image.mode, image.size)
        
        # Process each tile
        for tile in tiles:
            start_time = time.time()
            
            # Extract tile from source
            x1, y1, x2, y2 = tile.position
            tile_img = image.crop((x1, y1, x2, y2))
            
            # Process tile
            self.logger.debug(f"Processing tile {tile.index + 1}/{len(tiles)}")
            processed_tile = process_func(tile_img, tile, **process_kwargs)
            
            # Blend tile into output
            if self.blend_mode == 'none' or (tile.overlap_left == 0 and tile.overlap_top == 0):
                # No blending needed
                output.paste(processed_tile, (x1, y1))
            else:
                # Blend with existing content
                self._blend_tile(output, processed_tile, tile)
            
            # Save tile if requested
            if save_tiles and tile_dir:
                tile_path = tile_dir / f"tile_{tile.index:04d}.png"
                processed_tile.save(tile_path)
            
            # Update statistics
            self.tiles_processed += 1
            process_time = time.time() - start_time
            self.processing_times.append(process_time)
            
            # Progress callback
            progress = self.tiles_processed / self.total_tiles
            self.logger.info(f"Progress: {progress*100:.1f}% ({self.tiles_processed}/{self.total_tiles})")
        
        return output
    
    def _blend_tile(self, canvas: Image.Image, tile: Image.Image, 
                   tile_info: TileInfo) -> None:
        """
        Blend a tile into the canvas with proper overlap handling.
        
        Args:
            canvas: Output canvas
            tile: Processed tile
            tile_info: Tile information
        """
        x1, y1, x2, y2 = tile_info.position
        
        if self.blend_mode == 'gaussian':
            # Create blend masks for overlapping regions
            tile_array = np.array(tile)
            
            # Horizontal blending (left edge)
            if tile_info.overlap_left > 0:
                blend_width = tile_info.overlap_left
                weights = gaussian_weights(blend_width)
                
                for i in range(blend_width):
                    x_canvas = x1 + i
                    if 0 <= x_canvas < canvas.width:
                        # Get existing column
                        existing = np.array(canvas.crop((x_canvas, y1, x_canvas + 1, y2)))
                        # Blend with new column
                        alpha = weights[i]
                        blended = (1 - alpha) * existing + alpha * tile_array[:, i:i+1]
                        # Put back
                        canvas.paste(Image.fromarray(blended.astype(np.uint8)), (x_canvas, y1))
            
            # Vertical blending (top edge)
            if tile_info.overlap_top > 0:
                blend_height = tile_info.overlap_top
                weights = gaussian_weights(blend_height)
                
                for i in range(blend_height):
                    y_canvas = y1 + i
                    if 0 <= y_canvas < canvas.height:
                        # Get existing row
                        existing = np.array(canvas.crop((x1, y_canvas, x2, y_canvas + 1)))
                        # Blend with new row
                        alpha = weights[i]
                        blended = (1 - alpha) * existing + alpha * tile_array[i:i+1, :]
                        # Put back
                        canvas.paste(Image.fromarray(blended.astype(np.uint8)), (x1, y_canvas))
            
            # Paste non-overlapping region
            content_x1, content_y1, content_x2, content_y2 = tile_info.content_region
            content = tile.crop((
                content_x1 - x1,
                content_y1 - y1,
                content_x2 - x1,
                content_y2 - y1
            ))
            canvas.paste(content, (content_x1, content_y1))
            
        elif self.blend_mode == 'linear':
            # Simple linear blending
            # Create alpha mask for linear blend
            mask = Image.new('L', tile.size, 255)
            draw = Image.new('L', tile.size, 0)
            
            # Fade edges
            if tile_info.overlap_left > 0:
                for i in range(tile_info.overlap_left):
                    alpha = int(255 * i / tile_info.overlap_left)
                    draw.paste(alpha, (i, 0, i+1, tile.height))
            
            if tile_info.overlap_top > 0:
                for i in range(tile_info.overlap_top):
                    alpha = int(255 * i / tile_info.overlap_top)
                    draw.paste(alpha, (0, i, tile.width, i+1))
            
            # Composite
            canvas.paste(tile, (x1, y1), mask)
        
        else:
            # No blending
            canvas.paste(tile, (x1, y1))
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.processing_times:
            return {
                'tiles_processed': 0,
                'average_time': 0,
                'total_time': 0
            }
        
        return {
            'tiles_processed': self.tiles_processed,
            'total_tiles': self.total_tiles,
            'average_time': np.mean(self.processing_times),
            'total_time': sum(self.processing_times),
            'min_time': min(self.processing_times),
            'max_time': max(self.processing_times)
        }
    
    def create_tile_debug_image(self, image_size: Tuple[int, int],
                              tiles: List[TileInfo]) -> Image.Image:
        """
        Create debug visualization of tile layout.
        
        Args:
            image_size: (width, height)
            tiles: List of tiles
            
        Returns:
            Debug image showing tile boundaries
        """
        from PIL import ImageDraw
        
        debug_img = Image.new('RGB', image_size, (255, 255, 255))
        draw = ImageDraw.Draw(debug_img)
        
        # Color palette for tiles
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
        for i, tile in enumerate(tiles):
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = tile.position
            
            # Draw tile boundary
            draw.rectangle([x1, y1, x2-1, y2-1], outline=color, width=2)
            
            # Draw content region
            cx1, cy1, cx2, cy2 = tile.content_region
            draw.rectangle([cx1, cy1, cx2-1, cy2-1], outline=color, width=1)
            
            # Label tile
            draw.text((x1 + 5, y1 + 5), f"T{tile.index}", fill=color)
        
        return debug_img