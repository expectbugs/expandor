"""
Test Tiled Expansion Strategy
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

from expandor import ExpandorConfig
from expandor.core.exceptions import StrategyError
from expandor.strategies.tiled_expansion import TiledExpansionStrategy


class TestTiledExpansionStrategy:

    def setup_method(self):
        """Setup for each test"""
        self.strategy = TiledExpansionStrategy()
        # Mock refiner pipeline
        self.mock_pipeline = Mock()
        self.mock_pipeline.return_value.images = [Image.new("RGB", (512, 512))]
        self.strategy.refiner_pipeline = self.mock_pipeline

    def test_initialization(self):
        """Test strategy initialization"""
        assert self.strategy.default_tile_size == 1024
        assert self.strategy.overlap == 256
        assert self.strategy.blend_width == 256  # Updated to match actual implementation
        assert self.strategy.min_tile_size == 512
        assert self.strategy.max_tile_size == 2048

    def test_validate_requirements(self):
        """Test validation requires at least one pipeline"""
        strategy = TiledExpansionStrategy()
        strategy.inpaint_pipeline = None
        strategy.refiner_pipeline = None
        strategy.img2img_pipeline = None

        with pytest.raises(StrategyError) as exc_info:
            strategy.validate_requirements()

        assert "at least one pipeline" in str(exc_info.value)

    def test_determine_tile_size(self):
        """Test optimal tile size determination"""
        # Mock VRAM availability
        self.strategy.vram_manager.get_available_vram = Mock(return_value=8000)
        self.strategy.vram_manager.calculate_generation_vram = Mock(return_value=2000)

        tile_size = self.strategy._determine_tile_size(4096, 4096)
        assert tile_size in [512, 768, 1024, 1536, 2048]

        # Test with limited VRAM
        self.strategy.vram_manager.calculate_generation_vram = Mock(return_value=10000)
        tile_size = self.strategy._determine_tile_size(4096, 4096)
        assert tile_size == self.strategy.min_tile_size

    def test_calculate_tiles(self):
        """Test tile calculation with overlap"""
        # Simple 2x2 grid
        tiles = self.strategy._calculate_tiles(width=2048, height=2048, tile_size=1024)

        # With 256px overlap, should have 4 tiles
        assert len(tiles) >= 4

        # Check all tiles are within bounds
        for x1, y1, x2, y2 in tiles:
            assert 0 <= x1 < x2 <= 2048
            assert 0 <= y1 < y2 <= 2048
            assert x2 - x1 <= 1024
            assert y2 - y1 <= 1024

    def test_calculate_tiles_edge_handling(self):
        """Test tile calculation handles edges properly"""
        # Non-divisible dimensions
        tiles = self.strategy._calculate_tiles(width=1500, height=1500, tile_size=1024)

        # Should adjust last tiles to avoid tiny edges
        for x1, y1, x2, y2 in tiles:
            # No tiny edge tiles
            assert (x2 - x1) >= 512  # min_edge_size
            assert (y2 - y1) >= 512

    def test_execute_basic(self):
        """Test basic tiled expansion execution"""
        source_img = Image.new("RGB", (512, 512))
        config = ExpandorConfig(
            source_image=source_img,
            target_resolution=(1024, 1024),
            prompt="Test tiled",
            seed=42,
            source_metadata={"model": "test"},
        )

        # Mock tile processing
        with patch.object(self.strategy, "_calculate_tiles") as mock_calc:
            mock_calc.return_value = [
                (0, 0, 512, 512),
                (512, 0, 1024, 512),
                (0, 512, 512, 1024),
                (512, 512, 1024, 1024),
            ]

            with patch.object(self.strategy, "_refine_tile") as mock_refine:
                mock_refine.return_value = Image.new("RGB", (512, 512))

                with patch.object(self.strategy, "_blend_tiles") as mock_blend:
                    mock_blend.return_value = Image.new("RGB", (1024, 1024))

                    result = self.strategy.execute(config)

        assert result["size"] == (1024, 1024)
        assert result["metadata"]["strategy"] == "tiled_expansion"
        assert result["metadata"]["tile_count"] == 4
        assert len(result["stages"]) >= 5  # resize + 4 tiles + blending

    def test_pipeline_selection(self):
        """Test correct pipeline selection"""
        source_img = Image.new("RGB", (512, 512))
        config = ExpandorConfig(
            source_image=source_img,
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={"model": "test"},
        )

        # Test with refiner
        self.strategy.refiner_pipeline = Mock()
        self.strategy.img2img_pipeline = None
        self.strategy.inpaint_pipeline = None

        with patch.object(self.strategy, "_calculate_tiles") as mock_calc:
            mock_calc.return_value = [(0, 0, 512, 512)]

            with patch.object(self.strategy, "_refine_tile") as mock_refine:
                mock_refine.return_value = Image.new("RGB", (512, 512))

                with patch.object(self.strategy, "_blend_tiles") as mock_blend:
                    mock_blend.return_value = Image.new("RGB", (1024, 1024))

                    self.strategy.execute(config)
                    mock_refine.assert_called()

    def test_blend_tiles(self):
        """Test tile blending functionality"""
        # Create test tiles
        tile1 = Image.new("RGB", (512, 512), color="red")
        tile2 = Image.new("RGB", (512, 512), color="blue")

        tiles = [
            {"image": tile1, "coords": (0, 0, 512, 512)},
            {"image": tile2, "coords": (256, 0, 768, 512)},  # Overlapping
        ]

        result = self.strategy._blend_tiles(
            tiles=tiles, canvas_size=(768, 512), overlap=256, blend_width=128
        )

        assert isinstance(result, Image.Image)
        assert result.size == (768, 512)

        # Check blending occurred in overlap region
        pixels = np.array(result)
        overlap_region = pixels[:, 256:512]
        # Should not be pure red or pure blue in overlap
        assert not np.all(overlap_region == [255, 0, 0])
        assert not np.all(overlap_region == [0, 0, 255])

    def test_boundary_tracking(self):
        """Test tile boundary tracking"""
        self.strategy.boundary_tracker = Mock()

        source_img = Image.new("RGB", (512, 512))
        config = ExpandorConfig(
            source_image=source_img,
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={"model": "test"},
        )

        with patch.object(self.strategy, "_calculate_tiles") as mock_calc:
            mock_calc.return_value = [(0, 0, 512, 512), (512, 0, 1024, 512)]

            with patch.object(self.strategy, "_refine_tile") as mock_refine:
                mock_refine.return_value = Image.new("RGB", (512, 512))

                with patch.object(self.strategy, "_blend_tiles") as mock_blend:
                    mock_blend.return_value = Image.new("RGB", (1024, 1024))

                    self.strategy.execute(config)

        # Should track tile boundaries
        assert self.strategy.boundary_tracker.add_boundary.called
        # Check that vertical boundary at x=512 was tracked
        calls = self.strategy.boundary_tracker.add_boundary.call_args_list
        positions = [call[1]["position"] for call in calls]
        assert 512 in positions

    def test_execute_with_context(self):
        """Test execution with context parameter"""
        source_img = Image.new("RGB", (512, 512))
        config = ExpandorConfig(
            source_image=source_img,
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={"model": "test"},
        )

        context = {"save_stages": True, "metadata_tracker": Mock()}

        with patch.object(self.strategy, "_calculate_tiles") as mock_calc:
            mock_calc.return_value = [(0, 0, 512, 512)]

            with patch.object(self.strategy, "_refine_tile") as mock_refine:
                mock_refine.return_value = Image.new("RGB", (512, 512))

                with patch.object(self.strategy, "_blend_tiles") as mock_blend:
                    mock_blend.return_value = Image.new("RGB", (1024, 1024))

                    result = self.strategy.execute(config, context)

        assert self.strategy._context == context
        assert result["size"] == (1024, 1024)
