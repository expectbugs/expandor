"""
Unit tests for boundary analysis
"""

import pytest
from PIL import Image, ImageDraw
import numpy as np

from expandor.processors.boundary_analysis import BoundaryAnalyzer
from expandor.core.boundary_tracker import BoundaryTracker


class TestBoundaryAnalyzer:
    """Test boundary analysis"""
    
    @pytest.fixture
    def analyzer(self):
        return BoundaryAnalyzer()
    
    @pytest.fixture
    def test_image_with_seam(self):
        """Create test image with visible seam"""
        img = Image.new('RGB', (512, 512), color=(100, 100, 100))
        draw = ImageDraw.Draw(img)
        
        # Create visible vertical seam
        draw.rectangle([0, 0, 256, 512], fill=(50, 50, 50))
        draw.rectangle([256, 0, 512, 512], fill=(150, 150, 150))
        
        return img
    
    def test_boundary_analysis(self, analyzer, test_image_with_seam):
        """Test boundary analyzer finds issues"""
        tracker = BoundaryTracker()
        tracker.add_boundary(
            position=256,
            direction='vertical',
            step=0,
            expansion_size=256,
            source_size=(256, 512),
            target_size=(512, 512),
            method='test'
        )
        
        analysis = analyzer.analyze_boundaries(tracker, test_image_with_seam)
        
        assert analysis['total_boundaries'] == 1
        assert len(analysis['issues']) > 0
        assert len(analysis['recommendations']) > 0
        assert analysis['quality_score'] < 1.0
    
    def test_clean_boundaries(self, analyzer):
        """Test clean image has good quality score"""
        # Create smooth gradient
        img = Image.new('RGB', (512, 512))
        pixels = img.load()
        
        for x in range(512):
            for y in range(512):
                r = int((x / 512) * 255)
                g = int((y / 512) * 255)
                b = 128
                pixels[x, y] = (r, g, b)
        
        tracker = BoundaryTracker()
        tracker.add_boundary(
            position=256,
            direction='vertical',
            step=0,
            expansion_size=256,
            source_size=(256, 512),
            target_size=(512, 512),
            method='test'
        )
        
        analysis = analyzer.analyze_boundaries(tracker, img)
        
        assert analysis['quality_score'] >= 0.9
        assert len(analysis['issues']) == 0