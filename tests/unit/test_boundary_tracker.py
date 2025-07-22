"""
Test Boundary Tracker functionality
"""

import pytest
from expandor.core.boundary_tracker import BoundaryTracker, BoundaryInfo

class TestBoundaryTracker:
    
    def setup_method(self):
        """Setup for each test"""
        self.tracker = BoundaryTracker()
    
    def test_initialization(self):
        """Test tracker initialization"""
        assert len(self.tracker.boundaries) == 0
        assert len(self.tracker.progressive_boundaries) == 0
        assert len(self.tracker.horizontal_positions) == 0
        assert len(self.tracker.vertical_positions) == 0
    
    def test_add_boundary(self):
        """Test adding a boundary"""
        self.tracker.add_boundary(
            position=512,
            direction='horizontal',
            step=1,
            expansion_size=256,
            source_size=(512, 512),
            target_size=(768, 512),
            method='progressive'
        )
        
        assert len(self.tracker.boundaries) == 1
        boundary = self.tracker.boundaries[0]
        assert boundary.position == 512
        assert boundary.direction == 'horizontal'
        assert boundary.step == 1
        assert boundary.expansion_size == 256
        assert 512 in self.tracker.horizontal_positions
    
    def test_add_progressive_boundary(self):
        """Test add_progressive_boundary (checklist method)"""
        self.tracker.add_progressive_boundary(
            position=256,
            direction='vertical',
            step=2,
            expansion_size=128
        )
        
        assert len(self.tracker.progressive_boundaries) == 1
        pb = self.tracker.progressive_boundaries[0]
        assert pb['position'] == 256
        assert pb['direction'] == 'vertical'
        assert pb['step'] == 2
        assert pb['expansion_size'] == 128
    
    def test_add_progressive_boundaries(self):
        """Test automatic boundary detection"""
        # Horizontal expansion
        self.tracker.add_progressive_boundaries(
            current_size=(512, 512),
            target_size=(1024, 512),
            step=1
        )
        
        # Should add boundaries at left and right
        assert len(self.tracker.boundaries) == 2
        assert all(b.direction == 'horizontal' for b in self.tracker.boundaries)
        
        # Reset for vertical test
        self.tracker.reset()
        
        # Vertical expansion
        self.tracker.add_progressive_boundaries(
            current_size=(512, 512),
            target_size=(512, 768),
            step=1
        )
        
        # Should add boundaries at top and bottom
        assert len(self.tracker.boundaries) == 2
        assert all(b.direction == 'vertical' for b in self.tracker.boundaries)
    
    def test_get_critical_boundaries(self):
        """Test getting critical boundaries in dict format"""
        # Add some boundaries
        self.tracker.add_boundary(100, 'horizontal', 1, 50, (0,0), (0,0))
        self.tracker.add_boundary(200, 'horizontal', 2, 50, (0,0), (0,0))
        self.tracker.add_boundary(150, 'vertical', 1, 75, (0,0), (0,0))
        
        critical = self.tracker.get_critical_boundaries()
        
        assert isinstance(critical, dict)
        assert 'horizontal' in critical
        assert 'vertical' in critical
        assert critical['horizontal'] == [100, 200]
        assert critical['vertical'] == [150]
    
    def test_get_boundary_regions(self):
        """Test getting boundary regions for processing"""
        # Add boundaries
        self.tracker.add_boundary(100, 'horizontal', 1, 50, (0,0), (0,0))
        self.tracker.add_boundary(200, 'vertical', 1, 50, (0,0), (0,0))
        
        regions = self.tracker.get_boundary_regions(
            width=1000,
            height=800,
            padding=10
        )
        
        assert len(regions) == 2
        # Each region should be (x1, y1, x2, y2)
        for region in regions:
            assert isinstance(region, tuple)
            assert len(region) == 4
            x1, y1, x2, y2 = region
            assert 0 <= x1 < x2 <= 1000
            assert 0 <= y1 < y2 <= 800
    
    def test_get_all_boundaries(self):
        """Test getting all boundaries as dicts"""
        self.tracker.add_boundary(
            position=256,
            direction='horizontal',
            step=1,
            expansion_size=128,
            source_size=(512, 512),
            target_size=(768, 512),
            method='progressive',
            metadata={'side': 'left'}
        )
        
        all_boundaries = self.tracker.get_all_boundaries()
        
        assert len(all_boundaries) == 1
        b = all_boundaries[0]
        assert b['position'] == 256
        assert b['direction'] == 'horizontal'
        assert b['metadata']['side'] == 'left'
    
    def test_boundary_validation(self):
        """Test boundary validation"""
        with pytest.raises(ValueError):
            self.tracker.add_boundary(
                position=100,
                direction='diagonal',  # Invalid direction
                step=1,
                expansion_size=50,
                source_size=(100, 100),
                target_size=(200, 200)
            )
    
    def test_summarize(self):
        """Test boundary summary"""
        # Add various boundaries
        self.tracker.add_boundary(100, 'horizontal', 1, 200, (0,0), (0,0), 'progressive')
        self.tracker.add_boundary(200, 'vertical', 1, 150, (0,0), (0,0), 'progressive')
        self.tracker.add_boundary(300, 'horizontal', 2, 100, (0,0), (0,0), 'tiled')
        
        summary = self.tracker.summarize()
        
        assert summary['total_boundaries'] == 3
        assert summary['horizontal_count'] == 2
        assert summary['vertical_count'] == 1
        assert 'progressive' in summary['methods_used']
        assert 'tiled' in summary['methods_used']
        assert summary['largest_expansion'] == 200
    
    def test_reset(self):
        """Test resetting tracker"""
        # Add some data
        self.tracker.add_boundary(100, 'horizontal', 1, 50, (0,0), (0,0))
        self.tracker.add_progressive_boundary(200, 'vertical', 2)
        
        # Reset
        self.tracker.reset()
        
        assert len(self.tracker.boundaries) == 0
        assert len(self.tracker.progressive_boundaries) == 0
        assert len(self.tracker.horizontal_positions) == 0
        assert len(self.tracker.vertical_positions) == 0