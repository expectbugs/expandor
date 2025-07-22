"""
Unit tests for quality orchestration
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw

from expandor.processors.quality_orchestrator import QualityOrchestrator
from expandor.core.boundary_tracker import BoundaryTracker
from expandor.core.config import ExpandorConfig
from expandor.core.exceptions import QualityError


class TestQualityOrchestrator:
    """Test quality orchestration"""
    
    @pytest.fixture
    def test_image_with_seam(self):
        """Create test image with seam and save it"""
        img = Image.new('RGB', (512, 512), color=(100, 100, 100))
        draw = ImageDraw.Draw(img)
        
        # Create visible seam
        draw.rectangle([0, 0, 256, 512], fill=(50, 50, 50))
        draw.rectangle([256, 0, 512, 512], fill=(150, 150, 150))
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            return Path(tmp.name), img
    
    def test_quality_validation_fail_loud(self, test_image_with_seam):
        """Test quality validation fails loud on bad quality"""
        image_path, _ = test_image_with_seam
        
        # Create boundary tracker with seam location
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
        
        # Create config
        config = ExpandorConfig(
            source_image=Image.new('RGB', (256, 512)),
            target_resolution=(512, 512),
            prompt="Test",
            seed=42,
            source_metadata={},
            quality_preset='ultra',
            auto_refine=False  # Disable refinement for this test
        )
        
        # Create orchestrator
        orchestrator = QualityOrchestrator(
            {'quality_validation': {'quality_threshold': 0.9}}, 
            None
        )
        
        # Should fail loud due to quality issues
        with pytest.raises(QualityError) as exc_info:
            orchestrator.validate_and_refine(
                image_path,
                tracker,
                {},  # No pipelines
                config
            )
        
        assert "Unable to achieve required quality" in str(exc_info.value)
        
        # Cleanup
        image_path.unlink()
    
    def test_auto_fix_threshold(self, test_image_with_seam):
        """Test auto-fix threshold allows marginal quality"""
        image_path, _ = test_image_with_seam
        
        tracker = BoundaryTracker()
        config = ExpandorConfig(
            source_image=Image.new('RGB', (256, 512)),
            target_resolution=(512, 512),
            prompt="Test",
            seed=42,
            source_metadata={},
            quality_preset='balanced',
            auto_refine=False
        )
        
        # Create orchestrator with lower auto-fix threshold
        orchestrator = QualityOrchestrator(
            {
                'quality_validation': {
                    'quality_threshold': 0.9,
                    'auto_fix_threshold': 0.5
                }
            }, 
            None
        )
        
        # Should succeed with warning
        result = orchestrator.validate_and_refine(
            image_path,
            tracker,
            {},
            config
        )
        
        assert result['success']
        assert result['quality_score'] < 0.9  # Below ideal
        assert result['quality_score'] >= 0.5  # Above auto-fix
        
        # Cleanup
        image_path.unlink()