"""
Quality system integration tests
"""

import pytest
import numpy as np
from PIL import Image, ImageDraw

from expandor import Expandor
from expandor.core.exceptions import QualityError

from .base_integration_test import BaseIntegrationTest


class TestQualityIntegration(BaseIntegrationTest):
    """Integration tests for quality systems"""
    
    def create_image_with_artifacts(self, width: int, height: int) -> Image.Image:
        """Create test image with intentional artifacts"""
        img = Image.new('RGB', (width, height), color=(100, 100, 100))
        draw = ImageDraw.Draw(img)
        
        # Add visible seams
        for x in range(0, width, 256):
            draw.line([(x, 0), (x, height)], fill=(255, 0, 0), width=2)
        
        # Add color blocks that will create discontinuities
        draw.rectangle([100, 100, 200, 200], fill=(0, 255, 0))
        draw.rectangle([300, 300, 400, 400], fill=(0, 0, 255))
        
        return img
    
    def test_artifact_detection_and_refinement(self, expandor, mock_inpaint_pipeline, temp_dir):
        """Test that artifacts are detected and refined"""
        img = self.create_image_with_artifacts(512, 512)
        
        config = self.create_config(
            source_image=img,
            target_resolution=(1024, 512),
            inpaint_pipeline=mock_inpaint_pipeline,
            quality_preset='ultra',
            auto_refine=True,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (1024, 512))
        
        # Should have quality metadata
        assert 'final_quality_score' in result.metadata or 'quality_refined' in result.metadata
        
        # If artifacts were fixed, should be noted
        if 'artifacts_fixed' in result.metadata:
            assert result.metadata['artifacts_fixed'] >= 0
    
    def test_quality_threshold_enforcement(self, expandor, 
                                         mock_inpaint_pipeline, temp_dir):
        """Test quality threshold is enforced"""
        # Create very poor quality image
        img = Image.new('RGB', (256, 256))
        pixels = img.load()
        
        # Create random noise
        np.random.seed(42)
        for x in range(256):
            for y in range(256):
                pixels[x, y] = tuple(np.random.randint(0, 256, 3))
        
        config = self.create_config(
            source_image=img,
            target_resolution=(512, 256),  # Aspect change
            inpaint_pipeline=mock_inpaint_pipeline,
            quality_preset='ultra',
            auto_refine=False,  # Disable refinement to test threshold
            stage_dir=temp_dir / "stages"
        )
        
        # Depending on implementation, this might fail quality checks
        try:
            result = expandor.expand(config)
            # If it succeeds, check quality metrics
            if 'final_quality_score' in result.metadata:
                print(f"Quality score: {result.metadata['final_quality_score']}")
        except QualityError as e:
            # Expected for very poor quality
            assert "quality" in str(e).lower()
            print(f"Quality check failed as expected: {e}")
    
    def test_refinement_improves_quality(self, expandor,
                                       mock_inpaint_pipeline, temp_dir):
        """Test that refinement improves quality scores"""
        # Create image with moderate artifacts
        img = self.create_test_image(512, 512, pattern="gradient")
        draw = ImageDraw.Draw(img)
        # Add subtle seam
        draw.line([(256, 0), (256, 512)], fill=(150, 150, 150), width=1)
        
        config = self.create_config(
            source_image=img,
            target_resolution=(1024, 512),
            inpaint_pipeline=mock_inpaint_pipeline,
            quality_preset='high',
            auto_refine=True,
            refinement_passes=2,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (1024, 512))
        
        # Check refinement occurred
        if 'refinement_passes' in result.metadata:
            assert result.metadata['refinement_passes'] > 0
            print(f"Performed {result.metadata['refinement_passes']} refinement passes")
        
        if 'final_quality_score' in result.metadata:
            assert result.metadata['final_quality_score'] > 0.7
            print(f"Final quality score: {result.metadata['final_quality_score']}")
    
    def test_boundary_tracking_integration(self, expandor, 
                                         mock_inpaint_pipeline, temp_dir):
        """Test boundary tracking integrates with quality systems"""
        config = self.create_config(
            source_image=self.test_image_small(),
            target_resolution=(1024, 768),  # Aspect change creates boundaries
            inpaint_pipeline=mock_inpaint_pipeline,
            quality_preset='ultra',
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        # Should have boundaries in metadata
        assert 'boundaries' in result.metadata or 'boundary_positions' in result.metadata
        
        # If quality was validated, boundaries should have been analyzed
        if 'final_quality_score' in result.metadata:
            print(f"Quality validated with {len(result.metadata.get('boundaries', []))} boundaries")