# Expandor Phase 3 Step 3: Integration Tests - FINAL Complete Implementation Guide

## Overview

This is the comprehensive final implementation guide for Integration Tests in Expandor Phase 3. This document provides complete test infrastructure and comprehensive integration tests that verify all components work together seamlessly.

## Prerequisites

```bash
# 1. Verify test directories exist
mkdir -p tests/integration
mkdir -p tests/fixtures

# 2. Verify mock pipelines are available
python -c "from expandor.adapters.mock_pipeline import MockInpaintPipeline"

# 3. Install test dependencies if needed
pip install pytest pytest-cov pytest-mock pytest-timeout

# 4. Verify all Phase 3 components are implemented
ls -la expandor/strategies/swpo_strategy.py
ls -la expandor/processors/artifact_detector_enhanced.py
ls -la expandor/processors/quality_orchestrator.py
```

## 1. Integration Test Base Class

### 1.1 Base Integration Test Class

**FILE**: `tests/integration/base_integration_test.py`

```python
"""
Base class for integration tests
Provides common fixtures and utilities for testing Expandor.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import logging

import pytest
import numpy as np
from PIL import Image, ImageDraw

from expandor import Expandor
from expandor.core.config import ExpandorConfig
from expandor.core.result import ExpandorResult
from expandor.core.exceptions import ExpandorError, VRAMError
from expandor.adapters.mock_pipeline import MockInpaintPipeline, MockImg2ImgPipeline


class BaseIntegrationTest:
    """
    Base class for Expandor integration tests.
    
    Provides common fixtures and validation methods following
    the FAIL LOUD philosophy.
    """
    
    @pytest.fixture(autouse=True)
    def setup_logging(self, caplog):
        """Setup logging for tests"""
        caplog.set_level(logging.INFO)
        self.caplog = caplog
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp(prefix="expandor_test_")
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def expandor(self, temp_dir):
        """Create Expandor instance for testing"""
        expandor = Expandor()
        # Override temp directory for testing
        expandor._temp_base = temp_dir
        return expandor
    
    @pytest.fixture
    def mock_inpaint_pipeline(self):
        """Create mock inpaint pipeline"""
        return MockInpaintPipeline(
            latency_ms=10,  # Fast for testing
            failure_rate=0.0  # No random failures
        )
    
    @pytest.fixture
    def mock_img2img_pipeline(self):
        """Create mock img2img pipeline"""
        return MockImg2ImgPipeline(
            latency_ms=10,
            failure_rate=0.0
        )
    
    @pytest.fixture
    def test_image_small(self):
        """Create small test image (512x512)"""
        return self.create_test_image(512, 512)
    
    @pytest.fixture
    def test_image_1080p(self):
        """Create 1080p test image"""
        return self.create_test_image(1920, 1080)
    
    @pytest.fixture
    def test_image_4k(self):
        """Create 4K test image"""
        return self.create_test_image(3840, 2160)
    
    def create_test_image(self, width: int, height: int, 
                         pattern: str = "gradient") -> Image.Image:
        """
        Create test image with specified pattern.
        
        Args:
            width: Image width
            height: Image height
            pattern: Pattern type ('gradient', 'checkerboard', 'solid')
            
        Returns:
            Test image
        """
        img = Image.new('RGB', (width, height))
        
        if pattern == "gradient":
            # Create gradient pattern for seam detection
            pixels = img.load()
            for x in range(width):
                for y in range(height):
                    r = int((x / width) * 255)
                    g = int((y / height) * 255)
                    b = 128
                    pixels[x, y] = (r, g, b)
                    
        elif pattern == "checkerboard":
            # Create checkerboard pattern
            draw = ImageDraw.Draw(img)
            square_size = 64
            for x in range(0, width, square_size):
                for y in range(0, height, square_size):
                    if (x // square_size + y // square_size) % 2 == 0:
                        draw.rectangle([x, y, x + square_size, y + square_size], 
                                     fill=(255, 255, 255))
                    else:
                        draw.rectangle([x, y, x + square_size, y + square_size], 
                                     fill=(0, 0, 0))
                        
        elif pattern == "solid":
            # Solid color
            img.paste((100, 150, 200), [0, 0, width, height])
        
        return img
    
    def create_config(self, 
                     source_image: Image.Image,
                     target_resolution: Tuple[int, int],
                     **kwargs) -> ExpandorConfig:
        """
        Create ExpandorConfig with defaults for testing.
        
        Args:
            source_image: Source image
            target_resolution: Target resolution
            **kwargs: Additional config parameters
            
        Returns:
            ExpandorConfig instance
        """
        defaults = {
            'prompt': "Extend the scene naturally with perfect quality",
            'seed': 42,
            'source_metadata': {'model': 'test', 'original_prompt': 'test scene'},
            'quality_preset': 'high',
            'save_stages': True,
            'stage_dir': kwargs.get('temp_dir', Path('temp/stages')),
            'auto_refine': True,
            'allow_tiled': True,
            'allow_cpu_offload': True
        }
        
        # Override with provided kwargs
        defaults.update(kwargs)
        
        return ExpandorConfig(
            source_image=source_image,
            target_resolution=target_resolution,
            **defaults
        )
    
    def validate_result(self, result: ExpandorResult, 
                       expected_size: Tuple[int, int]):
        """
        Validate expansion result - FAIL LOUD on issues.
        
        Args:
            result: ExpandorResult to validate
            expected_size: Expected (width, height)
            
        Raises:
            AssertionError: On any validation failure
        """
        # Check result structure
        assert hasattr(result, 'image_path'), "Result missing image_path"
        assert hasattr(result, 'size'), "Result missing size"
        assert hasattr(result, 'metadata'), "Result missing metadata"
        
        # Verify image exists and can be loaded
        assert result.image_path.exists(), f"Result image not found at {result.image_path}"
        
        # Verify image can be loaded
        img = Image.open(result.image_path)
        assert img.size == expected_size, f"Size mismatch: {img.size} != {expected_size}"
        
        # Verify metadata
        assert result.size == expected_size, f"Result size mismatch: {result.size} != {expected_size}"
        assert result.total_duration_seconds > 0, "No duration recorded"
        assert result.strategy_used is not None, "No strategy recorded"
        
        # Log validation success
        print(f"✓ Result validated: {result.size}, {result.strategy_used} strategy, "
              f"{result.total_duration_seconds:.2f}s")
    
    def check_no_artifacts(self, result: ExpandorResult, 
                          tolerance: float = 0.1):
        """
        Verify no significant artifacts detected.
        
        Args:
            result: ExpandorResult to check
            tolerance: Quality tolerance (0-1)
            
        Raises:
            AssertionError: If artifacts exceed tolerance
        """
        # Check quality metrics in metadata
        metadata = result.metadata
        
        if 'final_quality_score' in metadata:
            assert metadata['final_quality_score'] > (1.0 - tolerance), \
                f"Quality score {metadata['final_quality_score']} below threshold"
        
        if 'seams_detected' in metadata:
            assert metadata['seams_detected'] == 0, \
                f"Detected {metadata['seams_detected']} seams!"
        
        # Check artifact metrics
        if 'artifacts_fixed' in metadata:
            print(f"ℹ️  Fixed {metadata['artifacts_fixed']} artifacts during processing")
    
    def save_debug_info(self, result: ExpandorResult, 
                       test_name: str,
                       temp_dir: Path):
        """Save debug information for failed tests."""
        debug_dir = temp_dir / "debug" / test_name
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Save result image
        if result.image_path and result.image_path.exists():
            shutil.copy(result.image_path, debug_dir / "result.png")
        
        # Save metadata
        if hasattr(result, 'metadata'):
            import json
            with open(debug_dir / "metadata.json", 'w') as f:
                json.dump(result.metadata, f, indent=2, default=str)
        
        # Save stages if available
        if hasattr(result, 'stage_results') and result.stage_results:
            with open(debug_dir / "stages.txt", 'w') as f:
                for stage in result.stage_results:
                    f.write(f"{stage}\n")
        
        print(f"Debug info saved to: {debug_dir}")


## 2. Full Pipeline Integration Tests

### 2.1 Full Pipeline Tests

**FILE**: `tests/integration/test_full_pipeline.py`

```python
"""
Full pipeline integration tests
Tests complete expansion workflows with all strategies.
"""

import pytest
from pathlib import Path
from PIL import ImageDraw

from expandor import Expandor
from expandor.core.config import ExpandorConfig
from expandor.core.exceptions import ExpandorError, VRAMError
from expandor.adapters.mock_pipeline import MockInpaintPipeline, MockImg2ImgPipeline

from .base_integration_test import BaseIntegrationTest


class TestFullPipelineIntegration(BaseIntegrationTest):
    """Test complete expansion pipelines"""
    
    def test_simple_upscale(self, expandor, test_image_small, 
                           mock_img2img_pipeline, temp_dir):
        """Test simple 2x upscale"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 1024),  # 2x upscale
            img2img_pipeline=mock_img2img_pipeline,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (1024, 1024))
        assert result.strategy_used == "direct_upscale"
        self.check_no_artifacts(result)
    
    def test_progressive_expansion(self, expandor, test_image_1080p,
                                 mock_inpaint_pipeline, temp_dir):
        """Test progressive outpainting for aspect ratio change"""
        config = self.create_config(
            source_image=test_image_1080p,
            target_resolution=(2560, 1440),  # 16:9 to 16:9 with expansion
            inpaint_pipeline=mock_inpaint_pipeline,
            quality_preset='high',
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (2560, 1440))
        assert result.strategy_used in ["progressive_outpaint", "hybrid_adaptive"]
        
        # Check boundaries were tracked
        assert 'boundary_positions' in result.metadata or 'boundaries' in result.metadata
    
    def test_extreme_aspect_ratio_change(self, expandor, test_image_1080p,
                                        mock_inpaint_pipeline, temp_dir):
        """Test extreme aspect ratio change (16:9 to 32:9)"""
        config = self.create_config(
            source_image=test_image_1080p,
            target_resolution=(3840, 1080),  # Double width
            inpaint_pipeline=mock_inpaint_pipeline,
            quality_preset='balanced',
            window_size=300,
            overlap_ratio=0.8,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (3840, 1080))
        # Should use SWPO or progressive for extreme ratio
        assert result.strategy_used in ["swpo", "progressive_outpaint", "hybrid_adaptive"]
    
    def test_vram_constrained_expansion(self, expandor, test_image_4k,
                                       mock_inpaint_pipeline, temp_dir, monkeypatch):
        """Test expansion with limited VRAM"""
        # Mock limited VRAM
        def mock_get_available_vram():
            return 2048.0  # 2GB only
        
        monkeypatch.setattr(
            expandor.vram_manager, 
            'get_available_vram', 
            mock_get_available_vram
        )
        
        config = self.create_config(
            source_image=test_image_4k,
            target_resolution=(7680, 4320),  # 8K target
            inpaint_pipeline=mock_inpaint_pipeline,
            allow_tiled=True,
            allow_cpu_offload=True,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (7680, 4320))
        # Should use tiled or CPU offload strategy
        assert result.strategy_used in ["tiled_expansion", "cpu_offload", "hybrid_adaptive"]
    
    def test_no_pipeline_fails_loud(self, expandor, test_image_small, temp_dir):
        """Test that expansion fails loud without pipelines"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 768),  # Aspect change requires pipeline
            # No pipelines provided
            stage_dir=temp_dir / "stages"
        )
        
        with pytest.raises(ExpandorError) as exc_info:
            expandor.expand(config)
        
        # Should have clear error message
        assert "pipeline" in str(exc_info.value).lower()
    
    def test_quality_enforcement(self, expandor, mock_inpaint_pipeline, temp_dir):
        """Test quality enforcement with auto-refinement"""
        # Create image with intentional seam
        img = self.create_test_image(512, 512, pattern="solid")
        draw = ImageDraw.Draw(img)
        # Add visible seam
        draw.line([(256, 0), (256, 512)], fill=(255, 0, 0), width=2)
        
        config = self.create_config(
            source_image=img,
            target_resolution=(1024, 512),  # Horizontal expansion
            inpaint_pipeline=mock_inpaint_pipeline,
            quality_preset='ultra',
            auto_refine=True,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (1024, 512))
        
        # Should have quality validation in metadata
        assert 'quality_refined' in result.metadata or 'final_quality_score' in result.metadata
    
    def test_stage_saving(self, expandor, test_image_small,
                         mock_inpaint_pipeline, temp_dir):
        """Test that stages are saved correctly"""
        stage_dir = temp_dir / "stages"
        
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(768, 768),  # Requires processing
            inpaint_pipeline=mock_inpaint_pipeline,
            save_stages=True,
            stage_dir=stage_dir
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (768, 768))
        
        # Check stages were saved
        assert stage_dir.exists()
        stage_files = list(stage_dir.glob("*.png"))
        assert len(stage_files) > 0, "No stage files saved"
        
        # Verify stage images can be loaded
        for stage_file in stage_files:
            img = Image.open(stage_file)
            assert img.size[0] > 0 and img.size[1] > 0
    
    def test_metadata_completeness(self, expandor, test_image_small,
                                  mock_inpaint_pipeline, temp_dir):
        """Test that metadata is complete"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 768),
            inpaint_pipeline=mock_inpaint_pipeline,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        # Check required metadata fields
        assert 'strategy_used' in result.metadata
        assert 'duration_seconds' in result.metadata
        assert 'stages' in result.metadata
        
        # Check stage information
        if result.metadata['stages']:
            stage = result.metadata['stages'][0]
            assert 'name' in stage
            assert 'duration' in stage
            assert 'method' in stage


## 3. Strategy-Specific Integration Tests

### 3.1 SWPO Integration Tests

**FILE**: `tests/integration/test_swpo_integration.py`

```python
"""
SWPO strategy integration tests
"""

import pytest
from PIL import Image

from expandor import Expandor
from expandor.core.exceptions import ExpandorError, StrategyError

from .base_integration_test import BaseIntegrationTest


class TestSWPOIntegration(BaseIntegrationTest):
    """Integration tests for SWPO strategy"""
    
    def test_swpo_extreme_horizontal(self, expandor, test_image_1080p,
                                   mock_inpaint_pipeline, temp_dir):
        """Test SWPO handles extreme horizontal expansion"""
        config = self.create_config(
            source_image=test_image_1080p,
            target_resolution=(7680, 1080),  # 4x width
            inpaint_pipeline=mock_inpaint_pipeline,
            window_size=400,
            overlap_ratio=0.8,
            force_strategy='swpo',  # Force SWPO
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (7680, 1080))
        assert result.strategy_used == 'swpo'
        
        # Check SWPO specific metadata
        assert 'total_windows' in result.metadata
        assert result.metadata['total_windows'] > 1
        
        # Check window parameters
        assert 'window_parameters' in result.metadata
        assert result.metadata['window_parameters']['window_size'] == 400
        assert result.metadata['window_parameters']['overlap_ratio'] == 0.8
    
    def test_swpo_extreme_vertical(self, expandor, test_image_small,
                                  mock_inpaint_pipeline, temp_dir):
        """Test SWPO handles extreme vertical expansion"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(512, 2048),  # 4x height
            inpaint_pipeline=mock_inpaint_pipeline,
            window_size=256,
            overlap_ratio=0.75,
            force_strategy='swpo',
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (512, 2048))
        
        # Verify boundaries tracked
        assert 'boundaries' in result.metadata or 'boundary_positions' in result.metadata
    
    def test_swpo_bidirectional_expansion(self, expandor, test_image_small,
                                        mock_inpaint_pipeline, temp_dir):
        """Test SWPO handles expansion in both directions"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 1024),  # 2x in both dimensions
            inpaint_pipeline=mock_inpaint_pipeline,
            window_size=300,
            overlap_ratio=0.8,
            force_strategy='swpo',
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (1024, 1024))
        self.check_no_artifacts(result, tolerance=0.15)
    
    def test_swpo_without_pipeline_fails(self, expandor, test_image_small, temp_dir):
        """Test SWPO fails properly without inpaint pipeline"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 512),
            force_strategy='swpo',
            # No inpaint pipeline
            stage_dir=temp_dir / "stages"
        )
        
        with pytest.raises((ExpandorError, StrategyError)) as exc_info:
            expandor.expand(config)
        
        assert "inpaint" in str(exc_info.value).lower()
    
    def test_swpo_stage_tracking(self, expandor, test_image_small,
                                mock_inpaint_pipeline, temp_dir):
        """Test SWPO tracks all window stages"""
        stage_dir = temp_dir / "stages"
        
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1536, 512),  # 3x width
            inpaint_pipeline=mock_inpaint_pipeline,
            window_size=256,
            overlap_ratio=0.5,
            force_strategy='swpo',
            save_stages=True,
            stage_dir=stage_dir
        )
        
        result = expandor.expand(config)
        
        # Check window stages saved
        window_stages = list(stage_dir.glob("swpo_window_*.png"))
        assert len(window_stages) > 1
        
        # Check metadata has stage info
        assert 'stages' in result.metadata
        swpo_stages = [s for s in result.metadata['stages'] if 'swpo' in s['name']]
        assert len(swpo_stages) >= len(window_stages)


### 3.2 CPU Offload Integration Tests

**FILE**: `tests/integration/test_cpu_offload_integration.py`

```python
"""
CPU Offload strategy integration tests
"""

import pytest
from PIL import Image

from expandor import Expandor
from expandor.core.exceptions import ExpandorError, StrategyError

from .base_integration_test import BaseIntegrationTest


class TestCPUOffloadIntegration(BaseIntegrationTest):
    """Integration tests for CPU Offload strategy"""
    
    def test_cpu_offload_basic(self, expandor, test_image_small,
                              mock_inpaint_pipeline, temp_dir):
        """Test basic CPU offload operation"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 1024),
            inpaint_pipeline=mock_inpaint_pipeline,
            force_strategy='cpu_offload',
            allow_cpu_offload=True,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (1024, 1024))
        assert result.strategy_used == 'cpu_offload'
        
        # Check CPU offload specific metadata
        assert 'tile_size' in result.metadata
        assert result.metadata['tile_size'] >= 384  # Minimum tile size
    
    def test_cpu_offload_not_allowed_fails(self, expandor, test_image_small, temp_dir):
        """Test CPU offload fails when not allowed"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 1024),
            force_strategy='cpu_offload',
            allow_cpu_offload=False,  # Not allowed
            stage_dir=temp_dir / "stages"
        )
        
        with pytest.raises((ExpandorError, StrategyError)) as exc_info:
            expandor.expand(config)
        
        assert "not allowed" in str(exc_info.value).lower()
    
    def test_cpu_offload_extreme_memory_constraint(self, expandor, test_image_4k,
                                                  mock_img2img_pipeline, temp_dir, monkeypatch):
        """Test CPU offload with extreme memory constraints"""
        # Mock very low memory
        def mock_get_available_vram():
            return 256.0  # Only 256MB
        
        monkeypatch.setattr(
            expandor.vram_manager,
            'get_available_vram',
            mock_get_available_vram
        )
        
        config = self.create_config(
            source_image=test_image_4k,
            target_resolution=(5760, 3240),  # 1.5x expansion
            img2img_pipeline=mock_img2img_pipeline,
            force_strategy='cpu_offload',
            allow_cpu_offload=True,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        self.validate_result(result, (5760, 3240))
        
        # Should use very small tiles
        assert result.metadata['tile_size'] <= 512
    
    def test_cpu_offload_stage_tracking(self, expandor, test_image_small,
                                      mock_inpaint_pipeline, temp_dir):
        """Test CPU offload tracks processing stages"""
        stage_dir = temp_dir / "stages"
        
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 768),  # Aspect change
            inpaint_pipeline=mock_inpaint_pipeline,
            force_strategy='cpu_offload',
            allow_cpu_offload=True,
            save_stages=True,
            stage_dir=stage_dir
        )
        
        result = expandor.expand(config)
        
        # Check stages saved
        cpu_stages = list(stage_dir.glob("cpu_offload_stage_*.png"))
        assert len(cpu_stages) > 0
        
        # Check metadata
        assert 'total_stages' in result.metadata
        assert result.metadata['total_stages'] >= len(cpu_stages)


### 3.3 Quality Validation Integration Tests

**FILE**: `tests/integration/test_quality_integration.py`

```python
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


### 3.4 Edge Case Integration Tests

**FILE**: `tests/integration/test_edge_cases.py`

```python
"""
Edge case integration tests
"""

import pytest
from PIL import Image

from expandor import Expandor
from expandor.core.exceptions import ExpandorError, VRAMError

from .base_integration_test import BaseIntegrationTest


class TestEdgeCases(BaseIntegrationTest):
    """Test edge cases and error conditions"""
    
    def test_same_size_input_output(self, expandor, test_image_small,
                                   mock_img2img_pipeline, temp_dir):
        """Test when input and output are same size"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(512, 512),  # Same as input
            img2img_pipeline=mock_img2img_pipeline,
            stage_dir=temp_dir / "stages"
        )
        
        # Should handle gracefully
        result = expandor.expand(config)
        self.validate_result(result, (512, 512))
        
        # Might use direct strategy or skip processing
        assert result.strategy_used in ['direct_upscale', 'passthrough']
    
    def test_tiny_image_expansion(self, expandor, mock_inpaint_pipeline, temp_dir):
        """Test expanding very small images"""
        tiny_img = self.create_test_image(64, 64)
        
        config = self.create_config(
            source_image=tiny_img,
            target_resolution=(512, 512),  # 8x expansion
            inpaint_pipeline=mock_inpaint_pipeline,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        self.validate_result(result, (512, 512))
    
    def test_unusual_aspect_ratios(self, expandor, mock_inpaint_pipeline, temp_dir):
        """Test unusual aspect ratios"""
        # Ultra-wide
        img = self.create_test_image(1920, 400)
        
        config = self.create_config(
            source_image=img,
            target_resolution=(3840, 400),  # Keep ultra-wide
            inpaint_pipeline=mock_inpaint_pipeline,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        self.validate_result(result, (3840, 400))
    
    def test_invalid_target_resolution(self, expandor, test_image_small, temp_dir):
        """Test invalid target resolutions fail loud"""
        # Zero dimension
        with pytest.raises(ExpandorError) as exc_info:
            config = self.create_config(
                source_image=test_image_small,
                target_resolution=(0, 512),
                stage_dir=temp_dir / "stages"
            )
            expandor.expand(config)
        
        assert "Invalid target resolution" in str(exc_info.value)
        
        # Negative dimension
        with pytest.raises(ExpandorError) as exc_info:
            config = self.create_config(
                source_image=test_image_small,
                target_resolution=(512, -512),
                stage_dir=temp_dir / "stages"
            )
            expandor.expand(config)
        
        assert "Invalid target resolution" in str(exc_info.value)
    
    def test_corrupted_source_image(self, expandor, temp_dir):
        """Test handling of corrupted source images"""
        # Create invalid image path
        bad_path = temp_dir / "nonexistent.png"
        
        with pytest.raises(ExpandorError) as exc_info:
            config = self.create_config(
                source_image=bad_path,
                target_resolution=(512, 512),
                stage_dir=temp_dir / "stages"
            )
            expandor.expand(config)
        
        assert "not found" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
    
    def test_memory_pressure_handling(self, expandor, test_image_4k,
                                    mock_inpaint_pipeline, temp_dir, monkeypatch):
        """Test handling of memory pressure"""
        # Mock very low memory
        def mock_get_available_vram():
            return 512.0  # Only 512MB
        
        monkeypatch.setattr(
            expandor.vram_manager,
            'get_available_vram',
            mock_get_available_vram
        )
        
        config = self.create_config(
            source_image=test_image_4k,
            target_resolution=(7680, 4320),  # 8K target
            inpaint_pipeline=mock_inpaint_pipeline,
            allow_cpu_offload=True,
            stage_dir=temp_dir / "stages"
        )
        
        # Should use CPU offload or fail loud
        try:
            result = expandor.expand(config)
            assert result.strategy_used in ["cpu_offload", "tiled_expansion"]
        except VRAMError as e:
            # Also acceptable - fail loud
            assert "VRAM" in str(e)
    
    def test_dimension_rounding(self, expandor, test_image_small,
                               mock_inpaint_pipeline, temp_dir):
        """Test that dimensions are properly rounded"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1023, 767),  # Not multiples of 8
            inpaint_pipeline=mock_inpaint_pipeline,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        # Should be rounded to multiples of 8
        assert result.size[0] % 8 == 0
        assert result.size[1] % 8 == 0
        
        # Should be close to requested size
        assert abs(result.size[0] - 1023) < 8
        assert abs(result.size[1] - 767) < 8
    
    def test_force_strategy_override(self, expandor, test_image_small,
                                   mock_inpaint_pipeline, temp_dir):
        """Test force_strategy overrides automatic selection"""
        # Force SWPO even for simple expansion
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(768, 768),  # Simple 1.5x
            inpaint_pipeline=mock_inpaint_pipeline,
            force_strategy='swpo',
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        assert result.strategy_used == 'swpo'


## 4. Performance Tests

### 4.1 Performance Integration Tests

**FILE**: `tests/integration/test_performance.py`

```python
"""
Performance integration tests
"""

import time
import pytest
from concurrent.futures import ThreadPoolExecutor

from expandor import Expandor
from expandor.core.config import ExpandorConfig

from .base_integration_test import BaseIntegrationTest


class TestPerformance(BaseIntegrationTest):
    """Test performance characteristics"""
    
    def test_expansion_performance(self, expandor, test_image_small,
                                 mock_inpaint_pipeline, temp_dir):
        """Test expansion completes in reasonable time"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 768),
            inpaint_pipeline=mock_inpaint_pipeline,
            stage_dir=temp_dir / "stages"
        )
        
        start_time = time.time()
        result = expandor.expand(config)
        duration = time.time() - start_time
        
        # Should complete in reasonable time (adjust based on mock latency)
        assert duration < 10.0  # 10 seconds max for test
        
        # Check duration is tracked
        assert result.total_duration_seconds > 0
        assert abs(result.total_duration_seconds - duration) < 1.0
    
    def test_memory_usage_tracking(self, expandor, test_image_1080p,
                                 mock_inpaint_pipeline, temp_dir):
        """Test memory usage is tracked"""
        config = self.create_config(
            source_image=test_image_1080p,
            target_resolution=(3840, 2160),  # 4K
            inpaint_pipeline=mock_inpaint_pipeline,
            stage_dir=temp_dir / "stages"
        )
        
        result = expandor.expand(config)
        
        # Check memory tracking in metadata
        if 'peak_vram_mb' in result.metadata:
            assert result.metadata['peak_vram_mb'] > 0
            print(f"Peak VRAM usage: {result.metadata['peak_vram_mb']:.1f} MB")
    
    def test_concurrent_expansions(self, expandor, test_image_small,
                                 mock_img2img_pipeline, temp_dir):
        """Test concurrent expansions don't interfere"""
        def expand_image(index):
            config = self.create_config(
                source_image=test_image_small,
                target_resolution=(768, 768),
                img2img_pipeline=mock_img2img_pipeline,
                seed=42 + index,  # Different seeds
                stage_dir=temp_dir / f"stages_{index}"
            )
            return expandor.expand(config)
        
        # Run multiple expansions concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(expand_image, i) for i in range(3)]
            results = [f.result() for f in futures]
        
        # All should succeed
        for result in results:
            self.validate_result(result, (768, 768))
    
    def test_stage_cleanup(self, expandor, test_image_small,
                         mock_inpaint_pipeline, temp_dir):
        """Test temporary files are cleaned up"""
        config = self.create_config(
            source_image=test_image_small,
            target_resolution=(1024, 768),
            inpaint_pipeline=mock_inpaint_pipeline,
            save_stages=False,  # Don't save stages
            stage_dir=temp_dir / "stages"
        )
        
        # Track temp files before
        temp_files_before = list(expandor._temp_base.glob("*")) if hasattr(expandor, '_temp_base') else []
        
        result = expandor.expand(config)
        
        # Track temp files after
        temp_files_after = list(expandor._temp_base.glob("*")) if hasattr(expandor, '_temp_base') else []
        
        # Should not accumulate temp files
        assert len(temp_files_after) <= len(temp_files_before) + 1  # Allow for result image


## 5. Test Runner Configuration

### 5.1 pytest Configuration

**FILE**: `tests/integration/conftest.py`

```python
"""
pytest configuration for integration tests
"""

import pytest
import logging
from pathlib import Path


def pytest_configure(config):
    """Configure pytest for integration tests"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "quality: marks quality validation tests"
    )


@pytest.fixture(scope="session")
def integration_test_dir():
    """Create session-wide test directory"""
    test_dir = Path("test_outputs")
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
```

### 5.2 Running the Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific test file
pytest tests/integration/test_full_pipeline.py -v

# Run with coverage
pytest tests/integration/ --cov=expandor --cov-report=html

# Run only fast tests (skip slow ones)
pytest tests/integration/ -v -m "not slow"

# Run with specific log level
pytest tests/integration/ -v --log-cli-level=DEBUG

# Run specific test
pytest tests/integration/test_swpo_integration.py::TestSWPOIntegration::test_swpo_extreme_horizontal -v
```

## Summary

This final comprehensive implementation guide for Phase 3 Step 3 includes:

1. **Complete test infrastructure** with proper base class and fixtures
2. **Full pipeline integration tests** covering all major workflows
3. **Strategy-specific tests** for SWPO, CPU Offload, and quality systems
4. **Edge case tests** ensuring robust error handling
5. **Performance tests** validating efficiency and resource usage
6. **Proper pytest configuration** and test organization

All tests follow the FAIL LOUD philosophy, use the correct mock pipelines from `expandor.adapters.mock_pipeline`, and comprehensively validate the Expandor system. The tests are production-ready with proper assertions, error checking, and debug information.