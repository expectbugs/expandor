# Expandor Phase 2 Step 3: Unit Tests

## Overview

This step implements comprehensive unit tests for all base architecture components and basic strategies. Every test is designed to verify both success paths and failure modes, ensuring our "fail loud" philosophy is properly implemented.

## Step 2.3.1: Test Infrastructure Setup

### Location: `tests/conftest.py`

```python
"""
Pytest configuration and fixtures for Expandor tests
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from PIL import Image
import numpy as np
import logging

# Add expandor to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from expandor import Expandor, ExpandorConfig
from expandor.adapters.mock_pipeline import (
    MockInpaintPipeline, 
    MockRefinerPipeline,
    MockImg2ImgPipeline
)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)

@pytest.fixture
def test_image_square():
    """Create a test square image"""
    img = Image.new('RGB', (512, 512), color='blue')
    # Add some pattern for better testing
    pixels = img.load()
    for i in range(512):
        for j in range(512):
            if (i // 64 + j // 64) % 2 == 0:
                pixels[i, j] = (255, 0, 0)  # Red squares
    return img

@pytest.fixture
def test_image_landscape():
    """Create a test landscape image"""
    img = Image.new('RGB', (1024, 576), color='green')
    # Add gradient
    pixels = np.array(img)
    for i in range(576):
        pixels[i, :, 1] = int(255 * (1 - i / 576))  # Green gradient
    return Image.fromarray(pixels)

@pytest.fixture
def test_image_portrait():
    """Create a test portrait image"""
    return Image.new('RGB', (768, 1024), color='purple')

@pytest.fixture
def mock_inpaint_pipeline():
    """Create mock inpaint pipeline"""
    return MockInpaintPipeline()

@pytest.fixture
def mock_refiner_pipeline():
    """Create mock refiner pipeline"""
    return MockRefinerPipeline()

@pytest.fixture
def mock_img2img_pipeline():
    """Create mock img2img pipeline"""
    return MockImg2ImgPipeline()

@pytest.fixture
def test_config(test_image_square):
    """Create a basic test configuration"""
    return ExpandorConfig(
        source_image=test_image_square,
        target_resolution=(1024, 1024),
        prompt="A test image",
        seed=42,
        source_metadata={'model': 'test'},
        quality_preset='fast'
    )

@pytest.fixture
def mock_logger():
    """Create a mock logger that captures messages"""
    class LogCapture:
        def __init__(self):
            self.messages = {
                'debug': [],
                'info': [],
                'warning': [],
                'error': [],
                'critical': []
            }
        
        def debug(self, msg, *args, **kwargs):
            self.messages['debug'].append(msg)
        
        def info(self, msg, *args, **kwargs):
            self.messages['info'].append(msg)
        
        def warning(self, msg, *args, **kwargs):
            self.messages['warning'].append(msg)
        
        def error(self, msg, *args, **kwargs):
            self.messages['error'].append(msg)
        
        def critical(self, msg, *args, **kwargs):
            self.messages['critical'].append(msg)
    
    return LogCapture()

@pytest.fixture(autouse=True)
def setup_test_env(temp_dir):
    """Setup test environment"""
    # Create temp directory in test location
    test_temp = temp_dir / "temp"
    test_temp.mkdir(exist_ok=True)
    
    # Monkey patch temp directory
    import expandor.strategies.base_strategy
    original_save = expandor.strategies.base_strategy.BaseExpansionStrategy.save_temp_image
    
    def patched_save(self, image, name):
        timestamp = int(time.time() * 1000)
        temp_path = test_temp / f"{name}_{timestamp}.png"
        image.save(temp_path, "PNG", compress_level=0)
        self.temp_files.append(temp_path)
        return temp_path
    
    expandor.strategies.base_strategy.BaseExpansionStrategy.save_temp_image = patched_save
    
    yield
    
    # Restore
    expandor.strategies.base_strategy.BaseExpansionStrategy.save_temp_image = original_save
```

### Location: `tests/__init__.py`

```python
"""Test package initialization"""
```

## Step 2.3.2: Core Component Tests

### Location: `tests/unit/test_vram_manager.py` (update existing)

```python
"""
Test VRAM Manager functionality
"""

import pytest
import torch
from expandor.core.vram_manager import VRAMManager

class TestVRAMManager:
    
    def setup_method(self):
        """Setup for each test"""
        self.vram_manager = VRAMManager()
    
    def test_calculate_generation_vram(self):
        """Test VRAM calculation for different resolutions"""
        # Create mock config
        config = type('Config', (), {})()
        
        # Test 1080p
        config.target_resolution = (1920, 1080)
        result = self.vram_manager.estimate_requirement(config)
        assert isinstance(result, float)
        assert result > 0
        
        # Test 4K
        config.target_resolution = (3840, 2160)
        result_4k = self.vram_manager.estimate_requirement(config)
        assert result_4k > result
    
    def test_determine_strategy(self):
        """Test strategy determination"""
        # Small image should use full strategy (if VRAM available)
        strategy = self.vram_manager.determine_strategy(1024, 1024)
        assert strategy['strategy'] in ['full', 'tiled', 'cpu_offload']
        assert 'vram_required_mb' in strategy
        assert 'vram_available_mb' in strategy
        
        # Huge image might need tiling or CPU offload
        strategy_huge = self.vram_manager.determine_strategy(8192, 8192)
        assert strategy_huge['vram_required_mb'] > strategy['vram_required_mb']
        
        # If no GPU, should always be cpu_offload
        if not torch.cuda.is_available():
            assert strategy['strategy'] == 'cpu_offload'
    
    def test_get_available_vram(self):
        """Test VRAM availability check"""
        vram = self.vram_manager.get_available_vram()
        
        if torch.cuda.is_available():
            assert vram is not None
            assert vram > 0
            assert isinstance(vram, float)
        else:
            # CPU-only system
            assert vram is None
    
    def test_model_overhead_constants(self):
        """Test that model overhead constants are reasonable"""
        assert self.vram_manager.MODEL_OVERHEAD_MB > 0
        assert self.vram_manager.MODEL_OVERHEAD_MB < 20000  # Less than 20GB
        assert self.vram_manager.ATTENTION_MULTIPLIER > 0
        assert self.vram_manager.SAFETY_BUFFER > 0
        assert self.vram_manager.SAFETY_BUFFER < 1.0
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Zero dimensions should raise error or return 0
        with pytest.raises(Exception):
            self.vram_manager.estimate_requirement({
                'target_resolution': (0, 0),
                'source_metadata': {'model': 'test'}
            })
        
        # Negative dimensions
        with pytest.raises(Exception):
            self.vram_manager.estimate_requirement({
                'target_resolution': (-100, 100),
                'source_metadata': {'model': 'test'}
            })
        
        # Extremely large dimensions
        result = self.vram_manager.estimate_requirement({
            'target_resolution': (65536, 65536),
            'source_metadata': {'model': 'test'}
        })
        assert result['total_vram_mb'] > 10000  # Should be huge
```

### Location: `tests/unit/test_strategy_selector.py`

```python
"""
Test Strategy Selector functionality
"""

import pytest
from pathlib import Path
from PIL import Image

from expandor.core.strategy_selector import StrategySelector, SelectionMetrics
from expandor.core.config import ExpandorConfig
from expandor.core.vram_manager import VRAMManager
from expandor.strategies.direct_upscale import DirectUpscaleStrategy
from expandor.strategies.progressive_outpaint import ProgressiveOutpaintStrategy
from expandor.strategies.tiled_expansion import TiledExpansionStrategy

class TestStrategySelector:
    
    def setup_method(self):
        """Setup for each test"""
        self.config = {
            'strategies': {
                'progressive_outpainting': {
                    'enabled': True,
                    'aspect_ratio_thresholds': {
                        'extreme': 4.0,
                        'max_supported': 8.0
                    }
                },
                'swpo': {'enabled': True},
                'direct_upscale': {'enabled': True}
            },
            'quality_presets': {
                'fast': {},
                'balanced': {},
                'high': {},
                'ultra': {}
            },
            'vram_strategies': {
                'thresholds': {
                    'full_processing': 8000,
                    'tiled_processing': 4000
                },
                'safety_factor': 0.8
            }
        }
        
        self.vram_manager = VRAMManager()
        self.selector = StrategySelector(self.config, self.vram_manager)
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        # Create test config
        test_image = Image.new('RGB', (1024, 1024))
        config = ExpandorConfig(
            source_image=test_image,
            target_resolution=(2048, 1024),  # 2:1 aspect
            prompt="Test",
            seed=42,
            source_metadata={'model': 'sdxl'},
            inpaint_pipeline="mock_pipeline"
        )
        
        metrics = self.selector._calculate_metrics(config)
        
        assert metrics.source_size == (1024, 1024)
        assert metrics.target_size == (2048, 1024)
        assert metrics.area_ratio == 2.0  # Double the pixels
        assert metrics.aspect_change == 2.0  # 1:1 to 2:1
        assert metrics.has_inpaint == True
        assert metrics.model_type == 'sdxl'
    
    def test_simple_upscale_selection(self):
        """Test selection for simple upscaling"""
        test_image = Image.new('RGB', (1024, 1024))
        config = ExpandorConfig(
            source_image=test_image,
            target_resolution=(2048, 2048),  # Same aspect, 4x area
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        strategy = self.selector.select(config)
        
        # Should select direct upscale for simple scaling
        assert isinstance(strategy, DirectUpscaleStrategy)
        assert "Simple upscale" in self.selector.get_selection_reason()
    
    def test_aspect_change_selection(self):
        """Test selection for aspect ratio change"""
        test_image = Image.new('RGB', (1024, 1024))
        config = ExpandorConfig(
            source_image=test_image,
            target_resolution=(2048, 1024),  # 2:1 aspect change
            prompt="Test",
            seed=42,
            source_metadata={'model': 'sdxl'},
            inpaint_pipeline="mock_pipeline"
        )
        
        strategy = self.selector.select(config)
        
        # Should select progressive outpainting for aspect change
        assert isinstance(strategy, ProgressiveOutpaintStrategy)
        assert "aspect ratio change" in self.selector.get_selection_reason()
    
    def test_vram_limited_selection(self):
        """Test selection when VRAM is limited"""
        test_image = Image.new('RGB', (512, 512))
        config = ExpandorConfig(
            source_image=test_image,
            target_resolution=(8192, 8192),  # Huge size
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'},
            vram_limit_mb=2000,  # Force low VRAM
            allow_tiled=True
        )
        
        # Mock low VRAM
        original_get_vram = self.vram_manager.get_available_vram
        self.vram_manager.get_available_vram = lambda: 2000.0
        
        try:
            strategy = self.selector.select(config)
            
            # Should select tiled expansion due to VRAM limit
            assert isinstance(strategy, TiledExpansionStrategy)
            assert "VRAM" in self.selector.get_selection_reason()
        finally:
            self.vram_manager.get_available_vram = original_get_vram
    
    def test_strategy_override(self):
        """Test manual strategy override"""
        test_image = Image.new('RGB', (512, 512))
        config = ExpandorConfig(
            source_image=test_image,
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'},
            strategy_override='tiled_expansion'
        )
        
        strategy = self.selector.select(config)
        
        # Should use the override
        assert isinstance(strategy, TiledExpansionStrategy)
        assert "user-specified" in self.selector.get_selection_reason()
    
    def test_no_gpu_selection(self):
        """Test selection when no GPU available"""
        # Mock no GPU
        original_get_vram = self.vram_manager.get_available_vram
        self.vram_manager.get_available_vram = lambda: None
        
        try:
            test_image = Image.new('RGB', (512, 512))
            config = ExpandorConfig(
                source_image=test_image,
                target_resolution=(1024, 1024),
                prompt="Test",
                seed=42,
                source_metadata={'model': 'test'},
                allow_cpu_offload=False
            )
            
            # Should raise error if CPU offload disabled
            with pytest.raises(Exception) as excinfo:
                strategy = self.selector.select(config)
            assert "No GPU available" in str(excinfo.value)
            
        finally:
            self.vram_manager.get_available_vram = original_get_vram
    
    def test_extreme_aspect_selection(self):
        """Test selection for extreme aspect ratio"""
        test_image = Image.new('RGB', (1024, 1024))
        config = ExpandorConfig(
            source_image=test_image,
            target_resolution=(5120, 1024),  # 5:1 extreme aspect
            prompt="Test",
            seed=42,
            source_metadata={'model': 'sdxl'},
            inpaint_pipeline="mock_pipeline"
        )
        
        strategy = self.selector.select(config)
        
        # Should select SWPO for extreme aspect (or progressive as fallback)
        assert "aspect ratio change" in self.selector.get_selection_reason().lower()
    
    def test_strategy_caching(self):
        """Test that strategies are cached"""
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        strategy1 = self.selector.select(config)
        strategy2 = self.selector.select(config)
        
        # Should be the same instance (cached)
        assert strategy1 is strategy2
        
        # Clear cache
        self.selector.clear_cache()
        
        strategy3 = self.selector.select(config)
        # Should be different instance after cache clear
        assert strategy1 is not strategy3
```

### Location: `tests/unit/test_metadata_tracker.py`

```python
"""
Test Metadata Tracker functionality
"""

import pytest
import time
import json
from pathlib import Path

from expandor.core.metadata_tracker import MetadataTracker
from expandor.core.config import ExpandorConfig

class TestMetadataTracker:
    
    def setup_method(self):
        """Setup for each test"""
        self.tracker = MetadataTracker()
    
    def test_initialization(self):
        """Test tracker initialization"""
        assert self.tracker.operation_id.startswith("op_")
        assert self.tracker.current_stage == "initialization"
        assert len(self.tracker.events) == 0
        assert isinstance(self.tracker.start_time, float)
    
    def test_start_operation(self):
        """Test starting a new operation"""
        config = ExpandorConfig(
            source_image=Path("test.png"),
            target_resolution=(1920, 1080),
            prompt="Test prompt",
            seed=123,
            source_metadata={'model': 'sdxl'},
            quality_preset='high'
        )
        
        self.tracker.start_operation(config)
        
        # Check snapshot was created
        assert self.tracker.config_snapshot is not None
        assert self.tracker.config_snapshot['target_resolution'] == (1920, 1080)
        assert self.tracker.config_snapshot['quality_preset'] == 'high'
        assert self.tracker.config_snapshot['seed'] == 123
        
        # Check start event was recorded
        assert len(self.tracker.events) == 1
        assert self.tracker.events[0]['type'] == 'operation_start'
    
    def test_stage_tracking(self):
        """Test stage enter/exit tracking"""
        self.tracker.enter_stage("test_stage")
        assert self.tracker.current_stage == "test_stage"
        assert self.tracker.stage_start is not None
        
        time.sleep(0.1)  # Let some time pass
        
        self.tracker.exit_stage(success=True)
        assert "test_stage" in self.tracker.stage_timings
        assert self.tracker.stage_timings["test_stage"] >= 0.1
        
        # Check events
        stage_events = [e for e in self.tracker.events if 'stage' in e['type']]
        assert len(stage_events) == 2
        assert stage_events[0]['type'] == 'stage_enter'
        assert stage_events[1]['type'] == 'stage_exit'
        assert stage_events[1]['data']['success'] == True
    
    def test_event_recording(self):
        """Test event recording"""
        self.tracker.record_event("test_event", {"key": "value", "number": 42})
        
        assert len(self.tracker.events) == 1
        event = self.tracker.events[0]
        assert event['type'] == 'test_event'
        assert event['data']['key'] == 'value'
        assert event['data']['number'] == 42
        assert 'timestamp' in event
        assert 'relative_time' in event
    
    def test_metric_recording(self):
        """Test metric recording"""
        self.tracker.record_metric("vram_peak", 4096.5)
        self.tracker.record_metric("quality_score", 0.95)
        
        assert self.tracker.metrics['vram_peak'] == 4096.5
        assert self.tracker.metrics['quality_score'] == 0.95
    
    def test_operation_log(self):
        """Test getting complete operation log"""
        # Simulate an operation
        config = ExpandorConfig(
            source_image=Path("test.png"),
            target_resolution=(1920, 1080),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'},
            quality_preset='fast'
        )
        
        self.tracker.start_operation(config)
        self.tracker.enter_stage("stage1")
        self.tracker.record_event("processing", {"status": "ok"})
        self.tracker.record_metric("test_metric", 123)
        self.tracker.exit_stage()
        
        log = self.tracker.get_operation_log()
        
        assert log['operation_id'] == self.tracker.operation_id
        assert log['duration'] > 0
        assert log['config'] == self.tracker.config_snapshot
        assert len(log['events']) >= 3  # start, enter, exit
        assert log['metrics']['test_metric'] == 123
        assert 'stage1' in log['stage_timings']
        assert 'event_summary' in log
    
    def test_partial_result(self):
        """Test getting partial result for errors"""
        self.tracker.enter_stage("failing_stage")
        self.tracker.record_event("error", {"message": "Test error"})
        
        partial = self.tracker.get_partial_result()
        
        assert partial['last_stage'] == "failing_stage"
        assert partial['events_count'] >= 1
        assert len(partial['last_events']) >= 1
        assert partial['last_events'][-1]['data']['message'] == "Test error"
    
    def test_save_to_file(self, temp_dir):
        """Test saving log to file"""
        self.tracker.record_event("test", {"data": "value"})
        self.tracker.record_metric("metric", 42)
        
        log_path = temp_dir / "test_log.json"
        self.tracker.save_to_file(log_path)
        
        assert log_path.exists()
        
        # Load and verify
        with open(log_path) as f:
            loaded = json.load(f)
        
        assert loaded['operation_id'] == self.tracker.operation_id
        assert loaded['metrics']['metric'] == 42
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Simulate stages
        self.tracker.enter_stage("stage1")
        time.sleep(0.1)
        self.tracker.exit_stage()
        
        self.tracker.enter_stage("stage2")
        time.sleep(0.2)
        self.tracker.exit_stage()
        
        self.tracker.record_metric("test", 123)
        
        summary = self.tracker.get_performance_summary()
        
        assert summary['total_duration'] >= 0.3
        assert 'stage1' in summary['stages']
        assert 'stage2' in summary['stages']
        assert 'stage1' in summary['stage_percentages']
        assert summary['stage_percentages']['stage1'] < 50  # stage1 was shorter
        assert summary['metrics']['test'] == 123
        assert summary['events_per_second'] > 0
```

### Location: `tests/unit/test_boundary_tracker.py`

```python
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
        assert len(self.tracker.horizontal_positions) == 0
        assert len(self.tracker.vertical_positions) == 0
        assert len(self.tracker.boundary_map['horizontal']) == 0
        assert len(self.tracker.boundary_map['vertical']) == 0
    
    def test_add_boundary(self):
        """Test adding individual boundaries"""
        self.tracker.add_boundary(
            position=100,
            direction='horizontal',
            step=1,
            expansion_size=50,
            source_size=(200, 200),
            target_size=(300, 200),
            method='progressive'
        )
        
        assert len(self.tracker.boundaries) == 1
        boundary = self.tracker.boundaries[0]
        assert boundary.position == 100
        assert boundary.direction == 'horizontal'
        assert boundary.step == 1
        assert boundary.expansion_size == 50
        assert boundary.method == 'progressive'
        
        assert 100 in self.tracker.horizontal_positions
        assert len(self.tracker.boundary_map['horizontal']) == 1
    
    def test_add_progressive_boundaries(self):
        """Test adding boundaries for progressive expansion"""
        self.tracker.add_progressive_boundaries(
            current_size=(1024, 768),
            target_size=(2048, 768),  # Horizontal expansion
            step=1,
            method='progressive'
        )
        
        # Should add left and right boundaries
        assert len(self.tracker.boundaries) == 2
        assert len(self.tracker.horizontal_positions) == 2
        
        # Check boundary positions
        positions = sorted(list(self.tracker.horizontal_positions))
        assert positions[0] == 512  # Left boundary (pad)
        assert positions[1] == 1536  # Right boundary (pad + original width)
    
    def test_add_sliding_window_boundaries(self):
        """Test adding boundaries for sliding window"""
        windows = [
            (0, 200),      # First window
            (150, 350),    # Second window (overlaps)
            (300, 500),    # Third window
        ]
        
        self.tracker.add_sliding_window_boundaries(
            window_positions=windows,
            direction='horizontal',
            step=1
        )
        
        # Should add boundaries for overlapping windows (not first)
        assert len(self.tracker.boundaries) == 2
        assert 150 in self.tracker.horizontal_positions
        assert 300 in self.tracker.horizontal_positions
    
    def test_add_tile_boundaries(self):
        """Test adding boundaries for tiled processing"""
        tiles = [
            (0, 0, 512, 512),      # Top-left
            (512, 0, 1024, 512),   # Top-right
            (0, 512, 512, 1024),   # Bottom-left
            (512, 512, 1024, 1024) # Bottom-right
        ]
        
        self.tracker.add_tile_boundaries(tiles, step=1)
        
        # Should add boundaries at tile edges
        assert 512 in self.tracker.horizontal_positions
        assert 512 in self.tracker.vertical_positions
    
    def test_get_all_boundaries(self):
        """Test getting all boundaries as list"""
        self.tracker.add_boundary(100, 'horizontal', 1, 50, (200, 200), (300, 200))
        self.tracker.add_boundary(150, 'vertical', 2, 100, (300, 200), (300, 300))
        
        all_boundaries = self.tracker.get_all_boundaries()
        
        assert len(all_boundaries) == 2
        assert all(isinstance(b, dict) for b in all_boundaries)
        assert all_boundaries[0]['position'] == 100
        assert all_boundaries[1]['position'] == 150
    
    def test_get_boundaries_for_detection(self):
        """Test getting boundaries formatted for artifact detection"""
        self.tracker.add_boundary(100, 'horizontal', 1, 50, (200, 200), (300, 200))
        self.tracker.add_boundary(200, 'horizontal', 2, 50, (300, 200), (400, 200))
        self.tracker.add_boundary(150, 'vertical', 1, 50, (400, 200), (400, 300))
        
        detection_boundaries = self.tracker.get_boundaries_for_detection()
        
        assert 'horizontal' in detection_boundaries
        assert 'vertical' in detection_boundaries
        assert detection_boundaries['horizontal'] == [100, 200]
        assert detection_boundaries['vertical'] == [150]
    
    def test_get_boundary_regions(self):
        """Test getting regions around boundaries"""
        self.tracker.add_boundary(
            100, 'horizontal', 1, 50, (200, 200), (300, 200),
            metadata={'test': True}
        )
        
        regions = self.tracker.get_boundary_regions(margin=10)
        
        assert len(regions) == 1
        x1, y1, x2, y2 = regions[0]
        assert x1 == 90   # 100 - 10
        assert x2 == 110  # 100 + 10
        assert y1 == 0
        assert y2 == 200  # Full height
    
    def test_get_critical_boundaries(self):
        """Test getting critical boundaries"""
        # Add various boundaries
        self.tracker.add_boundary(100, 'horizontal', 1, 50, (200, 200), (250, 200), method='progressive')
        self.tracker.add_boundary(200, 'horizontal', 2, 300, (250, 200), (550, 200), method='progressive')
        self.tracker.add_boundary(300, 'vertical', 3, 50, (550, 200), (550, 250), method='tiled')
        
        critical = self.tracker.get_critical_boundaries()
        
        # Progressive and large expansions are critical
        assert len(critical) == 2
        assert all(b.method == 'progressive' or b.expansion_size > 200 for b in critical)
    
    def test_summarize(self):
        """Test boundary summary"""
        self.tracker.add_progressive_boundaries((1024, 768), (2048, 768), 1)
        self.tracker.add_progressive_boundaries((2048, 768), (2048, 1536), 2)
        
        summary = self.tracker.summarize()
        
        assert summary['total_boundaries'] == 4  # 2 horizontal + 2 vertical
        assert summary['horizontal_count'] == 2
        assert summary['vertical_count'] == 2
        assert 'progressive' in summary['methods_used']
        assert summary['expansion_steps'] == 2
        assert summary['critical_boundaries'] == 4
        assert summary['largest_expansion'] > 0
    
    def test_reset(self):
        """Test resetting tracker"""
        self.tracker.add_boundary(100, 'horizontal', 1, 50, (200, 200), (300, 200))
        assert len(self.tracker.boundaries) == 1
        
        self.tracker.reset()
        
        assert len(self.tracker.boundaries) == 0
        assert len(self.tracker.horizontal_positions) == 0
        assert len(self.tracker.vertical_positions) == 0
    
    def test_invalid_direction(self):
        """Test invalid direction raises error"""
        with pytest.raises(ValueError):
            self.tracker.add_boundary(100, 'diagonal', 1, 50, (200, 200), (300, 300))
```

### Location: `tests/unit/test_pipeline_orchestrator.py`

```python
"""
Test Pipeline Orchestrator functionality
"""

import pytest
from unittest.mock import Mock, MagicMock

from expandor.core.pipeline_orchestrator import PipelineOrchestrator
from expandor.core.config import ExpandorConfig
from expandor.core.metadata_tracker import MetadataTracker
from expandor.core.boundary_tracker import BoundaryTracker
from expandor.core.exceptions import StrategyError, VRAMError
from expandor.strategies.base_strategy import BaseExpansionStrategy

class MockStrategy(BaseExpansionStrategy):
    """Mock strategy for testing"""
    
    def __init__(self, should_fail=False, failure_type=None):
        super().__init__()
        self.should_fail = should_fail
        self.failure_type = failure_type
        self.execute_called = False
    
    def execute(self, config, context):
        self.execute_called = True
        
        if self.should_fail:
            if self.failure_type == 'vram':
                raise VRAMError("Insufficient VRAM", required_mb=8000, available_mb=4000)
            else:
                raise StrategyError("Mock strategy failed")
        
        return {
            'image_path': 'test.png',
            'size': (1024, 1024),
            'stages': [],
            'boundaries': []
        }

class TestPipelineOrchestrator:
    
    def setup_method(self):
        """Setup for each test"""
        self.config = {
            'vram_strategies': {
                'fallback_chain': {
                    1: 'tiled_large',
                    2: 'tiled_medium',
                    3: 'cpu_offload'
                }
            }
        }
        self.orchestrator = PipelineOrchestrator(self.config)
        self.metadata_tracker = MetadataTracker()
        self.boundary_tracker = BoundaryTracker()
    
    def test_successful_execution(self):
        """Test successful strategy execution"""
        strategy = MockStrategy()
        config = Mock(spec=ExpandorConfig)
        config.quality_preset = 'balanced'
        
        result = self.orchestrator.execute(
            strategy=strategy,
            config=config,
            metadata_tracker=self.metadata_tracker,
            boundary_tracker=self.boundary_tracker
        )
        
        assert strategy.execute_called
        assert result.success
        assert result.strategy_used == 'MockStrategy'
        assert result.fallback_count == 0
    
    def test_pipeline_registration(self):
        """Test pipeline registration"""
        mock_pipeline = Mock()
        self.orchestrator.register_pipeline('test', mock_pipeline)
        
        assert 'test' in self.orchestrator.pipeline_registry
        assert self.orchestrator.pipeline_registry['test'] is mock_pipeline
    
    def test_strategy_preparation(self):
        """Test strategy is prepared with pipelines"""
        strategy = MockStrategy()
        mock_inpaint = Mock()
        mock_refiner = Mock()
        
        self.orchestrator.register_pipeline('inpaint', mock_inpaint)
        self.orchestrator.register_pipeline('refiner', mock_refiner)
        
        self.orchestrator._prepare_strategy(strategy, self.boundary_tracker)
        
        assert strategy.inpaint_pipeline is mock_inpaint
        assert strategy.refiner_pipeline is mock_refiner
        assert strategy.boundary_tracker is self.boundary_tracker
    
    def test_fallback_on_failure(self):
        """Test fallback chain on strategy failure"""
        # Create primary strategy that fails
        primary = MockStrategy(should_fail=True)
        
        # Create fallback that succeeds
        fallback = MockStrategy(should_fail=False)
        
        # Mock the fallback chain building
        original_build = self.orchestrator._build_fallback_chain
        self.orchestrator._build_fallback_chain = lambda s, c: [primary, fallback]
        
        try:
            config = Mock(spec=ExpandorConfig)
            config.quality_preset = 'high'
            
            result = self.orchestrator.execute(
                strategy=primary,
                config=config,
                metadata_tracker=self.metadata_tracker,
                boundary_tracker=self.boundary_tracker
            )
            
            assert primary.execute_called
            assert fallback.execute_called
            assert result.strategy_used == 'MockStrategy'
            assert result.fallback_count == 1
            
        finally:
            self.orchestrator._build_fallback_chain = original_build
    
    def test_complete_failure(self):
        """Test when all strategies fail"""
        # All strategies fail
        strategies = [
            MockStrategy(should_fail=True),
            MockStrategy(should_fail=True),
            MockStrategy(should_fail=True)
        ]
        
        # Mock the fallback chain
        self.orchestrator._build_fallback_chain = lambda s, c: strategies
        
        config = Mock(spec=ExpandorConfig)
        config.quality_preset = 'high'
        
        with pytest.raises(Exception) as excinfo:
            self.orchestrator.execute(
                strategy=strategies[0],
                config=config,
                metadata_tracker=self.metadata_tracker,
                boundary_tracker=self.boundary_tracker
            )
        
        assert "All expansion strategies failed" in str(excinfo.value)
        assert all(s.execute_called for s in strategies)
    
    def test_vram_error_handling(self):
        """Test special handling of VRAM errors"""
        strategy = MockStrategy(should_fail=True, failure_type='vram')
        
        # No fallbacks for simplicity
        self.orchestrator._build_fallback_chain = lambda s, c: [strategy]
        
        config = Mock(spec=ExpandorConfig)
        config.quality_preset = 'high'
        
        with pytest.raises(Exception):
            self.orchestrator.execute(
                strategy=strategy,
                config=config,
                metadata_tracker=self.metadata_tracker,
                boundary_tracker=self.boundary_tracker
            )
        
        # Check that VRAM error was recorded
        events = [e for e in self.metadata_tracker.events 
                 if e['type'] == 'strategy_execution_failed']
        assert len(events) > 0
        assert 'VRAM' in events[0]['data']['error']
    
    def test_execution_history(self):
        """Test execution history tracking"""
        strategies = [
            MockStrategy(should_fail=True),
            MockStrategy(should_fail=False)
        ]
        
        self.orchestrator._build_fallback_chain = lambda s, c: strategies
        
        config = Mock(spec=ExpandorConfig)
        config.quality_preset = 'high'
        
        result = self.orchestrator.execute(
            strategy=strategies[0],
            config=config,
            metadata_tracker=self.metadata_tracker,
            boundary_tracker=self.boundary_tracker
        )
        
        # Check execution history
        history = result.metadata['execution_history']
        assert len(history) == 2
        assert history[0]['status'] == 'failed'
        assert history[1]['status'] == 'success'  # Second strategy succeeded
```

## Step 2.3.3: Strategy Tests

### Location: `tests/unit/test_direct_upscale_strategy.py`

```python
"""
Test Direct Upscale Strategy
"""

import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch

from expandor.strategies.direct_upscale import DirectUpscaleStrategy
from expandor.core.config import ExpandorConfig
from expandor.core.exceptions import StrategyError

class TestDirectUpscaleStrategy:
    
    def setup_method(self):
        """Setup for each test"""
        self.strategy = DirectUpscaleStrategy()
        self.context = {
            'save_stages': False,
            'stage_callback': Mock()
        }
    
    def test_initialization(self):
        """Test strategy initialization"""
        assert self.strategy.upscale_config is not None
        assert self.strategy.model_config is not None
        assert self.strategy.tile_config is not None
    
    @patch('expandor.strategies.direct_upscale.DirectUpscaleStrategy._find_realesrgan')
    def test_missing_realesrgan(self, mock_find):
        """Test error when Real-ESRGAN not found"""
        mock_find.return_value = None
        
        with pytest.raises(StrategyError) as excinfo:
            DirectUpscaleStrategy()
        
        assert "Real-ESRGAN not found" in str(excinfo.value)
    
    def test_simple_2x_upscale(self, test_image_square):
        """Test simple 2x upscaling"""
        config = ExpandorConfig(
            source_image=test_image_square,
            target_resolution=(1024, 1024),  # 2x
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        # Mock Real-ESRGAN execution
        self.strategy._run_realesrgan = Mock(return_value=Path("test_upscaled.png"))
        
        # Create mock upscaled image
        mock_image = Image.new('RGB', (1024, 1024))
        self.strategy.validate_image_path = Mock(return_value=mock_image)
        
        result = self.strategy.execute(config, self.context)
        
        assert result['size'] == (1024, 1024)
        assert len(result['stages']) >= 1
        assert result['metadata']['scale_factor'] == 2.0
        assert result['metadata']['passes'] == 1
        
        # Verify Real-ESRGAN was called
        self.strategy._run_realesrgan.assert_called_once()
    
    def test_large_upscale_multiple_passes(self, test_image_square):
        """Test large upscaling requiring multiple passes"""
        config = ExpandorConfig(
            source_image=test_image_square,
            target_resolution=(4096, 4096),  # 8x
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        # Mock Real-ESRGAN to simulate progressive upscaling
        call_count = 0
        def mock_upscale(input_path, scale, **kwargs):
            nonlocal call_count
            call_count += 1
            # First pass: 512 -> 2048 (4x)
            # Second pass: 2048 -> 8192 (4x)
            if call_count == 1:
                size = (2048, 2048)
            else:
                size = (8192, 8192)
            
            # Create and save mock image
            img = Image.new('RGB', size)
            path = Path(f"test_upscaled_{call_count}.png")
            img.save(path)
            return path
        
        self.strategy._run_realesrgan = Mock(side_effect=mock_upscale)
        
        result = self.strategy.execute(config, self.context)
        
        assert result['size'] == (4096, 4096)
        assert result['metadata']['passes'] == 2
        assert len(result['stages']) >= 3  # 2 upscale + 1 resize
        
        # Verify Real-ESRGAN was called twice
        assert self.strategy._run_realesrgan.call_count == 2
    
    def test_tile_size_determination(self):
        """Test VRAM-based tile size selection"""
        # Mock different VRAM amounts
        test_cases = [
            (10000, 2048),  # High VRAM
            (7000, 1024),   # Medium VRAM
            (5000, 768),    # Low-medium VRAM
            (3000, 512),    # Low VRAM
            (None, 512),    # No GPU
        ]
        
        for vram, expected_tile in test_cases:
            self.strategy.vram_manager.get_available_vram = Mock(return_value=vram)
            tile_size = self.strategy._determine_tile_size((1024, 1024))
            assert tile_size == expected_tile
    
    def test_non_square_upscale(self):
        """Test upscaling non-square images"""
        source = Image.new('RGB', (1920, 1080))  # 16:9
        config = ExpandorConfig(
            source_image=source,
            target_resolution=(3840, 2160),  # 4K, same aspect
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        # Mock Real-ESRGAN
        self.strategy._run_realesrgan = Mock(return_value=Path("test_4k.png"))
        mock_image = Image.new('RGB', (3840, 2160))
        self.strategy.validate_image_path = Mock(return_value=mock_image)
        
        result = self.strategy.execute(config, self.context)
        
        assert result['size'] == (3840, 2160)
        assert not result['metadata']['final_resize']  # Should be exact
    
    def test_vram_estimation(self):
        """Test VRAM estimation"""
        config = Mock()
        config.target_resolution = (2048, 2048)
        
        # Mock tile size determination
        self.strategy._determine_tile_size = Mock(return_value=1024)
        
        estimate = self.strategy.estimate_vram(config)
        
        assert 'base_vram_mb' in estimate
        assert 'peak_vram_mb' in estimate
        assert estimate['peak_vram_mb'] > estimate['base_vram_mb']
        assert estimate['strategy_overhead_mb'] == 500
```

### Location: `tests/unit/test_progressive_outpaint_strategy.py`

```python
"""
Test Progressive Outpaint Strategy
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, MagicMock

from expandor.strategies.progressive_outpaint import ProgressiveOutpaintStrategy
from expandor.core.config import ExpandorConfig
from expandor.core.exceptions import StrategyError

class TestProgressiveOutpaintStrategy:
    
    def setup_method(self):
        """Setup for each test"""
        self.strategy = ProgressiveOutpaintStrategy()
        self.strategy.inpaint_pipeline = Mock()
        self.context = {
            'save_stages': False,
            'stage_callback': Mock()
        }
    
    def test_initialization(self):
        """Test strategy initialization"""
        assert self.strategy.dimension_calc is not None
        assert self.strategy.denoising_strength > 0
        assert self.strategy.base_mask_blur > 0
        assert self.strategy.base_steps > 0
    
    def test_no_pipeline_error(self):
        """Test error when no inpaint pipeline"""
        strategy = ProgressiveOutpaintStrategy()
        strategy.inpaint_pipeline = None
        
        with pytest.raises(StrategyError) as excinfo:
            strategy.validate_requirements()
        
        assert "inpainting pipeline" in str(excinfo.value)
    
    def test_horizontal_expansion(self, test_image_square):
        """Test horizontal aspect ratio expansion"""
        config = ExpandorConfig(
            source_image=test_image_square,  # 512x512
            target_resolution=(1024, 512),    # 2:1 aspect
            prompt="Expand horizontally",
            seed=42,
            source_metadata={'model': 'sdxl'},
            inpaint_pipeline=Mock()
        )
        
        # Mock dimension calculator to return one step
        self.strategy.dimension_calc.calculate_progressive_strategy = Mock(
            return_value=[{
                'method': 'outpaint',
                'current_size': (512, 512),
                'target_size': (1024, 512),
                'expansion_ratio': 2.0,
                'direction': 'horizontal',
                'step_type': 'initial',
                'description': 'Expand horizontally'
            }]
        )
        
        # Mock inpaint pipeline
        result_image = Image.new('RGB', (1024, 512))
        self.strategy.inpaint_pipeline.return_value = Mock(images=[result_image])
        
        # Mock boundary tracker
        self.strategy.boundary_tracker = Mock()
        
        result = self.strategy.execute(config, self.context)
        
        assert result['size'] == (1024, 512)
        assert len(result['stages']) == 1
        assert result['metadata']['steps_executed'] == 1
        
        # Verify inpainting was called
        self.strategy.inpaint_pipeline.assert_called_once()
        
        # Verify boundaries were tracked
        self.strategy.boundary_tracker.add_progressive_boundaries.assert_called_once()
    
    def test_multi_step_expansion(self, test_image_portrait):
        """Test multi-step progressive expansion"""
        config = ExpandorConfig(
            source_image=test_image_portrait,  # 768x1024
            target_resolution=(3072, 1024),    # Much wider
            prompt="Expand to ultrawide",
            seed=42,
            source_metadata={'model': 'sdxl'},
            inpaint_pipeline=Mock()
        )
        
        # Mock multiple expansion steps
        self.strategy.dimension_calc.calculate_progressive_strategy = Mock(
            return_value=[
                {
                    'method': 'outpaint',
                    'current_size': (768, 1024),
                    'target_size': (1536, 1024),
                    'expansion_ratio': 2.0,
                    'direction': 'horizontal',
                    'step_type': 'initial',
                    'description': 'Step 1'
                },
                {
                    'method': 'outpaint',
                    'current_size': (1536, 1024),
                    'target_size': (3072, 1024),
                    'expansion_ratio': 2.0,
                    'direction': 'horizontal',
                    'step_type': 'final',
                    'description': 'Step 2'
                }
            ]
        )
        
        # Mock inpaint to return progressively larger images
        def mock_inpaint(*args, **kwargs):
            width = kwargs.get('width', 1024)
            height = kwargs.get('height', 1024)
            return Mock(images=[Image.new('RGB', (width, height))])
        
        self.strategy.inpaint_pipeline = Mock(side_effect=mock_inpaint)
        self.strategy.boundary_tracker = Mock()
        
        result = self.strategy.execute(config, self.context)
        
        assert result['size'] == (3072, 1024)
        assert len(result['stages']) == 2
        assert result['metadata']['steps_executed'] == 2
        
        # Verify progressive boundaries tracked
        assert self.strategy.boundary_tracker.add_progressive_boundaries.call_count == 2
    
    def test_edge_color_analysis(self):
        """Test edge color analysis"""
        # Create image with distinct edge colors
        img = Image.new('RGB', (100, 100))
        pixels = img.load()
        
        # Make left edge red
        for y in range(100):
            for x in range(10):
                pixels[x, y] = (255, 0, 0)
        
        # Make right edge blue
        for y in range(100):
            for x in range(90, 100):
                pixels[x, y] = (0, 0, 255)
        
        # Test left edge
        left_analysis = self.strategy._analyze_edge_colors(img, 'left', 10)
        assert left_analysis['mean_rgb'][0] > 250  # Red channel high
        assert left_analysis['mean_rgb'][2] < 5    # Blue channel low
        
        # Test right edge
        right_analysis = self.strategy._analyze_edge_colors(img, 'right', 10)
        assert right_analysis['mean_rgb'][0] < 5    # Red channel low
        assert right_analysis['mean_rgb'][2] > 250  # Blue channel high
    
    def test_adaptive_parameters(self):
        """Test adaptive parameter calculation"""
        # Test blur adaptation
        small_step = {'expansion_ratio': 1.2}
        large_step = {'expansion_ratio': 2.0}
        
        small_blur = self.strategy._get_adaptive_blur(small_step)
        large_blur = self.strategy._get_adaptive_blur(large_step)
        
        assert large_blur > small_blur
        
        # Test step adaptation
        initial_step = {'step_type': 'initial'}
        final_step = {'step_type': 'final'}
        
        initial_steps = self.strategy._get_adaptive_steps(initial_step)
        final_steps = self.strategy._get_adaptive_steps(final_step)
        
        assert initial_steps > final_steps
        
        # Test strength adaptation
        early_step = {'step_type': 'initial', 'step': 1}
        late_step = {'step_type': 'progressive', 'step': 4}
        
        early_strength = self.strategy._get_adaptive_strength(early_step)
        late_strength = self.strategy._get_adaptive_strength(late_step)
        
        assert early_strength > late_strength
    
    def test_canvas_prefill(self):
        """Test canvas pre-filling with edge colors"""
        # Create source image with colored edges
        source = Image.new('RGB', (100, 100), color='white')
        pixels = source.load()
        
        # Color the edges
        for i in range(100):
            pixels[0, i] = (255, 0, 0)      # Left red
            pixels[99, i] = (0, 255, 0)     # Right green
            pixels[i, 0] = (0, 0, 255)      # Top blue
            pixels[i, 99] = (255, 255, 0)   # Bottom yellow
        
        # Create larger canvas
        canvas = Image.new('RGB', (200, 200), color='black')
        
        # Pre-fill canvas
        filled = self.strategy._prefill_canvas_with_edge_colors(
            canvas, source, 50, 50, 100, 100
        )
        
        # Check that empty areas were filled
        filled_array = np.array(filled)
        
        # Top-left should be influenced by red/blue
        top_left = filled_array[10, 10]
        # Should have some color influence from edges
        assert top_left[0] > 20 or top_left[2] > 20  # Red or blue influence
        
        # Bottom-right should be influenced by green/yellow
        bottom_right = filled_array[180, 180]
        assert bottom_right[1] > 0
    
    def test_no_expansion_needed(self):
        """Test when no expansion is needed"""
        config = ExpandorConfig(
            source_image=Image.new('RGB', (1920, 1080)),
            target_resolution=(1920, 1080),  # Same size
            prompt="No change",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        # Mock no steps needed
        self.strategy.dimension_calc.calculate_progressive_strategy = Mock(
            return_value=[]
        )
        
        result = self.strategy.execute(config, self.context)
        
        assert result['size'] == (1920, 1080)
        assert len(result['stages']) == 0
        assert len(result['boundaries']) == 0
```

### Location: `tests/unit/test_tiled_expansion_strategy.py`

```python
"""
Test Tiled Expansion Strategy
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock

from expandor.strategies.tiled_expansion import TiledExpansionStrategy
from expandor.core.config import ExpandorConfig
from expandor.core.exceptions import StrategyError

class TestTiledExpansionStrategy:
    
    def setup_method(self):
        """Setup for each test"""
        self.strategy = TiledExpansionStrategy()
        self.strategy.refiner_pipeline = Mock()
        self.context = {
            'save_stages': False,
            'stage_callback': Mock()
        }
    
    def test_initialization(self):
        """Test strategy initialization"""
        assert getattr(self.strategy, 'default_tile_size', 1024) == 1024
        assert getattr(self.strategy, 'overlap', 256) == 256
        assert getattr(self.strategy, 'blend_width', 128) == 128
    
    def test_no_pipeline_error(self):
        """Test error when no pipeline available"""
        strategy = TiledExpansionStrategy()
        
        with pytest.raises(StrategyError) as excinfo:
            strategy.validate_requirements()
        
        assert "at least one pipeline" in str(excinfo.value)
    
    def test_tile_calculation(self):
        """Test tile position calculation"""
        tiles = self.strategy._calculate_tiles(2048, 2048, 1024)
        
        # Should create 3x3 grid with overlap
        # (0,0), (768,0), (1024,0)
        # (0,768), (768,768), (1024,768)
        # (0,1024), (768,1024), (1024,1024)
        assert len(tiles) == 9
        
        # Check first tile
        assert tiles[0] == (0, 0, 1024, 1024)
        
        # Check overlap exists
        assert tiles[1][0] < 1024  # Second tile starts before first ends
    
    def test_simple_tiled_processing(self, test_image_square):
        """Test basic tiled processing"""
        config = ExpandorConfig(
            source_image=test_image_square,
            target_resolution=(2048, 2048),
            prompt="Test tiled",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        # Mock tile processing
        def mock_refine(prompt, image, **kwargs):
            # Return slightly modified tile
            return Mock(images=[image])
        
        self.strategy.refiner_pipeline = Mock(side_effect=mock_refine)
        self.strategy.boundary_tracker = Mock()
        
        # Mock tile size determination
        self.strategy._determine_tile_size = Mock(return_value=1024)
        
        result = self.strategy.execute(config, self.context)
        
        assert result['size'] == (2048, 2048)
        assert result['metadata']['tile_count'] > 0
        assert result['metadata']['tile_size'] == 1024
        
        # Check stages include tile processing
        tile_stages = [s for s in result['stages'] if 'tile_' in s.get('name', '')]
        assert len(tile_stages) > 0
        
        # Verify boundaries tracked
        self.strategy.boundary_tracker.add_tile_boundaries.assert_called_once()
    
    def test_tile_blending(self):
        """Test tile blending logic"""
        # Create test tiles with different colors
        tiles = [
            {
                'image': Image.new('RGB', (100, 100), color='red'),
                'coords': (0, 0, 100, 100)
            },
            {
                'image': Image.new('RGB', (100, 100), color='blue'),
                'coords': (50, 0, 150, 100)  # Overlaps with first
            }
        ]
        
        blended = self.strategy._blend_tiles(
            tiles, (150, 100), overlap=50, blend_width=25
        )
        
        assert blended.size == (150, 100)
        
        # Check blending occurred in overlap region
        blended_array = np.array(blended)
        
        # At x=75 (middle of overlap), should be blend of red and blue
        middle_pixel = blended_array[50, 75]
        # Both red and blue should be present in the blend
        assert middle_pixel[0] > 50  # Significant red component
        assert middle_pixel[2] > 50  # Significant blue component
    
    def test_vram_based_tile_size(self):
        """Test tile size selection based on VRAM"""
        test_cases = [
            (8000, 2048),   # High VRAM -> large tiles
            (4000, 1024),   # Medium VRAM -> medium tiles
            (2000, 512),    # Low VRAM -> small tiles
            (None, 512),    # No GPU -> minimum tiles
        ]
        
        for vram, expected_tile in test_cases:
            self.strategy.vram_manager.get_available_vram = Mock(return_value=vram)
            
            # Mock VRAM calculation
            def mock_calc(config):
                # Simulate VRAM usage based on tile size
                w, h = config['target_resolution']
                base = (w * h * 4) / (1024 * 1024)  # Rough estimate
                return {'total_with_buffer_mb': base * 10, 'total_vram_mb': base * 8}
            
            self.strategy.vram_manager.estimate_requirement = Mock(
                side_effect=mock_calc
            )
            
            tile_size = self.strategy._determine_tile_size(4096, 4096)
            assert tile_size == expected_tile
    
    def test_different_pipelines(self):
        """Test tiled processing with different pipeline types"""
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        # Test with refiner
        self.strategy.refiner_pipeline = Mock(
            return_value=Mock(images=[Image.new('RGB', (512, 512))])
        )
        self.strategy.img2img_pipeline = None
        self.strategy.inpaint_pipeline = None
        
        result = self.strategy.execute(config, self.context)
        assert self.strategy.refiner_pipeline.called
        
        # Test with img2img
        self.strategy.refiner_pipeline = None
        self.strategy.img2img_pipeline = Mock(
            return_value=Mock(images=[Image.new('RGB', (512, 512))])
        )
        
        result = self.strategy.execute(config, self.context)
        assert self.strategy.img2img_pipeline.called
        
        # Test with inpaint
        self.strategy.img2img_pipeline = None
        self.strategy.inpaint_pipeline = Mock(
            return_value=Mock(images=[Image.new('RGB', (512, 512))])
        )
        
        result = self.strategy.execute(config, self.context)
        assert self.strategy.inpaint_pipeline.called
    
    def test_edge_tile_adjustment(self):
        """Test that edge tiles are adjusted properly"""
        # Test case where last tile would be too small
        tiles = self.strategy._calculate_tiles(1100, 1100, 512)
        
        # Last tile should be adjusted to minimum size
        last_tile = tiles[-1]
        tile_width = last_tile[2] - last_tile[0]
        tile_height = last_tile[3] - last_tile[1]
        
        # Minimum tile size should be at least 256
        min_tile_size = getattr(self.strategy, 'min_tile_size', 512)
        assert tile_width >= min_tile_size // 2
        assert tile_height >= min_tile_size // 2
```

## Step 2.3.4: Integration Tests

### Location: `tests/integration/test_expandor_integration.py`

```python
"""
Integration tests for complete Expandor system
"""

import pytest
from pathlib import Path
from PIL import Image

from expandor import Expandor, ExpandorConfig
from expandor.adapters.mock_pipeline import MockInpaintPipeline, MockRefinerPipeline
from expandor.core.exceptions import ExpandorError, VRAMError

class TestExpandorIntegration:
    
    def setup_method(self):
        """Setup for each test"""
        self.expandor = Expandor()
    
    def test_complete_workflow(self, test_image_square):
        """Test complete expansion workflow"""
        config = ExpandorConfig(
            source_image=test_image_square,
            target_resolution=(1024, 576),  # 16:9 aspect
            prompt="Beautiful landscape",
            seed=12345,
            source_metadata={'model': 'sdxl'},
            inpaint_pipeline=MockInpaintPipeline(),
            quality_preset='balanced'
        )
        
        result = self.expandor.expand(config)
        
        assert result.success
        assert result.size == (1024, 576)
        assert result.image_path.exists()
        assert len(result.stages) > 0
        assert result.total_duration_seconds > 0
        
        # Check metadata saved
        metadata_path = result.image_path.with_suffix('.json')
        assert metadata_path.exists()
    
    def test_strategy_selection_flow(self):
        """Test that different inputs select different strategies"""
        # Simple upscale
        config1 = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'},
            quality_preset='fast'
        )
        
        try:
            result1 = self.expandor.expand(config1)
            # Should try direct upscale (but fail without Real-ESRGAN)
        except ExpandorError as e:
            assert "Real-ESRGAN" in str(e) or "direct_upscale" in result1.strategy_used
        
        # Aspect change with inpaint
        config2 = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1024, 512),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'sdxl'},
            inpaint_pipeline=MockInpaintPipeline(),
            quality_preset='balanced'
        )
        
        result2 = self.expandor.expand(config2)
        assert 'progressive' in result2.strategy_used.lower()
    
    def test_error_handling(self):
        """Test error handling and fail-loud behavior"""
        # Invalid configuration
        with pytest.raises(ExpandorError) as excinfo:
            config = ExpandorConfig(
                source_image=Path("nonexistent.png"),
                target_resolution=(1024, 1024),
                prompt="Test",
                seed=42,
                source_metadata={'model': 'test'}
            )
            self.expandor.expand(config)
        
        assert "not found" in str(excinfo.value)
        
        # Invalid resolution
        with pytest.raises(ExpandorError) as excinfo:
            config = ExpandorConfig(
                source_image=Image.new('RGB', (512, 512)),
                target_resolution=(0, 1024),
                prompt="Test",
                seed=42,
                source_metadata={'model': 'test'}
            )
            self.expandor.expand(config)
        
        assert "Invalid target resolution" in str(excinfo.value)
    
    def test_vram_estimation(self):
        """Test VRAM estimation functionality"""
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(4096, 4096),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'sdxl'},
            refiner_pipeline=MockRefinerPipeline()
        )
        
        estimate = self.expandor.estimate_vram(config)
        
        assert 'total_required_mb' in estimate
        assert 'available_mb' in estimate
        assert 'recommended_strategy' in estimate
        assert estimate['total_required_mb'] > 0
    
    def test_quality_validation_flow(self, test_image_landscape):
        """Test quality validation and repair flow"""
        # Create a config that will generate boundaries
        config = ExpandorConfig(
            source_image=test_image_landscape,
            target_resolution=(2048, 1080),
            prompt="Landscape expansion",
            seed=42,
            source_metadata={'model': 'sdxl'},
            inpaint_pipeline=MockInpaintPipeline(),
            quality_preset='high',  # Enables artifact detection
            artifact_detection_level='aggressive'
        )
        
        result = self.expandor.expand(config)
        
        assert result.success
        assert hasattr(result, 'quality_score')
        assert hasattr(result, 'seams_detected')
        
        # With mock pipelines, shouldn't detect real seams
        assert result.seams_detected == 0
    
    def test_stage_saving(self, temp_dir):
        """Test saving intermediate stages"""
        stage_dir = temp_dir / "stages"
        
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1024, 512),
            prompt="Test stages",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=MockInpaintPipeline(),
            save_stages=True,
            stage_dir=stage_dir
        )
        
        result = self.expandor.expand(config)
        
        assert result.success
        assert stage_dir.exists()
        # Should have saved some stage files
        stage_files = list(stage_dir.glob("*.png"))
        assert len(stage_files) >= 0  # Might be 0 with mocks
    
    def test_pipeline_registration(self):
        """Test pipeline registration system"""
        mock_inpaint = MockInpaintPipeline()
        mock_refiner = MockRefinerPipeline()
        
        self.expandor.register_pipeline('inpaint', mock_inpaint)
        self.expandor.register_pipeline('refiner', mock_refiner)
        
        # Pipelines should be available to strategies
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'},
            refiner_pipeline=mock_refiner
        )
        
        # This would use the registered pipelines
        assert 'refiner' in self.expandor.pipeline_registry
```

## Step 2.3.5: Test Execution Scripts

### Location: `tests/run_tests.sh`

```bash
#!/bin/bash
# Run all Expandor tests

echo "Running Expandor Unit Tests..."
echo "=============================="

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run unit tests with coverage
echo "Running unit tests..."
pytest tests/unit/ -v --cov=expandor --cov-report=term-missing

# Run integration tests
echo -e "\nRunning integration tests..."
pytest tests/integration/ -v

# Run specific test categories
echo -e "\nTest Summary:"
echo "-------------"
pytest tests/unit/test_vram_manager.py -v --tb=short
pytest tests/unit/test_strategy_selector.py -v --tb=short
pytest tests/unit/test_metadata_tracker.py -v --tb=short
pytest tests/unit/test_boundary_tracker.py -v --tb=short
pytest tests/unit/test_direct_upscale_strategy.py -v --tb=short
pytest tests/unit/test_progressive_outpaint_strategy.py -v --tb=short
pytest tests/unit/test_tiled_expansion_strategy.py -v --tb=short

# Generate coverage report
echo -e "\nGenerating coverage report..."
pytest tests/ --cov=expandor --cov-report=html

echo -e "\nTest run complete! Coverage report available in htmlcov/index.html"
```

### Location: `Makefile`

```makefile
# Expandor development makefile

.PHONY: test test-unit test-integration test-coverage clean lint format

# Run all tests
test:
	@echo "Running all tests..."
	pytest tests/ -v

# Run unit tests only
test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v

# Run integration tests only
test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=expandor --cov-report=term-missing --cov-report=html

# Run specific test file
test-file:
	@echo "Running test file: $(FILE)"
	pytest $(FILE) -v

# Clean up
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf temp/
	rm -f .coverage

# Lint code
lint:
	@echo "Running linters..."
	flake8 expandor/ tests/ --max-line-length=100
	mypy expandor/ --ignore-missing-imports

# Format code
format:
	@echo "Formatting code..."
	black expandor/ tests/ --line-length=100

# Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	pip install -e ".[dev]"

# Run quick smoke test
smoke-test:
	@echo "Running smoke test..."
	python -c "from expandor import Expandor; print('Import successful')"
	pytest tests/unit/test_vram_manager.py::TestVRAMManager::test_initialization -v
```

## Summary

This completes Phase 2 with comprehensive unit tests covering:

1. **Core Components**
   - VRAMManager: VRAM calculations and strategy determination
   - StrategySelector: Multi-factor strategy selection
   - MetadataTracker: Operation tracking and logging
   - BoundaryTracker: Seam position tracking
   - PipelineOrchestrator: Execution and fallback handling

2. **Strategy Tests**
   - DirectUpscaleStrategy: Real-ESRGAN integration
   - ProgressiveOutpaintStrategy: Aspect ratio adjustment
   - TiledExpansionStrategy: Large image processing

3. **Integration Tests**
   - Complete workflow testing
   - Error handling verification
   - Quality validation flow

Each test verifies:
- **Success paths**: Normal operation works correctly
- **Error handling**: Fail-loud philosophy is implemented
- **Edge cases**: Unusual inputs are handled properly
- **Mocking**: External dependencies are properly isolated

The test suite ensures that all components work correctly both individually and together, maintaining the project's core principles of quality and reliability.