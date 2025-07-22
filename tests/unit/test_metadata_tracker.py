"""
Test Metadata Tracker functionality
"""

import pytest
import time
import json
from pathlib import Path
from unittest.mock import Mock

from expandor import ExpandorConfig
from expandor.core.metadata_tracker import MetadataTracker

class TestMetadataTracker:
    
    def setup_method(self):
        """Setup for each test"""
        self.tracker = MetadataTracker()
    
    def test_initialization(self):
        """Test tracker initialization"""
        assert self.tracker.operation_id.startswith('op_')
        assert self.tracker.current_stage == 'initialization'
        assert len(self.tracker.events) == 0
        assert len(self.tracker.metrics) == 0
    
    def test_start_operation(self):
        """Test operation start tracking"""
        config = Mock(spec=ExpandorConfig)
        config.source_image = Path("/test/image.png")
        config.target_resolution = (1920, 1080)
        config.quality_preset = "balanced"
        config.prompt = "Test prompt"
        config.seed = 42
        config.source_metadata = {'model': 'test'}
        config.inpaint_pipeline = Mock()
        config.refiner_pipeline = None
        config.img2img_pipeline = None
        
        operation_id = self.tracker.start_operation(config)
        
        assert operation_id == self.tracker.operation_id
        assert self.tracker.config_snapshot is not None
        assert self.tracker.config_snapshot['target_resolution'] == (1920, 1080)
        assert len(self.tracker.events) > 0
        assert self.tracker.events[0]['type'] == 'operation_start'
    
    def test_track_operation(self):
        """Test operation tracking"""
        self.tracker.track_operation('test_op', {'data': 'value'})
        
        assert len(self.tracker.events) == 1
        assert self.tracker.events[0]['type'] == 'operation_test_op'
        assert self.tracker.events[0]['data']['data'] == 'value'
    
    def test_record_event(self):
        """Test event recording"""
        self.tracker.record_event('test_event', {'key': 'value'})
        
        assert len(self.tracker.events) == 1
        event = self.tracker.events[0]
        assert event['type'] == 'test_event'
        assert event['data']['key'] == 'value'
        assert 'timestamp' in event
        assert 'relative_time' in event
        assert event['stage'] == 'initialization'
    
    def test_stage_tracking(self):
        """Test stage enter/exit"""
        # Enter stage
        self.tracker.enter_stage('processing')
        
        assert self.tracker.current_stage == 'processing'
        assert any(e['type'] == 'stage_enter' for e in self.tracker.events)
        
        # Simulate some work
        time.sleep(0.1)
        
        # Exit stage
        self.tracker.exit_stage(success=True)
        
        assert 'processing' in self.tracker.stage_timings
        assert self.tracker.stage_timings['processing'] >= 0.1
        assert any(e['type'] == 'stage_exit' for e in self.tracker.events)
    
    def test_stage_failure(self):
        """Test stage exit with failure"""
        self.tracker.enter_stage('failing_stage')
        self.tracker.exit_stage(success=False, error="Test error")
        
        exit_event = next(e for e in self.tracker.events if e['type'] == 'stage_exit')
        assert exit_event['data']['success'] == False
        assert exit_event['data']['error'] == "Test error"
    
    def test_record_metric(self):
        """Test metric recording"""
        self.tracker.record_metric('vram_used', 1024.5)
        self.tracker.record_metric('quality_score', 0.95)
        
        assert self.tracker.metrics['vram_used'] == 1024.5
        assert self.tracker.metrics['quality_score'] == 0.95
    
    def test_add_stage(self):
        """Test add_stage method"""
        stage_data = {'image_path': '/test/path.png', 'metadata': {'size': (1024, 1024)}}
        self.tracker.add_stage('upscale', stage_data)
        
        stage_event = next(e for e in self.tracker.events if e['type'] == 'stage_metadata')
        assert stage_event['data']['stage'] == 'upscale'
        assert stage_event['data']['metadata'] == stage_data
    
    def test_get_summary(self):
        """Test get_summary method"""
        # Add some data
        self.tracker.record_event('test1', {})
        self.tracker.record_event('test2', {})
        self.tracker.record_metric('metric1', 100)
        
        summary = self.tracker.get_summary()
        
        assert summary['operation_id'] == self.tracker.operation_id
        assert summary['duration'] > 0
        assert len(summary['events']) == 2
        assert summary['metrics']['metric1'] == 100
        assert 'event_summary' in summary
    
    def test_get_partial_result(self):
        """Test partial result generation"""
        self.tracker.enter_stage('stage1')
        time.sleep(0.05)
        self.tracker.exit_stage()
        
        self.tracker.enter_stage('stage2')
        
        partial = self.tracker.get_partial_result()
        
        assert partial['operation_id'] == self.tracker.operation_id
        assert partial['current_stage'] == 'stage2'
        assert 'stage1' in partial['completed_stages']
        assert partial['event_count'] > 0
        assert partial['elapsed_time'] > 0
    
    def test_save_operation_log(self, temp_dir):
        """Test saving log to file"""
        self.tracker.record_event("test", {"data": "value"})
        self.tracker.record_metric("metric", 42)
        
        log_path = temp_dir / "test_log.json"
        self.tracker.save_operation_log(log_path)
        
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
        assert 'stage1' in summary['stage_timings']
        assert 'stage2' in summary['stage_timings']
        assert 'stage1' in summary['stage_percentages']
        assert summary['stage_percentages']['stage1'] < 50  # stage1 was shorter
        assert summary['metrics']['test'] == 123
        assert summary['events_per_second'] > 0