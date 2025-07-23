"""
Test Pipeline Orchestrator functionality
"""

import pytest
from unittest.mock import Mock, patch
from PIL import Image

from expandor import ExpandorConfig, ExpandorResult, VRAMError, StrategyError, ExpandorError
from expandor.core.pipeline_orchestrator import PipelineOrchestrator
from expandor.core.metadata_tracker import MetadataTracker
from expandor.core.boundary_tracker import BoundaryTracker
from expandor.strategies.base_strategy import BaseExpansionStrategy

class MockStrategy(BaseExpansionStrategy):
    """Mock strategy for testing"""
    
    def execute(self, config, context=None):
        if hasattr(self, 'should_fail') and self.should_fail:
            raise StrategyError("Mock strategy failed")
        
        return {
            'image_path': '/test/result.png',
            'size': (1024, 1024),
            'stages': [],
            'boundaries': [],
            'metadata': {'test': True}
        }
    
    def validate_requirements(self):
        if hasattr(self, 'invalid_requirements') and self.invalid_requirements:
            raise StrategyError("Requirements not met")

class TestPipelineOrchestrator:
    
    def setup_method(self):
        """Setup for each test"""
        self.config = {
            'vram_strategies': {
                'fallback_chain': {
                    1: 'tiled_large',
                    2: 'cpu_offload'
                }
            }
        }
        self.orchestrator = PipelineOrchestrator(self.config)
        self.metadata_tracker = MetadataTracker()
        self.boundary_tracker = BoundaryTracker()
    
    def test_register_pipeline(self):
        """Test pipeline registration"""
        mock_pipeline = Mock()
        self.orchestrator.register_pipeline('test', mock_pipeline)
        
        assert 'test' in self.orchestrator.pipeline_registry
        assert self.orchestrator.pipeline_registry['test'] == mock_pipeline
    
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
        
        assert isinstance(result, ExpandorResult)
        assert result.success == True
        assert result.strategy_used == 'MockStrategy'
        assert result.fallback_count == 0
    
    def test_strategy_preparation(self):
        """Test strategy preparation with pipelines"""
        strategy = MockStrategy()
        config = Mock(spec=ExpandorConfig)
        config.inpaint_pipeline = Mock()
        config.refiner_pipeline = Mock()
        config.img2img_pipeline = None
        
        self.orchestrator._prepare_strategy(
            strategy, config, self.boundary_tracker, self.metadata_tracker
        )
        
        assert strategy.inpaint_pipeline == config.inpaint_pipeline
        assert strategy.refiner_pipeline == config.refiner_pipeline
        assert strategy.boundary_tracker == self.boundary_tracker
        assert strategy.metadata_tracker == self.metadata_tracker
    
    def test_fallback_execution(self):
        """Test fallback chain execution"""
        # Primary strategy that will fail
        primary_strategy = MockStrategy()
        primary_strategy.should_fail = True
        
        # Mock fallback strategy
        fallback_strategy = MockStrategy()
        
        # Mock the fallback chain building
        with patch.object(self.orchestrator, '_build_fallback_chain') as mock_build:
            mock_build.return_value = [primary_strategy, fallback_strategy]
            
            config = Mock(spec=ExpandorConfig)
            config.quality_preset = 'ultra'
            
            result = self.orchestrator.execute(
                strategy=primary_strategy,
                config=config,
                metadata_tracker=self.metadata_tracker,
                boundary_tracker=self.boundary_tracker
            )
            
            assert result.strategy_used == 'MockStrategy'
            assert result.fallback_count == 1
            # Only failures are tracked in execution_history
            assert len(self.orchestrator.execution_history) == 1
            assert self.orchestrator.execution_history[0]['status'] == 'failed'
    
    def test_all_strategies_fail(self):
        """Test when all strategies fail"""
        strategy1 = MockStrategy()
        strategy1.should_fail = True
        
        strategy2 = MockStrategy()
        strategy2.should_fail = True
        
        with patch.object(self.orchestrator, '_build_fallback_chain') as mock_build:
            mock_build.return_value = [strategy1, strategy2]
            
            config = Mock(spec=ExpandorConfig)
            config.quality_preset = 'ultra'
            
            with pytest.raises(ExpandorError) as exc_info:
                self.orchestrator.execute(
                    strategy=strategy1,
                    config=config,
                    metadata_tracker=self.metadata_tracker,
                    boundary_tracker=self.boundary_tracker
                )
            
            assert "All expansion strategies failed" in str(exc_info.value)
    
    def test_vram_error_handling(self):
        """Test VRAM error triggers fallback"""
        primary_strategy = MockStrategy()
        primary_strategy.should_fail = True
        primary_strategy.execute = Mock(side_effect=VRAMError(
            operation="test",
            required_mb=8000,
            available_mb=4000
        ))
        
        fallback_strategy = MockStrategy()
        
        with patch.object(self.orchestrator, '_build_fallback_chain') as mock_build:
            mock_build.return_value = [primary_strategy, fallback_strategy]
            
            config = Mock(spec=ExpandorConfig)
            config.quality_preset = 'ultra'
            
            result = self.orchestrator.execute(
                strategy=primary_strategy,
                config=config,
                metadata_tracker=self.metadata_tracker,
                boundary_tracker=self.boundary_tracker
            )
            
            # Should have used fallback
            assert result.fallback_count == 1
    
    def test_no_fallback_in_fast_mode(self):
        """Test no fallbacks in fast quality mode"""
        primary_strategy = MockStrategy()
        primary_strategy.should_fail = True
        
        config = Mock(spec=ExpandorConfig)
        config.quality_preset = 'fast'
        
        with pytest.raises(ExpandorError):
            self.orchestrator.execute(
                strategy=primary_strategy,
                config=config,
                metadata_tracker=self.metadata_tracker,
                boundary_tracker=self.boundary_tracker
            )
        
        # Should not have tried fallbacks
        assert len(self.orchestrator.execution_history) == 1
    
    def test_context_creation(self):
        """Test execution context creation"""
        strategy = MockStrategy()
        
        # Override execute to capture context
        captured_context = None
        def capture_execute(config, context):
            nonlocal captured_context
            captured_context = context
            return {
                'image_path': '/test/result.png',
                'size': (1024, 1024),
                'stages': [],
                'boundaries': [],
                'metadata': {}
            }
        
        strategy.execute = capture_execute
        
        config = Mock(spec=ExpandorConfig)
        config.quality_preset = 'balanced'
        config.save_stages = True
        config.stage_dir = '/test/stages'
        
        self.orchestrator.execute(
            strategy=strategy,
            config=config,
            metadata_tracker=self.metadata_tracker,
            boundary_tracker=self.boundary_tracker
        )
        
        assert captured_context is not None
        assert captured_context['config'] == config
        assert captured_context['metadata_tracker'] == self.metadata_tracker
        assert captured_context['boundary_tracker'] == self.boundary_tracker
        assert 'stage_callback' in captured_context
        assert captured_context['save_stages'] == True