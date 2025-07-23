"""
Test Strategy Selector functionality
"""

import pytest
from unittest.mock import Mock
from PIL import Image

from expandor import ExpandorConfig
from expandor.core.strategy_selector import StrategySelector, SelectionMetrics
from expandor.core.vram_manager import VRAMManager

class TestStrategySelector:
    
    def setup_method(self):
        """Setup for each test"""
        # Create mock VRAM manager
        self.vram_manager = Mock(spec=VRAMManager)
        self.vram_manager.get_available_vram.return_value = 8000.0  # 8GB
        self.vram_manager.calculate_generation_vram.return_value = 2000.0  # 2GB
        
        # Create test config
        self.config = {
            'strategies': {
                'progressive_outpainting': {'enabled': True},
                'swpo': {'enabled': True},
                'direct_upscale': {'enabled': True}
            },
            'quality_presets': {
                'ultra': {},
                'balanced': {}
            },
            'vram_strategies': {
                'thresholds': {'tiled_processing': 4000},
                'safety_factor': 0.8
            }
        }
        
        self.selector = StrategySelector(
            config=self.config,
            vram_manager=self.vram_manager
        )
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=Mock(),
            refiner_pipeline=Mock()
        )
        
        metrics = self.selector._calculate_metrics(config)
        
        assert isinstance(metrics, SelectionMetrics)
        assert metrics.source_size == (512, 512)
        assert metrics.target_size == (1024, 1024)
        assert metrics.area_ratio == 4.0
        assert metrics.aspect_change == 1.0  # No aspect change
        assert metrics.has_inpaint == True
        assert metrics.has_refiner == True
        assert metrics.vram_available_mb == 8000.0
    
    def test_select_direct_upscale(self):
        """Test selection of direct upscale strategy"""
        # Simple upscale, no aspect change
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        strategy_name, reason, metrics = self.selector.select_strategy(config)
        
        # Accept either direct_upscale or hybrid_adaptive for simple upscale
        assert strategy_name in ['direct_upscale', 'hybrid_adaptive']
        # Reason varies by strategy
    
    def test_select_progressive_outpaint(self):
        """Test selection of progressive outpaint strategy"""
        # Significant aspect change
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1920, 512),  # 16:9 from 1:1
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=Mock()
        )
        
        strategy_name, reason, metrics = self.selector.select_strategy(config)
        
        assert strategy_name == 'progressive_outpaint'
        assert 'aspect ratio change' in reason
    
    def test_select_by_vram_constraint(self):
        """Test VRAM-based strategy selection"""
        # Simulate low VRAM
        self.vram_manager.get_available_vram.return_value = 2000.0  # 2GB
        self.vram_manager.calculate_generation_vram.return_value = 5000.0  # 5GB needed
        
        config = ExpandorConfig(
            source_image=Image.new('RGB', (4096, 4096)),
            target_resolution=(8192, 8192),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        strategy_name, reason, metrics = self.selector.select_strategy(config)
        
        # Accept either tiled or cpu_offload for low VRAM
        assert strategy_name in ['tiled_expansion', 'cpu_offload']
        assert 'VRAM' in reason  # Should mention VRAM in reason
    
    def test_select_cpu_offload(self):
        """Test CPU offload selection when no GPU"""
        # Simulate no GPU
        self.vram_manager.get_available_vram.return_value = 0
        
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        strategy_name, reason, metrics = self.selector.select_strategy(config)
        
        assert strategy_name == 'cpu_offload'
        assert 'No GPU available' in reason
    
    def test_strategy_override(self):
        """Test user strategy override"""
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'},
            strategy_override='tiled_expansion'
        )
        
        strategy_name, reason, metrics = self.selector.select_strategy(config)
        
        assert strategy_name == 'tiled_expansion'
        assert reason == 'User override'
    
    def test_extreme_ratio_swpo(self):
        """Test SWPO selection for extreme aspect changes"""
        # Extreme aspect change
        config = ExpandorConfig(
            source_image=Image.new('RGB', (512, 512)),
            target_resolution=(4096, 512),  # 8:1 aspect ratio
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'},
            inpaint_pipeline=Mock()
        )
        
        # Set extreme threshold
        self.config['strategies']['progressive_outpainting']['aspect_ratio_thresholds'] = {
            'extreme': 4.0
        }
        
        strategy_name, reason, metrics = self.selector.select_strategy(config)
        
        assert strategy_name == 'swpo'
        assert 'Extreme aspect ratio' in reason
    
    def test_get_available_strategies(self):
        """Test getting list of available strategies"""
        strategies = self.selector.get_available_strategies()
        
        assert isinstance(strategies, list)
        assert 'direct_upscale' in strategies
        assert 'progressive_outpaint' in strategies
        assert 'tiled_expansion' in strategies
        assert len(strategies) >= 6  # All 6 strategies