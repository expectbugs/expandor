"""
Test Progressive Outpaint Strategy
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from PIL import Image
import numpy as np

from expandor import ExpandorConfig
from expandor.strategies.progressive_outpaint import ProgressiveOutpaintStrategy
from expandor.core.exceptions import StrategyError

class TestProgressiveOutpaintStrategy:
    
    def setup_method(self):
        """Setup for each test"""
        self.strategy = ProgressiveOutpaintStrategy()
        # Mock inpaint pipeline
        self.mock_pipeline = Mock()
        self.mock_pipeline.return_value.images = [Image.new('RGB', (1024, 1024))]
        self.strategy.inpaint_pipeline = self.mock_pipeline
    
    def test_initialization(self):
        """Test strategy initialization"""
        assert hasattr(self.strategy, 'dimension_calc')
        assert hasattr(self.strategy, 'vram_manager')
        assert self.strategy.first_step_ratio == 1.4
        assert self.strategy.middle_step_ratio == 1.25
        assert self.strategy.final_step_ratio == 1.15
    
    def test_validate_requirements(self):
        """Test validation requires inpaint pipeline"""
        strategy = ProgressiveOutpaintStrategy()
        # Make sure there's no inpaint_pipeline attribute
        if hasattr(strategy, 'inpaint_pipeline'):
            delattr(strategy, 'inpaint_pipeline')
        
        with pytest.raises(RuntimeError) as exc_info:
            config = Mock()
            config.source_image = Image.new('RGB', (512, 512))
            config.target_resolution = (1024, 1024)
            config.inpaint_pipeline = None
            strategy.execute(config)
        
        assert "No inpainting pipeline" in str(exc_info.value)
    
    def test_analyze_edge_colors(self):
        """Test edge color analysis"""
        # Create test image with gradient
        img = Image.new('RGB', (100, 100))
        pixels = np.array(img)
        # Make top edge red gradient
        for x in range(100):
            pixels[0:10, x] = [x * 2, 0, 0]
        test_img = Image.fromarray(pixels.astype(np.uint8))
        
        result = self.strategy._analyze_edge_colors(test_img, 'top', sample_width=10)
        
        assert 'mean_rgb' in result
        assert 'median_rgb' in result
        assert 'is_uniform' in result
        assert 'color_variance' in result
    
    def test_execute_no_aspect_change(self):
        """Test execution with no aspect ratio change"""
        source_img = Image.new('RGB', (512, 512))
        config = ExpandorConfig(
            source_image=source_img,
            target_resolution=(1024, 1024),  # Same aspect ratio
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        # Mock dimension calculator to return no steps
        with patch.object(self.strategy.dimension_calc, 'calculate_progressive_strategy') as mock_calc:
            mock_calc.return_value = []  # No steps needed
            
            result = self.strategy.execute(config)
        
        assert result['size'] == (512, 512)  # No change
        assert len(result['stages']) == 0
        # When no steps, boundaries key may not be present
        assert result.get('boundaries', []) == []
    
    def test_execute_aspect_change(self):
        """Test execution with aspect ratio change"""
        source_img = Image.new('RGB', (512, 512))
        config = ExpandorConfig(
            source_image=source_img,
            target_resolution=(1920, 512),  # Wide aspect ratio
            prompt="Test landscape",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        # Mock dimension calculator to return expansion steps
        mock_steps = [
            {
                'current_size': (512, 512),
                'target_size': (716, 512),
                'expansion_ratio': 1.4,
                'direction': 'horizontal',
                'description': 'Step 1: Expand horizontally'
            },
            {
                'current_size': (716, 512),
                'target_size': (1920, 512),
                'expansion_ratio': 2.68,
                'direction': 'horizontal',
                'description': 'Step 2: Final horizontal expansion'
            }
        ]
        
        with patch.object(self.strategy.dimension_calc, 'calculate_progressive_strategy') as mock_calc:
            mock_calc.return_value = mock_steps
            
            # Mock execute_outpaint_step to return progressive sizes
            def mock_outpaint(image_path, prompt, step_info):
                target_w, target_h = step_info['target_size']
                # Create a temp file
                output_path = Path(f'/tmp/test_step_{target_w}x{target_h}.png')
                # Mock the output - just return the path
                return output_path
            
            with patch.object(self.strategy, '_execute_outpaint_step', side_effect=mock_outpaint):
                # Mock Image.open to return the expected sizes
                def mock_image_open(path):
                    # Return appropriate size based on the path
                    if '716x512' in str(path):
                        return Image.new('RGB', (716, 512))
                    elif '1920x512' in str(path):
                        return Image.new('RGB', (1920, 512))
                    else:
                        return Image.new('RGB', (512, 512))
                
                with patch('PIL.Image.open', side_effect=mock_image_open):
                    result = self.strategy.execute(config)
        
        assert result['size'] == (1920, 512)
        assert len(result['stages']) == 2
        assert len(result['boundaries']) == 2
    
    def test_execute_with_context(self):
        """Test execution with context parameter"""
        source_img = Image.new('RGB', (512, 512))
        config = ExpandorConfig(
            source_image=source_img,
            target_resolution=(768, 512),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        context = {
            'save_stages': True,
            'boundary_tracker': Mock()
        }
        
        # Mock dimension calculator
        with patch.object(self.strategy.dimension_calc, 'calculate_progressive_strategy') as mock_calc:
            mock_calc.return_value = [{
                'current_size': (512, 512),
                'target_size': (768, 512),
                'expansion_ratio': 1.5,
                'direction': 'horizontal',
                'description': 'Expand'
            }]
            
            with patch.object(self.strategy, '_execute_outpaint_step') as mock_outpaint:
                # Mock to return a path as the actual method does
                mock_outpaint.return_value = Path('/test/result.png')
                
                # Also mock Image.open to return the expected image
                with patch('PIL.Image.open') as mock_open:
                    mock_open.return_value = Image.new('RGB', (768, 512))
                    
                    result = self.strategy.execute(config, context)
        
        assert self.strategy._context == context
        assert result['size'] == (768, 512)
    
    def test_boundary_tracking(self):
        """Test boundary tracking during expansion"""
        self.strategy.boundary_tracker = Mock()
        
        source_img = Image.new('RGB', (512, 512))
        config = ExpandorConfig(
            source_image=source_img,
            target_resolution=(768, 512),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        # Mock expansion
        with patch.object(self.strategy.dimension_calc, 'calculate_progressive_strategy') as mock_calc:
            mock_calc.return_value = [{
                'current_size': (512, 512),
                'target_size': (768, 512),
                'expansion_ratio': 1.5,
                'direction': 'horizontal',
                'description': 'Expand'
            }]
            
            with patch.object(self.strategy, '_execute_outpaint_step') as mock_outpaint:
                # Mock to return a path as the actual method does
                mock_outpaint.return_value = Path('/test/result.png')
                
                # Also mock Image.open
                with patch('PIL.Image.open') as mock_open:
                    mock_open.return_value = Image.new('RGB', (768, 512))
                    
                    self.strategy.execute(config)
        
        # Should have tracked boundaries
        assert self.strategy.boundary_tracker.add_boundary.called
    
    def test_adaptive_parameters(self):
        """Test adaptive parameter calculation"""
        # Test first step
        step_info = {'expansion_ratio': 1.4, 'current_size': (512, 512)}
        blur = self.strategy._get_adaptive_blur(step_info)
        steps = self.strategy._get_adaptive_steps(step_info)
        guidance = self.strategy._get_adaptive_guidance(step_info)
        
        assert blur >= 32  # Base blur value
        assert steps >= 60  # More steps for first expansion
        assert guidance >= 7.0
        
        # Test large expansion
        step_info = {'expansion_ratio': 1.9, 'current_size': (1024, 1024)}
        blur_large = self.strategy._get_adaptive_blur(step_info)
        steps_large = self.strategy._get_adaptive_steps(step_info)
        
        assert blur_large > blur  # More blur for large expansion
        assert steps_large == steps  # Same base steps