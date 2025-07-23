"""
Test Direct Upscale Strategy
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from PIL import Image

from expandor import ExpandorConfig
from expandor.strategies.direct_upscale import DirectUpscaleStrategy
from expandor.core.exceptions import StrategyError, UpscalerError

class TestDirectUpscaleStrategy:
    
    def setup_method(self):
        """Setup for each test"""
        self.strategy = DirectUpscaleStrategy()
        # Mock Real-ESRGAN path
        self.strategy.realesrgan_path = Path('/mock/realesrgan')
    
    def test_initialization(self):
        """Test strategy initialization"""
        assert self.strategy.upscale_config['model'] == 'RealESRGAN_x4plus'
        assert 'RealESRGAN_x4plus' in self.strategy.model_config
        assert 'high_vram' in self.strategy.tile_config
    
    def test_validate_requirements_no_upscaler(self):
        """Test validation fails loud when no upscaler"""
        strategy = DirectUpscaleStrategy()
        strategy.realesrgan_path = None
        
        # Should FAIL LOUD - no silent fallbacks
        with pytest.raises(StrategyError) as exc_info:
            strategy.validate_requirements()
        
        assert "Real-ESRGAN not found" in str(exc_info.value)
    
    def test_calculate_upscale_passes(self):
        """Test upscale pass calculation"""
        # Small upscale - single pass
        passes = self.strategy._calculate_upscale_passes(
            source_size=(512, 512),
            target_size=(1024, 1024)
        )
        assert len(passes) == 1
        assert passes[0]['scale'] == 2
        
        # Large upscale - multiple passes
        passes = self.strategy._calculate_upscale_passes(
            source_size=(512, 512),
            target_size=(4096, 4096)
        )
        assert len(passes) > 1
        assert all(p['scale'] in [2, 3, 4] for p in passes)
    
    def test_determine_tile_size(self):
        """Test tile size determination"""
        # Mock VRAM availability
        self.strategy.vram_manager.get_available_vram = Mock(return_value=8000)
        
        tile_size = self.strategy._determine_tile_size((1024, 1024))
        assert tile_size in [512, 768, 1024, 2048]
        
        # Test CPU mode
        self.strategy.vram_manager.get_available_vram = Mock(return_value=None)
        tile_size = self.strategy._determine_tile_size((1024, 1024))
        assert tile_size == 512  # Should use low_vram setting
    
    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    def test_run_realesrgan_success(self, mock_exists, mock_run):
        """Test successful Real-ESRGAN execution"""
        # Setup mocks
        mock_exists.return_value = True
        mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
        
        input_path = Path('/test/input.png')
        result_path = self.strategy._run_realesrgan(
            input_path=input_path,
            scale=4,
            model_name='RealESRGAN_x4plus',
            tile_size=1024,
            fp32=True
        )
        
        # Check subprocess was called
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert str(input_path) in call_args
        assert '-s' in call_args
        assert '4' in call_args
    
    @patch('subprocess.run')
    def test_run_realesrgan_failure(self, mock_run):
        """Test Real-ESRGAN execution failure"""
        # Setup mock to fail with CalledProcessError
        from subprocess import CalledProcessError
        mock_run.side_effect = CalledProcessError(
            returncode=1,
            cmd=['realesrgan', '-i', 'input.png'],
            stderr='Error: Out of memory'
        )
        
        with pytest.raises(UpscalerError) as exc_info:
            self.strategy._run_realesrgan(
                input_path=Path('/test/input.png'),
                scale=4,
                model_name='RealESRGAN_x4plus',
                tile_size=1024,
                fp32=True
            )
        
        assert exc_info.value.tool_name == 'realesrgan'
        assert exc_info.value.exit_code == 1
        assert 'Out of memory' in str(exc_info.value)
    
    @patch.object(DirectUpscaleStrategy, '_run_realesrgan')
    def test_execute_simple_upscale(self, mock_run):
        """Test simple upscale execution"""
        # Mock Real-ESRGAN execution
        mock_run.return_value = Path('/test/upscaled.png')
        
        # Create test config
        source_img = Image.new('RGB', (512, 512))
        config = ExpandorConfig(
            source_image=source_img,
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        # Mock the upscaled image
        with patch('PIL.Image.open') as mock_open:
            mock_open.return_value = Image.new('RGB', (1024, 1024))
            
            result = self.strategy.execute(config)
        
        assert result['size'] == (1024, 1024)
        assert result['metadata']['strategy'] == 'direct_upscale'
        assert result['metadata']['passes'] == 1
        assert len(result['stages']) > 0
    
    @patch.object(DirectUpscaleStrategy, '_run_realesrgan')
    def test_execute_with_final_resize(self, mock_run):
        """Test upscale with final resize step"""
        # Mock Real-ESRGAN to return larger than target
        mock_run.return_value = Path('/test/upscaled.png')
        
        source_img = Image.new('RGB', (512, 512))
        config = ExpandorConfig(
            source_image=source_img,
            target_resolution=(1000, 1000),  # Not a power of 2
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        # Mock the upscaled image to be larger
        with patch('PIL.Image.open') as mock_open:
            mock_open.return_value = Image.new('RGB', (1024, 1024))
            
            result = self.strategy.execute(config)
        
        assert result['size'] == (1000, 1000)
        assert result['metadata']['final_resize'] == True
        # Should have upscale + resize stages
        assert len(result['stages']) >= 2
    
    @patch.object(DirectUpscaleStrategy, '_run_realesrgan')
    def test_execute_with_context(self, mock_run):
        """Test execution with context parameter"""
        # Setup mock to return a valid path
        output_path = Path('/tmp/upscaled.png')
        mock_run.return_value = output_path
        
        source_img = Image.new('RGB', (512, 512))
        config = ExpandorConfig(
            source_image=source_img,
            target_resolution=(1024, 1024),
            prompt="Test",
            seed=42,
            source_metadata={'model': 'test'}
        )
        
        context = {
            'save_stages': True,
            'metadata_tracker': Mock(),
            'temp_dir': Path('/tmp')
        }
        
        # Mock image loading to return expected size
        with patch('PIL.Image.open') as mock_open:
            mock_open.return_value = Image.new('RGB', (1024, 1024))
            
            result = self.strategy.execute(config, context)
        
        assert result['size'] == (1024, 1024)
        # strategy_used is not in direct result, it's added by orchestrator
        assert hasattr(self.strategy, '_context')
        assert self.strategy._context == context