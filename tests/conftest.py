"""
Test configuration and fixtures for Expandor
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from PIL import Image
import numpy as np

from expandor.adapters.mock_pipeline import MockInpaintPipeline, MockRefinerPipeline, MockImg2ImgPipeline

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
def test_image_paths():
    """Return paths to test images"""
    test_dir = Path(__file__).parent / "fixtures"
    return {
        'checkerboard': test_dir / "checkerboard_512x512.png",
        'gradient': test_dir / "gradient_1024x1024.png",
        'landscape': test_dir / "landscape_1344x768.png",
        'portrait': test_dir / "portrait_768x1344.png"
    }

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
def mock_pipelines(mock_inpaint_pipeline, mock_refiner_pipeline, mock_img2img_pipeline):
    """Get all mock pipelines"""
    return {
        'inpaint': mock_inpaint_pipeline,
        'refiner': mock_refiner_pipeline,
        'img2img': mock_img2img_pipeline
    }

@pytest.fixture
def expandor_config(test_image_square):
    """Create basic ExpandorConfig for testing"""
    from expandor import ExpandorConfig
    
    return ExpandorConfig(
        source_image=test_image_square,
        target_resolution=(1024, 1024),
        prompt="A beautiful test image",
        seed=42,
        source_metadata={'model': 'test'},
        quality_preset='balanced'
    )

@pytest.fixture
def expandor_with_mocks(mock_pipelines):
    """Create Expandor instance with mock pipelines"""
    from expandor import Expandor
    
    expandor = Expandor()
    
    # Register mock pipelines
    for name, pipeline in mock_pipelines.items():
        expandor.register_pipeline(name, pipeline)
    
    return expandor

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