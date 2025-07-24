"""
Integration tests for pipeline adapters
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from expandor.adapters import (
    A1111PipelineAdapter,
    BasePipelineAdapter,
    ComfyUIPipelineAdapter,
    DiffusersPipelineAdapter,
    MockPipelineAdapter,
)
from expandor.core import Expandor
from expandor.core.config import ExpandorConfig


class TestMockAdapter:
    """Test mock adapter functionality"""

    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter instance"""
        return MockPipelineAdapter()

    @pytest.fixture
    def test_image(self):
        """Create test image"""
        return Image.new("RGB", (512, 512), color=(100, 150, 200))

    def test_mock_adapter_interface(self, mock_adapter):
        """Test that mock adapter implements all required methods"""
        # Check all abstract methods are implemented
        assert hasattr(mock_adapter, "load_pipeline")
        assert hasattr(mock_adapter, "get_pipeline")
        assert hasattr(mock_adapter, "generate")
        assert hasattr(mock_adapter, "inpaint")
        assert hasattr(mock_adapter, "img2img")
        assert hasattr(mock_adapter, "refine")
        assert hasattr(mock_adapter, "enhance")
        assert hasattr(mock_adapter, "supports_operation")
        assert hasattr(mock_adapter, "cleanup")

    def test_mock_generation(self, mock_adapter):
        """Test mock image generation"""
        # Load pipeline
        mock_adapter.load_pipeline("sdxl")

        # Generate image
        result = mock_adapter.generate(
            prompt="test prompt", width=1024, height=1024, num_inference_steps=20
        )

        assert isinstance(result, Image.Image)
        assert result.size == (1024, 1024)

    def test_mock_inpainting(self, mock_adapter, test_image):
        """Test mock inpainting"""
        mock_adapter.load_pipeline("sdxl-inpaint")

        # Create mask
        mask = Image.new("L", test_image.size, 255)

        result = mock_adapter.inpaint(
            image=test_image, mask=mask, prompt="test inpaint", strength=0.8
        )

        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_mock_img2img(self, mock_adapter, test_image):
        """Test mock img2img"""
        mock_adapter.load_pipeline("sdxl")

        result = mock_adapter.img2img(
            image=test_image, prompt="transformed image", strength=0.7
        )

        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_mock_adapter_with_expandor(self, mock_adapter):
        """Test mock adapter integration with Expandor"""
        config = ExpandorConfig(target_width=1024, target_height=768, strategy="direct")

        expandor = Expandor(pipeline_adapter=mock_adapter, config=config)

        # Create test image
        test_image = Image.new("RGB", (512, 384), color=(100, 150, 200))

        # Expand image
        result = expandor.expand(test_image)

        assert result.success
        assert result.final_image.size == (1024, 768)

    def test_mock_operations_support(self, mock_adapter):
        """Test operation support checking"""
        assert mock_adapter.supports_operation("generate")
        assert mock_adapter.supports_operation("inpaint")
        assert mock_adapter.supports_operation("img2img")
        assert mock_adapter.supports_operation("enhance")

        # ControlNet not supported in mock
        assert not mock_adapter.supports_controlnet()
        assert not mock_adapter.supports_operation("controlnet_inpaint")


class TestDiffusersAdapter:
    """Test Diffusers adapter with mocked pipelines"""

    @pytest.fixture
    def mock_diffusers(self):
        """Mock diffusers module"""
        with patch("expandor.adapters.diffusers_adapter.diffusers") as mock:
            # Mock pipeline classes
            mock.StableDiffusionXLPipeline = MagicMock()
            mock.StableDiffusionXLInpaintPipeline = MagicMock()
            mock.StableDiffusionXLImg2ImgPipeline = MagicMock()
            mock.DiffusionPipeline = MagicMock()

            # Mock AutoPipeline
            mock.AutoPipelineForText2Image = MagicMock()
            mock.AutoPipelineForInpainting = MagicMock()
            mock.AutoPipelineForImage2Image = MagicMock()

            yield mock

    @pytest.fixture
    def diffusers_adapter(self, mock_diffusers):
        """Create Diffusers adapter with mocked imports"""
        return DiffusersPipelineAdapter(device="cuda")

    def test_pipeline_loading(self, diffusers_adapter, mock_diffusers):
        """Test pipeline loading with different model types"""
        # Mock pipeline instance
        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_diffusers.DiffusionPipeline.from_pretrained.return_value = mock_pipeline

        # Load SDXL pipeline
        diffusers_adapter.load_pipeline(
            "sdxl", model_id="stabilityai/stable-diffusion-xl-base-1.0"
        )

        assert "sdxl" in diffusers_adapter._pipelines
        mock_diffusers.DiffusionPipeline.from_pretrained.assert_called_once()

    def test_model_type_detection(self, diffusers_adapter):
        """Test automatic model type detection"""
        # Test various model ID patterns
        assert (
            diffusers_adapter._detect_model_type(
                "stabilityai/stable-diffusion-xl-base-1.0"
            )
            == "sdxl"
        )

        assert (
            diffusers_adapter._detect_model_type("runwayml/stable-diffusion-v1-5")
            == "sd15"
        )

        assert (
            diffusers_adapter._detect_model_type("stabilityai/stable-diffusion-2-1")
            == "sd2"
        )

        assert (
            diffusers_adapter._detect_model_type(
                "stabilityai/stable-diffusion-3-medium"
            )
            == "sd3"
        )

        assert (
            diffusers_adapter._detect_model_type("black-forest-labs/FLUX.1-dev")
            == "flux"
        )

    def test_lora_loading(self, diffusers_adapter, mock_diffusers):
        """Test LoRA loading and stacking"""
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline
        diffusers_adapter._pipelines["sdxl"] = mock_pipeline

        # Load LoRAs
        lora_configs = [
            {"path": "/path/to/lora1.safetensors", "weight": 0.7},
            {"path": "/path/to/lora2.safetensors", "weight": 0.5},
        ]

        with patch("os.path.exists", return_value=True):
            diffusers_adapter.load_loras("sdxl", lora_configs)

        # Should have called load_lora_weights
        assert mock_pipeline.load_lora_weights.call_count == 2

    def test_memory_management(self, diffusers_adapter, mock_diffusers):
        """Test memory efficient mode and cleanup"""
        # Mock pipeline with memory methods
        mock_pipeline = MagicMock()
        mock_pipeline.enable_model_cpu_offload = MagicMock()
        mock_pipeline.enable_sequential_cpu_offload = MagicMock()

        diffusers_adapter._pipelines["test"] = mock_pipeline

        # Enable memory efficient mode
        diffusers_adapter.enable_memory_efficient_mode()

        # Should enable CPU offload on all pipelines
        mock_pipeline.enable_model_cpu_offload.assert_called_once()

    def test_vram_estimation(self, diffusers_adapter):
        """Test VRAM estimation for different operations"""
        # Test base generation
        vram = diffusers_adapter.estimate_vram(
            "generate", width=1024, height=1024, model_type="sdxl"
        )
        assert vram > 0
        assert vram < 16000  # Should be reasonable

        # Larger resolution should need more VRAM
        vram_large = diffusers_adapter.estimate_vram(
            "generate", width=2048, height=2048, model_type="sdxl"
        )
        assert vram_large > vram

    def test_generation_with_mock(self, diffusers_adapter, mock_diffusers):
        """Test image generation with mocked pipeline"""
        # Create mock pipeline that returns image
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (1024, 1024))]
        mock_pipeline.return_value = mock_result

        diffusers_adapter._pipelines["sdxl"] = mock_pipeline

        # Generate image
        result = diffusers_adapter.generate(
            prompt="test prompt", width=1024, height=1024, num_inference_steps=20
        )

        assert isinstance(result, Image.Image)
        assert result.size == (1024, 1024)
        mock_pipeline.assert_called_once()

    def test_error_handling(self, diffusers_adapter, mock_diffusers):
        """Test error handling in adapter"""
        # Test with no pipeline loaded
        with pytest.raises(ValueError) as exc_info:
            diffusers_adapter.generate("test", width=512, height=512)
        assert "No pipeline loaded" in str(exc_info.value)

        # Test with missing model
        mock_diffusers.DiffusionPipeline.from_pretrained.side_effect = Exception(
            "Model not found"
        )

        with pytest.raises(Exception) as exc_info:
            diffusers_adapter.load_pipeline("sdxl", model_id="invalid/model")
        assert "Model not found" in str(exc_info.value)


class TestPlaceholderAdapters:
    """Test placeholder adapters (ComfyUI and A1111)"""

    def test_comfyui_placeholder(self):
        """Test ComfyUI adapter placeholder behavior"""
        adapter = ComfyUIPipelineAdapter()

        # All operations should raise NotImplementedError
        with pytest.raises(NotImplementedError) as exc_info:
            adapter.load_pipeline("sdxl")
        assert "not yet implemented" in str(exc_info.value)

        with pytest.raises(NotImplementedError) as exc_info:
            adapter.generate("test prompt")
        assert "not yet implemented" in str(exc_info.value)

        # supports_operation should return False
        assert not adapter.supports_operation("generate")
        assert not adapter.supports_operation("inpaint")

        # Server status should indicate not implemented
        status = adapter.get_server_status()
        assert status["status"] == "not_implemented"

    def test_a1111_placeholder(self):
        """Test A1111 adapter placeholder behavior"""
        adapter = A1111PipelineAdapter()

        # All operations should raise NotImplementedError
        with pytest.raises(NotImplementedError) as exc_info:
            adapter.load_pipeline("sdxl")
        assert "not yet implemented" in str(exc_info.value)

        with pytest.raises(NotImplementedError) as exc_info:
            adapter.generate("test prompt")
        assert "not yet implemented" in str(exc_info.value)

        # supports_operation should return False
        assert not adapter.supports_operation("generate")
        assert not adapter.supports_operation("img2img")

        # API status should indicate not implemented
        status = adapter.get_api_status()
        assert status["status"] == "not_implemented"


class TestAdapterIntegration:
    """Test adapter integration with Expandor core"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_adapter_selection(self):
        """Test adapter selection based on availability"""
        from expandor.config.pipeline_config import PipelineConfigurator

        configurator = PipelineConfigurator()

        # Mock adapter should always work
        adapter = configurator.create_adapter("mock")
        assert isinstance(adapter, MockPipelineAdapter)

        # Auto selection should fall back to mock
        with patch(
            "expandor.config.pipeline_config.DiffusersPipelineAdapter"
        ) as mock_diff:
            mock_diff.side_effect = ImportError("No diffusers")
            adapter = configurator.create_adapter("auto")
            assert isinstance(adapter, MockPipelineAdapter)

    def test_adapter_with_strategies(self):
        """Test different strategies work with adapters"""
        mock_adapter = MockPipelineAdapter()

        strategies = ["direct", "progressive", "tiled"]

        for strategy in strategies:
            config = ExpandorConfig(
                target_width=2048, target_height=1536, strategy=strategy
            )

            expandor = Expandor(pipeline_adapter=mock_adapter, config=config)

            test_image = Image.new("RGB", (1024, 768))
            result = expandor.expand(test_image)

            assert result.success, f"Strategy {strategy} failed"
            assert result.final_image.size == (2048, 1536)

    def test_adapter_cleanup(self):
        """Test adapter cleanup functionality"""
        mock_adapter = MockPipelineAdapter()

        # Load some pipelines
        mock_adapter.load_pipeline("sdxl")
        mock_adapter.load_pipeline("sd15")

        assert len(mock_adapter._pipelines) == 2

        # Cleanup
        mock_adapter.cleanup()

        # Should have cleared pipelines
        assert len(mock_adapter._pipelines) == 0

    def test_adapter_operation_checking(self):
        """Test operation support checking across adapters"""
        adapters = [
            MockPipelineAdapter(),
            ComfyUIPipelineAdapter(),
            A1111PipelineAdapter(),
        ]

        for adapter in adapters:
            # All should have the method
            assert hasattr(adapter, "supports_operation")

            # Mock should support most operations
            if isinstance(adapter, MockPipelineAdapter):
                assert adapter.supports_operation("generate")
                assert adapter.supports_operation("inpaint")
            else:
                # Placeholders don't support anything yet
                assert not adapter.supports_operation("generate")
                assert not adapter.supports_operation("inpaint")
