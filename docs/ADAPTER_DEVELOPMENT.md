# Pipeline Adapter Development Guide

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Base Adapter Interface](#base-adapter-interface)
4. [Implementation Guide](#implementation-guide)
5. [Required Methods](#required-methods)
6. [Optional Methods](#optional-methods)
7. [Error Handling](#error-handling)
8. [Memory Management](#memory-management)
9. [Testing Adapters](#testing-adapters)
10. [Example Implementations](#example-implementations)
11. [Best Practices](#best-practices)
12. [Integration](#integration)

## Overview

Pipeline adapters are the bridge between Expandor and various image generation backends. They provide a unified interface that allows Expandor to work with any image generation pipeline, whether it's Diffusers, ComfyUI, A1111, or a custom implementation.

### Design Principles

1. **Uniformity**: All adapters expose the same interface
2. **Flexibility**: Support pipeline-specific features without breaking compatibility
3. **Fail Loud**: Clear error messages, no silent failures
4. **Efficiency**: Minimize overhead and memory usage
5. **Extensibility**: Easy to add new capabilities

## Architecture

```
┌─────────────────┐
│    Expandor     │
└────────┬────────┘
         │
┌────────▼────────┐
│  BasePipeline   │ (Abstract Base Class)
│    Adapter      │
└────────┬────────┘
         │
    ┌────┴────┬────────┬──────────┐
    │         │        │          │
┌───▼───┐ ┌──▼───┐ ┌──▼───┐ ┌───▼───┐
│Diffusers│ │ComfyUI│ │A1111│ │Custom│
│Adapter │ │Adapter│ │Adapter│ │Adapter│
└────────┘ └───────┘ └──────┘ └───────┘
```

## Base Adapter Interface

### Abstract Base Class

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import torch

class BasePipelineAdapter(ABC):
    """Base class for all pipeline adapters"""
    
    def __init__(self, device: str = "cuda", dtype: str = "float16"):
        self.device = device
        self.dtype = dtype
        self._pipelines = {}
        self._current_pipeline = None
        
    @abstractmethod
    def load_pipeline(self, model_name: str, model_id: Optional[str] = None, **kwargs):
        """Load a specific model/pipeline"""
        pass
        
    @abstractmethod
    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """Generate an image from text"""
        pass
        
    @abstractmethod
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """Inpaint masked areas of an image"""
        pass
        
    @abstractmethod
    def img2img(
        self,
        image: Image.Image,
        prompt: str,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """Transform an existing image"""
        pass
        
    @abstractmethod
    def supports_operation(self, operation: str) -> bool:
        """Check if adapter supports a specific operation"""
        pass
        
    @abstractmethod
    def cleanup(self):
        """Clean up resources"""
        pass
```

## Implementation Guide

### Step 1: Create Your Adapter Class

```python
from expandor.adapters.base_adapter import BasePipelineAdapter
from typing import Optional, Dict, Any
from PIL import Image
import logging

class MyCustomAdapter(BasePipelineAdapter):
    """Custom adapter for MyPipeline"""
    
    def __init__(self, device: str = "cuda", dtype: str = "float16", **kwargs):
        super().__init__(device, dtype)
        self.logger = logging.getLogger(__name__)
        self.api_url = kwargs.get("api_url", "http://localhost:7860")
        self._supported_operations = {
            "generate", "img2img", "inpaint", "upscale"
        }
```

### Step 2: Implement Pipeline Loading

```python
def load_pipeline(self, model_name: str, model_id: Optional[str] = None, **kwargs):
    """Load pipeline with error handling"""
    try:
        self.logger.info(f"Loading pipeline: {model_name}")
        
        # Example: Load your custom pipeline
        if model_name == "sdxl":
            from mypipeline import MySDXLPipeline
            pipeline = MySDXLPipeline.from_pretrained(
                model_id or "stabilityai/stable-diffusion-xl-base-1.0",
                device=self.device,
                dtype=self._get_torch_dtype()
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        # Store pipeline
        self._pipelines[model_name] = pipeline
        self._current_pipeline = model_name
        
        self.logger.info(f"Successfully loaded {model_name}")
        
    except Exception as e:
        self.logger.error(f"Failed to load pipeline: {str(e)}")
        raise RuntimeError(f"Pipeline loading failed: {str(e)}")
```

### Step 3: Implement Core Operations

```python
def generate(
    self,
    prompt: str,
    width: int,
    height: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs
) -> Image.Image:
    """Generate image with comprehensive error handling"""
    
    # Validate inputs
    if not self._current_pipeline:
        raise ValueError("No pipeline loaded. Call load_pipeline first.")
        
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError(f"Dimensions must be multiples of 8, got {width}x{height}")
        
    try:
        # Get pipeline
        pipeline = self._pipelines[self._current_pipeline]
        
        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            import torch
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        # Generate image
        self.logger.debug(f"Generating {width}x{height} image")
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs
        )
        
        # Extract image from result
        if hasattr(result, 'images'):
            image = result.images[0]
        else:
            image = result
            
        return image
        
    except torch.cuda.OutOfMemoryError:
        self.logger.error("CUDA out of memory")
        raise RuntimeError("GPU out of memory. Try reducing resolution or using tiled mode.")
    except Exception as e:
        self.logger.error(f"Generation failed: {str(e)}")
        raise RuntimeError(f"Image generation failed: {str(e)}")
```

## Required Methods

### 1. load_pipeline

```python
def load_pipeline(self, model_name: str, model_id: Optional[str] = None, **kwargs):
    """
    Load a specific model/pipeline.
    
    Args:
        model_name: Identifier for the model type (e.g., "sdxl", "sd15")
        model_id: Model path or HuggingFace ID
        **kwargs: Additional pipeline-specific parameters
        
    Raises:
        RuntimeError: If pipeline loading fails
        ValueError: If model_name is not supported
    """
```

### 2. generate

```python
def generate(self, prompt: str, width: int, height: int, **kwargs) -> Image.Image:
    """
    Generate an image from text prompt.
    
    Args:
        prompt: Text description of the image
        width: Target width (must be multiple of 8)
        height: Target height (must be multiple of 8)
        **kwargs: Additional generation parameters
        
    Returns:
        PIL.Image: Generated image
        
    Raises:
        RuntimeError: If generation fails
        ValueError: If no pipeline is loaded
    """
```

### 3. inpaint

```python
def inpaint(
    self,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    **kwargs
) -> Image.Image:
    """
    Inpaint masked areas of an image.
    
    Args:
        image: Original image
        mask: Binary mask (white=inpaint, black=keep)
        prompt: Description for inpainted areas
        **kwargs: Additional parameters
        
    Returns:
        PIL.Image: Inpainted image
    """
```

### 4. img2img

```python
def img2img(
    self,
    image: Image.Image,
    prompt: str,
    strength: float = 0.8,
    **kwargs
) -> Image.Image:
    """
    Transform an existing image based on prompt.
    
    Args:
        image: Input image
        prompt: Transformation description
        strength: Denoising strength (0.0-1.0)
        **kwargs: Additional parameters
        
    Returns:
        PIL.Image: Transformed image
    """
```

### 5. supports_operation

```python
def supports_operation(self, operation: str) -> bool:
    """
    Check if adapter supports a specific operation.
    
    Args:
        operation: Operation name (e.g., "generate", "inpaint", "controlnet")
        
    Returns:
        bool: True if operation is supported
    """
    return operation in self._supported_operations
```

### 6. cleanup

```python
def cleanup(self):
    """
    Clean up resources and free memory.
    """
    for pipeline in self._pipelines.values():
        if hasattr(pipeline, 'to'):
            pipeline.to('cpu')
        del pipeline
    
    self._pipelines.clear()
    self._current_pipeline = None
    
    # Clear CUDA cache if available
    if self.device.startswith('cuda'):
        import torch
        torch.cuda.empty_cache()
```

## Optional Methods

### Enhanced Functionality

```python
def refine(
    self,
    image: Image.Image,
    prompt: str,
    strength: float = 0.3,
    **kwargs
) -> Image.Image:
    """Refine an image with minimal changes"""
    # Implementation optional
    pass

def enhance(
    self,
    image: Image.Image,
    enhancement_type: str = "detail",
    **kwargs
) -> Image.Image:
    """Apply enhancement to image"""
    # Implementation optional
    pass

def upscale(
    self,
    image: Image.Image,
    scale: float = 2.0,
    **kwargs
) -> Image.Image:
    """Upscale image using pipeline-specific method"""
    # Implementation optional
    pass
```

### ControlNet Support

```python
def supports_controlnet(self) -> bool:
    """Check if adapter supports ControlNet"""
    return False

def controlnet_inpaint(
    self,
    image: Image.Image,
    mask: Image.Image,
    control_image: Image.Image,
    prompt: str,
    control_type: str = "canny",
    **kwargs
) -> Image.Image:
    """Inpaint with ControlNet guidance"""
    if not self.supports_controlnet():
        raise NotImplementedError("ControlNet not supported")
```

### LoRA Management

```python
def load_loras(self, model_name: str, lora_configs: List[Dict[str, Any]]):
    """Load multiple LoRAs for a model"""
    pipeline = self._pipelines.get(model_name)
    if not pipeline:
        raise ValueError(f"Model {model_name} not loaded")
        
    for config in lora_configs:
        self._load_single_lora(pipeline, config)

def unload_loras(self, model_name: str):
    """Unload all LoRAs from a model"""
    pass
```

### Performance Methods

```python
def estimate_vram(
    self,
    operation: str,
    width: int,
    height: int,
    **kwargs
) -> float:
    """Estimate VRAM usage in MB"""
    base_usage = {
        "generate": 4000,
        "img2img": 4500,
        "inpaint": 5000
    }
    
    # Scale by resolution
    pixels = width * height
    scale = pixels / (1024 * 1024)  # Relative to 1024x1024
    
    return base_usage.get(operation, 4000) * scale

def enable_memory_efficient_mode(self):
    """Enable memory optimizations"""
    for pipeline in self._pipelines.values():
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()
        if hasattr(pipeline, 'enable_vae_slicing'):
            pipeline.enable_vae_slicing()
```

## Error Handling

### Fail Loud Philosophy

```python
class AdapterError(Exception):
    """Base exception for adapter errors"""
    pass

class PipelineNotLoadedError(AdapterError):
    """Raised when operation attempted without loaded pipeline"""
    pass

class OperationNotSupportedError(AdapterError):
    """Raised when unsupported operation is requested"""
    pass

class InvalidParametersError(AdapterError):
    """Raised when parameters are invalid"""
    pass

# Usage in adapter
def generate(self, prompt: str, width: int, height: int, **kwargs):
    if not self._current_pipeline:
        raise PipelineNotLoadedError(
            "No pipeline loaded. Call load_pipeline() first."
        )
        
    if width <= 0 or height <= 0:
        raise InvalidParametersError(
            f"Invalid dimensions: {width}x{height}. Must be positive."
        )
```

### Comprehensive Error Messages

```python
def _validate_image_size(self, width: int, height: int, model_name: str):
    """Validate image dimensions for model"""
    constraints = self._get_model_constraints(model_name)
    
    if width < constraints['min_size'] or height < constraints['min_size']:
        raise ValueError(
            f"Image size {width}x{height} too small for {model_name}. "
            f"Minimum size is {constraints['min_size']}x{constraints['min_size']}."
        )
        
    if width > constraints['max_size'] or height > constraints['max_size']:
        raise ValueError(
            f"Image size {width}x{height} too large for {model_name}. "
            f"Maximum size is {constraints['max_size']}x{constraints['max_size']}. "
            f"Consider using tiled processing."
        )
        
    if width % constraints['multiple'] != 0 or height % constraints['multiple'] != 0:
        raise ValueError(
            f"Image dimensions must be multiples of {constraints['multiple']} "
            f"for {model_name}. Got {width}x{height}."
        )
```

## Memory Management

### VRAM Monitoring

```python
def _get_available_vram(self) -> float:
    """Get available VRAM in MB"""
    if not self.device.startswith('cuda'):
        return float('inf')
        
    import torch
    return torch.cuda.get_device_properties(0).total_memory / 1024**2

def _check_vram_availability(self, operation: str, width: int, height: int):
    """Check if enough VRAM is available"""
    required = self.estimate_vram(operation, width, height)
    available = self._get_available_vram()
    
    if required > available * 0.9:  # Leave 10% buffer
        raise RuntimeError(
            f"Insufficient VRAM. Operation requires ~{required:.0f}MB, "
            f"but only {available:.0f}MB available. "
            f"Try: 1) Lower resolution, 2) Use tiled mode, 3) Enable CPU offload"
        )
```

### Automatic Cleanup

```python
def __enter__(self):
    """Context manager entry"""
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit with cleanup"""
    self.cleanup()
    
# Usage
with MyCustomAdapter() as adapter:
    adapter.load_pipeline("sdxl")
    image = adapter.generate("test", 1024, 1024)
# Automatically cleaned up
```

## Testing Adapters

### Unit Test Template

```python
import pytest
from PIL import Image
from expandor.adapters import MyCustomAdapter

class TestMyCustomAdapter:
    
    @pytest.fixture
    def adapter(self):
        """Create adapter instance"""
        return MyCustomAdapter(device="cpu")  # Use CPU for tests
        
    @pytest.fixture
    def test_image(self):
        """Create test image"""
        return Image.new('RGB', (512, 512), color='red')
        
    def test_initialization(self, adapter):
        """Test adapter initialization"""
        assert adapter.device == "cpu"
        assert len(adapter._pipelines) == 0
        
    def test_supports_operation(self, adapter):
        """Test operation support checking"""
        assert adapter.supports_operation("generate")
        assert adapter.supports_operation("inpaint")
        assert not adapter.supports_operation("invalid_op")
        
    def test_generate_without_pipeline(self, adapter):
        """Test error when generating without pipeline"""
        with pytest.raises(ValueError, match="No pipeline loaded"):
            adapter.generate("test", 512, 512)
            
    def test_dimension_validation(self, adapter):
        """Test dimension validation"""
        adapter.load_pipeline("test")
        
        # Invalid dimensions
        with pytest.raises(ValueError, match="multiples of"):
            adapter.generate("test", 513, 512)  # Not multiple of 8
```

### Integration Test Template

```python
def test_full_workflow(adapter):
    """Test complete generation workflow"""
    # Load pipeline
    adapter.load_pipeline("sdxl")
    
    # Generate base image
    image = adapter.generate(
        prompt="a beautiful landscape",
        width=1024,
        height=1024,
        seed=42
    )
    assert image.size == (1024, 1024)
    
    # Transform with img2img
    transformed = adapter.img2img(
        image=image,
        prompt="a beautiful landscape at sunset",
        strength=0.5
    )
    assert transformed.size == image.size
    
    # Cleanup
    adapter.cleanup()
```

## Example Implementations

### Minimal Adapter

```python
class MinimalAdapter(BasePipelineAdapter):
    """Minimal adapter implementation"""
    
    def load_pipeline(self, model_name: str, **kwargs):
        # Minimal loading logic
        self._current_pipeline = model_name
        
    def generate(self, prompt: str, width: int, height: int, **kwargs):
        # Return placeholder image
        return Image.new('RGB', (width, height), color='blue')
        
    def inpaint(self, image, mask, prompt, **kwargs):
        # Return original image as placeholder
        return image
        
    def img2img(self, image, prompt, **kwargs):
        # Return original image as placeholder
        return image
        
    def supports_operation(self, operation: str) -> bool:
        return operation in ["generate", "img2img", "inpaint"]
        
    def cleanup(self):
        self._current_pipeline = None
```

### API-Based Adapter

```python
class APIAdapter(BasePipelineAdapter):
    """Adapter for external API"""
    
    def __init__(self, api_url: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_url = api_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        
    def generate(self, prompt: str, width: int, height: int, **kwargs):
        response = self.session.post(
            f"{self.api_url}/generate",
            json={
                "prompt": prompt,
                "width": width,
                "height": height,
                **kwargs
            }
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"API error: {response.text}")
            
        # Convert response to image
        image_data = base64.b64decode(response.json()["image"])
        return Image.open(io.BytesIO(image_data))
```

## Best Practices

### 1. Consistent Parameter Names

```python
# Good: Use standard parameter names
def generate(self, prompt: str, width: int, height: int, 
            num_inference_steps: int = 50, guidance_scale: float = 7.5):
    pass

# Bad: Custom parameter names
def generate(self, text: str, w: int, h: int, steps: int = 50, cfg: float = 7.5):
    pass
```

### 2. Comprehensive Logging

```python
import logging

class WellLoggedAdapter(BasePipelineAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        
    def generate(self, prompt: str, width: int, height: int, **kwargs):
        self.logger.info(f"Generating {width}x{height} image")
        self.logger.debug(f"Prompt: {prompt[:50]}...")
        self.logger.debug(f"Parameters: {kwargs}")
        
        try:
            image = self._generate_internal(prompt, width, height, **kwargs)
            self.logger.info("Generation successful")
            return image
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}", exc_info=True)
            raise
```

### 3. Resource Management

```python
class ResourceManagedAdapter(BasePipelineAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._resource_lock = threading.Lock()
        self._active_operations = 0
        
    def generate(self, prompt: str, width: int, height: int, **kwargs):
        with self._resource_lock:
            self._active_operations += 1
            
        try:
            return self._generate_internal(prompt, width, height, **kwargs)
        finally:
            with self._resource_lock:
                self._active_operations -= 1
                
    def cleanup(self):
        # Wait for active operations
        while self._active_operations > 0:
            time.sleep(0.1)
            
        super().cleanup()
```

### 4. Type Hints and Documentation

```python
from typing import Optional, Dict, Any, Union, Literal

class WellDocumentedAdapter(BasePipelineAdapter):
    """
    Adapter for MyPipeline backend.
    
    This adapter provides integration with MyPipeline, supporting
    text-to-image generation, image-to-image transformation, and inpainting.
    
    Attributes:
        device: Torch device to use ('cuda', 'cpu', 'mps')
        dtype: Data type for models ('float32', 'float16', 'bfloat16')
        api_url: URL of the MyPipeline API endpoint
        
    Example:
        >>> adapter = MyPipelineAdapter(device="cuda")
        >>> adapter.load_pipeline("sdxl")
        >>> image = adapter.generate("a cat", 512, 512)
    """
    
    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        scheduler: Literal["ddim", "pndm", "lms", "euler"] = "euler",
        **kwargs
    ) -> Image.Image:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the desired image
            width: Image width in pixels (must be multiple of 8)
            height: Image height in pixels (must be multiple of 8)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance strength
            negative_prompt: Text describing what to avoid
            seed: Random seed for reproducibility
            scheduler: Noise scheduler to use
            **kwargs: Additional pipeline-specific parameters
            
        Returns:
            Generated PIL Image
            
        Raises:
            ValueError: If dimensions are invalid
            RuntimeError: If generation fails
            
        Note:
            For best results with SDXL, use dimensions that are
            multiples of 64 and total pixel count near 1024x1024.
        """
        # Implementation
        pass
```

## Integration

### Registering Your Adapter

```python
# In expandor/adapters/__init__.py
from .base_adapter import BasePipelineAdapter
from .mock_adapter import MockPipelineAdapter
from .diffusers_adapter import DiffusersPipelineAdapter
from .my_custom_adapter import MyCustomAdapter  # Add your adapter

__all__ = [
    'BasePipelineAdapter',
    'MockPipelineAdapter',
    'DiffusersPipelineAdapter',
    'MyCustomAdapter',  # Export it
]

# Adapter registry
ADAPTER_REGISTRY = {
    'mock': MockPipelineAdapter,
    'diffusers': DiffusersPipelineAdapter,
    'mycustom': MyCustomAdapter,  # Register it
}
```

### Using Your Adapter

```python
from expandor import Expandor
from expandor.adapters import MyCustomAdapter
from expandor.core.config import ExpandorConfig

# Create adapter
adapter = MyCustomAdapter(
    device="cuda",
    api_url="http://localhost:7860"
)

# Use with Expandor
config = ExpandorConfig(
    target_width=2048,
    target_height=1536
)

expandor = Expandor(adapter, config)
result = expandor.expand(input_image)
```

### CLI Integration

```yaml
# In expandor/config/adapters.yaml
adapters:
  mycustom:
    class: "expandor.adapters.MyCustomAdapter"
    display_name: "My Custom Pipeline"
    default_params:
      api_url: "http://localhost:7860"
    supported_models: ["sdxl", "sd15"]
    features:
      - generate
      - img2img
      - inpaint
```

## Advanced Topics

### Async Support

```python
import asyncio
from typing import Coroutine

class AsyncAdapter(BasePipelineAdapter):
    """Adapter with async support"""
    
    async def generate_async(
        self,
        prompt: str,
        width: int,
        height: int,
        **kwargs
    ) -> Image.Image:
        """Async generation method"""
        # Async implementation
        result = await self._api_call_async("generate", {
            "prompt": prompt,
            "width": width,
            "height": height,
            **kwargs
        })
        return self._process_result(result)
        
    def generate(self, prompt: str, width: int, height: int, **kwargs):
        """Sync wrapper for async method"""
        return asyncio.run(self.generate_async(prompt, width, height, **kwargs))
```

### Batch Processing

```python
def generate_batch(
    self,
    prompts: List[str],
    width: int,
    height: int,
    **kwargs
) -> List[Image.Image]:
    """Generate multiple images efficiently"""
    if not self.supports_batch():
        # Fallback to sequential
        return [self.generate(p, width, height, **kwargs) for p in prompts]
        
    # Batch implementation
    return self._generate_batch_internal(prompts, width, height, **kwargs)
```

### Progress Callbacks

```python
from typing import Callable

def generate(
    self,
    prompt: str,
    width: int,
    height: int,
    progress_callback: Optional[Callable[[float], None]] = None,
    **kwargs
) -> Image.Image:
    """Generate with progress tracking"""
    
    def internal_callback(step: int, total: int):
        if progress_callback:
            progress_callback(step / total)
            
    # Pass callback to pipeline
    return self._pipeline(
        prompt=prompt,
        width=width,
        height=height,
        callback=internal_callback,
        **kwargs
    )
```

## See Also

- [Base Adapter Source](../expandor/adapters/base_adapter.py)
- [Example Adapters](../expandor/adapters/)
- [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md) - Creating custom strategies
- [API Documentation](API.md) - Full API reference
- [Testing Guide](TESTING.md) - Testing your adapter