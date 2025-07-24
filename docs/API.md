# Expandor API Documentation

## Table of Contents

1. [Core API](#core-api)
2. [Configuration API](#configuration-api)
3. [Strategy API](#strategy-api)
4. [Adapter API](#adapter-api)
5. [Quality API](#quality-api)
6. [Utility API](#utility-api)
7. [CLI API](#cli-api)
8. [Examples](#examples)

## Core API

### Expandor Class

The main class for image expansion operations.

```python
from expandor import Expandor
from expandor.core.config import ExpandorConfig
from expandor.adapters import DiffusersPipelineAdapter

class Expandor:
    """Universal Image Resolution Adaptation System"""
    
    def __init__(
        self,
        pipeline_adapter: BasePipelineAdapter,
        config: ExpandorConfig
    ):
        """
        Initialize Expandor.
        
        Args:
            pipeline_adapter: Pipeline adapter instance
            config: Expandor configuration
        """
```

#### Methods

##### expand

```python
def expand(
    self,
    image: Union[Image.Image, str, Path],
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    **kwargs
) -> ExpansionResult:
    """
    Expand an image to target resolution.
    
    Args:
        image: Input image (PIL Image, file path, or Path object)
        prompt: Override config prompt
        negative_prompt: Override config negative prompt
        seed: Random seed for reproducibility
        progress_callback: Callback for progress updates
        **kwargs: Additional pipeline-specific parameters
        
    Returns:
        ExpansionResult object containing:
            - success: bool
            - final_image: PIL.Image or None
            - strategy_used: str
            - processing_time: float
            - metadata: Dict[str, Any]
            - error: Optional[str]
            - stages: List[PIL.Image] (if save_stages=True)
            
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If expansion fails
        
    Example:
        result = expandor.expand(
            "input.jpg",
            prompt="beautiful landscape, high quality",
            seed=42
        )
        if result.success:
            result.final_image.save("output.png")
    """
```

##### expand_batch

```python
def expand_batch(
    self,
    images: List[Union[Image.Image, str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    max_workers: Optional[int] = None,
    continue_on_error: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    **kwargs
) -> BatchResult:
    """
    Expand multiple images.
    
    Args:
        images: List of input images
        output_dir: Directory for output files
        max_workers: Maximum parallel workers
        continue_on_error: Continue if individual image fails
        progress_callback: Callback(current, total, filename)
        **kwargs: Parameters passed to expand()
        
    Returns:
        BatchResult object containing:
            - total: int
            - successful: int
            - failed: int
            - results: List[ExpansionResult]
            - errors: Dict[str, str]
    """
```

### ExpansionResult Class

```python
@dataclass
class ExpansionResult:
    """Result from image expansion"""
    
    success: bool
    final_image: Optional[Image.Image]
    strategy_used: str
    processing_time: float
    metadata: Dict[str, Any]
    error: Optional[str] = None
    stages: Optional[List[Image.Image]] = None
    
    @property
    def image_path(self) -> Optional[Path]:
        """Get saved image path if available"""
        return self.metadata.get('output_path')
        
    def save(
        self,
        path: Union[str, Path],
        format: Optional[str] = None,
        optimize: bool = True,
        **kwargs
    ) -> Path:
        """
        Save the final image.
        
        Args:
            path: Output file path
            format: Image format (auto-detected if None)
            optimize: Optimize file size
            **kwargs: Additional save parameters
            
        Returns:
            Path to saved file
        """
```

## Configuration API

### ExpandorConfig Class

```python
from expandor.core.config import ExpandorConfig

@dataclass
class ExpandorConfig:
    """Configuration for Expandor operations"""
    
    # Target dimensions
    target_width: int
    target_height: int
    
    # Strategy selection
    strategy: str = "auto"
    
    # Quality settings
    quality_preset: str = "balanced"
    inference_steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    denoise_strength: Optional[float] = None
    
    # Generation parameters
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    
    # Model settings
    model_type: str = "sdxl"
    device: str = "cuda"
    dtype: str = "float16"
    
    # Memory management
    enable_memory_efficient: bool = False
    vram_limit_mb: Optional[float] = None
    tile_size: int = 1024
    tile_overlap: int = 128
    clear_cache_frequency: int = 10
    
    # Quality control
    enable_artifacts_check: bool = True
    artifact_detection_threshold: float = 0.1
    max_artifact_repair_attempts: int = 3
    
    # Output settings
    output_format: str = "png"
    output_quality: int = 95
    save_stages: bool = False
    stage_dir: Optional[Path] = None
    save_metadata: bool = True
    
    # Advanced settings
    custom_pipeline_params: Optional[Dict[str, Any]] = None
    verbose: bool = False
```

### UserConfig Class

```python
from expandor.config import UserConfig, ModelConfig, LoRAConfig

@dataclass
class UserConfig:
    """User configuration for Expandor"""
    
    # Model configurations
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    
    # LoRA configurations
    loras: List[LoRAConfig] = field(default_factory=list)
    
    # Default settings
    default_model: str = "sdxl"
    default_quality: str = "balanced"
    default_strategy: str = "auto"
    
    # User preferences
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Performance settings
    performance: Dict[str, Any] = field(default_factory=dict)
    
    # Output settings
    output: Dict[str, Any] = field(default_factory=dict)
```

### Configuration Management

```python
from expandor.config import UserConfigManager

class UserConfigManager:
    """Manages user configuration files"""
    
    def load(self, path: Optional[Path] = None) -> UserConfig:
        """Load configuration from file"""
        
    def save(self, config: UserConfig, path: Optional[Path] = None):
        """Save configuration to file"""
        
    def create_example_config(self, path: Path):
        """Create example configuration file"""
        
    def merge_configs(
        self,
        base: UserConfig,
        override: Dict[str, Any]
    ) -> UserConfig:
        """Merge configurations with overrides"""
```

## Strategy API

### BaseStrategy Abstract Class

```python
from expandor.strategies import BaseStrategy

class BaseStrategy(ABC):
    """Base class for expansion strategies"""
    
    @abstractmethod
    def can_handle(
        self,
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
        expansion_factor: float,
        aspect_ratio_change: float
    ) -> bool:
        """Check if strategy can handle expansion"""
        
    @abstractmethod
    def estimate_vram(
        self,
        source_size: Tuple[int, int],
        target_size: Tuple[int, int]
    ) -> float:
        """Estimate VRAM usage in MB"""
        
    @abstractmethod
    def expand(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> StrategyResult:
        """Execute expansion strategy"""
```

### Strategy Selection

```python
from expandor.strategies import StrategySelector

class StrategySelector:
    """Selects appropriate expansion strategy"""
    
    def select_strategy(
        self,
        current_size: Tuple[int, int],
        target_size: Tuple[int, int],
        available_vram: float,
        user_preference: Optional[str] = None,
        quality_priority: bool = True
    ) -> Tuple[BaseStrategy, str]:
        """
        Select best strategy for expansion.
        
        Returns:
            Tuple of (strategy_instance, reason_string)
        """
        
    def register_strategy(
        self,
        name: str,
        strategy_class: Type[BaseStrategy],
        priority: int = 50
    ):
        """Register custom strategy"""
```

### Built-in Strategies

```python
from expandor.strategies import (
    DirectUpscaleStrategy,
    ProgressiveOutpaintStrategy,
    SWPOStrategy,
    TiledStrategy,
    HybridAdaptiveStrategy
)

# Direct upscale for small expansions
strategy = DirectUpscaleStrategy(adapter, config)

# Progressive for large expansions
strategy = ProgressiveOutpaintStrategy(adapter, config)

# SWPO for extreme aspect ratios
strategy = SWPOStrategy(adapter, config)

# Tiled for memory efficiency
strategy = TiledStrategy(adapter, config)

# Hybrid for adaptive processing
strategy = HybridAdaptiveStrategy(adapter, config)
```

## Adapter API

### BasePipelineAdapter Abstract Class

```python
from expandor.adapters import BasePipelineAdapter

class BasePipelineAdapter(ABC):
    """Base class for pipeline adapters"""
    
    @abstractmethod
    def load_pipeline(
        self,
        model_name: str,
        model_id: Optional[str] = None,
        **kwargs
    ):
        """Load a specific model/pipeline"""
        
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
        """Generate image from text"""
        
    @abstractmethod
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        **kwargs
    ) -> Image.Image:
        """Inpaint masked areas"""
        
    @abstractmethod
    def supports_operation(self, operation: str) -> bool:
        """Check if operation is supported"""
```

### Available Adapters

```python
from expandor.adapters import (
    DiffusersPipelineAdapter,
    ComfyUIPipelineAdapter,
    A1111PipelineAdapter,
    MockPipelineAdapter
)

# Diffusers (Hugging Face)
adapter = DiffusersPipelineAdapter(
    device="cuda",
    dtype="float16",
    enable_attention_slicing=True
)

# ComfyUI (when implemented)
adapter = ComfyUIPipelineAdapter(
    server_url="http://localhost:8188"
)

# Automatic1111 (when implemented)
adapter = A1111PipelineAdapter(
    api_url="http://localhost:7860"
)

# Mock for testing
adapter = MockPipelineAdapter()
```

## Quality API

### Artifact Detection

```python
from expandor.quality import SmartArtifactDetector

class SmartArtifactDetector:
    """Detects artifacts in expanded images"""
    
    def detect_all_artifacts(
        self,
        image: Image.Image,
        boundaries: Optional[List[Dict[str, Any]]] = None,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect all types of artifacts.
        
        Returns:
            {
                'has_artifacts': bool,
                'confidence': float,
                'locations': List[Dict],
                'types': List[str],
                'severity': str  # 'low', 'medium', 'high'
            }
        """
        
    def detect_seams(
        self,
        image: Image.Image,
        boundaries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect seams at boundaries"""
        
    def detect_color_artifacts(
        self,
        image: Image.Image
    ) -> List[Dict[str, Any]]:
        """Detect color discontinuities"""
```

### Quality Refinement

```python
from expandor.quality import SmartQualityRefiner

class SmartQualityRefiner:
    """Refines image quality and fixes artifacts"""
    
    def refine_artifacts(
        self,
        image: Image.Image,
        artifacts: Dict[str, Any],
        prompt: str,
        strength: float = 0.3,
        max_attempts: int = 3
    ) -> Image.Image:
        """Refine detected artifacts"""
        
    def enhance_details(
        self,
        image: Image.Image,
        enhancement_type: str = "adaptive",
        strength: float = 0.5
    ) -> Image.Image:
        """Enhance image details"""
```

### Boundary Tracking

```python
from expandor.quality import BoundaryTracker

class BoundaryTracker:
    """Tracks expansion boundaries"""
    
    def add_boundary(
        self,
        boundary_type: str,
        position: Tuple[int, int, int, int],
        metadata: Dict[str, Any]
    ):
        """Add a boundary for tracking"""
        
    def get_boundaries_for_detection(self) -> List[Dict[str, Any]]:
        """Get boundaries formatted for detection"""
        
    def merge_boundaries(
        self,
        threshold: int = 10
    ) -> List[Dict[str, Any]]:
        """Merge nearby boundaries"""
```

## Utility API

### Dimension Calculator

```python
from expandor.utils import DimensionCalculator

class DimensionCalculator:
    """Calculates and validates dimensions"""
    
    def __init__(self, model_type: str = "sdxl"):
        """Initialize with model constraints"""
        
    def adjust_dimensions(
        self,
        width: int,
        height: int
    ) -> Tuple[int, int]:
        """Adjust dimensions to model constraints"""
        
    def calculate_progressive_dimensions(
        self,
        source: Tuple[int, int],
        target: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Calculate progressive expansion steps"""
        
    def validate_dimensions(
        self,
        width: int,
        height: int
    ) -> Tuple[bool, Optional[str]]:
        """Validate dimensions for model"""
```

### VRAM Manager

```python
from expandor.utils import VRAMManager

class VRAMManager:
    """Manages VRAM usage and optimization"""
    
    def get_available_vram(self) -> float:
        """Get available VRAM in MB"""
        
    def estimate_operation_vram(
        self,
        operation: str,
        width: int,
        height: int,
        model_type: str = "sdxl"
    ) -> float:
        """Estimate VRAM for operation"""
        
    def get_memory_efficient_settings(
        self,
        target_vram: float
    ) -> Dict[str, Any]:
        """Get settings for target VRAM limit"""
        
    def clear_cache(self):
        """Clear CUDA cache"""
```

### Model Manager

```python
from expandor.utils import ModelManager

class ModelManager:
    """Manages model downloads and validation"""
    
    def download_model(
        self,
        model_id: str,
        model_type: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ) -> Path:
        """Download model if not cached"""
        
    def validate_model(
        self,
        model_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Validate model file"""
        
    def get_model_info(
        self,
        model_id: str
    ) -> Dict[str, Any]:
        """Get model information"""
        
    def list_available_models(
        self,
        model_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available models"""
```

### Metadata Manager

```python
from expandor.utils import MetadataManager

class MetadataManager:
    """Manages image and processing metadata"""
    
    def extract_metadata(
        self,
        image: Image.Image
    ) -> Dict[str, Any]:
        """Extract metadata from image"""
        
    def embed_metadata(
        self,
        image: Image.Image,
        metadata: Dict[str, Any]
    ) -> Image.Image:
        """Embed metadata in image"""
        
    def save_metadata(
        self,
        metadata: Dict[str, Any],
        path: Path,
        format: str = "json"
    ):
        """Save metadata to file"""
```

## CLI API

### CLI Command Structure

```python
from expandor.cli import create_parser, main

# Programmatic CLI usage
parser = create_parser()
args = parser.parse_args([
    "input.jpg",
    "--resolution", "4K",
    "--quality", "high",
    "--output", "output.png"
])

# Execute with args
main(args)
```

### CLI Commands

```python
from expandor.cli.commands import (
    SetupCommand,
    ProcessCommand,
    BatchCommand,
    ValidateCommand
)

# Setup wizard
cmd = SetupCommand()
cmd.execute()

# Process single image
cmd = ProcessCommand()
cmd.execute(
    input_path="input.jpg",
    resolution="3840x2160",
    output="output.png"
)

# Batch processing
cmd = BatchCommand()
cmd.execute(
    pattern="*.jpg",
    resolution="2K",
    output_dir="results/"
)
```

### Setup Wizard API

```python
from expandor.cli.setup_wizard import SetupWizard

wizard = SetupWizard()

# Run interactive setup
config = wizard.run()

# Or configure programmatically
config = wizard.configure_models(
    preferred_model="sdxl",
    model_path="/models/sdxl.safetensors"
)

config = wizard.configure_quality(
    default_quality="high",
    auto_select_strategy=True
)
```

## Examples

### Basic Usage

```python
from expandor import Expandor
from expandor.core.config import ExpandorConfig
from expandor.adapters import DiffusersPipelineAdapter
from PIL import Image

# Setup
adapter = DiffusersPipelineAdapter()
config = ExpandorConfig(
    target_width=3840,
    target_height=2160,
    quality_preset="high"
)

expandor = Expandor(adapter, config)

# Expand image
image = Image.open("photo.jpg")
result = expandor.expand(image)

if result.success:
    result.final_image.save("photo_4k.png")
    print(f"Expanded using {result.strategy_used} strategy")
    print(f"Processing time: {result.processing_time:.2f}s")
```

### Advanced Configuration

```python
from expandor.config import UserConfig, ModelConfig, LoRAConfig
from expandor.config.pipeline_config import PipelineConfigurator

# Create user configuration
user_config = UserConfig(
    models={
        'sdxl': ModelConfig(
            model_id='stabilityai/stable-diffusion-xl-base-1.0',
            dtype='float16',
            device='cuda'
        )
    },
    loras=[
        LoRAConfig(
            name='detail_enhancer',
            path='/loras/detail.safetensors',
            weight=0.7
        )
    ],
    default_quality='ultra'
)

# Create pipeline configurator
configurator = PipelineConfigurator()

# Generate Expandor config
expandor_config = configurator.create_expandor_config(
    user_config=user_config,
    target_size=(5120, 2880),
    strategy='progressive'
)

# Create adapter with LoRAs
adapter = configurator.create_adapter(
    adapter_type='diffusers',
    model_config=user_config.models['sdxl']
)

# Load LoRAs
adapter.load_loras('sdxl', [
    {'path': lora.path, 'weight': lora.weight}
    for lora in user_config.loras
])
```

### Custom Strategy

```python
from expandor.strategies import BaseStrategy, StrategyResult
from expandor.strategies import register_strategy

class MyCustomStrategy(BaseStrategy):
    @property
    def name(self):
        return "my_custom"
    
    def can_handle(self, source_size, target_size, *args):
        # Custom logic
        return True
        
    def estimate_vram(self, source_size, target_size):
        # Custom estimation
        return 8000  # 8GB
        
    def expand(self, image, target_size, prompt, **kwargs):
        # Custom expansion
        result = self._my_expansion_logic(image, target_size)
        return StrategyResult(
            success=True,
            final_image=result,
            stages=[image, result],
            metadata={'custom': True}
        )

# Register strategy
register_strategy('my_custom', MyCustomStrategy)

# Use it
config = ExpandorConfig(
    target_width=3840,
    target_height=2160,
    strategy='my_custom'
)
```

### Batch Processing with Progress

```python
from pathlib import Path
import concurrent.futures

def process_directory(
    input_dir: Path,
    output_dir: Path,
    target_resolution: Tuple[int, int],
    max_workers: int = 4
):
    """Process all images in directory"""
    
    # Setup
    adapter = DiffusersPipelineAdapter()
    config = ExpandorConfig(
        target_width=target_resolution[0],
        target_height=target_resolution[1]
    )
    expandor = Expandor(adapter, config)
    
    # Get all images
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    # Process with thread pool
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_single,
                expandor,
                img,
                output_dir
            ): img for img in images
        }
        
        # Process results
        for future in concurrent.futures.as_completed(futures):
            img_path = futures[future]
            try:
                result = future.result()
                results.append((img_path, result))
                print(f"✓ {img_path.name}")
            except Exception as e:
                print(f"✗ {img_path.name}: {e}")
                
    return results

def process_single(expandor, img_path, output_dir):
    """Process single image"""
    result = expandor.expand(img_path)
    if result.success:
        output_path = output_dir / f"{img_path.stem}_expanded{img_path.suffix}"
        result.save(output_path)
    return result
```

### Quality-Focused Pipeline

```python
from expandor.quality import SmartArtifactDetector, SmartQualityRefiner

# Expand with quality pipeline
config = ExpandorConfig(
    target_width=3840,
    target_height=2160,
    quality_preset='ultra',
    enable_artifacts_check=True,
    artifact_detection_threshold=0.05
)

expandor = Expandor(adapter, config)

# Custom quality pipeline
def expand_with_quality_check(image, expandor):
    # Initial expansion
    result = expandor.expand(image)
    
    if not result.success:
        return result
        
    # Additional quality check
    detector = SmartArtifactDetector(expandor.config)
    artifacts = detector.detect_all_artifacts(
        result.final_image,
        result.metadata.get('boundaries', [])
    )
    
    if artifacts['has_artifacts']:
        print(f"Found {len(artifacts['locations'])} artifacts, refining...")
        
        refiner = SmartQualityRefiner(
            expandor.pipeline_adapter,
            expandor.config
        )
        
        refined = refiner.refine_artifacts(
            result.final_image,
            artifacts,
            expandor.config.prompt or "",
            strength=0.3
        )
        
        result.final_image = refined
        result.metadata['quality_refined'] = True
        
    return result
```

### Memory-Constrained Usage

```python
# Configure for low VRAM (4GB)
config = ExpandorConfig(
    target_width=2560,
    target_height=1440,
    strategy='tiled',
    tile_size=512,
    tile_overlap=64,
    enable_memory_efficient=True,
    vram_limit_mb=3500,  # Leave buffer
    clear_cache_frequency=2
)

# Enable all memory optimizations
adapter = DiffusersPipelineAdapter()
adapter.enable_memory_efficient_mode()
adapter.enable_attention_slicing()
adapter.enable_vae_slicing()

# Monitor memory during expansion
from expandor.utils import VRAMManager

vram_manager = VRAMManager()

def expand_with_monitoring(expandor, image):
    # Check before
    before = vram_manager.get_available_vram()
    print(f"Available VRAM: {before:.0f}MB")
    
    # Expand
    result = expandor.expand(image)
    
    # Check after
    after = vram_manager.get_available_vram()
    print(f"VRAM used: {before - after:.0f}MB")
    
    return result
```

## See Also

- [CLI Usage Guide](CLI_USAGE.md) - Detailed CLI documentation
- [Configuration Guide](CONFIGURATION.md) - Configuration options
- [Adapter Development](ADAPTER_DEVELOPMENT.md) - Creating custom adapters
- [Strategy Development](STRATEGY_DEVELOPMENT.md) - Creating custom strategies
- [Examples](../examples/) - Example scripts and notebooks