# Strategy Development Guide

## Table of Contents

1. [Overview](#overview)
2. [Strategy Architecture](#strategy-architecture)
3. [Base Strategy Interface](#base-strategy-interface)
4. [Creating Custom Strategies](#creating-custom-strategies)
5. [Strategy Lifecycle](#strategy-lifecycle)
6. [Key Components](#key-components)
7. [VRAM Management](#vram-management)
8. [Boundary Tracking](#boundary-tracking)
9. [Quality Assurance](#quality-assurance)
10. [Testing Strategies](#testing-strategies)
11. [Example Implementations](#example-implementations)
12. [Best Practices](#best-practices)

## Overview

Strategies in Expandor define how images are expanded from their original resolution to the target resolution. Each strategy implements a different approach optimized for specific scenarios, hardware constraints, or quality requirements.

### Core Strategy Types

1. **Direct Upscale**: Simple one-step expansion
2. **Progressive Outpaint**: Multi-step gradual expansion
3. **SWPO**: Sliding Window Progressive Outpaint for extreme ratios
4. **Tiled**: Memory-efficient processing in tiles
5. **Hybrid**: Adaptive combination of strategies

## Strategy Architecture

```
┌─────────────────┐
│ StrategySelector│
└────────┬────────┘
         │ Selects
┌────────▼────────┐
│  BaseStrategy   │ (Abstract Base)
└────────┬────────┘
         │
    ┌────┴────┬────────┬──────────┬────────┐
    │         │        │          │        │
┌───▼───┐ ┌──▼───┐ ┌──▼───┐ ┌───▼───┐ ┌──▼───┐
│Direct  │ │Progr │ │SWPO  │ │Tiled │ │Custom│
│Strategy│ │essive│ │Strategy│ │Strategy│ │Strategy│
└────────┘ └──────┘ └───────┘ └────────┘ └───────┘
```

## Base Strategy Interface

### Abstract Base Class

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
from dataclasses import dataclass

@dataclass
class StrategyResult:
    """Result from strategy execution"""
    success: bool
    final_image: Optional[Image.Image]
    stages: List[Image.Image]
    metadata: Dict[str, Any]
    error: Optional[str] = None

class BaseStrategy(ABC):
    """Base class for all expansion strategies"""
    
    def __init__(self, pipeline_adapter, config):
        self.pipeline_adapter = pipeline_adapter
        self.config = config
        self.logger = self._setup_logger()
        
    @abstractmethod
    def can_handle(
        self,
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
        expansion_factor: float,
        aspect_ratio_change: float
    ) -> bool:
        """Check if strategy can handle the expansion"""
        pass
        
    @abstractmethod
    def estimate_vram(
        self,
        source_size: Tuple[int, int],
        target_size: Tuple[int, int]
    ) -> float:
        """Estimate VRAM usage in MB"""
        pass
        
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
        """Execute the expansion strategy"""
        pass
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name"""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """Strategy description"""
        pass
```

## Creating Custom Strategies

### Step 1: Define Your Strategy Class

```python
from expandor.strategies.base_strategy import BaseStrategy, StrategyResult
from expandor.utils.dimension_calculator import DimensionCalculator
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import numpy as np

class MyCustomStrategy(BaseStrategy):
    """Custom strategy implementation"""
    
    def __init__(self, pipeline_adapter, config):
        super().__init__(pipeline_adapter, config)
        self.dimension_calculator = DimensionCalculator(
            model_type=config.model_type
        )
        self.max_expansion_factor = 4.0
        self.stages = []
        
    @property
    def name(self) -> str:
        return "custom"
        
    @property
    def description(self) -> str:
        return "Custom expansion strategy with special optimizations"
```

### Step 2: Implement Capability Check

```python
def can_handle(
    self,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
    expansion_factor: float,
    aspect_ratio_change: float
) -> bool:
    """Determine if strategy can handle this expansion"""
    
    # Check expansion factor
    if expansion_factor > self.max_expansion_factor:
        self.logger.debug(
            f"Expansion factor {expansion_factor:.2f} exceeds maximum "
            f"{self.max_expansion_factor}"
        )
        return False
        
    # Check aspect ratio change
    if aspect_ratio_change > 2.0:
        self.logger.debug(
            f"Aspect ratio change {aspect_ratio_change:.2f} too extreme"
        )
        return False
        
    # Check minimum size requirements
    min_dimension = min(source_size[0], source_size[1])
    if min_dimension < 256:
        self.logger.debug(f"Source too small: {min_dimension}px")
        return False
        
    return True
```

### Step 3: Implement VRAM Estimation

```python
def estimate_vram(
    self,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int]
) -> float:
    """Estimate VRAM usage for this expansion"""
    
    # Base VRAM for model
    base_vram = {
        'sd15': 4000,
        'sdxl': 10000,
        'flux': 24000
    }.get(self.config.model_type, 8000)
    
    # Calculate based on largest intermediate size
    max_pixels = max(
        source_size[0] * source_size[1],
        target_size[0] * target_size[1]
    )
    
    # Add overhead for intermediate operations
    # ~4 bytes per pixel for latents, x2 for processing
    latent_vram = (max_pixels * 4 * 2) / (1024 * 1024)
    
    # Add safety margin
    total_vram = (base_vram + latent_vram) * 1.2
    
    self.logger.debug(
        f"Estimated VRAM: {total_vram:.0f}MB "
        f"(base: {base_vram}MB, latents: {latent_vram:.0f}MB)"
    )
    
    return total_vram
```

### Step 4: Implement Core Expansion Logic

```python
def expand(
    self,
    image: Image.Image,
    target_size: Tuple[int, int],
    prompt: str,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs
) -> StrategyResult:
    """Execute custom expansion strategy"""
    
    try:
        self.logger.info(
            f"Starting custom expansion: {image.size} → {target_size}"
        )
        
        # Initialize tracking
        self.stages = []
        metadata = {
            'strategy': self.name,
            'source_size': image.size,
            'target_size': target_size,
            'steps': []
        }
        
        # Save initial image
        self.stages.append(image.copy())
        
        # Implement your custom expansion logic
        result_image = self._custom_expansion_logic(
            image,
            target_size,
            prompt,
            negative_prompt,
            seed,
            metadata
        )
        
        # Save final result
        self.stages.append(result_image)
        
        return StrategyResult(
            success=True,
            final_image=result_image,
            stages=self.stages,
            metadata=metadata
        )
        
    except Exception as e:
        self.logger.error(f"Custom expansion failed: {str(e)}")
        return StrategyResult(
            success=False,
            final_image=None,
            stages=self.stages,
            metadata={'error': str(e)},
            error=str(e)
        )

def _custom_expansion_logic(
    self,
    image: Image.Image,
    target_size: Tuple[int, int],
    prompt: str,
    negative_prompt: Optional[str],
    seed: Optional[int],
    metadata: Dict[str, Any]
) -> Image.Image:
    """Core custom expansion implementation"""
    
    current_image = image
    current_size = image.size
    
    # Example: Multi-step expansion with custom logic
    steps = self._calculate_expansion_steps(current_size, target_size)
    
    for i, step_size in enumerate(steps):
        self.logger.info(f"Step {i+1}/{len(steps)}: {current_size} → {step_size}")
        
        # Your custom processing here
        current_image = self._process_step(
            current_image,
            step_size,
            prompt,
            negative_prompt,
            seed,
            step_num=i
        )
        
        current_size = step_size
        self.stages.append(current_image.copy())
        
        # Track metadata
        metadata['steps'].append({
            'step': i + 1,
            'size': step_size,
            'method': 'custom_process'
        })
    
    return current_image
```

## Strategy Lifecycle

### 1. Selection Phase

```python
class StrategySelector:
    """Selects appropriate strategy based on requirements"""
    
    def select_strategy(
        self,
        current_size: Tuple[int, int],
        target_size: Tuple[int, int],
        available_vram: float,
        user_preference: Optional[str] = None
    ) -> Tuple[BaseStrategy, str]:
        """Select best strategy"""
        
        # Calculate metrics
        expansion_factor = self._calculate_expansion_factor(
            current_size, target_size
        )
        aspect_ratio_change = self._calculate_aspect_ratio_change(
            current_size, target_size
        )
        
        # User preference takes precedence
        if user_preference and user_preference != 'auto':
            strategy = self._get_strategy(user_preference)
            if strategy.can_handle(
                current_size, target_size,
                expansion_factor, aspect_ratio_change
            ):
                return strategy, f"User requested: {user_preference}"
                
        # Auto selection logic
        for strategy_name, strategy in self.strategies.items():
            if strategy.can_handle(
                current_size, target_size,
                expansion_factor, aspect_ratio_change
            ):
                vram_needed = strategy.estimate_vram(current_size, target_size)
                if vram_needed <= available_vram:
                    return strategy, f"Auto-selected based on requirements"
                    
        # Fallback
        return self.strategies['direct'], "Fallback to direct strategy"
```

### 2. Preparation Phase

```python
def prepare_expansion(
    self,
    image: Image.Image,
    target_size: Tuple[int, int]
) -> Dict[str, Any]:
    """Prepare for expansion"""
    
    # Validate inputs
    self._validate_inputs(image, target_size)
    
    # Calculate steps
    steps = self._plan_expansion_steps(image.size, target_size)
    
    # Prepare prompts
    prompts = self._prepare_prompts(
        self.config.prompt,
        self.config.negative_prompt,
        steps
    )
    
    # Initialize boundaries tracking
    boundaries = []
    
    return {
        'steps': steps,
        'prompts': prompts,
        'boundaries': boundaries,
        'original_size': image.size
    }
```

### 3. Execution Phase

```python
def execute_expansion(
    self,
    image: Image.Image,
    preparation: Dict[str, Any]
) -> StrategyResult:
    """Execute the prepared expansion"""
    
    current_image = image
    stages = [image.copy()]
    
    for i, (step_size, prompt) in enumerate(
        zip(preparation['steps'], preparation['prompts'])
    ):
        # Execute step
        result = self._execute_single_step(
            current_image,
            step_size,
            prompt,
            step_num=i
        )
        
        # Validate result
        if not self._validate_step_result(result, step_size):
            raise RuntimeError(f"Step {i+1} validation failed")
            
        current_image = result
        stages.append(result.copy())
        
        # Update boundaries
        self._update_boundaries(
            preparation['boundaries'],
            step_size,
            i
        )
    
    return StrategyResult(
        success=True,
        final_image=current_image,
        stages=stages,
        metadata=preparation
    )
```

## Key Components

### Dimension Calculator Integration

```python
from expandor.utils.dimension_calculator import DimensionCalculator

class DimensionAwareStrategy(BaseStrategy):
    """Strategy with dimension calculation"""
    
    def __init__(self, pipeline_adapter, config):
        super().__init__(pipeline_adapter, config)
        self.dim_calc = DimensionCalculator(
            model_type=config.model_type
        )
        
    def _calculate_optimal_dimensions(
        self,
        current_size: Tuple[int, int],
        target_size: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Calculate optimal intermediate dimensions"""
        
        # Ensure dimensions are valid for model
        validated_target = self.dim_calc.adjust_dimensions(
            target_size[0],
            target_size[1]
        )
        
        # Calculate intermediate steps
        if self._needs_progressive_expansion(current_size, validated_target):
            steps = self.dim_calc.calculate_progressive_dimensions(
                current_size,
                validated_target
            )
        else:
            steps = [validated_target]
            
        return steps
```

### Mask Generation

```python
def _generate_expansion_mask(
    self,
    current_size: Tuple[int, int],
    target_size: Tuple[int, int],
    direction: str = 'all'
) -> Image.Image:
    """Generate mask for expansion areas"""
    
    mask = Image.new('L', target_size, 0)
    
    # Calculate offsets for centering
    offset_x = (target_size[0] - current_size[0]) // 2
    offset_y = (target_size[1] - current_size[1]) // 2
    
    # White = areas to generate
    # Black = areas to preserve
    
    if direction == 'all':
        # Mark center as preserve (black)
        mask.paste(
            Image.new('L', current_size, 0),
            (offset_x, offset_y)
        )
        # Everything else is white (generate)
        mask = Image.new('L', target_size, 255)
        mask.paste(
            Image.new('L', current_size, 0),
            (offset_x, offset_y)
        )
    elif direction == 'horizontal':
        # Expand only horizontally
        mask = Image.new('L', target_size, 0)
        # Left side
        mask.paste(Image.new('L', (offset_x, target_size[1]), 255), (0, 0))
        # Right side
        mask.paste(
            Image.new('L', (offset_x, target_size[1]), 255),
            (target_size[0] - offset_x, 0)
        )
    
    # Apply blur for smooth transitions
    mask = self._apply_mask_blur(mask, current_size, target_size)
    
    return mask
```

### Prompt Enhancement

```python
def _enhance_prompt_for_expansion(
    self,
    base_prompt: str,
    expansion_type: str,
    step_num: int
) -> str:
    """Enhance prompt for better expansion results"""
    
    # Add expansion-specific tokens
    expansion_tokens = {
        'outpaint': 'extending scenery, seamless continuation',
        'upscale': 'high detail, enhanced quality',
        'extreme_ratio': 'panoramic view, wide composition'
    }
    
    tokens = expansion_tokens.get(expansion_type, '')
    
    # Add quality tokens for later steps
    if step_num > 0:
        tokens += ', highly detailed, professional'
        
    # Combine with base prompt
    if tokens:
        enhanced = f"{base_prompt}, {tokens}"
    else:
        enhanced = base_prompt
        
    return enhanced
```

## VRAM Management

### Dynamic VRAM Monitoring

```python
import torch
import gc

class VRAMAwareStrategy(BaseStrategy):
    """Strategy with active VRAM management"""
    
    def _monitor_vram(self) -> Dict[str, float]:
        """Get current VRAM status"""
        if not torch.cuda.is_available():
            return {'available': float('inf'), 'used': 0, 'total': 0}
            
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Get VRAM info
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        used = torch.cuda.memory_allocated() / (1024**2)
        available = total - used
        
        return {
            'available': available,
            'used': used,
            'total': total,
            'percentage': (used / total) * 100
        }
        
    def _ensure_vram_available(self, required_mb: float):
        """Ensure enough VRAM is available"""
        status = self._monitor_vram()
        
        if status['available'] < required_mb:
            self.logger.warning(
                f"Low VRAM: {status['available']:.0f}MB available, "
                f"{required_mb:.0f}MB required"
            )
            
            # Try to free memory
            self._free_vram()
            
            # Check again
            status = self._monitor_vram()
            if status['available'] < required_mb:
                raise RuntimeError(
                    f"Insufficient VRAM: need {required_mb:.0f}MB, "
                    f"have {status['available']:.0f}MB"
                )
```

### Adaptive Processing

```python
def _adaptive_processing(
    self,
    image: Image.Image,
    target_size: Tuple[int, int],
    vram_limit: float
) -> Image.Image:
    """Adapt processing based on available VRAM"""
    
    # Estimate VRAM for full processing
    full_vram = self.estimate_vram(image.size, target_size)
    
    if full_vram <= vram_limit:
        # Process normally
        return self._process_full(image, target_size)
    else:
        # Fall back to memory-efficient mode
        self.logger.info(
            f"Switching to memory-efficient mode "
            f"(need {full_vram:.0f}MB, have {vram_limit:.0f}MB)"
        )
        
        # Option 1: Reduce batch size
        if hasattr(self.config, 'batch_size'):
            self.config.batch_size = 1
            
        # Option 2: Enable CPU offload
        if hasattr(self.pipeline_adapter, 'enable_cpu_offload'):
            self.pipeline_adapter.enable_cpu_offload()
            
        # Option 3: Process in smaller chunks
        return self._process_chunked(image, target_size)
```

## Boundary Tracking

### Comprehensive Boundary Management

```python
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

class BoundaryType(Enum):
    PROGRESSIVE = "progressive"
    WINDOW = "window"
    TILE = "tile"
    MASK = "mask"

@dataclass
class Boundary:
    """Represents an expansion boundary"""
    position: Tuple[int, int, int, int]  # x1, y1, x2, y2
    boundary_type: BoundaryType
    step_number: int
    strength: float
    direction: str  # 'horizontal', 'vertical', 'all'

class BoundaryTracker:
    """Tracks all boundaries during expansion"""
    
    def __init__(self):
        self.boundaries: List[Boundary] = []
        
    def add_progressive_boundary(
        self,
        old_size: Tuple[int, int],
        new_size: Tuple[int, int],
        step: int
    ):
        """Add boundary from progressive expansion"""
        
        # Calculate boundary positions
        offset_x = (new_size[0] - old_size[0]) // 2
        offset_y = (new_size[1] - old_size[1]) // 2
        
        # Add boundaries for each edge
        if offset_x > 0:
            # Left boundary
            self.boundaries.append(Boundary(
                position=(offset_x - 10, 0, offset_x + 10, new_size[1]),
                boundary_type=BoundaryType.PROGRESSIVE,
                step_number=step,
                strength=0.9,
                direction='vertical'
            ))
            # Right boundary
            self.boundaries.append(Boundary(
                position=(
                    new_size[0] - offset_x - 10, 0,
                    new_size[0] - offset_x + 10, new_size[1]
                ),
                boundary_type=BoundaryType.PROGRESSIVE,
                step_number=step,
                strength=0.9,
                direction='vertical'
            ))
            
    def get_boundaries_for_detection(self) -> List[Dict[str, Any]]:
        """Get boundaries formatted for artifact detection"""
        return [
            {
                'bbox': b.position,
                'type': b.boundary_type.value,
                'strength': b.strength,
                'priority': 10 - b.step_number  # Earlier boundaries = higher priority
            }
            for b in self.boundaries
        ]
```

## Quality Assurance

### Artifact Detection Integration

```python
from expandor.quality.artifact_detector import SmartArtifactDetector
from expandor.quality.quality_refiner import SmartQualityRefiner

class QualityAwareStrategy(BaseStrategy):
    """Strategy with integrated quality assurance"""
    
    def __init__(self, pipeline_adapter, config):
        super().__init__(pipeline_adapter, config)
        self.artifact_detector = SmartArtifactDetector(config)
        self.quality_refiner = SmartQualityRefiner(pipeline_adapter, config)
        self.boundary_tracker = BoundaryTracker()
        
    def _post_process_with_qa(
        self,
        image: Image.Image,
        metadata: Dict[str, Any]
    ) -> Image.Image:
        """Apply quality assurance post-processing"""
        
        if not self.config.enable_artifacts_check:
            return image
            
        # Detect artifacts
        artifacts = self.artifact_detector.detect_all_artifacts(
            image,
            boundaries=self.boundary_tracker.get_boundaries_for_detection()
        )
        
        if not artifacts['has_artifacts']:
            self.logger.info("No artifacts detected")
            return image
            
        # Refine detected artifacts
        self.logger.info(
            f"Detected {len(artifacts['locations'])} artifacts, refining..."
        )
        
        refined = self.quality_refiner.refine_artifacts(
            image,
            artifacts,
            prompt=metadata.get('prompt', ''),
            strength=0.3
        )
        
        return refined
```

### Validation Framework

```python
def _validate_expansion_result(
    self,
    result: Image.Image,
    target_size: Tuple[int, int],
    source_image: Image.Image
) -> bool:
    """Validate expansion result"""
    
    # Check dimensions
    if result.size != target_size:
        self.logger.error(
            f"Size mismatch: expected {target_size}, got {result.size}"
        )
        return False
        
    # Check image integrity
    if result.mode != source_image.mode:
        self.logger.warning(
            f"Mode changed: {source_image.mode} → {result.mode}"
        )
        
    # Check for completely black/white areas (common failure)
    arr = np.array(result)
    if np.all(arr == 0) or np.all(arr == 255):
        self.logger.error("Result is completely black or white")
        return False
        
    # Check for NaN or infinity values
    if result.mode == 'F':  # Float image
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            self.logger.error("Result contains NaN or infinity values")
            return False
            
    return True
```

## Testing Strategies

### Unit Testing Template

```python
import pytest
from PIL import Image
import numpy as np
from expandor.strategies import MyCustomStrategy
from expandor.adapters import MockPipelineAdapter

class TestMyCustomStrategy:
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance"""
        adapter = MockPipelineAdapter()
        config = ExpandorConfig(
            target_width=2048,
            target_height=1536,
            model_type='sdxl'
        )
        return MyCustomStrategy(adapter, config)
        
    @pytest.fixture
    def test_image(self):
        """Create test image"""
        return Image.new('RGB', (1024, 768), color='blue')
        
    def test_can_handle(self, strategy):
        """Test capability checking"""
        # Should handle moderate expansion
        assert strategy.can_handle(
            (1024, 768), (2048, 1536), 2.0, 0.0
        )
        
        # Should reject extreme expansion
        assert not strategy.can_handle(
            (512, 512), (8192, 8192), 16.0, 0.0
        )
        
    def test_vram_estimation(self, strategy):
        """Test VRAM estimation"""
        vram = strategy.estimate_vram((1024, 768), (2048, 1536))
        
        # Should be reasonable
        assert 1000 < vram < 20000  # Between 1GB and 20GB
        
        # Should scale with size
        vram_large = strategy.estimate_vram((1024, 768), (4096, 3072))
        assert vram_large > vram
        
    def test_expansion_success(self, strategy, test_image):
        """Test successful expansion"""
        result = strategy.expand(
            test_image,
            (2048, 1536),
            "test prompt",
            seed=42
        )
        
        assert result.success
        assert result.final_image is not None
        assert result.final_image.size == (2048, 1536)
        assert len(result.stages) >= 2  # At least input and output
```

### Integration Testing

```python
def test_strategy_with_quality_pipeline(strategy, test_image):
    """Test strategy with full quality pipeline"""
    
    # Enable all quality features
    strategy.config.enable_artifacts_check = True
    strategy.config.save_stages = True
    
    # Run expansion
    result = strategy.expand(
        test_image,
        (3840, 2160),
        "beautiful landscape with mountains",
        negative_prompt="blurry, artifacts"
    )
    
    assert result.success
    
    # Check metadata
    assert 'boundaries' in result.metadata
    assert 'artifact_detection' in result.metadata
    assert 'quality_metrics' in result.metadata
    
    # Check stages were saved
    assert len(result.stages) > 2
```

### Performance Testing

```python
import time

def test_strategy_performance(strategy, test_image):
    """Test strategy performance"""
    
    sizes = [
        (1920, 1080),   # HD
        (2560, 1440),   # 2K
        (3840, 2160),   # 4K
    ]
    
    for target_size in sizes:
        start_time = time.time()
        
        result = strategy.expand(
            test_image,
            target_size,
            "test"
        )
        
        elapsed = time.time() - start_time
        
        print(f"{test_image.size} → {target_size}: {elapsed:.2f}s")
        
        # Should complete in reasonable time
        assert elapsed < 300  # 5 minutes max
        assert result.success
```

## Example Implementations

### Intelligent Progressive Strategy

```python
class IntelligentProgressiveStrategy(BaseStrategy):
    """Progressive strategy with adaptive steps"""
    
    def expand(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
        prompt: str,
        **kwargs
    ) -> StrategyResult:
        
        # Calculate intelligent steps based on content
        steps = self._calculate_content_aware_steps(
            image,
            target_size
        )
        
        current = image
        stages = [image]
        
        for i, (size, strength) in enumerate(steps):
            # Adapt prompt based on progress
            step_prompt = self._adapt_prompt(prompt, i, len(steps))
            
            # Expand with adaptive strength
            current = self._expand_step(
                current,
                size,
                step_prompt,
                strength=strength
            )
            
            stages.append(current)
            
        return StrategyResult(
            success=True,
            final_image=current,
            stages=stages,
            metadata={'steps': steps}
        )
        
    def _calculate_content_aware_steps(
        self,
        image: Image.Image,
        target: Tuple[int, int]
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Calculate steps based on image content"""
        
        # Analyze image complexity
        complexity = self._analyze_complexity(image)
        
        if complexity > 0.7:
            # Complex image: more gradual steps
            return self._calculate_gradual_steps(image.size, target)
        else:
            # Simple image: fewer steps
            return self._calculate_aggressive_steps(image.size, target)
```

### Memory-Efficient Tiled Strategy

```python
class MemoryEfficientTiledStrategy(BaseStrategy):
    """Process large images in tiles with minimal memory"""
    
    def expand(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
        prompt: str,
        **kwargs
    ) -> StrategyResult:
        
        # Calculate optimal tile size based on available memory
        tile_size = self._calculate_optimal_tile_size(
            target_size,
            self._get_available_memory()
        )
        
        # Process in tiles
        result = Image.new(image.mode, target_size)
        tiles_processed = 0
        
        for tile_info in self._generate_tiles(target_size, tile_size):
            # Extract tile from source
            source_tile = self._extract_source_tile(
                image,
                tile_info,
                target_size
            )
            
            # Process tile
            processed_tile = self._process_tile(
                source_tile,
                tile_info,
                prompt
            )
            
            # Merge into result
            self._merge_tile(result, processed_tile, tile_info)
            
            tiles_processed += 1
            
            # Clear memory periodically
            if tiles_processed % 4 == 0:
                self._clear_memory()
                
        return StrategyResult(
            success=True,
            final_image=result,
            stages=[image, result],
            metadata={'tiles': tiles_processed, 'tile_size': tile_size}
        )
```

## Best Practices

### 1. Robust Error Handling

```python
def expand(self, image: Image.Image, target_size: Tuple[int, int], **kwargs):
    """Expansion with comprehensive error handling"""
    
    try:
        # Validate inputs early
        self._validate_inputs(image, target_size)
        
        # Monitor resources
        if not self._check_resources():
            raise RuntimeError("Insufficient resources")
            
        # Process with recovery
        result = None
        attempts = 0
        max_attempts = 3
        
        while result is None and attempts < max_attempts:
            try:
                result = self._attempt_expansion(image, target_size, **kwargs)
            except torch.cuda.OutOfMemoryError:
                self.logger.warning(f"OOM on attempt {attempts + 1}, recovering...")
                self._recover_from_oom()
                attempts += 1
                
        if result is None:
            raise RuntimeError(f"Failed after {max_attempts} attempts")
            
        return result
        
    except Exception as e:
        # Always return StrategyResult, even on failure
        return StrategyResult(
            success=False,
            final_image=None,
            stages=self.stages,
            metadata={'error': str(e)},
            error=str(e)
        )
```

### 2. Progress Tracking

```python
from typing import Callable

class ProgressTrackingStrategy(BaseStrategy):
    """Strategy with progress callbacks"""
    
    def expand(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
        prompt: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs
    ) -> StrategyResult:
        
        steps = self._plan_steps(image.size, target_size)
        total_steps = len(steps)
        
        for i, step in enumerate(steps):
            if progress_callback:
                progress = (i / total_steps) * 100
                progress_callback(progress, f"Processing step {i+1}/{total_steps}")
                
            # Process step
            result = self._process_step(step)
            
        if progress_callback:
            progress_callback(100, "Complete")
            
        return result
```

### 3. Metadata Rich Results

```python
def _create_detailed_metadata(
    self,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
    processing_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Create comprehensive metadata"""
    
    return {
        'strategy': {
            'name': self.name,
            'version': self.version,
            'description': self.description
        },
        'dimensions': {
            'source': source_size,
            'target': target_size,
            'expansion_factor': (
                target_size[0] * target_size[1]
            ) / (source_size[0] * source_size[1]),
            'aspect_ratio_change': abs(
                (target_size[0] / target_size[1]) -
                (source_size[0] / source_size[1])
            )
        },
        'processing': processing_info,
        'quality': {
            'boundaries_tracked': len(self.boundary_tracker.boundaries),
            'artifacts_detected': processing_info.get('artifacts_found', 0),
            'refinement_applied': processing_info.get('refined', False)
        },
        'performance': {
            'total_time': processing_info.get('elapsed_time', 0),
            'vram_peak': processing_info.get('peak_vram', 0),
            'steps': processing_info.get('total_steps', 0)
        }
    }
```

### 4. Strategy Registration

```python
# In expandor/strategies/__init__.py
from .base_strategy import BaseStrategy
from .direct_upscale import DirectUpscaleStrategy
from .progressive_outpaint import ProgressiveOutpaintStrategy
from .my_custom_strategy import MyCustomStrategy

# Strategy registry
STRATEGY_REGISTRY = {
    'direct': DirectUpscaleStrategy,
    'progressive': ProgressiveOutpaintStrategy,
    'custom': MyCustomStrategy,
}

def register_strategy(name: str, strategy_class: type):
    """Register a custom strategy"""
    if not issubclass(strategy_class, BaseStrategy):
        raise ValueError(f"{strategy_class} must inherit from BaseStrategy")
        
    STRATEGY_REGISTRY[name] = strategy_class
```

## Advanced Topics

### Hybrid Strategies

```python
class HybridAdaptiveStrategy(BaseStrategy):
    """Combines multiple strategies adaptively"""
    
    def __init__(self, pipeline_adapter, config):
        super().__init__(pipeline_adapter, config)
        
        # Initialize sub-strategies
        self.strategies = {
            'direct': DirectUpscaleStrategy(pipeline_adapter, config),
            'progressive': ProgressiveOutpaintStrategy(pipeline_adapter, config),
            'tiled': TiledStrategy(pipeline_adapter, config)
        }
        
    def expand(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
        **kwargs
    ) -> StrategyResult:
        
        # Analyze requirements
        analysis = self._analyze_expansion(image, target_size)
        
        # Choose strategy per region
        if analysis['needs_hybrid']:
            return self._hybrid_expansion(image, target_size, analysis, **kwargs)
        else:
            # Delegate to single strategy
            chosen = self.strategies[analysis['best_strategy']]
            return chosen.expand(image, target_size, **kwargs)
```

### Machine Learning Integration

```python
class MLGuidedStrategy(BaseStrategy):
    """Strategy guided by ML predictions"""
    
    def __init__(self, pipeline_adapter, config):
        super().__init__(pipeline_adapter, config)
        self.quality_predictor = self._load_quality_model()
        
    def _predict_optimal_parameters(
        self,
        image: Image.Image,
        target_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Use ML to predict optimal parameters"""
        
        features = self._extract_features(image, target_size)
        predictions = self.quality_predictor.predict(features)
        
        return {
            'strength': float(predictions['optimal_strength']),
            'steps': int(predictions['optimal_steps']),
            'strategy': predictions['recommended_strategy']
        }
```

## See Also

- [Base Strategy Source](../expandor/strategies/base_strategy.py)
- [Example Strategies](../expandor/strategies/)
- [ADAPTER_DEVELOPMENT.md](ADAPTER_DEVELOPMENT.md) - Creating pipeline adapters
- [Quality Systems](../expandor/quality/) - Artifact detection and refinement
- [API Documentation](API.md) - Full API reference