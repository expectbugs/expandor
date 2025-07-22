# Expandor - Universal Image Resolution Adaptation System

## Executive Summary

Expandor is a standalone, model-agnostic image resolution and aspect ratio adaptation system that will be developed independently and later integrated into the AI Wallpaper project. It extracts and unifies all pipeline logic currently embedded in individual model implementations, providing intelligent strategy selection, seamless integration with any image generation model, and maintaining the project's core philosophy of "quality over all" with no silent failures.

## Development Strategy

### Standalone Development Approach

To ensure zero risk to the working AI Wallpaper project, Expandor will be developed as a completely independent system:

1. **Separate Repository Structure**
   ```
   expandor/                    # New separate repository
   ├── README.md
   ├── LICENSE
   ├── setup.py
   ├── requirements.txt
   ├── tests/
   │   ├── fixtures/           # Test images and data
   │   ├── unit/              # Unit tests
   │   ├── integration/       # Integration tests
   │   └── performance/       # Performance benchmarks
   ├── examples/
   │   ├── basic_usage.py
   │   ├── custom_strategy.py
   │   └── test_images/
   ├── docs/
   │   ├── api/
   │   ├── strategies/
   │   └── integration_guide.md
   └── expandor/              # Main package
       └── [source code structure below]
   ```

2. **Development Isolation**
   - Copy necessary components from ai-wallpaper (with attribution)
   - No direct dependencies on ai-wallpaper codebase
   - Mock interfaces for testing without real models
   - Comprehensive test suite with synthetic data

3. **Integration Path**
   - Develop adapters that match ai-wallpaper's interfaces
   - Test with extracted ai-wallpaper components
   - Package as pip-installable module
   - Final integration via dependency or git submodule

## Current State Analysis (Updated)

### Existing Components in AI Wallpaper

1. **SDXL Pipeline** (most sophisticated)
   - Progressive outpainting with SWPO (Sliding Window Progressive Outpainting)
   - VRAM-aware multi-strategy refinement (full/tiled/CPU offload)
   - Smart multi-pass refinement with artifact detection
   - Real-ESRGAN upscaling
   - Aspect adjustment before refinement
   - Tracks seam positions in `generation_metadata` for artifact detection
   - Multiple refinement passes based on severity (up to 5 passes)
   - Progressive expansion ratios already optimized (2.0 → 1.5 → 1.3)

2. **FLUX Pipeline** (simplest)
   - Fixed generation at 1920x1088
   - 4x upscale to 8K
   - Downsample to 4K

3. **DALL-E/GPT-Image Pipelines**
   - API generation at 1024x1024
   - Center crop to 16:9
   - 4x upscale
   - Downsample to 4K

4. **Critical Components to Extract**
   - `VRAMCalculator`: Determines refinement strategies based on available VRAM
   - `AspectAdjuster`: Progressive outpainting and SWPO implementation
   - `SmartQualityRefiner`: Multi-pass refinement with artifact detection
   - Artifact detection logic from `SmartArtifactDetector` (to be reimplemented)
   - `RealESRGANUpscaler`: High-quality upscaling
   - `HighQualityDownsampler`: Lanczos downsampling
   - `TiledRefiner`: Large image refinement for VRAM-limited situations
   - `ResolutionManager`: Sophisticated calculations with model-specific constraints
   - Generation metadata tracking system for boundary positions

## Expandor Architecture Design (Updated)

### Core Philosophy
- **Quality Over All**: No compromises, no fallbacks
- **Fail Loud**: All errors are explicit and informative
- **Model Agnostic**: Works with any image source
- **VRAM Aware**: Intelligent resource management
- **Boundary Tracking**: Zero tolerance for visible seams
- **Extensible**: Easy to add new methods

### System Components

```python
expandor/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── expandor.py              # Main entry point
│   ├── strategy_selector.py     # VRAM-aware strategy selection
│   ├── pipeline_orchestrator.py # Executes selected strategies
│   ├── metadata_tracker.py      # Tracks boundaries and operations
│   ├── vram_manager.py          # VRAM calculation and management
│   ├── boundary_tracker.py      # Track expansion seams
│   └── constraint_validator.py  # Model constraints (8x/16x multiples)
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py         # VRAM-aware base class
│   ├── progressive_outpaint.py  # Progressive expansion
│   ├── swpo_strategy.py         # Sliding Window Progressive Outpaint
│   ├── direct_upscale.py        # Real-ESRGAN only
│   ├── tiled_expansion.py       # VRAM-limited processing
│   ├── cpu_offload.py          # Last resort strategy
│   ├── hybrid_adaptive.py       # Intelligent combination
│   └── experimental/
│       └── coherent_tiles.py    # SD3.5 coherent tile generation
├── processors/
│   ├── __init__.py
│   ├── refinement/
│   │   ├── multi_pass.py        # Smart quality refinement
│   │   ├── tiled_refiner.py    # Tiled refinement
│   │   └── seam_repair.py      # Automatic seam fixing
│   ├── artifact_removal.py     # Aggressive artifact detection
│   ├── upscaling.py            # Real-ESRGAN wrapper
│   ├── downsampling.py         # High-quality downsampling
│   ├── edge_analysis.py        # Edge color analysis
│   ├── mask_generator.py       # Intelligent mask creation
│   └── cpu_offload.py         # CPU-based processing
├── utils/
│   ├── __init__.py
│   ├── dimension_calculator.py  # Resolution calculations
│   ├── vram_estimator.py       # VRAM usage prediction
│   ├── quality_validator.py    # Output validation
│   ├── model_constraints.py    # Model-specific requirements
│   └── image_io.py            # Lossless save/load
├── adapters/
│   ├── __init__.py
│   ├── ai_wallpaper.py        # Adapter for ai-wallpaper integration
│   ├── diffusers.py           # Direct diffusers pipeline support
│   └── mock_pipeline.py       # For testing without real models
└── config/
    ├── __init__.py
    ├── strategies.yaml          # Strategy configurations
    ├── quality_presets.yaml     # Quality level definitions
    ├── vram_strategies.yaml     # VRAM thresholds and fallbacks
    ├── model_constraints.yaml   # Model-specific requirements
    └── artifact_detection.yaml  # Seam detection settings
```

### Core API Design (Enhanced)

```python
from dataclasses import dataclass
from typing import Union, Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path
from PIL import Image

@dataclass
class ExpandorConfig:
    """Comprehensive configuration for expansion operation"""
    # Core inputs
    source_image: Union[Path, Image.Image]
    target_resolution: Tuple[int, int]
    prompt: str
    seed: int
    
    # Source information
    source_metadata: Dict[str, Any]  # Model, generation size, etc.
    generation_metadata: Optional[Dict] = None  # Existing boundaries, etc.
    
    # Pipeline access (optional - strategies work without them too)
    inpaint_pipeline: Optional[Any] = None
    refiner_pipeline: Optional[Any] = None
    img2img_pipeline: Optional[Any] = None
    
    # Quality & strategy
    quality_preset: str = "ultra"
    strategy_override: Optional[str] = None
    
    # VRAM management
    vram_limit_mb: Optional[float] = None
    allow_cpu_offload: bool = True
    allow_tiled: bool = True
    
    # Progressive/SWPO parameters
    window_size: int = 200
    overlap_ratio: float = 0.8
    denoising_strength: float = 0.95
    min_strength: float = 0.20
    max_strength: float = 0.95
    
    # Refinement parameters
    refinement_passes: Optional[int] = None  # Auto-determine if None
    artifact_detection_level: str = "aggressive"
    
    # Tracking and debugging
    save_stages: bool = False
    stage_dir: Optional[Path] = None
    stage_save_callback: Optional[Callable] = None
    verbose: bool = False

@dataclass
class StageResult:
    """Result from a single processing stage"""
    name: str
    method: str
    input_size: Tuple[int, int]
    output_size: Tuple[int, int]
    duration_seconds: float
    vram_used_mb: float
    artifacts_detected: int = 0
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ExpandorResult:
    """Comprehensive result from expansion operation"""
    # Core results
    image_path: Path
    size: Tuple[int, int]
    success: bool = True
    
    # Stage tracking
    stages: List[StageResult]
    boundaries: List[Dict]  # Seam positions for detection
    
    # Quality metrics
    seams_detected: int = 0
    artifacts_fixed: int = 0
    refinement_passes: int = 0
    quality_score: float = 1.0
    
    # Resource usage
    vram_peak_mb: float
    total_duration_seconds: float
    strategy_used: str
    fallback_count: int = 0
    
    # Full metadata (includes generation_metadata updates)
    metadata: Dict[str, Any]
    
    # Error information (if success=False)
    error: Optional[Exception] = None
    error_stage: Optional[str] = None

class Expandor:
    """Universal image expansion and adaptation system"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize Expandor with optional config override"""
        self.config = self._load_config(config_path)
        self.strategy_selector = StrategySelector(self.config)
        self.orchestrator = PipelineOrchestrator(self.config)
        self.metadata_tracker = MetadataTracker()
        self.vram_manager = VRAMManager()
        self.pipeline_registry = {}
        
    def expand(self, config: ExpandorConfig) -> ExpandorResult:
        """
        Main expansion method with comprehensive error handling
        
        Args:
            config: Expansion configuration
            
        Returns:
            ExpandorResult with image path, metadata, and metrics
            
        Raises:
            ExpandorError: On unrecoverable errors (fail loud)
        """
        # Input validation
        self._validate_config(config)
        
        # Pre-execution setup
        self.metadata_tracker.start_operation(config)
        
        try:
            # Select strategy with VRAM awareness
            strategy = self.strategy_selector.select(config)
            
            # Execute with fallback chain
            result = self.orchestrator.execute(strategy, config)
            
            # Post-execution validation
            result = self._validate_and_repair(result, config)
            
            return result
            
        except Exception as e:
            # Fail loud with comprehensive error info
            raise ExpandorError(
                f"Expansion failed at {self.metadata_tracker.current_stage}: {str(e)}",
                stage=self.metadata_tracker.current_stage,
                config=config,
                partial_result=self.metadata_tracker.get_partial_result()
            )
    
    def register_pipeline(self, name: str, pipeline: Any):
        """Register a pipeline for strategies to use"""
        self.pipeline_registry[name] = pipeline
        
    def estimate_vram(self, config: ExpandorConfig) -> Dict[str, float]:
        """Estimate VRAM requirements before execution"""
        return self.vram_manager.estimate_requirement(config)
```

### Strategy Selection Logic (VRAM-Aware)

```python
class StrategySelector:
    """Intelligent strategy selection with VRAM awareness"""
    
    def select(self, config: ExpandorConfig) -> BaseStrategy:
        """Select optimal strategy based on multiple factors"""
        
        # Calculate metrics
        metrics = self._calculate_metrics(config)
        
        # Check VRAM constraints first
        vram_available = self.vram_manager.get_available_vram()
        vram_estimate = self.vram_manager.estimate_requirement(config)
        vram_required = vram_estimate['total_with_buffer_mb']
        
        # Force VRAM-friendly strategies if needed
        if vram_required > vram_available * 0.8:
            if config.allow_tiled:
                return TiledExpansionStrategy()
            elif config.allow_cpu_offload:
                return CPUOffloadStrategy()
            else:
                raise VRAMError(
                    operation="strategy_selection",
                    required_mb=vram_required,
                    available_mb=vram_available,
                    message=f"Insufficient VRAM: need {vram_required:.0f}MB, have {vram_available:.0f}MB available"
                )
        
        # Multi-factor decision matrix
        if metrics['is_extreme_ratio'] and metrics['has_inpaint']:
            return SWPOStrategy()  # Best for extreme aspect changes
            
        elif metrics['aspect_change'] > 1.5 and metrics['has_inpaint']:
            return ProgressiveOutpaintStrategy()
            
        elif metrics['area_ratio'] < 4 and metrics['aspect_change'] < 1.1:
            return DirectUpscaleStrategy()
            
        elif metrics['area_ratio'] > 16 and config.quality_preset == 'ultra':
            # Massive upscale - use tiled approach
            return TiledExpansionStrategy()
            
        else:
            # Intelligent hybrid approach
            return AdaptiveHybridStrategy()
    
    def _calculate_metrics(self, config: ExpandorConfig) -> Dict[str, Any]:
        """Calculate all decision metrics"""
        # Get dimensions
        if isinstance(config.source_image, Path):
            from PIL import Image
            with Image.open(config.source_image) as img:
                source_w, source_h = img.size
        else:
            source_w, source_h = config.source_image.size
            
        target_w, target_h = config.target_resolution
        
        # Calculate ratios
        source_aspect = source_w / source_h
        target_aspect = target_w / target_h
        
        return {
            'source_size': (source_w, source_h),
            'target_size': (target_w, target_h),
            'area_ratio': (target_w * target_h) / (source_w * source_h),
            'aspect_change': max(target_aspect/source_aspect, source_aspect/target_aspect),
            'absolute_width_change': abs(target_w - source_w),
            'absolute_height_change': abs(target_h - source_h),
            'has_inpaint': config.inpaint_pipeline is not None,
            'has_refiner': config.refiner_pipeline is not None,
            'is_extreme_ratio': max(target_aspect/source_aspect, source_aspect/target_aspect) > 4.0,
            'model_type': config.source_metadata.get('model', 'unknown')
        }
```

### Quality Presets (Updated)

```yaml
quality_presets:
  ultra:
    description: "Maximum quality, no time limits"
    refinement_passes: 5
    artifact_detection: aggressive
    seam_repair_attempts: 3
    upscale_model: RealESRGAN_x4plus
    tile_size: 512
    use_fp32: true
    denoising_strength_decay: 0.95
    
  high:
    description: "95% quality, 50% faster"
    refinement_passes: 3
    artifact_detection: standard
    seam_repair_attempts: 2
    upscale_model: RealESRGAN_x4plus
    tile_size: 768
    use_fp32: true
    denoising_strength_decay: 0.90
    
  balanced:
    description: "90% quality, 75% faster"
    refinement_passes: 2
    artifact_detection: light
    seam_repair_attempts: 1
    upscale_model: RealESRGAN_x4plus
    tile_size: 1024
    use_fp32: false
    denoising_strength_decay: 0.85
    
  fast:
    description: "85% quality, maximum speed"
    refinement_passes: 1
    artifact_detection: disabled
    seam_repair_attempts: 0
    upscale_model: RealESRGAN_x2plus
    tile_size: 1536
    use_fp32: false
    denoising_strength_decay: 0.80
```

## Development & Testing Plan

### Phase 1: Repository Setup & Core Extraction (Week 1)

1. **Create Expandor Repository**
   ```bash
   # Create new repository
   mkdir expandor
   cd expandor
   git init
   
   # Setup Python package structure
   python -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

2. **Extract Core Components**
   - Copy with attribution and modification:
     - `VRAMCalculator` → `expandor/core/vram_manager.py`
     - `ResolutionManager` calculations → `expandor/utils/dimension_calculator.py`
     - `AspectAdjuster` logic → `expandor/strategies/progressive_outpaint.py`
     - Create new artifact detection → `expandor/processors/artifact_removal.py`
   
3. **Create Mock Interfaces**
   ```python
   # expandor/adapters/mock_pipeline.py
   class MockInpaintPipeline:
       """Mock pipeline for testing without real models"""
       def __call__(self, prompt, image, mask_image, **kwargs):
           # Return slightly modified image for testing
           return MockPipelineOutput(images=[image])
   ```

### Phase 2: Core Implementation (Week 2)

1. **Implement Base Architecture**
   - Strategy selector with VRAM awareness
   - Pipeline orchestrator with fallback chains
   - Metadata tracking system
   - Boundary tracking for seam detection

2. **Basic Strategies**
   - DirectUpscaleStrategy (simplest)
   - ProgressiveOutpaintStrategy (from AspectAdjuster)
   - TiledExpansionStrategy (for testing VRAM limits)

3. **Unit Tests**
   ```python
   # tests/unit/test_strategy_selection.py
   def test_vram_limited_selection():
       config = ExpandorConfig(
           source_image=test_image_1080p,
           target_resolution=(7680, 4320),
           vram_limit_mb=4000  # Force tiled strategy
       )
       strategy = selector.select(config)
       assert isinstance(strategy, TiledExpansionStrategy)
   ```

### Phase 3: Advanced Features (Week 3)

1. **Complex Strategies**
   - SWPO implementation with window management
   - CPU offload strategy
   - Adaptive hybrid strategy

2. **Quality Systems**
   - Multi-pass refinement
   - Artifact detection and repair
   - Boundary tracking integration

3. **Integration Tests**
   ```python
   # tests/integration/test_full_pipeline.py
   def test_extreme_aspect_change():
       # Test 16:9 to 32:9 expansion
       result = expandor.expand(config)
       assert result.seams_detected == 0
       assert result.quality_score > 0.95
   ```

### Phase 4: AI Wallpaper Compatibility (Week 4)

1. **Create Adapters**
   ```python
   # expandor/adapters/ai_wallpaper.py
   class AIWallpaperAdapter:
       """Adapter to match ai-wallpaper interfaces"""
       
       def adapt_metadata(self, ai_wallpaper_metadata):
           """Convert ai-wallpaper metadata format"""
           return {
               'progressive_boundaries': ai_wallpaper_metadata.get('progressive_boundaries', []),
               'generation_metadata': ai_wallpaper_metadata
           }
   ```

2. **Test with Extracted Components**
   - Use real images from ai-wallpaper
   - Verify metadata compatibility
   - Ensure identical quality output

3. **Performance Benchmarks**
   - Compare timing with original implementations
   - Verify VRAM usage matches expectations
   - Test fallback chains

### Phase 5: Integration Preparation (Week 5)

1. **Package for Distribution**
   ```python
   # setup.py
   setup(
       name='expandor',
       version='1.0.0',
       packages=find_packages(),
       install_requires=[
           'torch>=2.0.0',
           'pillow>=10.0.0',
           'numpy>=1.24.0',
           'opencv-python>=4.8.0',
       ],
       extras_require={
           'ai-wallpaper': ['diffusers>=0.25.0'],
       }
   )
   ```

2. **Documentation**
   - API reference
   - Integration guide for ai-wallpaper
   - Strategy development guide
   - Migration checklist

3. **Final Testing**
   - Full test suite with 100% coverage
   - Integration tests with mock ai-wallpaper
   - Performance regression tests

## Integration with AI Wallpaper

### Option 1: Package Dependency
```bash
# In ai-wallpaper
pip install expandor

# In code
from expandor import Expandor, ExpandorConfig
expandor = Expandor()
```

### Option 2: Git Submodule
```bash
# In ai-wallpaper
git submodule add https://github.com/user/expandor.git lib/expandor
git submodule update --init
```

### Phased Integration Approach

1. **Phase 1: Optional Flag**
   ```python
   # In models.yaml
   sdxl:
     use_expandor: false  # Enable per-model for testing
   ```

2. **Phase 2: Parallel Testing**
   ```python
   if self.config.get('use_expandor', False):
       result = self._expand_with_expandor(image, target_size)
   else:
       result = self._legacy_pipeline(image, target_size)
   ```

3. **Phase 3: Default with Fallback**
   ```python
   try:
       result = self._expand_with_expandor(image, target_size)
   except ExpandorError:
       self.logger.warning("Expandor failed, using legacy pipeline")
       result = self._legacy_pipeline(image, target_size)
   ```

4. **Phase 4: Full Migration**
   - Remove legacy pipeline code
   - Expandor becomes the only expansion system

## Success Metrics

- **Zero Quality Regression**: Output identical or better than original
- **Performance**: No more than 10% slower than original
- **Reliability**: Zero failures that original pipeline handles
- **VRAM Efficiency**: Better fallback handling than original
- **Integration Ease**: < 100 lines to integrate per model
- **Standalone Viability**: Works without ai-wallpaper dependencies

## Benefits of Standalone Development

1. **Risk Mitigation**
   - Zero chance of breaking working system
   - Can test extensively before integration
   - Easy rollback if issues found

2. **Clean Development**
   - No accidental dependencies on ai-wallpaper
   - Forces proper abstraction
   - Easier to understand and maintain

3. **Broader Applicability**
   - Can be used by other projects
   - Becomes a portfolio piece
   - Potential for community contributions

4. **Better Testing**
   - Can use synthetic test data
   - Mock all external dependencies
   - Comprehensive test coverage

This approach ensures Expandor is thoroughly tested and production-ready before touching the working AI Wallpaper system, maintaining the project's "quality over all" philosophy throughout development.