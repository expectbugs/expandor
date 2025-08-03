#!/usr/bin/env python3
"""
Advanced Features Example

This example demonstrates advanced Expandor features including:
- LoRA stacking and management
- ControlNet integration (placeholder)
- Custom strategies
- Memory optimization
- Metadata tracking
- Stage saving and debugging
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add expandor to path for development testing
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from expandor import Expandor
from expandor.adapters import DiffusersPipelineAdapter, MockPipelineAdapter
from expandor.config import LoRAConfig, ModelConfig, UserConfig
from expandor.config.lora_manager import LoRAManager
from expandor.core.config import ExpandorConfig
from expandor.strategies import StrategySelector
from expandor.utils.logging_utils import setup_logger
from expandor.utils.metadata_utils import MetadataManager

# Set up module-level logger
logger = setup_logger(__name__)


def create_test_image_with_details():
    """Create a detailed test image with various features"""
    img = Image.new("RGB", (1024, 768), color=(50, 100, 150))
    draw = ImageDraw.Draw(img)

    # Add some geometric shapes
    draw.rectangle([100, 100, 300, 300], fill=(200, 50, 50))
    draw.ellipse([400, 200, 600, 400], fill=(50, 200, 50))
    draw.polygon([(700, 100), (850, 200), (750, 350), (650, 250)], fill=(200, 200, 50))

    # Add text (if font available)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
        draw.text((50, 50), "Expandor Test Image", fill=(255, 255, 255), font=font)
        draw.text((50, 700), "Advanced Features Demo", fill=(255, 255, 255), font=font)
    except ImportError:
        logger.debug("Default font not available, skipping text overlay")

    # Add some noise/texture
    pixels = img.load()
    for i in range(0, 1024, 50):
        for j in range(0, 768, 50):
            noise = np.random.randint(-20, 20, 3)
            r, g, b = pixels[i, j]
            pixels[i, j] = (
                max(0, min(255, r + noise[0])),
                max(0, min(255, g + noise[1])),
                max(0, min(255, b + noise[2])),
            )

    return img


def main():
    print("=== Advanced Expandor Features ===\n")

    # Create test image
    test_image = create_test_image_with_details()
    test_image.save("advanced_test_input.png")

    # Example 1: LoRA Stacking and Management
    print("1. LoRA Stacking and Conflict Resolution")
    print("-" * 50)

    # Create LoRA manager
    lora_manager = LoRAManager()

    # Define multiple LoRAs
    loras = [
        LoRAConfig(name="style_anime", type="style", weight=0.8),
        LoRAConfig(name="detail_enhance", type="detail", weight=0.6),
        LoRAConfig(name="quality_boost", type="quality", weight=0.5),
        # This would conflict: LoRAConfig(name='style_realistic', type='style', weight=0.7)
    ]

    print("Original LoRAs:")
    for lora in loras:
        print(f"  - {lora.name}: type={lora.type}, weight={lora.weight}")

    # Check compatibility and resolve weights
    try:
        lora_manager.check_compatibility(loras)
        resolved_loras = lora_manager.resolve_lora_stack(loras)

        print("\nResolved LoRA stack:")
        total_weight = 0
        for lora in resolved_loras:
            print(f"  - {lora.name}: weight={lora.weight:.3f}")
            total_weight += lora.weight
        print(f"  Total weight: {total_weight:.3f}")

        # Get recommended inference steps
        base_steps = 40
        recommended_steps = lora_manager.get_recommended_inference_steps(
            resolved_loras, base_steps
        )
        print(
            f"\nRecommended inference steps: {recommended_steps} (base: {base_steps})"
        )

    except Exception as e:
        print(f"LoRA conflict detected: {e}")

    # Example 2: Custom Strategy Implementation
    print("\n2. Strategy Selection and Customization")
    print("-" * 50)

    # Test strategy selector
    selector = StrategySelector()

    test_cases = [
        ((1024, 768), (2048, 1536)),  # 2x expansion
        ((1024, 768), (5120, 1440)),  # Extreme aspect change
        ((512, 512), (4096, 4096)),  # 8x expansion
    ]

    for current, target in test_cases:
        strategy, reason = selector.select_strategy(
            current_size=current,
            target_size=target,
            available_vram=8000,  # 8GB
            user_preference="auto",
        )
        print(f"\n{current} → {target}:")
        print(f"  Selected: {strategy}")
        print(f"  Reason: {reason}")

    # Example 3: Memory Optimization
    print("\n3. Memory-Optimized Processing")
    print("-" * 50)

    # Create low-memory configuration
    low_mem_config = ExpandorConfig(
        target_width=3840,
        target_height=2160,
        strategy="tiled",
        tile_size=512,
        tile_overlap=64,
        enable_memory_efficient=True,
        clear_cache_frequency=5,  # Clear cache every 5 operations
        quality_preset="balanced",
    )

    print("Low-memory configuration:")
    print(f"  - Strategy: {low_mem_config.strategy}")
    print(f"  - Tile size: {low_mem_config.tile_size}")
    print(f"  - Tile overlap: {low_mem_config.tile_overlap}")
    print(f"  - Cache clearing: every {low_mem_config.clear_cache_frequency} ops")

    # Process with memory monitoring
    adapter = MockPipelineAdapter()
    expandor = Expandor(pipeline_adapter=adapter, config=low_mem_config)

    # Simulate memory monitoring
    print("\nProcessing with memory monitoring...")
    result = expandor.expand(test_image)

    if result.success:
        print(f"✓ Success with tiled processing")
        if hasattr(result, "metadata") and result.metadata:
            if "memory_usage" in result.metadata:
                print(f"  Peak memory: {result.metadata['memory_usage']:.2f} MB")

    # Example 4: Metadata Tracking
    print("\n4. Comprehensive Metadata Tracking")
    print("-" * 50)

    # Create config with metadata saving enabled
    metadata_config = ExpandorConfig(
        target_width=2560,
        target_height=1440,
        save_metadata=True,
        metadata_format="json",
        include_generation_params=True,
        include_performance_metrics=True,
    )

    expandor = Expandor(pipeline_adapter=adapter, config=metadata_config)
    result = expandor.expand(test_image)

    if result.success:
        # Create comprehensive metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "version": "0.4.0",
            "input": {
                "size": test_image.size,
                "mode": test_image.mode,
                "format": "PNG",
            },
            "output": {
                "size": result.final_image.size,
                "strategy_used": result.strategy_used,
                "processing_time": result.processing_time,
            },
            "configuration": {
                "quality_preset": metadata_config.quality_preset,
                "artifacts_check": metadata_config.enable_artifacts_check,
                "stages_saved": metadata_config.save_stages,
            },
            "generation_params": {
                "seed": getattr(result, "seed", "N/A"),
                "prompt": getattr(result, "prompt", "N/A"),
                "negative_prompt": getattr(result, "negative_prompt", "N/A"),
            },
            "performance": {
                "total_time": result.processing_time,
                "stages": getattr(result, "stage_times", {}),
                "memory_peak": getattr(result, "peak_memory", "N/A"),
            },
        }

        # Save metadata
        metadata_path = Path("advanced_output_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved comprehensive metadata to: {metadata_path}")
        print("Metadata summary:")
        print(f"  - Input size: {metadata['input']['size']}")
        print(f"  - Output size: {metadata['output']['size']}")
        print(f"  - Strategy: {metadata['output']['strategy_used']}")
        print(f"  - Processing time: {metadata['output']['processing_time']:.2f}s")

    # Example 5: Stage Saving and Debugging
    print("\n5. Stage Saving for Debugging")
    print("-" * 50)

    # Create debug configuration
    debug_config = ExpandorConfig(
        target_width=2048,
        target_height=1536,
        strategy="progressive",
        save_stages=True,
        stage_dir=Path("debug_stages"),
        stage_format="png",
        save_masks=True,
        save_metadata_per_stage=True,
        verbose=True,
    )

    # Create stages directory
    debug_config.stage_dir.mkdir(exist_ok=True)

    print(f"Debug configuration:")
    print(f"  - Saving stages to: {debug_config.stage_dir}")
    print(f"  - Stage format: {debug_config.stage_format}")
    print(f"  - Save masks: {debug_config.save_masks}")

    expandor = Expandor(pipeline_adapter=adapter, config=debug_config)
    result = expandor.expand(test_image)

    if result.success:
        print(f"\n✓ Debug expansion complete")

        # List saved stages
        stages = list(debug_config.stage_dir.glob("*.png"))
        print(f"  Saved {len(stages)} stage files:")
        for stage in sorted(stages):
            print(f"    - {stage.name}")

    # Example 6: Advanced Error Handling
    print("\n6. Advanced Error Handling and Recovery")
    print("-" * 50)

    # Create config with strict validation
    strict_config = ExpandorConfig(
        target_width=3840,
        target_height=2160,
        enable_artifacts_check=True,
        artifact_detection_threshold=0.05,  # Very strict
        max_artifact_repair_attempts=3,
        fallback_on_error=True,
        fallback_strategy="direct",
    )

    print("Strict validation configuration:")
    print(f"  - Artifact threshold: {strict_config.artifact_detection_threshold}")
    print(f"  - Max repair attempts: {strict_config.max_artifact_repair_attempts}")
    print(f"  - Fallback strategy: {strict_config.fallback_strategy}")

    # Example 7: Custom Pipeline Parameters
    print("\n7. Custom Pipeline Parameters")
    print("-" * 50)

    # Create config with custom pipeline parameters
    custom_params_config = ExpandorConfig(
        target_width=2560,
        target_height=1440,
        custom_pipeline_params={
            "guidance_scale": 8.5,
            "num_inference_steps": 60,
            "eta": 0.0,
            "generator_seed": 42,
            "scheduler": "DPMSolverMultistepScheduler",
            "scheduler_params": {
                "use_karras_sigmas": True,
                "algorithm_type": "dpmsolver++",
            },
        },
    )

    print("Custom pipeline parameters:")
    for key, value in custom_params_config.custom_pipeline_params.items():
        print(f"  - {key}: {value}")

    # Example 8: Batch Configuration Profiles
    print("\n8. Configuration Profiles")
    print("-" * 50)

    # Define different profiles for different use cases
    profiles = {
        "wallpaper": {
            "description": "High-quality desktop wallpaper",
            "config": ExpandorConfig(
                target_width=3840,
                target_height=2160,
                quality_preset="ultra",
                strategy="progressive",
                enable_artifacts_check=True,
            ),
        },
        "web_banner": {
            "description": "Optimized for web use",
            "config": ExpandorConfig(
                target_width=1920,
                target_height=600,
                quality_preset="balanced",
                strategy="swpo",  # For extreme aspect ratio
                output_format="jpeg",
                output_quality=85,
            ),
        },
        "print_ready": {
            "description": "High-resolution for printing",
            "config": ExpandorConfig(
                target_width=7680,
                target_height=4320,
                quality_preset="ultra",
                strategy="tiled",  # Memory efficient for huge size
                tile_size=1024,
                save_lossless=True,
            ),
        },
    }

    print("Available configuration profiles:")
    for name, profile in profiles.items():
        config = profile["config"]
        print(f"\n{name}: {profile['description']}")
        print(f"  - Resolution: {config.target_width}x{config.target_height}")
        print(f"  - Quality: {config.quality_preset}")
        print(f"  - Strategy: {config.strategy}")

    print("\n=== Advanced Features Example Complete ===")
    print("\nCheck the output files and debug_stages directory for results!")


if __name__ == "__main__":
    main()
