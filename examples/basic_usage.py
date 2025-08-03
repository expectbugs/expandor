#!/usr/bin/env python3
"""
Basic Expandor Usage Example

This example demonstrates the simplest way to use Expandor to expand an image
to a higher resolution using the default settings.
"""

import sys
from pathlib import Path

# Add expandor to path for development testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from expandor import Expandor
from expandor.adapters import MockPipelineAdapter
from expandor.core.config import ExpandorConfig


def main():
    # Create a sample image (or load your own)
    print("Creating sample image...")
    sample_image = Image.new("RGB", (1024, 768), color=(100, 150, 200))
    sample_image.save("sample_input.png")
    print(f"Sample image created: 1024x768")

    # Method 1: Using Expandor directly with minimal configuration
    print("\n=== Method 1: Direct Expandor Usage ===")

    # Create Expandor instance with mock adapter for demo
    # In production, use DiffusersPipelineAdapter or your preferred adapter
    adapter = MockPipelineAdapter()
    expandor = Expandor(pipeline_adapter=adapter)

    # Create configuration for target resolution
    config = ExpandorConfig(
        source_image=sample_image,
        target_resolution=(3840, 2160),  # 4K resolution
        prompt="a high quality image",  # Basic prompt for validation
        seed=42,
        source_metadata={"original_size": sample_image.size},
        strategy="direct",  # Use direct upscale for mock adapter
    )

    # Expand the image
    print(
        f"Expanding to {config.target_resolution[0]}x{config.target_resolution[1]}..."
    )
    result = expandor.expand(config)

    if result.success:
        print(f"✓ Expansion successful!")
        print(f"  Final size: {result.size}")
        print(f"  Strategy used: {result.strategy_used}")
        print(f"  Processing time: {result.total_duration_seconds:.2f}s")
        print(f"  Output saved to: {result.image_path}")
        print(f"  Saved to: expanded_output.png")
    else:
        print(f"✗ Expansion failed: {result.error}")

    # Method 2: Using different quality presets
    print("\n=== Method 2: Quality Preset Examples ===")

    quality_presets = ["fast", "balanced", "high", "ultra"]

    for quality in quality_presets:
        config = ExpandorConfig(
            source_image=sample_image,
            target_resolution=(2560, 1440),
            prompt="a high quality image",
            seed=42,
            source_metadata={"original_size": sample_image.size},
            quality_preset=quality,
        )

        expandor = Expandor(pipeline_adapter=adapter)
        print(f"\nTesting {quality} quality...")

        result = expandor.expand(config)
        if result.success:
            print(f"  ✓ Success - Time: {result.processing_time:.2f}s")
            print(f"    Output saved to: {result.image_path}")

    # Method 3: Using specific strategies
    print("\n=== Method 3: Strategy Examples ===")

    strategies = {
        "direct": "Fast upscaling for simple expansions",
        "progressive": "Multi-step expansion for large size increases",
        "tiled": "Memory-efficient processing for limited VRAM",
    }

    for strategy, description in strategies.items():
        print(f"\n{strategy}: {description}")

        config = ExpandorConfig(
            source_image=sample_image,
            target_resolution=(2048, 1536),
            prompt="a high quality image",
            seed=42,
            source_metadata={"original_size": sample_image.size},
            strategy_override=strategy,
        )

        expandor = Expandor(pipeline_adapter=adapter)
        result = expandor.expand(config)

        if result.success:
            print(f"  ✓ Success - Time: {result.processing_time:.2f}s")
            if hasattr(result, "metadata") and result.metadata:
                print(f"  Metadata: {result.metadata.get('stages', 'N/A')} stages")

    # Method 4: Extreme aspect ratio change
    print("\n=== Method 4: Extreme Aspect Ratio ===")

    # Convert 16:9 to 32:9 (super ultrawide)
    config = ExpandorConfig(
        source_image=sample_image,
        target_resolution=(5120, 1440),
        prompt="a high quality image",
        seed=42,
        source_metadata={"original_size": sample_image.size},
        strategy_override="swpo",  # Use SWPO for extreme aspect ratios
    )

    expandor = Expandor(pipeline_adapter=adapter)
    print(f"Converting 4:3 (1024x768) to 32:9 (5120x1440)...")

    result = expandor.expand(config)
    if result.success:
        print(f"✓ Extreme expansion successful!")
        print(f"  Aspect ratio change: 1.33:1 → 3.56:1")
        print(f"  Output saved to: {result.image_path}")

    print("\n=== Example Complete ===")
    print("Check the output files to see the results!")


if __name__ == "__main__":
    main()
