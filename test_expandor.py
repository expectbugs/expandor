#!/usr/bin/env python
"""Test Expandor with MockPipelineAdapter"""

import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from expandor import Expandor
from expandor.adapters import MockPipelineAdapter

# Test configuration
input_image = "/home/user/Pictures/backgrounds/wp2491458-gentoo-wallpapers.jpg"
output_path = "/home/user/ai-wallpaper/expandor/test_output_4k.png"

print(f"Testing Expandor with MockPipelineAdapter")
print(f"Input: {input_image}")
print(f"Target: 3840x2160 (4K)")
print(f"Quality: ultra\n")

# Create mock adapter
adapter = MockPipelineAdapter(
    model_id="mock-sdxl",
    device="cuda",
    dtype="fp16"
)

# Create Expandor instance
expandor = Expandor(pipeline_adapter=adapter)

# Load input image
with Image.open(input_image) as img:
    print(f"Original size: {img.size}")
    print(f"Aspect ratio: {img.width/img.height:.2f}")

# Create config
from expandor.core.config import ExpandorConfig

config = ExpandorConfig(
    source_image=Path(input_image),
    target_width=3840,
    target_height=2160,
    prompt="A beautiful landscape",
    quality_preset="ultra",
    save_stages=True
)

# Expand to 4K
print("\nExpanding to 4K...")
result = expandor.expand(config)

print(f"\nExpansion complete!")
print(f"Output size: {result.final_image.size}")
print(f"Strategy used: {result.strategy_used}")
print(f"Processing time: {result.metadata.get('total_time', 'N/A')}s")
print(f"Stages saved: {len(result.intermediate_stages)}")

# Save output
result.final_image.save(output_path, compress_level=0)
print(f"\nSaved to: {output_path}")