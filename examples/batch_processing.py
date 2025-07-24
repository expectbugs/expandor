#!/usr/bin/env python3
"""
Batch Processing Example

This example demonstrates how to process multiple images efficiently,
including parallel processing and progress tracking.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from expandor import Expandor
from expandor.adapters import MockPipelineAdapter
from expandor.core.config import ExpandorConfig
from expandor.utils.logging_utils import setup_logger


def create_sample_images(num_images=5):
    """Create sample images for demonstration"""
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)

    print(f"Creating {num_images} sample images...")

    for i in range(num_images):
        # Create images with different sizes and colors
        width = 800 + i * 100
        height = 600 + i * 75
        color = (50 + i * 30, 100 + i * 20, 150 + i * 10)

        img = Image.new("RGB", (width, height), color=color)
        img_path = sample_dir / f"sample_{i:03d}.png"
        img.save(img_path)

        print(f"  Created: {img_path.name} ({width}x{height})")

    return sample_dir


def process_single_image(args):
    """Process a single image (used for parallel processing)"""
    img_path, output_dir, config_dict = args

    try:
        # Load image
        image = Image.open(img_path)

        # Create config from dict
        config = ExpandorConfig(**config_dict)

        # Create adapter and expandor
        adapter = MockPipelineAdapter()
        expandor = Expandor(pipeline_adapter=adapter, config=config)

        # Expand image
        result = expandor.expand(image)

        if result.success:
            # Save with descriptive filename
            output_name = f"{img_path.stem}_expanded_{config.target_width}x{config.target_height}.png"
            output_path = output_dir / output_name
            result.final_image.save(output_path)

            return {
                "success": True,
                "input": img_path.name,
                "output": output_path.name,
                "time": result.processing_time,
                "size": result.final_image.size,
            }
        else:
            return {
                "success": False,
                "input": img_path.name,
                "error": str(result.error),
            }

    except Exception as e:
        return {"success": False, "input": img_path.name, "error": str(e)}


def main():
    logger = setup_logger(__name__)

    # Create sample images
    sample_dir = create_sample_images(10)

    # Create output directory
    output_dir = Path("batch_output")
    output_dir.mkdir(exist_ok=True)

    print("\n=== Batch Processing Example ===")

    # Method 1: Sequential Processing
    print("\n1. Sequential Processing")
    print("-" * 40)

    # Get all PNG images
    image_files = list(sample_dir.glob("*.png"))
    print(f"Found {len(image_files)} images to process")

    # Configuration for all images
    config = ExpandorConfig(
        target_width=2560, target_height=1440, quality_preset="balanced"
    )

    # Create adapter once for sequential processing
    adapter = MockPipelineAdapter()
    expandor = Expandor(pipeline_adapter=adapter, config=config)

    start_time = time.time()
    results = []

    # Process with progress bar
    for img_path in tqdm(image_files[:3], desc="Processing"):
        image = Image.open(img_path)
        result = expandor.expand(image)

        if result.success:
            output_path = output_dir / f"{img_path.stem}_sequential.png"
            result.final_image.save(output_path)
            results.append((img_path.name, True, result.processing_time))
        else:
            results.append((img_path.name, False, 0))

    sequential_time = time.time() - start_time
    print(f"\nSequential processing completed in {sequential_time:.2f}s")

    # Method 2: Parallel Processing
    print("\n2. Parallel Processing")
    print("-" * 40)

    # Prepare arguments for parallel processing
    config_dict = config.__dict__.copy()
    process_args = [(img_path, output_dir, config_dict) for img_path in image_files]

    start_time = time.time()
    parallel_results = []

    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_img = {
            executor.submit(process_single_image, args): args[0]
            for args in process_args
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(image_files), desc="Processing") as pbar:
            for future in as_completed(future_to_img):
                result = future.result()
                parallel_results.append(result)
                pbar.update(1)

                if result["success"]:
                    tqdm.write(
                        f"  ✓ {result['input']} → {result['output']} "
                        f"({result['time']:.2f}s)"
                    )
                else:
                    tqdm.write(f"  ✗ {result['input']}: {result['error']}")

    parallel_time = time.time() - start_time
    print(f"\nParallel processing completed in {parallel_time:.2f}s")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")

    # Method 3: Batch Processing with Different Resolutions
    print("\n3. Multi-Resolution Batch")
    print("-" * 40)

    resolutions = [(1920, 1080, "FHD"), (2560, 1440, "QHD"), (3840, 2160, "4K")]

    # Process first image at multiple resolutions
    test_image = Image.open(image_files[0])

    for width, height, name in resolutions:
        config = ExpandorConfig(
            target_width=width,
            target_height=height,
            quality_preset="fast",  # Use fast for demo
        )

        expandor = Expandor(pipeline_adapter=adapter, config=config)
        result = expandor.expand(test_image)

        if result.success:
            output_path = output_dir / f"multisize_{name}.png"
            result.final_image.save(output_path)
            print(f"  ✓ Generated {name}: {width}x{height}")

    # Method 4: Smart Batch Processing with Auto Strategy
    print("\n4. Smart Batch with Auto Strategy")
    print("-" * 40)

    # Process images with automatic strategy selection based on size
    for img_path in image_files[:3]:
        image = Image.open(img_path)

        # Calculate size increase factor
        target_w, target_h = 3840, 2160
        factor = max(target_w / image.width, target_h / image.height)

        # Auto-select strategy based on factor
        if factor > 3:
            strategy = "progressive"
            print(f"\n{img_path.name}: Large expansion ({factor:.1f}x) → Progressive")
        elif factor > 1.5:
            strategy = "direct"
            print(f"\n{img_path.name}: Moderate expansion ({factor:.1f}x) → Direct")
        else:
            strategy = "refine"
            print(f"\n{img_path.name}: Small expansion ({factor:.1f}x) → Refine")

        config = ExpandorConfig(
            target_width=target_w, target_height=target_h, strategy=strategy
        )

        expandor = Expandor(pipeline_adapter=adapter, config=config)
        result = expandor.expand(image)

        if result.success:
            print(f"  ✓ Success: {result.processing_time:.2f}s")

    # Summary
    print("\n=== Batch Processing Summary ===")
    print(f"Total images processed: {len(parallel_results)}")

    successful = sum(1 for r in parallel_results if r["success"])
    failed = len(parallel_results) - successful

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful > 0:
        avg_time = sum(r["time"] for r in parallel_results if r["success"]) / successful
        print(f"Average processing time: {avg_time:.2f}s per image")

    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
