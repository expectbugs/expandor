#!/usr/bin/env python3
"""
Custom Configuration Example

This example shows how to use custom configurations, override defaults,
and work with user-specific settings.
"""

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import yaml
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from expandor import Expandor
from expandor.adapters import MockPipelineAdapter
from expandor.config import LoRAConfig, ModelConfig, UserConfig, UserConfigManager
from expandor.config.pipeline_config import PipelineConfigurator
from expandor.core.config import ExpandorConfig
from expandor.utils.config_loader import ConfigLoader


@contextmanager
def temporary_config():
    """Create a temporary config file for testing"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yield Path(f.name)
    Path(f.name).unlink()


def main():
    print("=== Custom Configuration Examples ===\n")

    # Create sample image
    sample_image = Image.new("RGB", (1024, 768), color=(100, 150, 200))

    # Example 1: Creating and Using User Configuration
    print("1. User Configuration")
    print("-" * 40)

    # Create user config programmatically
    user_config = UserConfig(
        models={
            "sdxl": ModelConfig(
                model_id="stabilityai/stable-diffusion-xl-base-1.0",
                dtype="float16",
                device="cuda",
                custom_pipeline_params={"variant": "fp16", "use_safetensors": True},
            ),
            "sd15": ModelConfig(
                model_id="runwayml/stable-diffusion-v1-5",
                dtype="float32",
                device="cuda",
            ),
        },
        loras=[
            LoRAConfig(
                name="detail_enhancer",
                path="./loras/detail_v2.safetensors",
                weight=0.7,
                type="detail",
            ),
            LoRAConfig(
                name="anime_style",
                path="./loras/anime_style.safetensors",
                weight=0.5,
                type="style",
            ),
        ],
        default_quality="high",
        default_strategy="progressive",
        preferences={
            "save_metadata": True,
            "auto_select_strategy": True,
            "preserve_original_prompt": False,
        },
        performance={"max_workers": 4, "tile_size": 1024, "cache_size": 2048},
        output={"format": "png", "quality": 95, "optimize": True},
    )

    print("Created user configuration with:")
    print(f"  - {len(user_config.models)} models configured")
    print(f"  - {len(user_config.loras)} LoRAs available")
    print(f"  - Default quality: {user_config.default_quality}")
    print(f"  - Default strategy: {user_config.default_strategy}")

    # Example 2: Using Configuration with Expandor
    print("\n2. Applying User Configuration")
    print("-" * 40)

    # Create pipeline configurator
    pipeline_config = PipelineConfigurator()

    # Create Expandor config from user config
    expandor_config = pipeline_config.create_expandor_config(
        user_config=user_config,
        target_size=(2560, 1440),
        quality="ultra",  # Override default quality
    )

    print(f"Expandor configuration:")
    print(f"  - Target: {expandor_config.target_width}x{expandor_config.target_height}")
    print(f"  - Quality: {expandor_config.quality_preset}")
    print(f"  - Strategy: {expandor_config.strategy}")

    # Use with Expandor
    adapter = MockPipelineAdapter()
    expandor = Expandor(pipeline_adapter=adapter, config=expandor_config)

    result = expandor.expand(sample_image)
    if result.success:
        print(f"  ✓ Expansion successful!")

    # Example 3: Loading Configuration from YAML
    print("\n3. YAML Configuration")
    print("-" * 40)

    # Create a custom YAML config
    yaml_config = """
# Custom Expandor Configuration
models:
  flux:
    model_id: "black-forest-labs/FLUX.1-dev"
    dtype: bfloat16
    device: cuda
    custom_pipeline_params:
      guidance_scale: 7.5
      num_inference_steps: 50

loras:
  - name: photorealistic
    path: /models/loras/photorealistic_v3.safetensors
    weight: 0.6
    type: style
    enabled: true
    
  - name: face_enhance
    path: /models/loras/face_detail.safetensors
    weight: 0.4
    type: subject
    enabled: true

quality_overrides:
  ultra:
    inference_steps: 100
    cfg_scale: 8.0
    denoise_strength: 0.95
  fast:
    inference_steps: 20
    cfg_scale: 5.0
    denoise_strength: 0.7

default_quality: balanced
default_strategy: auto

preferences:
  auto_select_strategy: true
  save_metadata: true
  metadata_format: json
  preserve_original_prompt: true
  
performance:
  max_workers: 8
  tile_size: 1024
  cache_size: 4096
  clear_cache_on_error: true
  
output:
  format: png
  quality: 100
  optimize: false
  save_stages: true
  stage_format: png
"""

    with temporary_config() as config_path:
        # Write config to temporary file
        with open(config_path, "w") as f:
            f.write(yaml_config)

        print(f"Created custom config at: {config_path}")

        # Load with ConfigLoader
        loader = ConfigLoader(config_path.parent)
        loaded_config = loader.load_config_file(config_path.name)

        print(f"Loaded configuration sections:")
        for key in loaded_config.keys():
            print(f"  - {key}")

    # Example 4: Environment-Specific Configuration
    print("\n4. Environment-Specific Configuration")
    print("-" * 40)

    # Create different configs for different environments
    configs = {
        "development": ExpandorConfig(
            target_width=1920,
            target_height=1080,
            quality_preset="fast",
            strategy="direct",
            save_stages=True,
            enable_artifacts_check=False,  # Skip for speed in dev
        ),
        "production": ExpandorConfig(
            target_width=3840,
            target_height=2160,
            quality_preset="ultra",
            strategy="auto",
            save_stages=False,
            enable_artifacts_check=True,
            artifact_detection_threshold=0.1,  # Strict quality
        ),
        "low_memory": ExpandorConfig(
            target_width=2560,
            target_height=1440,
            quality_preset="balanced",
            strategy="tiled",  # Use tiled for low memory
            tile_size=512,
            enable_memory_efficient=True,
        ),
    }

    # Use different configs based on environment
    import os

    env = os.getenv("EXPANDOR_ENV", "development")
    config = configs.get(env, configs["development"])

    print(f"Using {env} configuration:")
    print(f"  - Quality: {config.quality_preset}")
    print(f"  - Strategy: {config.strategy}")
    print(f"  - Artifacts check: {config.enable_artifacts_check}")

    # Example 5: Configuration Validation and Merging
    print("\n5. Configuration Merging")
    print("-" * 40)

    # Base configuration
    base_config = {
        "quality_preset": "balanced",
        "strategy": "auto",
        "inference_steps": 40,
        "cfg_scale": 7.0,
    }

    # User overrides
    user_overrides = {
        "quality_preset": "high",
        "inference_steps": 60,
        "custom_setting": "value",
    }

    # Merge configurations
    final_config = {**base_config, **user_overrides}

    print("Configuration merge result:")
    for key, value in final_config.items():
        source = "override" if key in user_overrides else "base"
        print(f"  - {key}: {value} (from {source})")

    # Example 6: Dynamic Configuration Based on Input
    print("\n6. Dynamic Configuration")
    print("-" * 40)

    def get_optimal_config(image: Image.Image, target_size: tuple) -> ExpandorConfig:
        """Dynamically determine optimal configuration based on input"""

        # Calculate expansion factor
        factor_w = target_size[0] / image.width
        factor_h = target_size[1] / image.height
        max_factor = max(factor_w, factor_h)

        # Determine strategy
        if max_factor > 4:
            strategy = "swpo"  # Extreme expansion
        elif max_factor > 2:
            strategy = "progressive"
        else:
            strategy = "direct"

        # Determine quality based on image size
        total_pixels = target_size[0] * target_size[1]
        if total_pixels > 8_000_000:  # >8MP
            quality = "fast"  # Faster for very large images
        elif total_pixels > 4_000_000:  # >4MP
            quality = "balanced"
        else:
            quality = "high"

        # Check if aspect ratio is changing significantly
        aspect_change = abs(
            (target_size[0] / target_size[1]) - (image.width / image.height)
        )
        enable_artifacts = aspect_change > 0.5  # Enable for significant changes

        config = ExpandorConfig(
            target_width=target_size[0],
            target_height=target_size[1],
            strategy=strategy,
            quality_preset=quality,
            enable_artifacts_check=enable_artifacts,
        )

        return config

    # Test dynamic configuration
    test_sizes = [(2048, 1152), (5120, 1440), (1920, 1080)]

    for size in test_sizes:
        config = get_optimal_config(sample_image, size)
        print(f"\nTarget {size[0]}x{size[1]}:")
        print(f"  - Strategy: {config.strategy}")
        print(f"  - Quality: {config.quality_preset}")
        print(f"  - Artifacts check: {config.enable_artifacts_check}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
