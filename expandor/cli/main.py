"""
Main CLI entry point for Expandor
"""

import glob
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch

from ..config.pipeline_config import PipelineConfigurator
from ..config.user_config import UserConfigManager
from ..core.exceptions import (ExpandorError, QualityError, StrategyError,
                               VRAMError)
from ..core.expandor import Expandor
from ..utils.logging_utils import setup_logger
from .args import create_parser, validate_args
from .commands import parse_resolution
from .process import process_single_image
from .setup_wizard import SetupWizard
from .utils import generate_output_path, test_configuration


def setup_controlnet(args, logger):
    """Set up ControlNet configuration files"""
    from ..utils.config_defaults import (create_default_controlnet_config,
                                         update_vram_strategies_with_defaults)
    from ..utils.config_loader import ConfigLoader

    # Determine config directory
    config_dir = args.config.parent if args.config else (
        Path.home() / ".config" / "expandor")
    config_dir.mkdir(parents=True, exist_ok=True)

    # Initialize config loader
    config_loader = ConfigLoader(config_dir, logger=logger)

    # Check existing files
    controlnet_config_path = config_dir / "controlnet_config.yaml"
    vram_config_path = config_dir / "vram_strategies.yaml"

    # Handle ControlNet config
    force = hasattr(args, 'force') and args.force
    if controlnet_config_path.exists() and not force:
        logger.warning(
            f"ControlNet config already exists: {controlnet_config_path}")
        logger.info("Use --force to overwrite")
    else:
        try:
            config = create_default_controlnet_config()
            config_loader.save_config_file(
                "controlnet_config.yaml",
                config,
                user_config=True
            )
            logger.info(
                f"✓ Created ControlNet config: {controlnet_config_path}")
        except Exception as e:
            logger.error(f"Failed to create ControlNet config: {e}")
            return 1

    # Update VRAM strategies if needed
    try:
        if vram_config_path.exists():
            # Load existing config
            vram_config = config_loader.load_config_file(
                "vram_strategies.yaml")

            # Check if operation_estimates exists
            if "operation_estimates" not in vram_config:
                logger.info(
                    "Adding operation_estimates to vram_strategies.yaml")
                updates = update_vram_strategies_with_defaults()
                vram_config.update(updates)

                config_loader.save_config_file(
                    "vram_strategies.yaml",
                    vram_config,
                    user_config=True
                )
                logger.info("✓ Updated VRAM strategies config")
            else:
                logger.info("VRAM strategies already has operation_estimates")
        else:
            logger.warning(
                f"vram_strategies.yaml not found at {vram_config_path}\n"
                "Run 'expandor --setup' to create base configuration first"
            )
    except Exception as e:
        logger.error(f"Failed to update VRAM config: {e}")
        return 1

    logger.info("\n✓ ControlNet setup complete!")
    logger.info("You can now use ControlNet features with Expandor")
    return 0


def main():
    """Main CLI entry point"""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("expandor.cli", level=log_level)

    # Handle special commands
    if args.setup:
        wizard = SetupWizard()
        wizard.run()
        return 0

    if hasattr(args, 'setup_controlnet') and args.setup_controlnet:
        return setup_controlnet(args, logger)

    if args.init_config:
        import shutil
        import yaml
        
        user_config_dir = Path.home() / ".config" / "expandor"
        user_config_dir.mkdir(parents=True, exist_ok=True)
        user_config_path = user_config_dir / "config.yaml"
        
        if user_config_path.exists():
            logger.info(f"User config already exists at {user_config_path}")
            response = input("Overwrite? (y/N): ")
            if response.lower() != 'y':
                return 0
        
        # Copy template and update version
        template_path = Path(__file__).parent.parent / "config" / "examples" / "user_config_example.yaml"
        
        # Read template and update version to 2.0
        with open(template_path, 'r') as f:
            template_config = yaml.safe_load(f)
        
        # Update to current version
        template_config['version'] = '2.0'
        
        # Write updated config
        with open(user_config_path, 'w') as f:
            yaml.dump(template_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"✓ Created user configuration at {user_config_path}")
        logger.info("Edit this file to customize Expandor for your system.")
        logger.info("\nCommon customizations:")
        logger.info("  - Set 'vram.limit_mb' for your GPU memory")
        logger.info("  - Adjust 'core.quality_preset' for default quality")
        logger.info("  - Configure 'paths' for your storage locations")
        logger.info("  - Add custom quality presets under 'custom_presets'")
        return 0

    if args.test:
        success = test_configuration(args.config)
        return 0 if success else 1

    try:
        # Validate arguments
        validate_args(args)

        # Load user configuration
        user_config_manager = UserConfigManager(args.config)
        user_config = user_config_manager.load()

        # Create pipeline configurator
        configurator = PipelineConfigurator(args.config)

        # Get list of input files
        if "*" in str(args.input):
            input_files = [Path(f) for f in glob.glob(str(args.input))]
            if not input_files:
                logger.error(f"No files matching pattern: {args.input}")
                return 1
        else:
            input_files = [args.input]

        logger.info(f"Processing {len(input_files)} image(s)")

        # Initialize Expandor with pipeline adapter
        logger.info("Initializing Expandor...")

        # Create pipeline adapter using factory method
        if args.model is None:
            raise ValueError(
                "Model name is required!\n"
                "Solutions:\n"
                "1. Specify with --model flag (e.g., --model sdxl)\n"
                "2. Set model in your config file\n"
                "3. Use expandor --setup to configure defaults"
            )
        model_name = args.model
        try:
            adapter = configurator.create_adapter(
                model_name, adapter_type="auto")
            logger.info(f"Created adapter for model: {model_name}")
        except ValueError as e:
            logger.error(f"Failed to create adapter: {e}")
            logger.info("Run 'expandor --setup' to configure models")
            return 1

        # Collect all LoRAs to apply
        loras_to_apply = []

        # Add manually specified LoRAs
        if args.lora:
            for lora_name in args.lora:
                # Find LoRA config by name
                for lora_config in user_config.loras:
                    if lora_config.name == lora_name and lora_config.enabled:
                        loras_to_apply.append(lora_config)
                        break
                else:
                    # FAIL LOUD - requested LoRA not found
                    raise ValueError(
                        f"LoRA '{lora_name}' not found or disabled in configuration. "
                        f"Available LoRAs: {[lora.name for lora in user_config.loras if lora.enabled]}"
                    )

        # Add auto-applied LoRAs based on prompt
        if args.prompt:
            auto_loras = user_config_manager.get_applicable_loras(args.prompt)
            for lora in auto_loras:
                if lora not in loras_to_apply:  # Avoid duplicates
                    loras_to_apply.append(lora)

        # Use LoRAManager to resolve conflicts and calculate weights
        if loras_to_apply:
            from ..config.lora_manager import LoRAConflictError, LoRAManager

            lora_manager = LoRAManager(logger)

            try:
                resolved_loras = lora_manager.resolve_lora_stack(
                    loras_to_apply)

                # Apply resolved LoRAs with adjusted weights
                for lora_config, adjusted_weight in resolved_loras:
                    adapter.load_lora(
                        lora_config.path,
                        adjusted_weight,  # Use adjusted weight
                        lora_config.name,
                    )
                    logger.info(
                        f"Loaded LoRA: {
                            lora_config.name} (weight: {
                            adjusted_weight:.2f})"
                    )

                # Get recommended inference steps
                recommended_steps = lora_manager.get_recommended_inference_steps(
                    resolved_loras)
                logger.info(
                    f"Recommended inference steps for LoRA stack: {recommended_steps}")

            except LoRAConflictError as e:
                # FAIL LOUD - LoRA conflicts are unrecoverable
                logger.error(f"LoRA conflict: {e}")
                raise

        # Create Expandor instance with adapter
        expandor = Expandor(pipeline_adapter=adapter, logger=logger)

        # Process each image
        success_count = 0
        # Add progress bar for batch processing (QUALITY OVER ALL: good UX)
        from tqdm import tqdm

        for i, input_path in enumerate(
            tqdm(input_files, desc="Processing images", unit="image"), 1
        ):
            logger.info(
                f"\n[{i}/{len(input_files)}] Processing {input_path.name}")

            # Parse resolution - may need to load image for multipliers
            if (
                args.resolution.endswith("x")
                and args.resolution[:-1].replace(".", "").isdigit()
            ):
                # Multiplier format - need to load image to get dimensions
                from PIL import Image

                with Image.open(input_path) as img:
                    current_resolution = img.size
                    target_resolution = parse_resolution(
                        args.resolution, current_resolution
                    )
                    logger.info(
                        f"Applying {
                            args.resolution} multiplier: {current_resolution} -> {target_resolution}")
            else:
                # Standard resolution format
                target_resolution = parse_resolution(args.resolution)

            # Generate output path
            if args.output and len(input_files) == 1:
                output_path = args.output
            else:
                output_path = generate_output_path(
                    input_path,
                    target_resolution,  # Now using parsed tuple
                    args.output_dir,
                    args.output_format,
                )

            # Generate seed
            if args.seed is not None:
                seed = args.seed + i - 1  # Increment seed for batch
            else:
                seed = abs(hash(f"{input_path}_{datetime.now()}")) % (2**32)

            # Create config
            config = configurator.create_expandor_config(
                source_image=input_path,
                target_resolution=target_resolution,  # Now using parsed tuple
                prompt=args.prompt or f"high quality {model_name} expansion",
                seed=seed,
                quality_preset=args.quality,
                strategy_override=args.strategy,
                save_stages=args.save_stages,
                stage_dir=args.stage_dir or Path("./expandor_stages"),
                verbose=args.verbose,
                artifact_detection_level=(
                    "disabled" if args.no_artifact_detection else None
                ),
                vram_limit_mb=(
                    args.vram_limit if args.vram_limit else None
                ),  # Override user config if specified
            )

            # Add negative prompt
            if args.negative_prompt:
                config.source_metadata["negative_prompt"] = args.negative_prompt

            # Process image
            try:
                if process_single_image(
                    expandor, input_path, output_path, config, args, logger
                ):
                    success_count += 1
            finally:
                # Always cleanup between images to prevent memory leaks
                if hasattr(expandor, "cleanup_iteration"):
                    expandor.cleanup_iteration()

                # Force garbage collection every 5 images
                if i % 5 == 0:
                    import gc

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    logger.debug(f"Performed memory cleanup after {i} images")

        # Final summary
        logger.info(f"\n{'=' * 50}")
        logger.info(
            f"Completed: {success_count}/{len(input_files)} images processed successfully"
        )

        return 0 if success_count == len(input_files) else 1

    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        return 130
    except ExpandorError as e:
        # Expandor-specific errors with details
        logger.error(f"Expandor Error: {e}")
        if hasattr(e, "stage"):
            logger.error(f"  Stage: {e.stage}")
        if hasattr(e, "details"):
            logger.error(f"  Details: {e.details}")
        logger.info("This is likely a configuration or resource issue.")
        return 1
    except VRAMError as e:
        logger.error(f"VRAM Error: {e}")
        logger.info("Try: --strategy cpu_offload or reduce resolution")
        logger.info("Or: expandor --vram-limit <MB> to set a lower limit")
        return 2
    except StrategyError as e:
        logger.error(f"Strategy Error: {e}")
        logger.info("Try a different strategy with --strategy")
        logger.info(
            "Available strategies: direct_upscale, progressive_outpaint, tiled_expansion, swpo"
        )
        return 3
    except QualityError as e:
        logger.error(f"Quality Error: {e}")
        logger.info("Output did not meet quality requirements")
        logger.info("Try: --quality ultra or --no-artifact-detection")
        return 4
    except ValueError as e:
        # Configuration errors
        logger.error(f"Configuration Error: {e}")
        logger.info("Run 'expandor --setup' to fix configuration issues")
        return 5
    except ImportError as e:
        # Missing dependencies
        logger.error(f"Import Error: {e}")
        logger.info("Missing required dependencies. Install with:")
        logger.info("  pip install expandor[diffusers]  # For AI models")
        logger.info("  pip install expandor[all]       # For all features")
        return 6
    except Exception as e:
        # This is our "fail loud" - don't hide unexpected errors
        logger.error(f"UNEXPECTED ERROR - THIS IS A BUG!")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {e}")
        logger.error(f"Full traceback:")
        logger.error(traceback.format_exc())
        logger.error(
            "Please report this at: https://github.com/expandor/expandor/issues"
        )
        logger.error("Include the full error output above in your report.")
        raise  # Re-raise for full traceback


if __name__ == "__main__":
    sys.exit(main())
