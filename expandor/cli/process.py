"""
Process single image function for CLI
"""

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from ..core.config import ExpandorConfig
from ..core.expandor import Expandor
from ..utils.path_resolver import PathResolver


def process_single_image(
    expandor: Expandor,
    input_path: Path,
    output_path: Path,
    config: ExpandorConfig,
    args: Any,
    logger: logging.Logger,
) -> bool:
    """
    Process a single image

    Args:
        expandor: Expandor instance
        input_path: Input image path
        output_path: Output image path
        config: ExpandorConfig
        args: Command line arguments
        logger: Logger instance

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing: {input_path} -> {output_path}")

        # Load the image
        image = Image.open(input_path)

        # Update config with any CLI overrides
        if args.negative_prompt and not config.negative_prompt:
            config.negative_prompt = args.negative_prompt

        # Process image
        result = expandor.expand(config)

        if args.dry_run:
            # Dry run - just report what would happen
            logger.info("DRY RUN: Expansion validated successfully")
            logger.info(f"  Would save to: {output_path}")
            logger.info(f"  Strategy: {result.strategy_used}")
            # Check if would_succeed key exists in metadata
            if "would_succeed" not in result.metadata:
                raise ValueError(
                    "Missing 'would_succeed' key in dry run metadata!\n"
                    "This indicates an error in the expansion strategy.\n"
                    "Please report this issue."
                )
            if result.metadata["would_succeed"]:
                logger.info("  Status: Would succeed ✓")
            else:
                logger.warning(
                    "  Status: Would fail due to insufficient resources ✗")
            return True

        # Check if successful
        if not result.success:
            logger.error(f"Expansion failed: {result.error}")
            return False

        # Save output (non-dry-run)
        # Use PathResolver for consistent path handling
        path_resolver = PathResolver(logger)
        output_dir = path_resolver.resolve_path(output_path.parent, create=True, path_type="directory")
        
        # Load the image from result path if not already loaded
        if result.image is None:
            logger.debug(f"Loading result image from: {result.image_path}")
            result.image = Image.open(result.image_path)

        # Handle different output formats
        # Use ConfigurationManager for all output settings - NO HARDCODED VALUES
        from ..core.configuration_manager import ConfigurationManager
        config_manager = ConfigurationManager()
        
        # Get quality configuration for current preset
        quality_config = config_manager.get_value(f'quality_presets.{config.quality_preset}')

        if args.output_format and args.output_format != "png":
            if args.output_format in ["jpg", "jpeg"]:
                # FAIL LOUD - get required JPEG configuration
                jpeg_quality = config_manager.get_value('output.formats.jpeg.quality')
                jpeg_optimize = config_manager.get_value('output.formats.jpeg.optimize')
                
                result.image.save(
                    output_path, "JPEG",
                    quality=jpeg_quality,
                    optimize=jpeg_optimize
                )
            elif args.output_format == "webp":
                # FAIL LOUD - get required WebP configuration
                webp_quality = config_manager.get_value('output.formats.webp.quality')
                webp_lossless = config_manager.get_value('output.formats.webp.lossless')
                
                result.image.save(
                    output_path, "WEBP",
                    quality=webp_quality,
                    lossless=webp_lossless
                )
            else:
                result.image.save(
                    output_path, args.output_format.upper())
        else:
            # PNG by default
            # FAIL LOUD - get required PNG configuration
            png_compress_level = config_manager.get_value('output.formats.png.compression')
            
            result.image.save(
                output_path, "PNG",
                compress_level=png_compress_level
            )

        # Save metadata if stages are being saved (implies user wants metadata)
        if config.save_stages:
            metadata_path = output_path.with_suffix(".json")
            import json

            with open(metadata_path, "w") as f:
                # Get JSON config from quality config - FAIL LOUD if missing
                if 'json' not in quality_config:
                    raise ValueError(
                        "JSON configuration missing from quality config!\n"
                        "This should be defined in the quality preset.\n"
                        "Please check your configuration files."
                    )
                json_config = quality_config['json']
                
                # Get indent value - FAIL LOUD if missing
                if 'indent' not in json_config:
                    raise ValueError(
                        "JSON indent configuration missing!\n"
                        "Please add 'indent' to the json section of your quality config."
                    )
                
                json.dump(
                    result.metadata,
                    f,
                    indent=json_config['indent'])
            logger.debug(f"Saved metadata to: {metadata_path}")

        # Report results
        logger.info(f"✓ Success! Saved to: {output_path}")
        logger.info(f"  Strategy used: {result.strategy_used}")
        logger.info(f"  Processing time: {result.processing_time:.1f}s")

        if hasattr(result, "stages") and result.stages and args.save_stages:
            logger.info(f"  Stages saved: {len(result.stages)} images")

        return True

    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        if args.verbose:
            import traceback

            logger.debug(traceback.format_exc())
        return False
