"""
Process single image function for CLI
"""

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from ..core.config import ExpandorConfig
from ..core.expandor import Expandor


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
        result = expandor.expand(image=image, config=config, dry_run=args.dry_run)

        if args.dry_run:
            # Dry run - just report what would happen
            logger.info("DRY RUN: Expansion validated successfully")
            logger.info(f"  Would save to: {output_path}")
            logger.info(f"  Strategy: {result.strategy_used}")
            if result.metadata.get("would_succeed", True):
                logger.info("  Status: Would succeed ✓")
            else:
                logger.warning("  Status: Would fail due to insufficient resources ✗")
            return True

        # Check if successful
        if not result.success:
            logger.error(f"Expansion failed: {result.error}")
            return False

        # Save output (non-dry-run)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle different output formats
        # Load output quality config
        try:
            from ..utils.config_loader import ConfigLoader
            loader = ConfigLoader()
            quality_config = loader.load_config("output_quality.yaml")
        except:
            raise ValueError("Failed to load output quality configuration")
        
        if args.output_format and args.output_format != "png":
            if args.output_format in ["jpg", "jpeg"]:
                jpeg_config = quality_config.get('jpeg', {})
                result.final_image.save(
                    output_path, "JPEG", 
                    quality=jpeg_config.get('quality', 95), 
                    optimize=jpeg_config.get('optimize', True)
                )
            elif args.output_format == "webp":
                webp_config = quality_config.get('webp', {})
                result.final_image.save(
                    output_path, "WEBP", 
                    quality=webp_config.get('quality', 95), 
                    lossless=webp_config.get('lossless', False)
                )
            else:
                result.final_image.save(output_path, args.output_format.upper())
        else:
            # PNG by default
            png_config = quality_config.get('png', {})
            result.final_image.save(
                output_path, "PNG", 
                compress_level=png_config.get('compress_level', 1)
            )

        # Save metadata if requested
        if config.save_metadata:
            metadata_path = output_path.with_suffix(".json")
            import json

            with open(metadata_path, "w") as f:
                json_config = quality_config.get('json', {})
                json.dump(result.metadata, f, indent=json_config.get('indent', 2))
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
