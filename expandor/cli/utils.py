"""
CLI utility functions
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from ..config.pipeline_config import PipelineConfigurator
from ..config.user_config import UserConfigManager
from ..core.result import ExpandorResult


def generate_output_path(
    input_path: Path,
    resolution: Tuple[int, int],
    output_dir: Optional[Path] = None,
    output_format: Optional[str] = None,
) -> Path:
    """
    Generate output path based on input and parameters

    Args:
        input_path: Input file path
        resolution: Target resolution tuple
        output_dir: Output directory (optional)
        output_format: Output format (optional)

    Returns:
        Generated output path
    """
    # Determine output directory
    if output_dir:
        base_dir = output_dir
    else:
        base_dir = input_path.parent

    # Generate filename
    width, height = resolution
    base_name = input_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine extension
    if output_format:
        ext = f".{output_format}"
    else:
        # Get extension from input file or use configured default
        if input_path.suffix:
            ext = input_path.suffix
        else:
            # FAIL LOUD if no extension and no default configured
            from ..core.configuration_manager import ConfigurationManager
            config_manager = ConfigurationManager()
            try:
                ext = config_manager.get_value("output.default_extension")
            except ValueError:
                raise ValueError(
                    "Cannot determine output file extension!\n"
                    "Input file has no extension and no default configured.\n"
                    "Solutions:\n"
                    "1. Use --format to specify output format\n"
                    "2. Configure output.default_extension in config\n"
                    "3. Use input files with extensions"
                )

    # Build filename
    output_name = f"{base_name}_{width}x{height}_{timestamp}{ext}"

    return base_dir / output_name


def print_summary(result: ExpandorResult, output_path: Path):
    """
    Print detailed processing summary

    Args:
        result: ExpandorResult object
        output_path: Output file path
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("EXPANSION COMPLETE")
    logger.info("=" * 60)

    logger.info(f"\nOutput: {output_path}")
    logger.info(f"Final size: {result.final_size[0]}x{result.final_size[1]}")
    logger.info(f"Strategy: {result.strategy_used}")
    logger.info(f"Total time: {result.total_time:.2f}s")

    if result.vram_peak_mb:
        logger.info(f"Peak VRAM: {result.vram_peak_mb:.0f}MB")

    # Stage breakdown
    if result.stages:
        logger.info(f"\nStages ({len(result.stages)}):")
        for i, stage in enumerate(result.stages, 1):
            logger.info(f"  {i}. {stage.name}: {stage.duration:.2f}s")
            if stage.vram_used_mb:
                logger.info(f"     VRAM: {stage.vram_used_mb:.0f}MB")

    # Metadata
    if result.metadata:
        logger.info("\nMetadata:")
        if "boundaries_tracked" in result.metadata:
            logger.info(
                f"  Boundaries tracked: {
                    result.metadata['boundaries_tracked']}"
            )
        if "artifacts_detected" in result.metadata:
            logger.info(
                f"  Artifacts detected: {
                    result.metadata['artifacts_detected']}"
            )
        if "artifacts_repaired" in result.metadata:
            logger.info(
                f"  Artifacts repaired: {
                    result.metadata['artifacts_repaired']}"
            )
        if "refinement_passes" in result.metadata:
            logger.info(
                f"  Refinement passes: {
                    result.metadata['refinement_passes']}"
            )

    logger.info("=" * 60)


def print_stage_info(stage_name: str, info: str):
    """Print stage information in a formatted way"""
    logger = logging.getLogger(__name__)
    logger.info(f"[{stage_name}] {info}")


def test_configuration(config_path: Optional[Path] = None) -> bool:
    """
    Test configuration and model availability

    Args:
        config_path: Custom config path

    Returns:
        True if all tests pass
    """
    logger = logging.getLogger("expandor.test")

    logger.info("Testing Expandor Configuration")
    logger.info("=" * 50)

    # Load user config
    logger.info("\n1. Loading user configuration...")
    try:
        manager = UserConfigManager(config_path)
        user_config = manager.load()
        logger.info("✓ User configuration loaded successfully")
        logger.info(f"  Config path: {manager.config_path}")
    except Exception as e:
        logger.info(f"✗ Failed to load user configuration: {e}")
        return False

    # Validate models
    logger.info("\n2. Validating model configurations...")
    configurator = PipelineConfigurator(config_path)
    validation = configurator.validate_models(user_config.models)

    all_valid = True
    for model_name, is_valid in validation.items():
        model_config = user_config.models.get(model_name)
        if model_config and model_config.enabled:
            status = "✓" if is_valid else "✗"
            logger.info(f"  {status} {model_name}: ", end="")

            if model_config.path:
                logger.info(f"local path = {model_config.path}")
            elif model_config.model_id:
                logger.info(f"HuggingFace = {model_config.model_id}")
            else:
                logger.info("no path or model_id configured")

            if not is_valid:
                all_valid = False

    # Check LoRAs
    if user_config.loras:
        logger.info("\n3. Checking LoRA configurations...")
        for lora in user_config.loras:
            if lora.enabled:
                exists = Path(lora.path).exists()
                status = "✓" if exists else "✗"
                logger.info(f"  {status} {lora.name}: {lora.path}")
                if not exists:
                    all_valid = False

    # Check VRAM
    logger.info("\n4. Checking GPU/VRAM...")
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_mb = vram_bytes / (1024 * 1024)
            logger.info(f"✓ CUDA available: {device_name}")
            logger.info(f"  Total VRAM: {vram_mb:.0f}MB")

            # Check current usage
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            logger.info(f"  Currently allocated: {allocated:.0f}MB")
            logger.info(f"  Available: {vram_mb - allocated:.0f}MB")
        else:
            logger.info("✗ CUDA not available - will use CPU (slow)")
            all_valid = False
    except Exception as e:
        logger.info(f"✗ Failed to check CUDA: {e}")
        all_valid = False

    # Summary
    logger.info("\n" + "=" * 50)
    if all_valid:
        logger.info("✓ All tests passed! Expandor is ready to use.")
    else:
        logger.info("✗ Some tests failed. Please check your configuration.")
        logger.info("  Run 'expandor --setup' to reconfigure.")

    return all_valid
