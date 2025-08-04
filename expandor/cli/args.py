"""
Command-line argument parsing for Expandor
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

from .. import __version__


def parse_resolution(
    resolution_str: str, current_resolution: Optional[Tuple[int, int]] = None
) -> Tuple[int, int]:
    """
    Parse resolution string into tuple

    Supports formats:
    - "3840x2160" - Explicit dimensions
    - "4K" (3840x2160) - Named presets
    - "1080p" (1920x1080) - Standard resolutions
    - "ultrawide" (3440x1440) - Aspect ratio presets
    - "21:9" (2560x1080) - Aspect ratios
    - "2x", "1.5x" - Multipliers (requires current_resolution)

    Args:
        resolution_str: Resolution string
        current_resolution: Current resolution for multiplier calculations

    Returns:
        Tuple of (width, height)
    """
    # Handle multipliers like "2x", "1.5x"
    if resolution_str.endswith("x") and len(resolution_str) > 1:
        multiplier_str = resolution_str[:-1]
        try:
            multiplier = float(multiplier_str)
            if multiplier <= 0:
                raise ValueError("Multiplier must be positive")

            if not current_resolution:
                raise argparse.ArgumentTypeError(
                    f"Multiplier format '{resolution_str}' requires knowing current resolution. "
                    "Use absolute resolution (e.g., '3840x2160') or load the image first."
                )

            new_width = int(current_resolution[0] * multiplier)
            new_height = int(current_resolution[1] * multiplier)
            return (new_width, new_height)

        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Invalid multiplier format: {resolution_str}. "
                f"Use a number followed by 'x' (e.g., '2x', '1.5x'). Error: {e}"
            )

    # Preset resolutions
    presets = {
        # Standard resolutions
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "4k": (3840, 2160),
        "5k": (5120, 2880),
        "8k": (7680, 4320),
        # Aspect ratio presets
        "16:9": (1920, 1080),
        "21:9": (2560, 1080),
        "32:9": (3840, 1080),
        "9:16": (1080, 1920),  # Portrait
        # Named presets
        "ultrawide": (3440, 1440),
        "superultrawide": (5120, 1440),
        "portrait4k": (2160, 3840),
        "square4k": (3840, 3840),
        # Mobile wallpapers
        "iphone": (1170, 2532),
        "ipad": (2048, 2732),
        "android": (1440, 3200),
    }

    # Check presets first (case insensitive)
    resolution_lower = resolution_str.lower()
    if resolution_lower in presets:
        return presets[resolution_lower]

    # Try to parse WIDTHxHEIGHT format
    if "x" in resolution_str:
        try:
            width, height = resolution_str.split("x")
            return (int(width), int(height))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid resolution format: {resolution_str}. "
                "Use WIDTHxHEIGHT (e.g., 3840x2160) or a preset (e.g., 4K)"
            )

    raise argparse.ArgumentTypeError(
        f"Unknown resolution: {resolution_str}. "
        "Use WIDTHxHEIGHT format or presets: 4K, 1080p, ultrawide, etc."
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser"""

    parser = argparse.ArgumentParser(
        prog="expandor",
        description="Universal Image Resolution Adaptation System",
        epilog="Examples:\n"
        "  expandor image.jpg --resolution 4K\n"
        "  expandor photo.png --resolution 3840x1080 --quality ultra\n"
        '  expandor art.jpg --resolution ultrawide --model flux --prompt "epic landscape"\n'
        "  expandor batch/*.jpg --resolution 1080p --output-dir upscaled/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional arguments
    parser.add_argument(
        "input",
        type=Path,
        nargs='?',  # Make optional for --test and --setup
        help="Input image path or pattern (supports wildcards for batch processing)",
    )

    # Required arguments (but not for --test or --setup)
    parser.add_argument(
        "-r",
        "--resolution",
        type=str,  # Changed from parse_resolution to str to allow deferred parsing
        required=False,  # Will validate in validate_args based on other flags
        help="Target resolution (e.g., 3840x2160, 4K, ultrawide, 16:9, 2x, 1.5x)",
    )

    # Output arguments
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path (default: input_name_resolution.ext)",
    )

    parser.add_argument(
        "--output-dir", type=Path, help="Output directory for batch processing"
    )

    parser.add_argument(
        "--output-format",
        choices=["png", "jpg", "jpeg", "webp"],
        help="Output format (default: same as input or PNG)",
    )

    # Quality and processing
    parser.add_argument(
        "-q",
        "--quality",
        choices=["fast", "balanced", "high", "ultra"],
        help="Quality preset (default: from user config or balanced)",
    )

    parser.add_argument(
        "-m",
        "--model",
        choices=["sdxl", "sd3", "flux"],
        default="sdxl",
        help="Model to use for expansion (default: sdxl)",
    )

    parser.add_argument(
        "--strategy",
        choices=[
            "direct",
            "progressive",
            "swpo",
            "tiled",
            "cpu_offload",
            "hybrid",
            "auto"],  # Add auto option
        help="Force specific expansion strategy (auto = let system choose)",
    )

    # Enhancement
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Enhancement prompt for better quality expansion",
    )

    parser.add_argument(
        "--negative-prompt",
        type=str,
        help="Negative prompt to avoid certain features")

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility")

    # Advanced options
    parser.add_argument(
        "--lora",
        action="append",
        help="Apply LoRA by name (can be used multiple times)",
    )

    parser.add_argument(
        "--no-artifact-detection",
        action="store_true",
        help="Disable automatic artifact detection and repair",
    )

    parser.add_argument(
        "--no-controlnet",
        action="store_true",
        help="Disable ControlNet even if available",
    )

    parser.add_argument(
        "--vram-limit",
        type=int,
        help="VRAM limit in MB (default: auto-detect)")

    # Debugging and output
    parser.add_argument(
        "--save-stages",
        action="store_true",
        help="Save intermediate processing stages")

    parser.add_argument(
        "--stage-dir",
        type=Path,
        help="Directory for saving stages (default: ./expandor_stages)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without processing",
    )

    # Special commands
    parser.add_argument(
        "--setup", action="store_true", help="Run interactive setup wizard"
    )

    parser.add_argument(
        "--setup-controlnet",
        action="store_true",
        help="Set up ControlNet configuration files")

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing configuration files (used with --setup-controlnet)")

    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Initialize user configuration file")

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test configuration and model availability")

    parser.add_argument(
        "--config",
        type=Path,
        help="Use custom config file instead of ~/.config/expandor/config.yaml",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate parsed arguments

    Args:
        args: Parsed arguments

    Raises:
        ValueError: If arguments are invalid
    """
    # Special cases that don't need input/resolution
    if args.setup or args.test or (
        hasattr(
            args,
            'setup_controlnet') and args.setup_controlnet) or args.init_config:
        return

    # For normal operation, input and resolution are required
    if not args.input:
        raise ValueError(
            "Input file is required (except for --setup, --test, --setup-controlnet, or --init-config)")

    if not args.resolution:
        raise ValueError("Resolution is required (use -r or --resolution)")

    # Check if input exists (unless it's a pattern)
    if "*" not in str(args.input) and not args.input.exists():
        raise ValueError(
            f"File not found: {args.input}\n"
            f"Please check the file path and try again."
        )

    # Validate output options
    if args.output and args.output_dir:
        raise ValueError("Cannot specify both --output and --output-dir")

    # Resolution validation is now done after parsing in main.py
    # since args.resolution is a string at this point
