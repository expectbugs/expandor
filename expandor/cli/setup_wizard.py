"""
Interactive setup wizard for Expandor
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ..config.user_config import (LoRAConfig, ModelConfig, UserConfig,
                                  UserConfigManager)
from ..utils.logging_utils import setup_logger
from ..utils.path_resolver import PathResolver


class SetupWizard:
    """Interactive setup wizard for first-time configuration"""

    def __init__(self):
        self.logger = setup_logger(__name__)
        self.user_config_manager = UserConfigManager()
        self.config = UserConfig()

    def run(self):
        """Run the interactive setup wizard"""
        self._print_welcome()

        # Check if config already exists
        if self.user_config_manager.config_path.exists():
            if not self._confirm("Configuration already exists. Overwrite?"):
                self.logger.info("Setup cancelled.")
                return

        # Run setup steps
        self._setup_device()  # Ask for device preference first
        self._setup_models()
        self._setup_quality()
        self._setup_performance()
        self._setup_loras()
        self._setup_output()

        # Save configuration
        self._save_config()

        # Offer to test
        if self._confirm("\nWould you like to test the configuration?"):
            self._test_config()

        self._print_complete()

    def _setup_device(self):
        """Setup device preference"""
        self.logger.info("\n--- Device Configuration ---")
        self.logger.info("Select the device to use for processing.")

        device = self._get_choice(
            "Select device:",
            {
                "1": ("CUDA (NVIDIA GPU) - Recommended", "cuda"),
                "2": ("CPU - Slower but works everywhere", "cpu"),
                "3": ("MPS (Apple Silicon) - For Mac", "mps"),
            },
            default="1"
        )
        self.config.default_device = device
        self.logger.info(f"Device set to: {device}")

    def _print_welcome(self):
        """Print welcome message"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPANDOR SETUP WIZARD")
        self.logger.info("=" * 60)
        self.logger.info(
            "\nThis wizard will help you configure Expandor for first use.")
        self.logger.info("Press Ctrl+C at any time to cancel.\n")

    def _setup_models(self):
        """Setup model configurations"""
        self.logger.info("\n--- Model Configuration ---")
        self.logger.info(
            "Expandor supports multiple AI models for image expansion.")
        self.logger.info(
            "You can use local model files or download from HuggingFace.\n")

        # SDXL (default model)
        self.logger.info("1. Stable Diffusion XL (SDXL) - Recommended")
        if self._confirm("   Enable SDXL?", default=True):
            use_local = self._confirm(
                "   Do you have SDXL downloaded locally?")
            if use_local:
                path = self._get_path(
                    "   Enter path to SDXL model directory: ", path_type="directory")
                self.config.models["sdxl"] = ModelConfig(
                    path=str(path), dtype="fp16", device=self.config.default_device)
            else:
                self.logger.info(
                    "   Will use HuggingFace model (downloads on first use)")
                self.config.models["sdxl"] = ModelConfig(
                    model_id="stabilityai/stable-diffusion-xl-base-1.0",
                    variant="fp16",
                    dtype="fp16",
                    device=self.config.default_device,
                )

            # SDXL Refiner
            if self._confirm(
                    "   Enable SDXL Refiner? (improves quality but slower)"):
                self.config.models["sdxl_refiner"] = ModelConfig(
                    model_id="stabilityai/stable-diffusion-xl-refiner-1.0",
                    variant="fp16",
                    dtype="fp16",
                    device=self.config.default_device,
                    enabled=True,
                )

        # SD3
        self.logger.info("\n2. Stable Diffusion 3")
        if self._confirm("   Enable SD3?"):
            self.config.models["sd3"] = ModelConfig(
                model_id="stabilityai/stable-diffusion-3-medium",
                dtype="fp16",
                device=self.config.default_device,
                enabled=True,
            )

        # FLUX
        self.logger.info("\n3. FLUX")
        if self._confirm("   Enable FLUX?"):
            model_id = self._get_choice(
                "   Select FLUX variant:", {
                    "1": (
                        "FLUX.1-schnell (fast)", "black-forest-labs/FLUX.1-schnell"), "2": (
                        "FLUX.1-dev (quality)", "black-forest-labs/FLUX.1-dev"), }, )
            self.config.models["flux"] = ModelConfig(
                model_id=model_id,
                dtype="fp16",
                device=self.config.default_device,
                enabled=True)

        # Real-ESRGAN
        self.logger.info("\n4. Real-ESRGAN (for upscaling)")
        self.logger.info(
            "   Real-ESRGAN will be downloaded automatically when needed.")
        self.config.models["realesrgan"] = ModelConfig(
            dtype="fp32", device=self.config.default_device)

        # ControlNet
        self.logger.info("\n5. ControlNet (optional, improves quality)")
        if self._confirm("   Enable ControlNet support?"):
            self.config.models["controlnet_tile"] = ModelConfig(
                model_id="lllyasviel/control_v11f1e_sd15_tile",
                dtype="fp16",
                device=self.config.default_device,
                enabled=True,
            )

    def _setup_quality(self):
        """Setup quality preferences"""
        self.logger.info("\n--- Quality Settings ---")

        quality = self._get_choice(
            "Select default quality preset:",
            {
                "1": (
                    "fast - Quick results, good quality",
                    "fast"),
                "2": (
                    "balanced - Good balance of speed and quality",
                    "balanced"),
                "3": (
                    "high - High quality, slower",
                    "high"),
                "4": (
                    "ultra - Maximum quality, no time limit",
                    "ultra"),
            },
        )
        self.config.default_quality = quality

        self.config.auto_artifact_detection = self._confirm(
            "Enable automatic artifact detection and repair?", default=True
        )

    def _setup_performance(self):
        """Setup performance settings"""
        self.logger.info("\n--- Performance Settings ---")

        # VRAM limit
        if self._confirm("Set a VRAM limit? (recommended for shared systems)"):
            vram_mb = self._get_number(
                "Enter VRAM limit in MB (e.g., 8192 for 8GB): ",
                min_val=1024,
                max_val=49152,
            )
            self.config.max_vram_usage_mb = int(vram_mb)

        # CPU offload
        self.config.allow_cpu_offload = self._confirm(
            "Allow CPU offloading when VRAM is limited?", default=True
        )

        # Tiled processing
        self.config.allow_tiled_processing = self._confirm(
            "Allow tiled processing for large images?", default=True
        )

        # Cache directory
        if self._confirm("Set a custom cache directory for models?"):
            cache_dir = self._get_path(
                "Enter cache directory path: ", create=True, path_type="directory")
            self.config.cache_directory = str(cache_dir)

    def _setup_loras(self):
        """Setup LoRA configurations"""
        self.logger.info("\n--- LoRA Configuration ---")
        self.logger.info(
            "LoRAs are small model modifications that can enhance specific aspects.")

        if not self._confirm("Do you have any LoRA files to configure?"):
            return

        while True:
            self.logger.info("\nConfiguring new LoRA:")

            name = input("LoRA name (for reference): ").strip()
            if not name:
                break

            path = self._get_path("LoRA file path: ", must_exist=True, path_type="file")

            weight = self._get_number(
                "LoRA weight (0.1-2.0, default 1.0): ",
                default=1.0,
                min_val=0.1,
                max_val=2.0,
            )

            # Auto-apply keywords
            keywords = []
            self.logger.info(
                "Enter keywords that will auto-apply this LoRA (one per line, empty to finish):"
            )
            while True:
                keyword = input("  Keyword: ").strip()
                if not keyword:
                    break
                keywords.append(keyword)

            lora = LoRAConfig(
                name=name,
                path=str(path),
                weight=float(weight),
                auto_apply_keywords=keywords,
                enabled=True,
            )

            self.config.loras.append(lora)

            if not self._confirm("Add another LoRA?"):
                break

    def _setup_output(self):
        """Setup output preferences"""
        self.logger.info("\n--- Output Settings ---")

        # Output format
        fmt = self._get_choice(
            "Default output format:",
            {
                "1": ("PNG (lossless, larger files)", "png"),
                "2": ("JPEG (lossy, smaller files)", "jpg"),
                "3": ("WebP (modern format, good compression)", "webp"),
            },
        )
        self.config.output_format = fmt

        # Default output directory
        if self._confirm("Set a default output directory?"):
            output_dir = self._get_path(
                "Enter output directory path: ", create=True, path_type="directory")
            self.config.default_output_dir = str(output_dir)

        # Save stages
        self.config.save_intermediate_stages = self._confirm(
            "Save intermediate processing stages by default?", default=False
        )

        # Verbose logging
        self.config.verbose_logging = self._confirm(
            "Enable verbose logging by default?", default=False
        )

    def _save_config(self):
        """Save the configuration"""
        self.logger.info("\n--- Saving Configuration ---")

        try:
            self.user_config_manager.save(self.config)
            self.logger.info(
                f"✓ Configuration saved to: {
                    self.user_config_manager.config_path}"
            )

            # Also create example config
            self.user_config_manager.create_example_config()

        except Exception as e:
            self.logger.info(f"✗ Failed to save configuration: {e}")
            sys.exit(1)

    def _test_config(self):
        """Test the configuration"""
        self.logger.info("\n--- Testing Configuration ---")

        # Import test function
        from .utils import test_configuration

        success = test_configuration(self.user_config_manager.config_path)

        if not success:
            self.logger.info("\nSome tests failed. You may need to:")
            self.logger.info("- Download missing models")
            self.logger.info("- Check file paths")
            self.logger.info("- Ensure CUDA is properly installed")

    def _print_complete(self):
        """Print completion message"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("SETUP COMPLETE!")
        self.logger.info("=" * 60)
        self.logger.info("\nYou can now use Expandor with commands like:")
        self.logger.info("  expandor image.jpg --resolution 4K")
        self.logger.info(
            "  expandor photo.png --resolution 3840x1080 --quality ultra")
        self.logger.info("\nFor more options, run: expandor --help")
        self.logger.info("\nTo reconfigure later, run: expandor --setup")

    # Utility methods

    def _confirm(self, prompt: str, default: bool = False) -> bool:
        """Get yes/no confirmation"""
        default_str = "Y/n" if default else "y/N"
        response = input(f"{prompt} [{default_str}] ").strip().lower()

        if not response:
            return default

        return response in ["y", "yes", "true", "1"]

    def _get_choice(self, prompt: str, choices: Dict[str, tuple]) -> Any:
        """Get choice from options"""
        self.logger.info(f"\n{prompt}")

        for key, (description, _) in choices.items():
            self.logger.info(f"  {key}. {description}")

        while True:
            response = input("\nChoice: ").strip()
            if response in choices:
                return choices[response][1]
            self.logger.info("Invalid choice. Please try again.")

    def _get_path(
        self, prompt: str, must_exist: bool = False, create: bool = False,
        path_type: str = "auto"
    ) -> Path:
        """Get a file/directory path with validation
        
        Args:
            prompt: Prompt to show user
            must_exist: Path must already exist
            create: Create path if it doesn't exist
            path_type: 'file', 'directory', or 'auto' (default)
        """
        while True:
            path_str = input(prompt).strip()

            if not path_str:
                self.logger.info("Path cannot be empty. Please try again.")
                continue

            # Use PathResolver for consistent path handling
            path_resolver = PathResolver(self.logger)
            
            try:
                path = path_resolver.resolve_path(path_str, create=False, path_type=path_type)

                # Validate path isn't trying to access system directories
                forbidden_paths = ["/sys", "/proc", "/dev", "/boot"]
                path_str_lower = str(path).lower()
                if any(path_str_lower.startswith(fp)
                       for fp in forbidden_paths):
                    self.logger.info(
                        f"Error: Cannot access system directory: {path}")
                    continue

                # Check permissions
                if must_exist or create:
                    # Determine which path to check permissions on
                    test_path = path if path.exists() else path.parent

                    # Ensure parent exists for permission check
                    if not test_path.exists():
                        self.logger.info(
                            f"Error: Parent directory does not exist: {
                                test_path.parent}"
                        )
                        continue

                    # Check write permissions
                    if not os.access(test_path, os.W_OK):
                        self.logger.info(
                            f"Error: No write permission for {test_path}")
                        self.logger.info(
                            "Please choose a different location or fix permissions.")
                        continue

                    # Check read permissions if path must exist
                    if must_exist and not os.access(path, os.R_OK):
                        self.logger.info(
                            f"Error: No read permission for {path}")
                        continue

                # Validate the path exists if required
                if must_exist and not path.exists():
                    self.logger.info(f"Error: Path does not exist: {path}")
                    continue

                # Create the path if requested
                if create and not path.exists():
                    try:
                        if path.suffix:  # It's a file
                            path.parent.mkdir(parents=True, exist_ok=True)
                            # Create empty file
                            path.touch(exist_ok=True)
                        else:  # It's a directory
                            path.mkdir(parents=True, exist_ok=True)
                        self.logger.info(f"✓ Created: {path}")
                    except PermissionError:
                        self.logger.info(
                            f"Error: Permission denied creating {path}")
                        self.logger.info(
                            "Please choose a different location or run with appropriate permissions."
                        )
                        continue
                    except OSError as e:
                        self.logger.info(f"Error: Failed to create path: {e}")
                        continue

                # Final validation - ensure we can actually use this path
                if path.exists():
                    if path.is_dir() and path.suffix:
                        self.logger.info(
                            f"Error: {path} exists as a directory but you specified a file")
                        continue
                    elif path.is_file() and not path.suffix and create:
                        self.logger.info(
                            f"Error: {path} exists as a file but you want to create a directory")
                        continue

                return path

            except ValueError as e:
                self.logger.info(f"Error: Invalid path: {e}")
                continue
            except Exception as e:
                self.logger.info(
                    f"Error: Unexpected error processing path: {e}")
                self.logger.info(
                    "Please enter a valid file or directory path.")
                continue

    def _get_number(
        self,
        prompt: str,
        default: Optional[float] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> float:
        """Get a numeric value"""
        while True:
            response = input(prompt).strip()

            if not response and default is not None:
                return default

            try:
                value = float(response)

                if min_val is not None and value < min_val:
                    self.logger.info(f"Value must be at least {min_val}")
                    continue

                if max_val is not None and value > max_val:
                    self.logger.info(f"Value must be at most {max_val}")
                    continue

                return value

            except ValueError:
                self.logger.info("Please enter a valid number")
