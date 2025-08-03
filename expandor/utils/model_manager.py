"""
Model management utilities for Expandor
Handles model downloading, caching, and availability checking
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from huggingface_hub import HfApi, model_info, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

from ..config.user_config import ModelConfig
from ..utils.logging_utils import setup_logger
from ..utils.path_resolver import PathResolver


class ModelManager:
    """
    Manages model downloading and availability checking
    FAIL LOUD: Any issues with models cause immediate failure
    QUALITY OVER ALL: Ensures models are properly downloaded and cached
    """

    def __init__(self,
                 cache_dir: Optional[Path] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize model manager

        Args:
            cache_dir: Custom cache directory for models
            logger: Logger instance
        """
        self.logger = logger or setup_logger(__name__)
        self.path_resolver = PathResolver(self.logger)
        self.cache_dir = cache_dir or self._get_default_cache_dir()
        self.api = HfApi()

        # Ensure cache directory exists using PathResolver
        self.cache_dir = self.path_resolver.resolve_path(self.cache_dir, create=True, path_type="directory")
        self.logger.info(f"Model cache directory: {self.cache_dir}")

    def _get_default_cache_dir(self) -> Path:
        """Get default cache directory"""
        # Use HuggingFace default cache if available
        hf_cache = os.environ.get("HF_HOME") or os.environ.get(
            "HUGGINGFACE_HUB_CACHE")
        if hf_cache:
            return Path(hf_cache)

        # Use ConfigurationManager for cache directory - NO HARDCODED VALUES
        from ..core.configuration_manager import ConfigurationManager
        config_manager = ConfigurationManager()
        
        # Get cache directory from configuration
        try:
            cache_dir = config_manager.get_value('paths.cache_dir')
            return Path(cache_dir)
        except ValueError:
            # FAIL LOUD if cache directory not configured
            raise ValueError(
                "Cache directory not configured. "
                "Please add 'paths.cache_dir' to your configuration file or "
                "set HF_HOME/HUGGINGFACE_HUB_CACHE environment variable."
            )

    def download_model(
        self,
        model_id: str,
        variant: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        force_download: bool = False,
    ) -> Path:
        """
        Download a model from HuggingFace Hub with progress bar

        FAIL LOUD: Authentication errors and missing models cause immediate failure
        QUALITY OVER ALL: Verifies download integrity

        Args:
            model_id: HuggingFace model ID (e.g., "stabilityai/stable-diffusion-xl-base-1.0")
            variant: Model variant (e.g., "fp16", "fp32")
            revision: Specific model revision/commit
            token: HuggingFace token for gated models
            force_download: Force re-download even if cached

        Returns:
            Path to downloaded model directory

        Raises:
            ValueError: If model not found or authentication fails
            RuntimeError: If download fails
        """
        self.logger.info(f"Downloading model: {model_id}")
        if variant:
            self.logger.info(f"  Variant: {variant}")

        # Check if we need authentication
        token = (
            token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )

        try:
            # Check model accessibility first
            try:
                info = model_info(model_id, revision=revision, token=token)

                # Check if model is gated and we have access
                if hasattr(info, "gated") and info.gated:
                    if not token:
                        # FAIL LOUD - gated model needs authentication
                        raise ValueError(
                            f"Model '{model_id}' is gated and requires authentication.\n"
                            f"Please set HF_TOKEN environment variable or run:\n"
                            f"  huggingface-cli login\n"
                            f"Then try again."
                        )
                    self.logger.info(
                        "Accessing gated model with authentication")

            except RepositoryNotFoundError:
                # FAIL LOUD - model doesn't exist
                raise ValueError(
                    f"Model '{model_id}' not found on HuggingFace Hub.\n"
                    f"Please check the model ID and try again.\n"
                    f"You can search for models at: https://huggingface.co/models"
                )
            except Exception as e:
                if "401" in str(e):
                    # FAIL LOUD - authentication failed
                    raise ValueError(
                        f"Authentication failed for model '{model_id}'.\n"
                        f"This model may be gated or private.\n"
                        f"Please ensure your HF_TOKEN is valid and has access to this model."
                    )
                raise

            # Prepare download kwargs
            download_kwargs = {
                "repo_id": model_id,
                "cache_dir": str(self.cache_dir),
                "force_download": force_download,
                "token": token,
                "local_dir_use_symlinks": False,  # Use actual files for reliability
            }

            if revision:
                download_kwargs["revision"] = revision

            # Add variant-specific file patterns if specified
            if variant:
                # Common variant patterns
                variant_patterns = {
                    "fp16": ["*fp16*", "*.fp16.*", "*f16*"],
                    "fp32": ["*fp32*", "*.fp32.*", "*f32*"],
                    "int8": ["*int8*", "*.int8.*", "*8bit*"],
                }

                if variant in variant_patterns:
                    download_kwargs["allow_patterns"] = variant_patterns[variant]
                else:
                    self.logger.warning(
                        f"Unknown variant '{variant}', downloading all files"
                    )

            # Download with progress
            self.logger.info("Starting download...")

            # Create custom progress bar for download
            # Note: snapshot_download shows its own progress, so we just wrap
            # it
            try:
                model_path = snapshot_download(**download_kwargs)

                # Verify download
                model_path = Path(model_path)
                if not model_path.exists():
                    raise RuntimeError(
                        f"Download appeared to succeed but model path doesn't exist: {model_path}")

                # Check for essential files
                essential_patterns = [
                    "*.bin", "*.safetensors", "*.ckpt", "*.pth"]
                has_model_files = any(list(model_path.glob(pattern))
                                      for pattern in essential_patterns)

                if not has_model_files:
                    # FAIL LOUD - no model files found
                    raise RuntimeError(
                        f"Download completed but no model files found in {model_path}\n"
                        f"Expected files matching: {essential_patterns}\n"
                        f"This may indicate a partial or corrupted download."
                    )

                self.logger.info(
                    f"✓ Model downloaded successfully to: {model_path}")
                return model_path

            except Exception as e:
                # FAIL LOUD with helpful message
                raise RuntimeError(
                    f"Failed to download model '{model_id}': {e}\n"
                    f"Possible solutions:\n"
                    f"  1. Check your internet connection\n"
                    f"  2. Ensure you have enough disk space in {self.cache_dir}\n"
                    f"  3. Try again with --force-download flag\n"
                    f"  4. Manually download from https://huggingface.co/{model_id}"
                )

        except Exception as e:
            # Re-raise with context
            self.logger.error(f"Model download failed: {e}")
            raise

    def verify_model_availability(
        self, model_config: ModelConfig, token: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Verify if a model is available locally or on HuggingFace

        Args:
            model_config: Model configuration to verify
            token: Optional HuggingFace token

        Returns:
            Tuple of (is_available, status_message)
        """
        # Check local path first
        if model_config.path:
            path = Path(model_config.path)
            if path.exists():
                # Verify it's a valid model file/directory
                if path.is_file():
                    # Check file extension
                    valid_extensions = [
                        ".bin", ".safetensors", ".ckpt", ".pth", ".pt"]
                    if path.suffix.lower() in valid_extensions:
                        return True, f"Local model file found: {path}"
                    else:
                        return (
                            False,
                            f"File exists but has invalid extension: {
                                path.suffix}",
                        )
                elif path.is_dir():
                    # Check for model files in directory
                    model_files = []
                    for ext in [".bin", ".safetensors", ".ckpt", ".pth"]:
                        model_files.extend(path.glob(f"*{ext}"))

                    if model_files:
                        return (
                            True,
                            f"Local model directory found with {
                                len(model_files)} model files",
                        )
                    else:
                        return (
                            False, f"Directory exists but contains no model files: {path}", )
            else:
                return False, f"Local path does not exist: {path}"

        # Check HuggingFace model
        if model_config.model_id:
            try:
                # Try to get model info
                token = token or os.environ.get("HF_TOKEN")
                info = model_info(model_config.model_id, token=token)

                # Check if it's cached locally
                cache_path = self._get_model_cache_path(model_config.model_id)
                if cache_path and cache_path.exists():
                    return (
                        True,
                        f"Model cached locally: {
                            model_config.model_id}",
                    )
                else:
                    # Model exists on HF but not cached
                    size_str = (
                        self._format_model_size(info)
                        if hasattr(info, "size")
                        else "unknown size"
                    )
                    return (
                        True,
                        f"Model available on HuggingFace ({size_str}): {
                            model_config.model_id}",
                    )

            except RepositoryNotFoundError:
                return (
                    False,
                    f"Model not found on HuggingFace: {
                        model_config.model_id}",
                )
            except Exception as e:
                if "401" in str(e):
                    return (
                        False,
                        f"Authentication required for model: {
                            model_config.model_id}",
                    )
                else:
                    return False, f"Error checking model: {e}"

        return False, "No model path or ID specified"

    def _get_model_cache_path(self, model_id: str) -> Optional[Path]:
        """Get the cache path for a model if it exists"""
        # Check various possible cache locations
        safe_model_id = model_id.replace("/", "--")

        # Check in cache directory
        possible_paths = [
            self.cache_dir / "hub" / f"models--{safe_model_id}",
            self.cache_dir / safe_model_id,
            self.cache_dir / model_id.split("/")[-1],
        ]

        for path in possible_paths:
            if path.exists() and path.is_dir():
                # Verify it contains model files
                if any(path.glob("*.bin")) or any(path.glob("*.safetensors")):
                    return path

        return None

    def _format_model_size(self, info: Any) -> str:
        """Format model size for display"""
        if not hasattr(info, "size"):
            return "unknown size"

        size_bytes = info.size
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    def get_model_info(
        self, model_id: str, token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed information about a model

        Args:
            model_id: HuggingFace model ID
            token: Optional HuggingFace token

        Returns:
            Dictionary with model information
        """
        try:
            token = token or os.environ.get("HF_TOKEN")
            info = model_info(model_id, token=token)

            return {
                "id": info.id,
                "author": info.author,
                "last_modified": (
                    str(info.lastModified) if hasattr(info, "lastModified") else None
                ),
                "private": info.private,
                "gated": getattr(info, "gated", False),
                "disabled": getattr(info, "disabled", False),
                "tags": info.tags if hasattr(info, "tags") else [],
                "pipeline_tag": getattr(info, "pipeline_tag", None),
                "library_name": getattr(info, "library_name", None),
                "size": self._format_model_size(info),
                "downloads": getattr(info, "downloads", 0),
                "likes": getattr(info, "likes", 0),
            }

        except Exception as e:
            # FAIL LOUD
            raise ValueError(f"Failed to get model info for '{model_id}': {e}")

    def clear_cache(self, model_id: Optional[str] = None):
        """
        Clear model cache

        Args:
            model_id: Specific model to clear, or None to clear all
        """
        if model_id:
            # Clear specific model
            cache_path = self._get_model_cache_path(model_id)
            if cache_path and cache_path.exists():
                import shutil

                shutil.rmtree(cache_path)
                self.logger.info(f"Cleared cache for model: {model_id}")
            else:
                self.logger.warning(f"No cache found for model: {model_id}")
        else:
            # Clear entire cache (with confirmation in real usage)
            self.logger.warning(
                "Clearing entire model cache is not implemented for safety"
            )
            self.logger.warning(
                "Please manually delete cache directory if needed: {self.cache_dir}"
            )

    def estimate_download_size(
        self, model_id: str, variant: Optional[str] = None
    ) -> int:
        """
        Estimate download size for a model

        Args:
            model_id: HuggingFace model ID
            variant: Model variant

        Returns:
            Estimated size in bytes
        """
        try:
            info = model_info(model_id)

            # If model has size info, use it
            if hasattr(info, "size"):
                base_size = info.size

                # Adjust for variant (rough estimates)
                if variant == "fp16":
                    return int(base_size * 0.5)  # FP16 is roughly half of FP32
                elif variant == "int8":
                    # INT8 is roughly quarter of FP32
                    return int(base_size * 0.25)

                return base_size

            # Fallback estimates based on model type
            if "xl" in model_id.lower():
                return 7 * 1024 * 1024 * 1024  # 7GB for SDXL models
            elif "flux" in model_id.lower():
                return 15 * 1024 * 1024 * 1024  # 15GB for FLUX models
            else:
                return 5 * 1024 * 1024 * 1024  # 5GB default

        except (AttributeError, ValueError, OSError, ConnectionError) as e:
            # Return conservative estimate
            self.logger.warning(
                f"Could not get model info for {model_id}: {e}")
            return 5 * 1024 * 1024 * 1024  # 5GB default
