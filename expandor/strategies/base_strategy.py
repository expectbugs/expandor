"""
Base Strategy Class for all expansion strategies
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image

from ..core.config import ExpandorConfig
from ..core.exceptions import StrategyError
from ..core.result import StageResult
from ..core.vram_manager import VRAMManager
from ..processors.artifact_removal import ArtifactDetector
from ..utils.path_resolver import PathResolver


class BaseExpansionStrategy(ABC):
    """Base class for all expansion strategies"""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize base strategy

        Args:
            config: Strategy-specific configuration
            metrics: Selection metrics from StrategySelector
            logger: Logger instance
        """
        self.config = config or {}
        self.metrics = metrics
        self.logger = logger or logging.getLogger(__name__)
        self.vram_manager = VRAMManager(self.logger)
        self.artifact_detector = ArtifactDetector(self.logger)

        # Pipeline placeholders - will be injected by orchestrator
        self.inpaint_pipeline = None
        self.refiner_pipeline = None
        self.img2img_pipeline = None

        # Tracking
        self.boundary_tracker = None  # Injected by orchestrator
        self.metadata_tracker = None  # Injected by orchestrator
        self.stage_results = []
        self.temp_files = []

        # Validate required parameters
        self._validate_required_params()

        # Initialize strategy-specific components
        self._initialize()

    def _validate_required_params(self):
        """Validate all required parameters are present"""
        # Get strategy name from class
        strategy_name = self.__class__.__name__.lower()
        if 'progressiveoutpaint' in strategy_name:
            strategy_name = 'progressive_outpaint'
        elif 'cpuoffload' in strategy_name:
            strategy_name = 'cpu_offload'
        elif 'tiledexpansion' in strategy_name:
            strategy_name = 'tiled_expansion'

        # Get required params from config if available
        if hasattr(self, 'config') and 'required_params' in self.config:
            required = self.config['required_params']
            params = self.config.get('parameters', self.config)

            missing = []
            for param in required:
                if param not in params:
                    missing.append(param)

            if missing:
                raise ValueError(
                    f"{self.__class__.__name__} missing required parameters: {missing}\n"
                    f"Add these to your strategy configuration or provide at runtime."
                )

    def _initialize(self):
        """Override to perform strategy-specific initialization"""

    @abstractmethod
    def execute(
        self, config: ExpandorConfig, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the expansion strategy

        Args:
            config: ExpandorConfig instance
            context: Execution context from orchestrator

        Returns:
            Dictionary with results including:
            - image_path: Path to result image
            - size: Final image size
            - stages: List of stage results
            - boundaries: List of boundary positions
            - metadata: Additional metadata
        """

    def validate_requirements(self):
        """
        Validate that strategy has required components

        Raises:
            StrategyError: If requirements not met
        """
        # Override in subclasses to check specific requirements

    def estimate_vram(self, config: ExpandorConfig) -> Dict[str, float]:
        """
        Estimate VRAM requirements for this strategy

        Args:
            config: ExpandorConfig instance

        Returns:
            Dictionary with VRAM estimates
        """
        target_w, target_h = config.target_resolution
        base_req = self.vram_manager.calculate_generation_vram(
            target_w, target_h)

        return {
            "base_vram_mb": base_req,
            "peak_vram_mb": base_req * 1.2,  # 20% buffer
            "strategy_overhead_mb": 0,  # Override in subclasses
        }

    def check_vram(self, required_mb: float) -> bool:
        """
        Check if required VRAM is available

        Args:
            required_mb: Required VRAM in MB

        Returns:
            True if VRAM available, False otherwise
        """
        available_mb = self.vram_manager.get_available_vram()
        if available_mb is None:
            return False
        return available_mb >= required_mb

    def check_vram_requirements(
            self, width: int, height: int) -> Dict[str, Any]:
        """Check if operation can fit in VRAM"""
        return self.vram_manager.determine_strategy(width, height)

    def track_boundary(
            self,
            position: int,
            direction: str,
            step: int,
            **kwargs):
        """Track expansion boundary for seam detection"""
        if self.boundary_tracker:
            self.boundary_tracker.add_boundary(
                position=position, direction=direction, step=step, **kwargs
            )

    def record_stage(
        self,
        name: str,
        method: str,
        input_size: Tuple[int, int],
        output_size: Tuple[int, int],
        start_time: float,
        **kwargs,
    ):
        """Record a stage completion"""
        stage = StageResult(
            name=name,
            method=method,
            input_size=input_size,
            output_size=output_size,
            duration_seconds=time.time() - start_time,
            vram_used_mb=self.vram_manager.get_available_vram() or 0.0,
            **kwargs,
        )

        self.stage_results.append(stage)

        # Call context callback if provided
        if (
            hasattr(self, "_context")
            and self._context
            and "stage_callback" in self._context
        ):
            self._context["stage_callback"](stage)

    def save_stage(self, name: str, image_path: Path,
                   metadata: Dict[str, Any]):
        """
        Save stage information

        Args:
            name: Stage name
            image_path: Path to stage image
            metadata: Stage metadata
        """
        if self.metadata_tracker:
            self.metadata_tracker.add_stage(
                name, {"image_path": str(image_path), "metadata": metadata}
            )

    def save_temp_image(self, image: Image.Image, name: str) -> Path:
        """Save temporary image and track for cleanup"""
        timestamp = int(time.time() * 1000)

        # Get temp directory from context or use default
        temp_base = Path("temp")
        if hasattr(
                self,
                "_context") and self._context and "temp_dir" in self._context:
            temp_base = self._context["temp_dir"]

        # Get file format config
        from ..core.configuration_manager import ConfigurationManager
        config_manager = ConfigurationManager()
        default_ext = config_manager.get_value("output.default_extension")
        png_compression = config_manager.get_value("output.formats.png.compression")
        
        temp_path = temp_base / f"{name}_{timestamp}{default_ext}"
        path_resolver = PathResolver(self.logger)
        path_resolver.resolve_path(temp_path.parent, create=True, path_type="directory")

        # Use configured PNG compression
        image.save(temp_path, "PNG", compress_level=png_compression)
        self.temp_files.append(temp_path)

        return temp_path

    def cleanup(self):
        """Clean up temporary files and clear cache"""
        for path in self.temp_files:
            try:
                if path.exists():
                    path.unlink()
            except Exception as e:
                self.logger.debug(f"Failed to delete temp file {path}: {e}")
        self.temp_files.clear()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _cleanup_temp_files(self, keep_last: int = 5):
        """Clean up old temp files to prevent memory growth"""
        if len(self.temp_files) > keep_last:
            # Clean oldest files
            for file_path in self.temp_files[:-keep_last]:
                try:
                    if Path(file_path).exists():
                        Path(file_path).unlink()
                except Exception as e:
                    self.logger.debug(
                        f"Failed to clean temp file {file_path}: {e}")
            # Keep only recent files
            self.temp_files = self.temp_files[-keep_last:]

    def validate_image_path(self, path: Path) -> Image.Image:
        """Validate and load image from path"""
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            image = Image.open(path)
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as e:
            raise StrategyError(f"Failed to load image {path}: {str(e)}")

    def validate_inputs(self, config: ExpandorConfig) -> None:
        """Validate configuration inputs"""
        if config.target_resolution[0] <= 0 or config.target_resolution[1] <= 0:
            raise ValueError(
                f"Invalid target resolution: {
                    config.target_resolution}"
            )

        if not config.prompt:
            raise ValueError("Prompt cannot be empty")

        if config.seed < 0:
            raise ValueError(f"Invalid seed: {config.seed}")
