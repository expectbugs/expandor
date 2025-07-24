"""
Base Strategy Class for all expansion strategies
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from ..core.config import ExpandorConfig
from ..core.exceptions import StrategyError, VRAMError
from ..core.result import StageResult
from ..core.vram_manager import VRAMManager
from ..processors.artifact_removal import ArtifactDetector


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

        # Initialize strategy-specific components
        self._initialize()

    def _initialize(self):
        """Override to perform strategy-specific initialization"""
        pass

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
        pass

    def validate_requirements(self):
        """
        Validate that strategy has required components

        Raises:
            StrategyError: If requirements not met
        """
        # Override in subclasses to check specific requirements
        pass

    def estimate_vram(self, config: ExpandorConfig) -> Dict[str, float]:
        """
        Estimate VRAM requirements for this strategy

        Args:
            config: ExpandorConfig instance

        Returns:
            Dictionary with VRAM estimates
        """
        target_w, target_h = config.target_resolution
        base_req = self.vram_manager.calculate_generation_vram(target_w, target_h)

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

    def check_vram_requirements(self, width: int, height: int) -> Dict[str, Any]:
        """Check if operation can fit in VRAM"""
        return self.vram_manager.determine_strategy(width, height)

    def track_boundary(self, position: int, direction: str, step: int, **kwargs):
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

    def save_stage(self, name: str, image_path: Path, metadata: Dict[str, Any]):
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
        if hasattr(self, "_context") and self._context and "temp_dir" in self._context:
            temp_base = self._context["temp_dir"]

        temp_path = temp_base / f"{name}_{timestamp}.png"
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        # Use lossless PNG
        image.save(temp_path, "PNG", compress_level=0)
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
