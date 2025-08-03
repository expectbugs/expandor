"""
Metadata tracking for Expandor operations
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class MetadataTracker:
    """
    Tracks all metadata during expansion operations

    Provides:
    - Operation event logging
    - Stage tracking
    - Performance metrics
    - Debug information collection
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize metadata tracker"""
        self.logger = logger or logging.getLogger(__name__)
        self.reset()

    def reset(self):
        """Reset tracker for new operation"""
        self.operation_id = f"op_{int(time.time() * 1000)}"
        self.start_time = time.time()
        self.current_stage = "initialization"
        self.events = []
        self.metrics = {}
        self.config_snapshot = None
        self.stage_timings = {}
        self.stage_start = None

    def start_operation(self, config) -> str:
        """
        Start tracking a new operation

        Args:
            config: ExpandorConfig instance

        Returns:
            operation_id string
        """
        self.reset()

        # Snapshot configuration
        self.config_snapshot = {
            "source_image": (
                str(config.source_image)
                if isinstance(config.source_image, Path)
                else "<PIL.Image>"
            ),
            "target_resolution": config.get_target_resolution(),
            "quality_preset": config.quality_preset,
            "prompt": (
                config.prompt[:100] + "..."
                if len(config.prompt) > 100
                else config.prompt
            ),
            "negative_prompt": config.negative_prompt,
            "seed": config.seed,
            "strategy": config.strategy,
            "strategy_override": config.strategy_override,
            "denoising_strength": config.denoising_strength,
            "guidance_scale": config.guidance_scale,
            "num_inference_steps": config.num_inference_steps,
            "model_type": config.source_metadata.get("model", "unknown"),
        }

        # Record start event
        self.record_event(
            "operation_start", {
                "operation_id": self.operation_id, "config": self.config_snapshot}, )

        self.logger.debug(f"Started tracking operation: {self.operation_id}")
        return self.operation_id

    def track_operation(self, operation_name: str, metadata: Dict[str, Any]):
        """
        Track a specific operation (alias for record_event)

        Args:
            operation_name: Name of the operation
            metadata: Operation metadata
        """
        self.record_event(f"operation_{operation_name}", metadata)

    def enter_stage(self, stage_name: str):
        """
        Mark entry into a new stage

        Args:
            stage_name: Name of the stage
        """
        # End previous stage timing
        if self.stage_start is not None:
            elapsed = time.time() - self.stage_start
            self.stage_timings[self.current_stage] = elapsed

        # Start new stage
        self.current_stage = stage_name
        self.stage_start = time.time()

        self.record_event(
            "stage_enter",
            {"stage": stage_name, "timestamp": datetime.now().isoformat()},
        )

    def exit_stage(self, success: bool = True, error: Optional[str] = None):
        """
        Mark exit from current stage

        Args:
            success: Whether stage completed successfully
            error: Error message if failed
        """
        if self.stage_start is not None:
            elapsed = time.time() - self.stage_start
            self.stage_timings[self.current_stage] = elapsed
            self.stage_start = None

        self.record_event(
            "stage_exit",
            {
                "stage": self.current_stage,
                "success": success,
                "duration": self.stage_timings.get(self.current_stage, 0),
                "error": error,
            },
        )

    def record_event(self, event_type: str, data: Dict[str, Any]):
        """
        Record an event during operation

        Args:
            event_type: Type of event (e.g., "strategy_selected", "vram_check")
            data: Event data
        """
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "relative_time": time.time() - self.start_time,
            "stage": self.current_stage,
            "data": data,
        }

        self.events.append(event)

        # Log significant events
        if event_type in ["strategy_selected", "stage_complete", "error"]:
            self.logger.debug(f"Event: {event_type} - {data}")

    def record_metric(self, metric_name: str, value: Any):
        """
        Record a performance metric

        Args:
            metric_name: Name of metric
            value: Metric value
        """
        self.metrics[metric_name] = value

    def add_stage(self, stage_name: str, metadata: Dict[str, Any]):
        """
        Add metadata for a specific stage (alias for add_stage_metadata)

        Args:
            stage_name: Name of stage
            metadata: Stage-specific metadata
        """
        self.add_stage_metadata(stage_name, metadata)

    def add_stage_metadata(self, stage_name: str, metadata: Dict[str, Any]):
        """
        Add metadata for a specific stage

        Args:
            stage_name: Name of stage
            metadata: Stage-specific metadata
        """
        self.record_event(
            "stage_metadata", {
                "stage": stage_name, "metadata": metadata})

    def get_summary(self) -> Dict[str, Any]:
        """
        Get operation summary (alias for get_operation_log)

        Returns:
            Dictionary with operation summary
        """
        return self.get_operation_log()

    def get_operation_log(self) -> Dict[str, Any]:
        """
        Get complete operation log

        Returns:
            Dictionary with all operation data
        """
        return {
            "operation_id": self.operation_id,
            "duration": time.time() - self.start_time,
            "config": self.config_snapshot,
            "events": self.events,
            "metrics": self.metrics,
            "stage_timings": self.stage_timings,
            "event_summary": self._summarize_events(),
        }

    def get_partial_result(self) -> Dict[str, Any]:
        """
        Get partial result for in-progress operation

        Returns:
            Dictionary with partial operation data
        """
        return {
            "operation_id": self.operation_id,
            "elapsed_time": time.time() - self.start_time,
            "current_stage": self.current_stage,
            "completed_stages": list(self.stage_timings.keys()),
            "event_count": len(self.events),
            "last_event": self.events[-1] if self.events else None,
        }

    def save_operation_log(self, filepath: Path):
        """
        Save operation log to file

        Args:
            filepath: Path to save log file
        """
        log_data = self.get_operation_log()

        with open(filepath, "w") as f:
            json.dump(log_data, f, indent=2, default=str)

        self.logger.debug(f"Saved operation log to: {filepath}")

    def _summarize_events(self) -> Dict[str, Any]:
        """Create summary of events by type"""
        summary = {}

        for event in self.events:
            event_type = event["type"]
            if event_type not in summary:
                summary[event_type] = {
                    "count": 0,
                    "first_at": event["relative_time"],
                    "last_at": event["relative_time"],
                }

            summary[event_type]["count"] += 1
            summary[event_type]["last_at"] = event["relative_time"]

        return summary

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary

        Returns:
            Dictionary with performance metrics
        """
        total_time = time.time() - self.start_time

        # Calculate stage percentages
        stage_percentages = {}
        for stage, duration in self.stage_timings.items():
            stage_percentages[stage] = (duration / total_time) * 100

        return {
            "total_duration": total_time,
            "stage_timings": self.stage_timings,
            "stage_percentages": stage_percentages,
            "metrics": self.metrics,
            "event_count": len(
                self.events),
            "events_per_second": len(
                self.events) /
            total_time if total_time > 0 else 0,
        }
