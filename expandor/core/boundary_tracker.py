"""
Boundary tracking for seam detection in expansions
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class BoundaryInfo:
    """Information about an expansion boundary"""

    position: int  # Pixel position of boundary
    direction: str  # 'horizontal' or 'vertical'
    step: int  # Which expansion step created this
    expansion_size: int  # How many pixels were added
    source_size: Tuple[int, int]  # Size before expansion
    target_size: Tuple[int, int]  # Size after expansion
    method: str  # Method used (e.g., 'outpaint', 'sliding_window')
    metadata: Dict = field(default_factory=dict)  # Additional info


class BoundaryTracker:
    """
    Tracks expansion boundaries for artifact detection

    This is critical for detecting seams where new content meets old.
    Every expansion creates potential seam locations that must be tracked.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize boundary tracker"""
        self.logger = logger or logging.getLogger(__name__)
        self.reset()

    def reset(self):
        """Reset tracker for new operation"""
        self.boundaries: List[BoundaryInfo] = []
        # For checklist compatibility
        self.progressive_boundaries: List[Dict] = []
        self.horizontal_positions: Set[int] = set()
        self.vertical_positions: Set[int] = set()
        self.boundary_map: Dict[str, List[BoundaryInfo]] = {
            "horizontal": [],
            "vertical": [],
        }

    def add_boundary(
        self,
        position: int,
        direction: str,
        step: int,
        expansion_size: int,
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
        method: str = "unknown",
        metadata: Optional[Dict] = None,
    ):
        """
        Add a boundary to track

        Args:
            position: Pixel position where new content meets old
            direction: 'horizontal' or 'vertical'
            step: Expansion step number
            expansion_size: Pixels added in this expansion
            source_size: Image size before expansion
            target_size: Image size after expansion
            method: Method used for expansion
            metadata: Additional tracking data
        """
        if direction not in ["horizontal", "vertical"]:
            raise ValueError(f"Invalid direction: {direction}")

        boundary = BoundaryInfo(
            position=position,
            direction=direction,
            step=step,
            expansion_size=expansion_size,
            source_size=source_size,
            target_size=target_size,
            method=method,
            metadata=metadata or {},
        )

        self.boundaries.append(boundary)
        self.boundary_map[direction].append(boundary)

        # Track unique positions
        if direction == "horizontal":
            self.horizontal_positions.add(position)
        else:
            self.vertical_positions.add(position)

        self.logger.debug(
            f"Added {direction} boundary at {position} "
            f"(step {step}, {expansion_size}px expansion)"
        )

    def add_progressive_boundary(
        self,
        position: int,
        direction: str,
        step: int,
        expansion_size: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Add a progressive boundary (checklist compatible method)

        Args:
            position: Boundary position
            direction: 'horizontal' or 'vertical'
            step: Expansion step number
            expansion_size: Size of expansion
            metadata: Additional data
        """
        # Add to simple list for checklist compatibility
        boundary_dict = {
            "position": position,
            "direction": direction,
            "step": step}
        if expansion_size is not None:
            boundary_dict["expansion_size"] = expansion_size
        if metadata:
            boundary_dict.update(metadata)

        self.progressive_boundaries.append(boundary_dict)

        # Also add to main tracking system
        if expansion_size is not None:
            self.add_boundary(
                position=position,
                direction=direction,
                step=step,
                expansion_size=expansion_size,
                # Will be overridden by add_progressive_boundaries
                source_size=(0, 0),
                target_size=(0, 0),
                method="progressive",
                metadata=metadata,
            )

    def add_progressive_boundaries(
        self,
        current_size: Tuple[int, int],
        target_size: Tuple[int, int],
        step: int,
        method: str = "progressive",
    ):
        """
        Add boundaries for a progressive expansion

        Automatically determines which edges expanded and tracks them.

        Args:
            current_size: Size before expansion (width, height)
            target_size: Size after expansion (width, height)
            step: Expansion step number
            method: Method name for tracking
        """
        curr_w, curr_h = current_size
        tgt_w, tgt_h = target_size

        # Track horizontal expansion (width change)
        if tgt_w > curr_w:
            # Check which side(s) expanded
            left_expansion = (tgt_w - curr_w) // 2
            right_expansion = tgt_w - curr_w - left_expansion

            if left_expansion > 0:
                self.add_boundary(
                    position=left_expansion,
                    direction="horizontal",
                    step=step,
                    expansion_size=left_expansion,
                    source_size=current_size,
                    target_size=target_size,
                    method=method,
                    metadata={"side": "left"},
                )

            if right_expansion > 0:
                self.add_boundary(
                    position=curr_w + left_expansion,
                    direction="horizontal",
                    step=step,
                    expansion_size=right_expansion,
                    source_size=current_size,
                    target_size=target_size,
                    method=method,
                    metadata={"side": "right"},
                )

        # Track vertical expansion (height change)
        if tgt_h > curr_h:
            # Check which side(s) expanded
            top_expansion = (tgt_h - curr_h) // 2
            bottom_expansion = tgt_h - curr_h - top_expansion

            if top_expansion > 0:
                self.add_boundary(
                    position=top_expansion,
                    direction="vertical",
                    step=step,
                    expansion_size=top_expansion,
                    source_size=current_size,
                    target_size=target_size,
                    method=method,
                    metadata={"side": "top"},
                )

            if bottom_expansion > 0:
                self.add_boundary(
                    position=curr_h + top_expansion,
                    direction="vertical",
                    step=step,
                    expansion_size=bottom_expansion,
                    source_size=current_size,
                    target_size=target_size,
                    method=method,
                    metadata={"side": "bottom"},
                )

    def get_critical_boundaries(self) -> Dict[str, List[int]]:
        """
        Get critical boundaries in dict format (checklist compatible)

        Returns:
            Dictionary with 'horizontal' and 'vertical' position lists
        """
        return {
            "horizontal": sorted(list(self.horizontal_positions)),
            "vertical": sorted(list(self.vertical_positions)),
        }

    def get_boundary_regions(
        self, width: int, height: int, padding: int = 10
    ) -> List[Tuple[int, int, int, int]]:
        """
        Get regions around boundaries for focused processing (checklist compatible)

        Args:
            width: Image width
            height: Image height
            padding: Pixels on each side of boundary

        Returns:
            List of (x1, y1, x2, y2) regions
        """
        regions = []

        # Horizontal boundaries create vertical strips
        for x in self.horizontal_positions:
            x1 = max(0, x - padding)
            x2 = min(width, x + padding)
            regions.append((x1, 0, x2, height))

        # Vertical boundaries create horizontal strips
        for y in self.vertical_positions:
            y1 = max(0, y - padding)
            y2 = min(height, y + padding)
            regions.append((0, y1, width, y2))

        return regions

    def get_all_boundaries(self) -> List[Dict]:
        """
        Get all boundaries as serializable list

        Returns:
            List of boundary dictionaries
        """
        return [
            {
                "position": b.position,
                "direction": b.direction,
                "step": b.step,
                "expansion_size": b.expansion_size,
                "source_size": b.source_size,
                "target_size": b.target_size,
                "method": b.method,
                "metadata": b.metadata,
            }
            for b in self.boundaries
        ]

    def get_boundaries_for_detection(self) -> Dict[str, List[int]]:
        """
        Get boundaries formatted for artifact detection

        Returns:
            Dictionary with 'horizontal' and 'vertical' position lists
        """
        return self.get_critical_boundaries()

    def summarize(self) -> Dict[str, Any]:
        """
        Get summary of tracked boundaries

        Returns:
            Summary dictionary
        """
        return {
            "total_boundaries": len(self.boundaries),
            "horizontal_count": len(self.horizontal_positions),
            "vertical_count": len(self.vertical_positions),
            "methods_used": list(set(b.method for b in self.boundaries)),
            "expansion_steps": len(set(b.step for b in self.boundaries)),
            "critical_boundaries": len(self.boundaries),
            "largest_expansion": max(
                (b.expansion_size for b in self.boundaries), default=0
            ),
        }
