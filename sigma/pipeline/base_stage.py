"""Common abstractions shared by every pipeline stage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Protocol

from sigma.data.frame_record import FrameIORecord

LOGGER = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Container that each stage returns.

    Attributes:
        data: Dictionary containing the primary outputs from the stage.
            Keys depend on the specific stage (e.g., "reconstruction", "motion", "inpainting").
        visualization_assets: Dictionary containing data suitable for visualization.
            Examples: depth_maps, point_clouds, confidence_maps, camera_poses, etc.
    """

    data: Dict[str, Any] = field(default_factory=dict)
    visualization_assets: Dict[str, Any] = field(default_factory=dict)


class BaseStage(Protocol):
    """Generic protocol for every stage within the SIGMA pipeline."""

    name: str

    def setup(self) -> None:
        """Load checkpoints or expensive state before processing begins."""

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        """Run the stage on a single frame and return dictionaries of outputs."""

    def process_batch(
        self,
        frame_records: Dict[int, FrameIORecord],
        progress_fn: Callable[[int, int], None] | None = None,
    ) -> Dict[int, StageResult]:
        """Process all frames in batch mode.

        The default implementation calls :meth:`process` sequentially for each
        frame.  Subclasses may override this to batch GPU forward passes.

        Args:
            frame_records: All frame records keyed by frame index.
            progress_fn: Optional ``(current, total)`` callback for progress reporting.

        Returns:
            Mapping of frame index to :class:`StageResult`.
        """

    def teardown(self) -> None:
        """Free any heavyweight resources allocated within setup."""
