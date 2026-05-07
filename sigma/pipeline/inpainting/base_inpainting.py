"""Baseclass for the dynamic object removal stage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict

from sigma.data.frame_record import FrameIORecord

from sigma.pipeline.base_stage import BaseStage, StageResult

LOGGER = logging.getLogger(__name__)


@dataclass
class InpaintingOutputs:
    """Pack the produced inpainted frame and metadata."""

    inpainted_image: Any
    confidence_map: Any


class BaseInpainter(BaseStage):
    """Abstract interface used by all inpainting methods."""

    name: str = "base_inpainter"

    def setup(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:  # pragma: no cover - placeholder
        """Inpaint the provided frame using masks and motion cues stored on ``FrameIORecord``."""
        raise NotImplementedError

    def process_batch(
        self,
        frame_records: Dict[int, FrameIORecord],
        progress_fn: Callable[[int, int], None] | None = None,
    ) -> Dict[int, StageResult]:
        """Process all frames sequentially (default implementation).

        Frames that already have ``inpainted_image`` set (e.g. by the
        preprocessor) are skipped.  Override in subclasses for parallel /
        vectorized inpainting.
        """
        sorted_indices = sorted(frame_records.keys())
        total = len(sorted_indices)
        results: Dict[int, StageResult] = {}

        for count, frame_idx in enumerate(sorted_indices):
            record = frame_records[frame_idx]
            if record.inpainted_image is not None:
                # Already inpainted (e.g. by preprocessor).
                if progress_fn:
                    progress_fn(count + 1, total)
                continue
            results[frame_idx] = self.process(frame_records, frame_idx)
            if progress_fn:
                progress_fn(count + 1, total)

        return results

    def teardown(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError
