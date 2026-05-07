"""Baseclass for motion estimation strategies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np

from sigma.data.frame_record import FrameIORecord

from sigma.pipeline.base_stage import BaseStage, StageResult

LOGGER = logging.getLogger(__name__)


@dataclass
class MotionOutputs:
    """Structured outputs from the motion stage."""

    background_mask: Any
    motion_vectors: Any
    fundamental_matrix: Any = None
    warped_image: Any | None = None


class BaseMotionEstimator(BaseStage):
    """Abstract motion estimator that returns masks and motion fields."""

    name: str = "base_motion"

    def setup(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:  # pragma: no cover - placeholder
        """Estimate background mask and motion vectors using tracked frame records."""
        raise NotImplementedError

    def process_batch(
        self,
        frame_records: Dict[int, FrameIORecord],
        progress_fn: Callable[[int, int], None] | None = None,
    ) -> Dict[int, StageResult]:
        """Process all frames sequentially (default implementation).

        Frame 0 is given an empty motion result (no previous frame).
        Override in subclasses for batched GPU forward passes.
        """
        sorted_indices = sorted(frame_records.keys())
        total = len(sorted_indices)
        results: Dict[int, StageResult] = {}

        for count, frame_idx in enumerate(sorted_indices):
            record = frame_records[frame_idx]

            # First frame — no previous frame available.
            if frame_idx == sorted_indices[0] or (frame_idx - 1) not in frame_records:
                h, w = record.origin_image.shape[:2]
                empty_payload = {
                    "curr_mask": np.zeros((h, w), dtype=np.uint8),
                    "warped_prev_to_curr": record.origin_image,
                    "optical_flow": np.zeros((h, w, 2), dtype=np.float32),
                    "static_curr": record.origin_image,
                    "fundamental_matrix": None,
                }
                results[frame_idx] = StageResult(
                    data={"motion": empty_payload}, visualization_assets={}
                )
            else:
                results[frame_idx] = self.process(frame_records, frame_idx)

            if progress_fn:
                progress_fn(count + 1, total)

        return results

    def teardown(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError
