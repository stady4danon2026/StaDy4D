"""Blank inpainter: zeros out dynamic-object regions instead of filling them."""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from sigma.data.frame_record import FrameIORecord
from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.inpainting.base_inpainting import BaseInpainter, InpaintingOutputs


class BlankInpainter(BaseInpainter):
    """Replaces moving-object regions with black pixels (no inpainting).

    Uses the motion mask stored on each FrameIORecord (0 = moving, 1 = static)
    to zero out dynamic regions before reconstruction.  This lets VGGT see only
    the static background, with moving regions simply absent from the image.
    """

    name = "blank_inpainter"

    def __init__(self, **_: Any) -> None:
        self.ready = False

    def setup(self) -> None:
        self.ready = True

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        if not self.ready:
            raise RuntimeError("BlankInpainter.setup() must be called first.")

        record = frame_records.get(frame_idx)
        if record is None or record.origin_image is None:
            outputs = InpaintingOutputs(inpainted_image=None, confidence_map=None)
            return StageResult(data={"inpainting": outputs}, visualization_assets={"inpainted": None})

        image = record.origin_image
        mask = record.mask  # uint8: 0 = moving region, 1 = static background

        if mask is not None:
            # mask = 1 means foreground/car (from grounded_sam), 0 means background.
            # Invert so we zero out the car region and keep the background.
            keep = (1 - mask)[..., np.newaxis]
            masked = (image * keep).astype(image.dtype)
        else:
            masked = image.copy()

        record.inpainted_image = masked
        outputs = InpaintingOutputs(inpainted_image=masked, confidence_map=None)
        return StageResult(data={"inpainting": outputs}, visualization_assets={"inpainted": masked})

    def process_batch(
        self,
        frame_records: Dict[int, FrameIORecord],
        progress_fn: Callable[[int, int], None] | None = None,
    ) -> Dict[int, StageResult]:
        """Process all frames in parallel — blank inpainting has no inter-frame deps."""
        if not self.ready:
            raise RuntimeError("BlankInpainter.setup() must be called first.")

        sorted_indices = sorted(frame_records.keys())
        total = len(sorted_indices)
        results: Dict[int, StageResult] = {}

        for count, frame_idx in enumerate(sorted_indices):
            record = frame_records[frame_idx]

            # Skip frames already inpainted (e.g. by preprocessor).
            if record.inpainted_image is not None:
                if progress_fn:
                    progress_fn(count + 1, total)
                continue

            if record.origin_image is None:
                if progress_fn:
                    progress_fn(count + 1, total)
                continue

            image = record.origin_image
            mask = record.mask

            if mask is not None:
                keep = (1 - mask)[..., np.newaxis]
                masked = (image * keep).astype(image.dtype)
            else:
                masked = image.copy()

            record.inpainted_image = masked
            outputs = InpaintingOutputs(inpainted_image=masked, confidence_map=None)
            results[frame_idx] = StageResult(
                data={"inpainting": outputs},
                visualization_assets={"inpainted": masked},
            )
            if progress_fn:
                progress_fn(count + 1, total)

        return results

    def teardown(self) -> None:
        self.ready = False
