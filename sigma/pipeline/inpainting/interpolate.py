"""Interpolation inpainter: fills masked regions using OpenCV inpainting."""

from __future__ import annotations

from typing import Any, Callable, Dict

import cv2
import numpy as np

from sigma.data.frame_record import FrameIORecord
from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.inpainting.base_inpainting import BaseInpainter, InpaintingOutputs


class InterpolateInpainter(BaseInpainter):
    """Fills masked regions via OpenCV interpolation-based inpainting.

    Uses cv2.inpaint (Telea or Navier-Stokes) to smoothly interpolate
    pixel values from the surrounding unmasked region into the mask.
    Much faster than diffusion-based methods, no GPU required.
    """

    name = "interpolate_inpainter"

    def __init__(self, method: str = "telea", radius: int = 5, **_: Any) -> None:
        """
        Args:
            method: "telea" (fast marching) or "ns" (Navier-Stokes).
            radius: Inpainting neighborhood radius in pixels.
        """
        self.method = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        self.radius = radius
        self.ready = False

    def setup(self) -> None:
        self.ready = True

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        if not self.ready:
            raise RuntimeError("InterpolateInpainter.setup() must be called first.")

        record = frame_records.get(frame_idx)
        if record is None or record.origin_image is None:
            outputs = InpaintingOutputs(inpainted_image=None, confidence_map=None)
            return StageResult(data={"inpainting": outputs}, visualization_assets={})

        image = record.origin_image
        mask = record.mask

        if mask is not None:
            # mask: 1=foreground (inpaint), 0=background (keep)
            mask_uint8 = mask if mask.max() > 1 else (mask * 255).astype(np.uint8)
            inpainted = cv2.inpaint(image, mask_uint8, self.radius, self.method)
        else:
            inpainted = image.copy()

        record.inpainted_image = inpainted
        outputs = InpaintingOutputs(inpainted_image=inpainted, confidence_map=None)
        return StageResult(
            data={"inpainting": outputs},
            visualization_assets={"inpainted": inpainted},
        )

    def process_batch(
        self,
        frame_records: Dict[int, FrameIORecord],
        progress_fn: Callable[[int, int], None] | None = None,
    ) -> Dict[int, StageResult]:
        """Process all frames — no inter-frame deps, CPU-only."""
        if not self.ready:
            raise RuntimeError("InterpolateInpainter.setup() must be called first.")

        sorted_indices = sorted(frame_records.keys())
        total = len(sorted_indices)
        results: Dict[int, StageResult] = {}

        for count, frame_idx in enumerate(sorted_indices):
            record = frame_records[frame_idx]

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
                mask_uint8 = mask if mask.max() > 1 else (mask * 255).astype(np.uint8)
                inpainted = cv2.inpaint(image, mask_uint8, self.radius, self.method)
            else:
                inpainted = image.copy()

            record.inpainted_image = inpainted
            outputs = InpaintingOutputs(inpainted_image=inpainted, confidence_map=None)
            results[frame_idx] = StageResult(
                data={"inpainting": outputs},
                visualization_assets={"inpainted": inpainted},
            )
            if progress_fn:
                progress_fn(count + 1, total)

        return results

    def teardown(self) -> None:
        self.ready = False
