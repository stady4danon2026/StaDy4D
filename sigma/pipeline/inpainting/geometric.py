"""Geometric inpainting placeholder implementation."""

from __future__ import annotations

from typing import Any, Dict

from sigma.data.frame_record import FrameIORecord
from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.inpainting.base_inpainting import BaseInpainter, InpaintingOutputs


class GeometricInpainter(BaseInpainter):
    """Stub for patch-based geometric inpainting guided by motion masks."""

    name = "geometric_inpainter"

    def __init__(self, patch_size: int = 9, blending: str = "poisson", **_: Any) -> None:
        self.patch_size = patch_size
        self.blending = blending
        self.prepared = False

    def setup(self) -> None:
        self.prepared = True

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        """Fill moving-object regions using geometric priors.

        Args:
            frame_records: Stored frames keyed by index.
            frame_idx: Current timestep to inpaint.

        Returns:
            StageResult whose ``data`` embeds an :class:`InpaintingOutputs`.
        """
        if not self.prepared:
            raise RuntimeError("GeometricInpainter.setup() must be invoked before process().")

        outputs = InpaintingOutputs(inpainted_image=None, confidence_map=None)
        if frame_idx in frame_records:
            frame_records[frame_idx].inpainted_image = outputs.inpainted_image
        vis_assets = {"inpainted": None}
        return StageResult(data={"inpainting": outputs}, visualization_assets=vis_assets)

    def teardown(self) -> None:
        self.prepared = False
