"""Motion stage that loads pre-computed masks from disk.

Used to plug external segmenters (SAM3, SAM3+head union, etc.) into the
SIGMA reconstruction pipeline without re-running them.

Expected layout::
    <mask_root>/<scene>/<camera>/mask/mask_NNNN.png   # 0/255 binary
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict

import cv2
import numpy as np

from sigma.data.frame_record import FrameIORecord
from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.motion.base_motion import BaseMotionEstimator

LOGGER = logging.getLogger(__name__)


class PrecomputedMaskMotionEstimator(BaseMotionEstimator):
    """Read masks from disk; no model loaded.

    Args:
        mask_root: Directory containing ``<scene>/<camera>/mask/`` PNGs.
        scene: Scene name (passed by Hydra config or runner).
        camera: Camera name.
        morph_dilate: Optional dilation in pixels (0 = off).
    """

    name = "precomputed_motion"

    def __init__(
        self,
        mask_root: str,
        scene: str = "",
        camera: str = "",
        morph_dilate: int = 0,
        threshold: int = 127,
        **kwargs: Any,
    ) -> None:
        self.mask_root = Path(mask_root)
        self.scene = scene
        self.camera = camera
        self.morph_dilate = morph_dilate
        self.threshold = threshold
        self.ready = False

    def setup(self) -> None:
        scene_cam_dir = self.mask_root / self.scene / self.camera / "mask"
        if not scene_cam_dir.is_dir():
            LOGGER.warning(
                "PrecomputedMaskMotionEstimator: %s does not exist; will return empty masks",
                scene_cam_dir,
            )
        else:
            LOGGER.info(
                "PrecomputedMaskMotionEstimator: reading masks from %s",
                scene_cam_dir,
            )
        self.ready = True

    def teardown(self) -> None:
        self.ready = False

    def _load_mask(self, frame_idx: int, H: int, W: int) -> np.ndarray:
        f = self.mask_root / self.scene / self.camera / "mask" / f"mask_{frame_idx:04d}.png"
        if not f.exists():
            return np.zeros((H, W), dtype=np.uint8)
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return np.zeros((H, W), dtype=np.uint8)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m = (m > self.threshold).astype(np.uint8)
        if self.morph_dilate > 0 and m.any():
            kernel = np.ones((self.morph_dilate, self.morph_dilate), np.uint8)
            m = cv2.dilate(m, kernel)
        return m

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        record = frame_records[frame_idx]
        frame = record.origin_image
        H, W = frame.shape[:2]
        mask = self._load_mask(frame_idx, H, W)
        payload = {
            "curr_mask": mask,
            "warped_prev_to_curr": frame,
            "optical_flow": np.zeros((H, W, 2), dtype=np.float32),
            "static_curr": frame,
            "fundamental_matrix": None,
        }
        return StageResult(data={"motion": payload}, visualization_assets={"mask": mask})

    def process_batch(
        self,
        frame_records: Dict[int, FrameIORecord],
        progress_fn: Callable[[int, int], None] | None = None,
    ) -> Dict[int, StageResult]:
        sorted_indices = sorted(frame_records.keys())
        total = len(sorted_indices)
        results: Dict[int, StageResult] = {}
        for k, frame_idx in enumerate(sorted_indices):
            results[frame_idx] = self.process(frame_records, frame_idx)
            if progress_fn:
                progress_fn(k + 1, total)
        return results
