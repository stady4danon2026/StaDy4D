"""Tiny U-Net student head, distilled from GroundedSAM masks.

Replaces the heavy DINO+SAM motion stage at inference: a single forward
pass of a small CNN per frame.  Trained offline by ``script/train_dynamic_head.py``.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F

from sigma.data.frame_record import FrameIORecord
from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.motion.base_motion import BaseMotionEstimator
from sigma.pipeline.motion.student_unet import StudentUNet

LOGGER = logging.getLogger(__name__)


class StudentMotionEstimator(BaseMotionEstimator):
    """Single-pass dynamic-mask predictor (UNet trained to mimic GroundedSAM)."""

    name = "student_motion"

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        img_size: int = 320,
        threshold: float = 0.5,
        min_blob_area: int = 64,
        **kwargs: Any,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.img_size = img_size
        self.threshold = threshold
        self.min_blob_area = min_blob_area
        self.model: StudentUNet | None = None
        self.ready = False

    def setup(self) -> None:
        LOGGER.info("Loading student UNet from %s", self.checkpoint)
        self.model = StudentUNet().to(self.device).eval()
        state = torch.load(self.checkpoint, map_location=self.device)
        self.model.load_state_dict(state)
        self.ready = True

    def teardown(self) -> None:
        self.model = None
        self.ready = False

    @staticmethod
    def _preprocess(frames: np.ndarray, size: int) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
        """Convert (N, H, W, 3) uint8 → (N, 3, size, size) tensor in [0, 1] with metadata for un-pad."""
        N, H, W = frames.shape[:3]
        scale = size / max(H, W)
        nh, nw = int(H * scale), int(W * scale)
        x = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        pad_h, pad_w = size - nh, size - nw
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (H, W, nh, nw)

    @staticmethod
    def _postprocess(logits: torch.Tensor, info: tuple[int, int, int, int],
                      threshold: float) -> np.ndarray:
        """(N, 1, S, S) logits → (N, H, W) uint8 binary masks."""
        H, W, nh, nw = info
        prob = torch.sigmoid(logits)
        # Crop pad and resize back to (H, W)
        prob = prob[:, :, :nh, :nw]
        prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
        bin_ = (prob > threshold).squeeze(1).cpu().numpy().astype(np.uint8)
        return bin_

    def _filter_blobs(self, masks: np.ndarray) -> np.ndarray:
        """Drop connected components below ``min_blob_area`` pixels."""
        import cv2
        out = np.zeros_like(masks)
        for i in range(masks.shape[0]):
            m = masks[i]
            if m.sum() == 0:
                continue
            n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            keep = np.zeros_like(m)
            for k in range(1, n_lab):
                if stats[k, cv2.CC_STAT_AREA] >= self.min_blob_area:
                    keep[lab == k] = 1
            out[i] = keep
        return out

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        record = frame_records[frame_idx]
        frame = record.origin_image
        return self._run_one(frame)

    def _run_one(self, frame: np.ndarray) -> StageResult:
        x, info = self._preprocess(frame[None], self.img_size)
        x = x.to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        mask = self._postprocess(logits, info, self.threshold)[0]
        mask = self._filter_blobs(mask[None])[0]
        H, W = frame.shape[:2]
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
        if not self.ready:
            raise RuntimeError("StudentMotionEstimator.setup() must be called first.")

        sorted_indices = sorted(frame_records.keys())
        total = len(sorted_indices)
        results: Dict[int, StageResult] = {}

        # Stack all frames into one batch for a single forward.
        frames_np = np.stack(
            [frame_records[i].origin_image for i in sorted_indices], axis=0
        )
        x, info = self._preprocess(frames_np, self.img_size)
        x = x.to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        masks = self._postprocess(logits, info, self.threshold)
        masks = self._filter_blobs(masks)

        for k, frame_idx in enumerate(sorted_indices):
            frame = frame_records[frame_idx].origin_image
            H, W = frame.shape[:2]
            payload = {
                "curr_mask": masks[k],
                "warped_prev_to_curr": frame,
                "optical_flow": np.zeros((H, W, 2), dtype=np.float32),
                "static_curr": frame,
                "fundamental_matrix": None,
            }
            results[frame_idx] = StageResult(
                data={"motion": payload},
                visualization_assets={"mask": masks[k]},
            )
            if progress_fn:
                progress_fn(k + 1, total)

        return results
