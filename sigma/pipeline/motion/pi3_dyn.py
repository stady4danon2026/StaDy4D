"""Pi3-feature dynamic-mask motion stage with photometric TTT.

At ``process_batch`` time:
  1. Load Pi3X (or accept a shared instance), put it in eval/frozen mode.
  2. Forward all frames in one batch; capture conf_decoder features + the
     model's depth/pose/intrinsics.
  3. Generate photometric pseudo-labels from the captured geometry.
  4. Optionally fine-tune the DynHead for ``ttt_steps`` iterations.
  5. Predict masks with the (adapted) head and pack them into StageResults.

The reconstruction stage downstream still runs Pi3 on its own (current
behaviour) — we don't try to share Pi3 forward across stages here, because
that would entangle motion + reconstruction in the runner.  The cost is one
extra Pi3 forward per scene; in exchange the motion stage stays a drop-in.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch

from sigma.data.frame_record import FrameIORecord
from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.motion.base_motion import BaseMotionEstimator
from sigma.pipeline.motion.pi3_dyn_features import Pi3FeatureExtractor
from sigma.pipeline.motion.pi3_dyn_head import DynHead
from sigma.pipeline.motion.pi3_dyn_ttt import (
    PhotometricLabels,
    compute_photometric_labels,
    predict_masks,
    run_ttt,
)

LOGGER = logging.getLogger(__name__)


def _compute_target_size(H: int, W: int, pixel_limit: int = 255000) -> tuple[int, int]:
    """Mirror Pi3 adapter sizing: multiples of 14, within pixel_limit."""
    scale = math.sqrt(pixel_limit / max(W * H, 1))
    W_f, H_f = W * scale, H * scale
    k, m = round(W_f / 14), round(H_f / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_f / H_f:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


class Pi3DynMotionEstimator(BaseMotionEstimator):
    """DynHead-on-Pi3 motion stage with optional photometric TTT.

    Args:
        checkpoint: Path to a pretrained DynHead checkpoint.
        pi3_checkpoint: HuggingFace ID / path for Pi3X weights.
        device: Compute device.
        threshold: Sigmoid threshold for binarizing the dyn map.
        pixel_limit: Pi3 input pixel budget (used for resize).
        ttt_steps: Photometric-TTT iterations (0 → skip TTT, just inference).
        ttt_lr: Adam LR for TTT.
        ttt_batch_frames: Frames per TTT step.
        photometric_threshold: Residual threshold for the pseudo-label.
        photometric_consensus: # neighbors that must agree to flag dynamic.
        photometric_neighbors: Frame offsets used in consensus.
        min_blob_area: Drop CC blobs smaller than this many pixels.
    """

    name = "pi3_dyn_motion"

    def __init__(
        self,
        checkpoint: str | None = None,
        pi3_checkpoint: str = "yyfz233/Pi3X",
        device: str = "cuda",
        threshold: float = 0.5,
        pixel_limit: int = 255000,
        ttt_steps: int = 200,
        ttt_lr: float = 1e-3,
        ttt_batch_frames: int = 8,
        photometric_threshold: float = 0.10,
        photometric_consensus: int = 3,
        photometric_neighbors: tuple[int, ...] = (-3, -1, 1, 3),
        min_blob_area: int = 64,
        **kwargs: Any,
    ) -> None:
        self.checkpoint = checkpoint
        self.pi3_checkpoint = pi3_checkpoint
        self.device = device
        self.threshold = threshold
        self.pixel_limit = pixel_limit
        self.ttt_steps = ttt_steps
        self.ttt_lr = ttt_lr
        self.ttt_batch_frames = ttt_batch_frames
        self.photometric_threshold = photometric_threshold
        self.photometric_consensus = photometric_consensus
        self.photometric_neighbors = tuple(photometric_neighbors)
        self.min_blob_area = min_blob_area

        self.pi3_model: torch.nn.Module | None = None
        self.dyn_head: DynHead | None = None
        self.ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def setup(self) -> None:
        from pi3.models.pi3x import Pi3X

        LOGGER.info("Loading Pi3X for motion stage from %s", self.pi3_checkpoint)
        self.pi3_model = Pi3X.from_pretrained(self.pi3_checkpoint).to(self.device).eval()
        for p in self.pi3_model.parameters():
            p.requires_grad_(False)

        if self.checkpoint and Path(self.checkpoint).exists():
            LOGGER.info("Loading DynHead checkpoint from %s", self.checkpoint)
            self.dyn_head = DynHead.load(self.checkpoint, map_location=self.device).to(self.device)
        else:
            LOGGER.warning("No DynHead checkpoint provided/found; initializing fresh head.")
            self.dyn_head = DynHead().to(self.device)
        self.ready = True

    def teardown(self) -> None:
        self.pi3_model = None
        self.dyn_head = None
        self.ready = False

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        # Single-frame mode is not meaningful for this stage (TTT needs the
        # whole scene).  Fall back to running the cached head if present.
        raise RuntimeError(
            "Pi3DynMotionEstimator requires process_batch (TTT needs all frames)."
        )

    # ------------------------------------------------------------------
    # Batch entry point
    # ------------------------------------------------------------------
    def process_batch(
        self,
        frame_records: Dict[int, FrameIORecord],
        progress_fn: Callable[[int, int], None] | None = None,
    ) -> Dict[int, StageResult]:
        if not self.ready:
            raise RuntimeError("Pi3DynMotionEstimator.setup() must be called first.")

        sorted_idx = sorted(frame_records.keys())
        total = len(sorted_idx)
        if progress_fn:
            progress_fn(0, total)

        # ----- Phase 1: build Pi3 input + run forward, capture features -----
        imgs_tensor, H_orig, W_orig, H_pi3, W_pi3 = self._stack_frames(frame_records, sorted_idx)
        extractor = Pi3FeatureExtractor(self.pi3_model)

        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        with extractor.capture():
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = self.pi3_model(imgs_tensor)

        feats = extractor.features                                  # (N, num_patches, 1024)
        patch_h, patch_w = extractor.patch_hw

        # ----- Phase 2: extract geometry needed for photometric labels -----
        # Pi3X output is (1, N, ...); squeeze the batch dim.
        local_pts = out["local_points"][0].float()                  # (N, H_pi3, W_pi3, 3)
        c2w = out["camera_poses"][0].float()                        # (N, 4, 4)
        depth = local_pts[..., 2].clamp_min(0.0)                    # (N, H_pi3, W_pi3)

        # Intrinsics — assume principal point at center, derive fx/fy from local_pts.
        K = self._derive_intrinsics_batch(local_pts, H_pi3, W_pi3)  # (N, 3, 3)

        rgb_pi3 = imgs_tensor[0].to(feats.device, dtype=torch.float32)  # (N, 3, H_pi3, W_pi3)

        # ----- Phase 3: photometric pseudo-labels -----
        if self.ttt_steps > 0:
            LOGGER.info("Computing photometric pseudo-labels for %d frames", total)
            with torch.no_grad():
                labels = compute_photometric_labels(
                    rgb=rgb_pi3,
                    depth=depth,
                    K=K,
                    c2w=c2w,
                    neighbors=self.photometric_neighbors,
                    residual_threshold=self.photometric_threshold,
                    consensus=self.photometric_consensus,
                )

            # ----- Phase 4: TTT -----
            LOGGER.info("Running photometric TTT (%d steps, lr=%g)", self.ttt_steps, self.ttt_lr)
            stats = run_ttt(
                self.dyn_head,
                conf_features=feats.to(torch.float32),
                patch_h=patch_h,
                patch_w=patch_w,
                labels=labels,
                steps=self.ttt_steps,
                lr=self.ttt_lr,
                batch_frames=self.ttt_batch_frames,
            )
            LOGGER.info("TTT done: final_loss=%.4f pos_frac=%.3f",
                         stats["final_loss"], stats["pos_frac"])

        # ----- Phase 5: predict masks at original resolution -----
        masks_pi3 = predict_masks(
            self.dyn_head, feats.to(torch.float32),
            patch_h=patch_h, patch_w=patch_w,
            out_hw=(H_pi3, W_pi3), threshold=self.threshold,
        )  # (N, H_pi3, W_pi3) uint8

        # Resize back to origin resolution + blob filter, then pack StageResults.
        results = self._build_results(frame_records, sorted_idx, masks_pi3,
                                       H_orig=H_orig, W_orig=W_orig,
                                       progress_fn=progress_fn, total=total)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _stack_frames(
        self,
        frame_records: Dict[int, FrameIORecord],
        sorted_idx: list[int],
    ) -> tuple[torch.Tensor, int, int, int, int]:
        """Stack origin frames into a (1, N, 3, H_pi3, W_pi3) tensor."""
        from PIL import Image
        from torchvision import transforms

        first = frame_records[sorted_idx[0]].origin_image
        H_orig, W_orig = first.shape[:2]
        TARGET_W, TARGET_H = _compute_target_size(H_orig, W_orig, self.pixel_limit)
        to_tensor = transforms.ToTensor()

        tensors = []
        for i in sorted_idx:
            frame = frame_records[i].origin_image
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            pil = Image.fromarray(frame).resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            tensors.append(to_tensor(pil))
        imgs = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)  # (1, N, 3, H, W)
        return imgs, H_orig, W_orig, TARGET_H, TARGET_W

    @staticmethod
    def _derive_intrinsics_batch(local_pts: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Vectorized version of pi3_adapter._derive_intrinsics over the N axis."""
        N = local_pts.shape[0]
        cx, cy = W / 2.0, H / 2.0
        device = local_pts.device

        u = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W).expand(N, H, W)
        v = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1).expand(N, H, W)
        x = local_pts[..., 0]
        y = local_pts[..., 1]
        z = local_pts[..., 2]

        valid_z = z > 0.01
        mask_fx = valid_z & ((u - cx).abs() > W * 0.1)
        mask_fy = valid_z & ((v - cy).abs() > H * 0.1)

        default_fx = W / (2 * math.tan(math.radians(60)))
        default_fy = H / (2 * math.tan(math.radians(60)))

        Ks = torch.zeros((N, 3, 3), device=device, dtype=torch.float32)
        for i in range(N):
            try:
                if mask_fx[i].sum() > 10:
                    fx = float(torch.median(
                        (u[i][mask_fx[i]] - cx) / (x[i][mask_fx[i]] / z[i][mask_fx[i]])
                    ).item())
                else:
                    fx = default_fx
                if mask_fy[i].sum() > 10:
                    fy = float(torch.median(
                        (v[i][mask_fy[i]] - cy) / (y[i][mask_fy[i]] / z[i][mask_fy[i]])
                    ).item())
                else:
                    fy = default_fy
            except Exception:
                fx, fy = default_fx, default_fy
            fx = float(np.clip(fx, W * 0.2, W * 5.0))
            fy = float(np.clip(fy, H * 0.2, H * 5.0))
            Ks[i, 0, 0] = fx
            Ks[i, 1, 1] = fy
            Ks[i, 0, 2] = cx
            Ks[i, 1, 2] = cy
            Ks[i, 2, 2] = 1.0
        return Ks

    def _build_results(
        self,
        frame_records: Dict[int, FrameIORecord],
        sorted_idx: list[int],
        masks_pi3: np.ndarray,
        H_orig: int,
        W_orig: int,
        progress_fn: Callable[[int, int], None] | None,
        total: int,
    ) -> Dict[int, StageResult]:
        import cv2

        results: Dict[int, StageResult] = {}
        for k, frame_idx in enumerate(sorted_idx):
            mask_small = masks_pi3[k]
            if mask_small.shape != (H_orig, W_orig):
                mask = cv2.resize(mask_small, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
            else:
                mask = mask_small.copy()

            if self.min_blob_area > 0 and mask.any():
                n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                keep = np.zeros_like(mask)
                for cc in range(1, n_lab):
                    if stats[cc, cv2.CC_STAT_AREA] >= self.min_blob_area:
                        keep[lab == cc] = 1
                mask = keep

            frame = frame_records[frame_idx].origin_image
            payload = {
                "curr_mask": mask.astype(np.uint8),
                "warped_prev_to_curr": frame,
                "optical_flow": np.zeros((H_orig, W_orig, 2), dtype=np.float32),
                "static_curr": frame,
                "fundamental_matrix": None,
            }
            results[frame_idx] = StageResult(
                data={"motion": payload},
                visualization_assets={"mask": mask.astype(np.uint8)},
            )
            if progress_fn:
                progress_fn(k + 1, total)
        return results
