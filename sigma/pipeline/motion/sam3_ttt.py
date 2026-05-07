"""End-to-end motion stage: TTT-adapted DynHead ∪ SAM3 text-prompted detection.

Per scene (called once via process_batch):
  1. Pi3 forward on all frames → cache conf_decoder features + capture depth/pose.
  2. GroundedSAM on K evenly-spaced keyframes → labels for TTT.
  3. TTT N steps to fine-tune the warm-started DynHead per scene.
  4. DynHead predict on all frames → head_mask.
  5. SAM3 forward on all frames with text → vehicle candidate masks (filtered by score).
  6. Final mask = head_mask ∪ SAM3_mask (union).

Replaces grounded_sam / student / pi3_dyn motion stages — one stage that
computes everything end-to-end without intermediate disk caching.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sigma.data.frame_record import FrameIORecord
from sigma.pipeline.base_stage import StageResult
from sigma.pipeline.motion.base_motion import BaseMotionEstimator

LOGGER = logging.getLogger(__name__)


def _focal_bce(logits, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    a = alpha * target + (1 - alpha) * (1 - target)
    return a * (1 - pt).pow(gamma) * bce


def _compute_target_size(H: int, W: int, pixel_limit: int) -> tuple[int, int]:
    scale = math.sqrt(pixel_limit / max(W * H, 1))
    Wf, Hf = W * scale, H * scale
    k, m = round(Wf / 14), round(Hf / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > Wf / Hf:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


class Sam3TttUnionMotionEstimator(BaseMotionEstimator):
    """SAM3 + TTT-adapted DynHead union (end-to-end).

    Args:
        head_checkpoint: Pretrained DynHead .pt (warm start).
        pi3_checkpoint: HuggingFace ID for Pi3X.
        sam3_checkpoint: HuggingFace ID for SAM3.
        text: Open-vocabulary prompt for SAM3.
        score_thr: SAM3 candidate score threshold.
        keyframes: # GSAM keyframes for TTT supervision.
        ttt_steps: TTT optimizer steps.
        ttt_lr: TTT learning rate.
        head_threshold: DynHead sigmoid threshold.
        head_morph_close, head_morph_dilate, head_min_blob: head mask cleanup.
        do_ttt: If False, skip TTT (use frozen pooled head).
        do_head: If False, output SAM3-only masks.
    """

    name = "sam3_ttt_union_motion"

    def __init__(
        self,
        head_checkpoint: str = "checkpoints/pi3_dyn_head_pooled.pt",
        pi3_checkpoint: str = "yyfz233/Pi3X",
        sam3_checkpoint: str = "facebook/sam3",
        device: str = "cuda",
        text: str = "car. truck. bus. motorcycle. bicycle. pedestrian.",
        score_thr: float = 0.3,
        keyframes: int = 5,
        ttt_steps: int = 30,
        ttt_lr: float = 5e-4,
        ttt_batch_frames: int = 4,
        head_threshold: float = 0.45,
        head_morph_close: int = 5,
        head_morph_dilate: int = 3,
        head_min_blob: int = 100,
        max_coverage: float = 0.50,
        head_cov_max_for_union: float = 1.0,
        pixel_limit: int = 255000,
        do_ttt: bool = True,
        do_head: bool = True,
        do_sam3: bool = True,
        **kwargs: Any,
    ) -> None:
        self.head_checkpoint = head_checkpoint
        self.pi3_checkpoint = pi3_checkpoint
        self.sam3_checkpoint = sam3_checkpoint
        self.device = device
        self.text = text
        self.score_thr = score_thr
        self.keyframes = keyframes
        self.ttt_steps = ttt_steps
        self.ttt_lr = ttt_lr
        self.ttt_batch_frames = ttt_batch_frames
        self.head_threshold = head_threshold
        self.head_morph_close = head_morph_close
        self.head_morph_dilate = head_morph_dilate
        self.head_min_blob = head_min_blob
        self.max_coverage = max_coverage
        self.head_cov_max_for_union = head_cov_max_for_union
        self.pixel_limit = pixel_limit
        self.do_ttt = do_ttt
        self.do_head = do_head
        self.do_sam3 = do_sam3

        self.pi3 = None
        self.gsam = None
        self.sam3_proc = None
        self.sam3_model = None
        self.head_init_state = None
        self.ready = False

    # ----------------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------------
    def setup(self) -> None:
        from pi3.models.pi3x import Pi3X
        from transformers import Sam3Processor, Sam3Model
        from sigma.pipeline.motion.pi3_dyn_head import DynHead

        if self.do_head or self.do_ttt:
            from sigma.pipeline._model_registry import get_or_load_pi3
            LOGGER.info("Loading Pi3X for motion stage (shared cache)")
            self.pi3 = get_or_load_pi3(self.pi3_checkpoint, self.device)

        # SAM3 provides BOTH the per-frame mask candidates AND the keyframe
        # supervision for TTT. GSAM is no longer needed.
        if self.do_sam3 or self.do_ttt:
            LOGGER.info("Loading SAM3: %s", self.sam3_checkpoint)
            self.sam3_proc = Sam3Processor.from_pretrained(self.sam3_checkpoint)
            self.sam3_model = Sam3Model.from_pretrained(self.sam3_checkpoint).to(self.device).eval()
            for p in self.sam3_model.parameters():
                p.requires_grad_(False)

        if self.do_head and Path(self.head_checkpoint).exists():
            LOGGER.info("Loading head warm-start from %s", self.head_checkpoint)
            self.head_init_state = DynHead.load(self.head_checkpoint, map_location="cpu").state_dict()

        LOGGER.info("Sam3TttUnion config: score_thr=%.3f  max_coverage=%.2f  head_cov_max_for_union=%.2f",
                     self.score_thr, self.max_coverage, self.head_cov_max_for_union)
        self.ready = True

    def teardown(self) -> None:
        if self.gsam is not None:
            try:
                self.gsam.teardown()
            except Exception:
                pass
        self.pi3 = None
        self.gsam = None
        self.sam3_proc = None
        self.sam3_model = None
        self.head_init_state = None
        self.ready = False

    def process(self, frame_records: Dict[int, FrameIORecord], frame_idx: int) -> StageResult:
        raise RuntimeError("Sam3TttUnionMotionEstimator is batch-only; use process_batch().")

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _stack_pi3_input(self, frames: list[np.ndarray]) -> tuple[torch.Tensor, int, int, int, int]:
        from PIL import Image
        from torchvision import transforms

        H_orig, W_orig = frames[0].shape[:2]
        TARGET_W, TARGET_H = _compute_target_size(H_orig, W_orig, self.pixel_limit)
        to_t = transforms.ToTensor()
        tensors = [to_t(Image.fromarray(f).resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS))
                   for f in frames]
        imgs = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)
        return imgs, H_orig, W_orig, TARGET_H, TARGET_W

    def _post_head_mask(self, m: np.ndarray) -> np.ndarray:
        if self.head_morph_close > 0 and m.any():
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                                  np.ones((self.head_morph_close, self.head_morph_close), np.uint8))
        if self.head_morph_dilate > 0 and m.any():
            m = cv2.dilate(m, np.ones((self.head_morph_dilate, self.head_morph_dilate), np.uint8))
        if self.head_min_blob > 0 and m.any():
            n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            keep = np.zeros_like(m)
            for cc in range(1, n_lab):
                if stats[cc, cv2.CC_STAT_AREA] >= self.head_min_blob:
                    keep[lab == cc] = 1
            m = keep
        return m

    def _run_sam3_one(self, frame_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image
        pil = Image.fromarray(frame_rgb)
        inp = self.sam3_proc(images=pil, text=self.text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.sam3_model(**inp)
        scores = out.pred_logits[0].sigmoid().cpu().numpy()
        masks_lr = (out.pred_masks[0].sigmoid() > 0.5).cpu().numpy()
        keep = scores > self.score_thr
        H, W = frame_rgb.shape[:2]
        if keep.sum() == 0:
            return np.zeros((H, W), dtype=np.uint8)
        masks_lr = masks_lr[keep]
        merged = np.zeros((H, W), dtype=bool)
        for c in range(masks_lr.shape[0]):
            m = cv2.resize(masks_lr[c].astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            merged |= m
        return merged.astype(np.uint8)

    # ----------------------------------------------------------------------
    # Main batch entry point
    # ----------------------------------------------------------------------
    def process_batch(
        self,
        frame_records: Dict[int, FrameIORecord],
        progress_fn: Callable[[int, int], None] | None = None,
    ) -> Dict[int, StageResult]:
        if not self.ready:
            raise RuntimeError("Sam3TttUnionMotionEstimator.setup() must be called first.")

        from sigma.pipeline.motion.pi3_dyn_features import Pi3FeatureExtractor
        from sigma.pipeline.motion.pi3_dyn_head import DynHead

        sorted_idx = sorted(frame_records.keys())
        N = len(sorted_idx)
        frames = [frame_records[i].origin_image for i in sorted_idx]
        H_orig, W_orig = frames[0].shape[:2]
        results: Dict[int, StageResult] = {}

        head_masks_orig = np.zeros((N, H_orig, W_orig), dtype=np.uint8)
        if self.do_head:
            from sigma.pipeline._pi3_cache import ensure_pi3_outputs
            ph, pw = ensure_pi3_outputs(frame_records, self.pi3,
                                         device=self.device,
                                         pixel_limit=self.pixel_limit,
                                         capture_conf_features=True)
            TH, TW = frame_records[sorted_idx[0]].pi3_input_h, frame_records[sorted_idx[0]].pi3_input_w
            feats = torch.stack([frame_records[i].pi3_conf_features for i in sorted_idx], dim=0).to(torch.float32)

            head = DynHead().to(self.device)
            if self.head_init_state is not None:
                head.load_state_dict(self.head_init_state)

            # ---- TTT (SAM3-supervised keyframes) ----
            if self.do_ttt and self.sam3_model is not None and self.ttt_steps > 0:
                K = min(self.keyframes, N)
                kf_indices = sorted(set(int(round(i * (N - 1) / max(K - 1, 1))) for i in range(K)))
                K = len(kf_indices)
                # Run SAM3 on the K keyframes to produce supervision labels.
                kf_masks_orig = np.stack(
                    [self._run_sam3_one(frames[orig_i]) for orig_i in kf_indices], axis=0
                )
                kf_masks_t = torch.from_numpy(kf_masks_orig).float().unsqueeze(1).to(self.device)
                kf_labels = F.interpolate(kf_masks_t, size=(TH, TW), mode="nearest").squeeze(1)

                head.train()
                opt = torch.optim.Adam(head.parameters(), lr=self.ttt_lr)
                kf_feats = feats[kf_indices]
                gen = torch.Generator(device=self.device).manual_seed(0)
                for _ in range(self.ttt_steps):
                    idx = torch.randint(0, K, (min(self.ttt_batch_frames, K),),
                                        device=self.device, generator=gen)
                    logits = head(kf_feats[idx], patch_h=ph, patch_w=pw)
                    if logits.shape[-2:] != kf_labels.shape[-2:]:
                        logits = F.interpolate(logits.unsqueeze(1), size=kf_labels.shape[-2:],
                                                mode="bilinear", align_corners=False).squeeze(1)
                    loss = _focal_bce(logits, kf_labels[idx]).mean()
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            head.eval()

            # ---- DynHead predict on all frames ----
            with torch.no_grad():
                preds = []
                for s in range(0, N, 8):
                    lg = head(feats[s:s + 8], patch_h=ph, patch_w=pw)
                    if lg.shape[-2:] != (TH, TW):
                        lg = F.interpolate(lg.unsqueeze(1), size=(TH, TW),
                                            mode="bilinear", align_corners=False).squeeze(1)
                    preds.append((torch.sigmoid(lg) > self.head_threshold).cpu().numpy().astype(np.uint8))
                head_masks_pi3 = np.concatenate(preds, axis=0)
            for i in range(N):
                m = cv2.resize(head_masks_pi3[i], (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
                head_masks_orig[i] = self._post_head_mask(m)

        # ---- SAM3 ----
        sam3_masks = np.zeros((N, H_orig, W_orig), dtype=np.uint8)
        if self.do_sam3:
            for i in range(N):
                sam3_masks[i] = self._run_sam3_one(frames[i])
                if progress_fn:
                    progress_fn(i + 1, N)

        # ---- Union + per-frame coverage cap + conditional union ----
        # Conditional union: only add SAM3 when head's mask is sparse
        # (head_cov < head_cov_max_for_union). This protects against
        # stacking SAM3 over-fire on top of an already-dense head mask.
        for k, frame_idx in enumerate(sorted_idx):
            sam3_m = sam3_masks[k]
            head_m = head_masks_orig[k]
            if self.max_coverage > 0 and float(sam3_m.mean()) > self.max_coverage:
                sam3_m = np.zeros_like(sam3_m)
            if float(head_m.mean()) > self.head_cov_max_for_union:
                mask = head_m            # head already dense → don't add SAM3
            else:
                mask = sam3_m | head_m   # union
            payload = {
                "curr_mask": mask.astype(np.uint8),
                "warped_prev_to_curr": frames[k],
                "optical_flow": np.zeros((H_orig, W_orig, 2), dtype=np.float32),
                "static_curr": frames[k],
                "fundamental_matrix": None,
            }
            results[frame_idx] = StageResult(
                data={"motion": payload},
                visualization_assets={"mask": mask.astype(np.uint8)},
            )

        return results
