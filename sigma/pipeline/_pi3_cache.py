"""Per-scene Pi3X forward cache shared across pipeline stages.

Avoids running Pi3X twice — once for motion-stage features and again for
reconstruction depth/poses/cloud. The first stage that needs Pi3 outputs
calls ``ensure_pi3_outputs(frame_records, model, ...)`` which:

1. Checks if frame_records already have ``pi3_local_points`` etc. populated.
2. If yes, returns immediately (cache hit).
3. If no, runs one Pi3X forward on all frames, populates frame_records,
   captures conf_decoder features via a forward hook.

Subsequent stages call the same function and get the cached outputs for free.

The cache is per-scene: ``run_pipeline_for_camera`` resets ``frame_records``
between scenes, so the cache is automatically invalidated.
"""
from __future__ import annotations

import logging
import math
from typing import Dict

import numpy as np
import torch

from sigma.data.frame_record import FrameIORecord

LOGGER = logging.getLogger(__name__)


def _compute_target_size(H: int, W: int, pixel_limit: int = 255000) -> tuple[int, int]:
    scale = math.sqrt(pixel_limit / max(W * H, 1))
    Wf, Hf = W * scale, H * scale
    k, m = round(Wf / 14), round(Hf / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > Wf / Hf:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def has_pi3_outputs(frame_records: Dict[int, FrameIORecord]) -> bool:
    """True if frame_records already contain Pi3 outputs from a prior call."""
    if not frame_records:
        return False
    rec = next(iter(frame_records.values()))
    return rec.pi3_local_points is not None


def ensure_pi3_outputs(
    frame_records: Dict[int, FrameIORecord],
    pi3_model,
    *,
    device: str = "cuda",
    pixel_limit: int = 255000,
    capture_conf_features: bool = True,
) -> tuple[int, int]:
    """Run Pi3X once on all frames if not already cached. Returns (patch_h, patch_w)."""
    if has_pi3_outputs(frame_records):
        rec = next(iter(frame_records.values()))
        return rec.pi3_patch_h, rec.pi3_patch_w

    from PIL import Image
    from torchvision import transforms

    sorted_idx = sorted(frame_records.keys())
    frames = [frame_records[i].origin_image for i in sorted_idx]
    H_orig, W_orig = frames[0].shape[:2]
    TARGET_W, TARGET_H = _compute_target_size(H_orig, W_orig, pixel_limit)
    patch_h, patch_w = TARGET_H // 14, TARGET_W // 14

    to_t = transforms.ToTensor()
    tensors = []
    for f in frames:
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        pil = Image.fromarray(f).resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        tensors.append(to_t(pil))
    imgs = torch.stack(tensors, dim=0).unsqueeze(0).to(device)   # (1, N, 3, H, W)

    # Optional: hook conf_decoder to grab features (used by Sam3TttUnion DynHead)
    captured = {"feats": None}
    handle = None
    if capture_conf_features and hasattr(pi3_model, "conf_decoder"):
        patch_start_idx = int(getattr(pi3_model, "patch_start_idx", 5))
        def _hook(module, inp, out):
            captured["feats"] = out[:, patch_start_idx:].float().detach()
        handle = pi3_model.conf_decoder.register_forward_hook(_hook)

    amp = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
           else torch.float16)
    LOGGER.info("[pi3-cache] running single Pi3X forward on %d frames", len(sorted_idx))
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=amp):
            out = pi3_model(imgs)

    if handle is not None:
        handle.remove()

    # Populate frame_records — keep on GPU as fp16/fp32 for downstream stages.
    local_pts = out["local_points"][0]      # (N, H, W, 3)
    world_pts = out["points"][0]
    cam_poses = out["camera_poses"][0]      # (N, 4, 4)
    conf_lg = out["conf"][0]                # (N, H, W, 1)
    # metric is a (B,)=(1,) tensor — preserve the leading batch dim for downstream `.view(B,...)`.
    metric = out["metric"]
    feats = captured["feats"]               # (N, num_patches, 1024) or None

    for i, idx in enumerate(sorted_idx):
        rec = frame_records[idx]
        rec.pi3_local_points = local_pts[i]
        rec.pi3_world_points = world_pts[i]
        rec.pi3_camera_pose = cam_poses[i]
        rec.pi3_conf_logits = conf_lg[i]
        rec.pi3_metric = metric
        rec.pi3_conf_features = feats[i] if feats is not None else None
        rec.pi3_patch_h = patch_h
        rec.pi3_patch_w = patch_w
        rec.pi3_input_h = TARGET_H
        rec.pi3_input_w = TARGET_W

    return patch_h, patch_w


def stack_pi3_outputs(frame_records: Dict[int, FrameIORecord], indices: list[int] | None = None):
    """Helper: stack cached per-frame Pi3 outputs back to (N, ...) tensors."""
    if indices is None:
        indices = sorted(frame_records.keys())
    local_pts = torch.stack([frame_records[i].pi3_local_points for i in indices], dim=0)
    world_pts = torch.stack([frame_records[i].pi3_world_points for i in indices], dim=0)
    cam_poses = torch.stack([frame_records[i].pi3_camera_pose for i in indices], dim=0)
    conf_lg = torch.stack([frame_records[i].pi3_conf_logits for i in indices], dim=0)
    return {
        "local_points": local_pts.unsqueeze(0),     # (1, N, H, W, 3) to mimic Pi3 output
        "points": world_pts.unsqueeze(0),
        "camera_poses": cam_poses.unsqueeze(0),
        "conf": conf_lg.unsqueeze(0),
        "metric": frame_records[indices[0]].pi3_metric,
    }
