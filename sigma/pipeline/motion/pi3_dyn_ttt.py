"""Per-scene photometric test-time training for the Pi3 dyn-head.

At test time:
  1. Pi3 forward (already done by reconstruction stage) provides depth, poses,
     intrinsics, and conf_decoder features (captured via Pi3FeatureExtractor).
  2. Photometric N-neighbor consensus produces a noisy dyn pseudo-label per
     frame, plus a per-pixel trust mask (low-texture / depth-edge regions
     excluded from the loss).
  3. The DynHead is fine-tuned for ``steps`` iterations of masked BCE on the
     cached features — Pi3 stays frozen.

The loss is constructed once and reused: features and pseudo-labels are
cached tensors, so each step is a single ConvHead forward + BCE backward.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Photometric pseudo-label generation (vectorized over frames)
# ------------------------------------------------------------------

def _texture_mask(rgb: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    """rgb: (N, 3, H, W) float in [0,1] → (N, H, W) bool, True where textured."""
    gray = rgb.mean(dim=1, keepdim=True)  # (N, 1, H, W)
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    sy = sx.transpose(-1, -2)
    gx = F.conv2d(gray, sx, padding=1)
    gy = F.conv2d(gray, sy, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy).squeeze(1)
    return mag > eps


def _depth_edge_mask(depth: torch.Tensor, rtol: float = 0.10) -> torch.Tensor:
    """depth: (N, H, W) → (N, H, W) bool, True where on a depth discontinuity."""
    d = depth.unsqueeze(1)  # (N, 1, H, W)
    pad = F.pad(d, (1, 1, 1, 1), mode="replicate")
    nmax = F.max_pool2d(pad, 3, stride=1)
    nmin = -F.max_pool2d(-pad, 3, stride=1)
    rel = (nmax - nmin) / nmin.clamp_min(1e-3)
    return rel.squeeze(1) > rtol


def _warp_i_to_j(
    rgb_j: torch.Tensor,        # (3, H, W)
    depth_i: torch.Tensor,       # (H, W)
    K: torch.Tensor,             # (3, 3)
    c2w_i: torch.Tensor,         # (4, 4)
    c2w_j: torch.Tensor,         # (4, 4)
) -> tuple[torch.Tensor, torch.Tensor]:
    H, W = depth_i.shape
    device = depth_i.device

    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pix = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # (H, W, 3)
    K_inv = torch.linalg.inv(K)
    rays = pix @ K_inv.T
    pts_cam_i = rays * depth_i.unsqueeze(-1)
    pts_h = torch.cat([pts_cam_i, torch.ones_like(pts_cam_i[..., :1])], dim=-1)
    T_ij = torch.linalg.inv(c2w_j) @ c2w_i
    pts_cam_j = (pts_h.view(-1, 4) @ T_ij.T).view(H, W, 4)[..., :3]

    z_j = pts_cam_j[..., 2]
    u_j = K[0, 0] * pts_cam_j[..., 0] / z_j.clamp_min(1e-3) + K[0, 2]
    v_j = K[1, 1] * pts_cam_j[..., 1] / z_j.clamp_min(1e-3) + K[1, 2]

    gx = (u_j / (W - 1)) * 2 - 1
    gy = (v_j / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    sampled = F.grid_sample(
        rgb_j.unsqueeze(0), grid, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    ).squeeze(0)
    valid = (z_j > 1e-3) & (gx.abs() <= 1) & (gy.abs() <= 1)
    return sampled, valid


@dataclass
class PhotometricLabels:
    """Cached tensors used by the TTT loop."""

    pseudo: torch.Tensor       # (N, H, W) float in {0, 1}
    trust: torch.Tensor        # (N, H, W) float weight in [0, 1]


def compute_photometric_labels(
    rgb: torch.Tensor,           # (N, 3, H, W) float in [0, 1]
    depth: torch.Tensor,          # (N, H, W) metric depth
    K: torch.Tensor,              # (N, 3, 3) intrinsics (per frame)
    c2w: torch.Tensor,            # (N, 4, 4) cam-to-world
    neighbors: tuple[int, ...] = (-3, -1, 1, 3),
    residual_threshold: float = 0.10,
    consensus: int = 3,
    texture_eps: float = 0.05,
    depth_edge_rtol: float = 0.10,
) -> PhotometricLabels:
    """Multi-neighbor photometric consensus → pseudo-label + trust mask."""
    N, _, H, W = rgb.shape
    device = rgb.device

    trust = _texture_mask(rgb, eps=texture_eps) & ~_depth_edge_mask(depth, rtol=depth_edge_rtol)
    pseudo = torch.zeros((N, H, W), device=device, dtype=torch.float32)
    valid_total = torch.zeros_like(pseudo)
    votes = torch.zeros((N, H, W), device=device, dtype=torch.int32)

    for off in neighbors:
        for i in range(N):
            j = i + off
            if j < 0 or j >= N:
                continue
            warped, valid = _warp_i_to_j(rgb[j], depth[i], K[i], c2w[i], c2w[j])
            resid = (warped - rgb[i]).abs().mean(dim=0)
            high = (resid > residual_threshold) & valid & trust[i]
            votes[i] += high.int()
            valid_total[i] += valid.float()

    dyn = (votes >= consensus).float()
    # Trust weight = original trust mask AND at least 2 neighbors were valid
    # (otherwise we can't form consensus).
    trust_weight = (trust.float()) * (valid_total >= max(2, consensus)).float()
    return PhotometricLabels(pseudo=dyn, trust=trust_weight)


# ------------------------------------------------------------------
# TTT loop
# ------------------------------------------------------------------

def run_ttt(
    dyn_head: nn.Module,
    conf_features: torch.Tensor,    # (N, num_patches, dec_embed_dim)  [cached]
    patch_h: int,
    patch_w: int,
    labels: PhotometricLabels,
    steps: int = 200,
    lr: float = 1e-3,
    batch_frames: int = 8,
    pos_weight: float | None = None,
    log_every: int = 50,
) -> dict:
    """Train ``dyn_head`` in-place on cached features + pseudo-labels.

    Args:
        dyn_head: nn.Module mapping (Bf, num_patches, C) → (Bf, H, W) logits.
        conf_features: per-frame conf-decoder features (Pi3 frozen, cached).
        labels: photometric pseudo-labels + trust mask, full-res (N, H, W).
        steps: optimizer steps.
        batch_frames: how many frames per step (random subset of the N).
        pos_weight: BCE positive class weight; ``None`` → auto from label freq.

    Returns:
        Dict with training stats (final_loss, pos_frac).
    """
    device = conf_features.device
    N = conf_features.shape[0]
    H, W = labels.pseudo.shape[-2:]

    pseudo = labels.pseudo.to(device)
    trust = labels.trust.to(device)

    pos_frac = float(pseudo[trust > 0].mean().item()) if trust.sum() > 0 else 0.5
    pos_frac = max(min(pos_frac, 0.95), 0.05)
    if pos_weight is None:
        pos_weight = (1.0 - pos_frac) / pos_frac
    pw = torch.tensor([pos_weight], device=device, dtype=torch.float32)

    dyn_head.train()
    opt = torch.optim.Adam(dyn_head.parameters(), lr=lr)

    last_loss = float("nan")
    for step in range(steps):
        idx = torch.randint(0, N, (min(batch_frames, N),), device=device)
        feats = conf_features[idx]                          # (Bf, num_patches, C)
        gt = pseudo[idx]                                    # (Bf, H, W)
        w = trust[idx]                                      # (Bf, H, W)

        logits = dyn_head(feats, patch_h=patch_h, patch_w=patch_w)
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits.unsqueeze(1), size=(H, W),
                                    mode="bilinear", align_corners=False).squeeze(1)

        bce = F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pw, reduction="none")
        denom = w.sum().clamp_min(1.0)
        loss = (bce * w).sum() / denom

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

        if log_every and (step + 1) % log_every == 0:
            LOGGER.info("TTT step %d/%d loss=%.4f pos_frac=%.3f", step + 1, steps, last_loss, pos_frac)

    dyn_head.eval()
    return {"final_loss": last_loss, "pos_frac": pos_frac, "steps": steps}


@torch.no_grad()
def predict_masks(
    dyn_head: nn.Module,
    conf_features: torch.Tensor,
    patch_h: int,
    patch_w: int,
    out_hw: tuple[int, int] | None = None,
    threshold: float = 0.5,
    chunk: int = 16,
) -> np.ndarray:
    """Run trained dyn_head on cached features → (N, H, W) uint8 masks."""
    N = conf_features.shape[0]
    masks: list[np.ndarray] = []
    H_t, W_t = out_hw if out_hw is not None else (patch_h * 14, patch_w * 14)
    for s in range(0, N, chunk):
        feats = conf_features[s : s + chunk]
        logits = dyn_head(feats, patch_h=patch_h, patch_w=patch_w)
        if (logits.shape[-2], logits.shape[-1]) != (H_t, W_t):
            logits = F.interpolate(logits.unsqueeze(1), size=(H_t, W_t),
                                    mode="bilinear", align_corners=False).squeeze(1)
        prob = torch.sigmoid(logits)
        masks.append((prob > threshold).cpu().numpy().astype(np.uint8))
    return np.concatenate(masks, axis=0)
