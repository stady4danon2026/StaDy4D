#!/usr/bin/env python3
"""Photometric teacher: geometric pseudo-mask of dynamic pixels.

Given Pi3X-predicted depth + camera_poses + intrinsics + RGB frames,
warp each frame *i* into its neighbors *j* using:

    p_j = K @ (R_ji @ K^-1 @ [u, v, 1] @ depth_i + t_ji)

Sample frame *j* RGB at p_j; compare to frame *i*'s RGB. Pixels with
high residual across MULTIPLE neighbor pairs are flagged dynamic.

This is the geometric "auto-mask" idea (MonoDepth2 etc.) generalized
to N-frame consensus.  Used as the teacher for the TTT student head.

Usage:
    python script/photometric_teacher.py \
        --pred outputs/_eval_batch/scene_T03_008_NEW_fill \
        --gt-rgb StaDy4D/short/test/scene_T03_008/dynamic/cam_00_car_forward/rgb.mp4 \
        --out comparison_vis/teacher_T03_008.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _decode_video(p: Path) -> np.ndarray:
    r = imageio.get_reader(str(p))
    out = [np.asarray(f) for f in r]
    r.close()
    return np.stack(out, axis=0)


def _load_pred_seq(pred_dir: Path):
    depth_files = sorted((pred_dir / "depth").glob("*.npy"))
    ext_files = sorted((pred_dir / "extrinsics").glob("*.npy"))
    intr_files = sorted((pred_dir / "intrinsics").glob("*.npy"))
    n = min(len(depth_files), len(ext_files), len(intr_files))
    depths = np.stack([np.load(depth_files[i]).astype(np.float32) for i in range(n)])
    exts = np.stack([np.load(ext_files[i]).astype(np.float32) for i in range(n)])    # (n, 3, 4)
    intrs = np.stack([np.load(intr_files[i]).astype(np.float32) for i in range(n)])  # (n, 3, 3)
    # Promote 3x4 c2w → 4x4
    c2ws = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    c2ws[:, :3, :] = exts
    return depths, c2ws, intrs


def warp_i_to_j(rgb_j: torch.Tensor, depth_i: torch.Tensor, K: torch.Tensor,
                c2w_i: torch.Tensor, c2w_j: torch.Tensor) -> torch.Tensor:
    """Warp rgb_j to frame i's pixel grid using depth_i + relative pose.

    Args:
        rgb_j: (3, H, W) target frame RGB in [0, 1].
        depth_i: (H, W) source frame depth.
        K: (3, 3) shared intrinsic.
        c2w_i, c2w_j: (4, 4) camera-to-world poses.

    Returns:
        (3, H, W) frame j sampled at frame i's pixels (where geometry says
        the same world point projects).  Values outside the frame are 0.
    """
    H, W = depth_i.shape
    device = depth_i.device

    # Pixel grid for frame i.
    v, u = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                          torch.arange(W, device=device, dtype=torch.float32),
                          indexing="ij")
    ones = torch.ones_like(u)
    pix_i = torch.stack([u, v, ones], dim=-1)                     # (H, W, 3)

    # Back-project depth_i to camera-i 3D points.
    K_inv = torch.linalg.inv(K)
    rays = pix_i @ K_inv.T                                          # (H, W, 3)
    pts_cam_i = rays * depth_i.unsqueeze(-1)                        # (H, W, 3)

    # Homogenise + transform i → world → j.
    pts_h = torch.cat([pts_cam_i, torch.ones_like(pts_cam_i[..., :1])], dim=-1)
    w2c_j = torch.linalg.inv(c2w_j)
    T_ij = w2c_j @ c2w_i                                            # (4, 4)
    pts_cam_j = (pts_h.view(-1, 4) @ T_ij.T).view(H, W, 4)[..., :3] # (H, W, 3)

    # Project into frame j.
    z_j = pts_cam_j[..., 2]
    u_j = K[0, 0] * pts_cam_j[..., 0] / z_j.clamp_min(1e-3) + K[0, 2]
    v_j = K[1, 1] * pts_cam_j[..., 1] / z_j.clamp_min(1e-3) + K[1, 2]

    # Normalize to [-1, 1] for grid_sample.
    grid_x = (u_j / (W - 1)) * 2 - 1
    grid_y = (v_j / (H - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)       # (1, H, W, 2)

    sampled = F.grid_sample(
        rgb_j.unsqueeze(0), grid, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    ).squeeze(0)                                                     # (3, H, W)
    valid = (z_j > 1e-3) & (grid_x.abs() <= 1) & (grid_y.abs() <= 1)
    return sampled, valid


def _texture_weight(rgb: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """Per-pixel gradient magnitude; low-texture regions get downweighted.

    rgb: (3, H, W) in [0, 1].  Returns (H, W) in [0, 1].
    """
    gray = rgb.mean(dim=0, keepdim=True).unsqueeze(0)            # (1, 1, H, W)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2).squeeze()                # (H, W)
    return (mag > eps).float()


def _depth_edge(depth: torch.Tensor, rtol: float = 0.05) -> torch.Tensor:
    """Mark depth discontinuities (huge relative jumps to neighbors)."""
    d = depth.unsqueeze(0).unsqueeze(0)                          # (1, 1, H, W)
    pad = F.pad(d, (1, 1, 1, 1), mode="replicate")
    neigh_max = F.max_pool2d(pad, 3, stride=1)
    neigh_min = -F.max_pool2d(-pad, 3, stride=1)
    rel = (neigh_max - neigh_min) / neigh_min.clamp_min(1e-3)
    return (rel.squeeze() > rtol)


def compute_pseudo_mask(
    depths: np.ndarray,                      # (N, H, W)
    c2ws: np.ndarray,                        # (N, 4, 4)
    intrs: np.ndarray,                       # (N, 3, 3)
    rgbs: np.ndarray,                        # (N, H, W, 3) uint8
    neighbor_offsets=(-3, -1, 1, 3),
    threshold: float = 0.10,                 # photometric residual threshold
    consensus: int = 3,                      # # neighbors that must agree
    texture_eps: float = 0.05,               # min gradient magnitude to trust residual
    depth_edge_rtol: float = 0.10,           # mask out depth-edge pixels
    morph_iters: int = 1,                    # close+open iterations on the binary mask
    device: str = "cuda",
) -> np.ndarray:
    """Returns (N, H, W) bool dynamic mask via N-neighbor photometric consensus."""
    N, H, W = depths.shape
    device = device if torch.cuda.is_available() else "cpu"
    depths_t = torch.from_numpy(depths).to(device)
    c2ws_t = torch.from_numpy(c2ws).to(device)
    intrs_t = torch.from_numpy(intrs).to(device)
    rgbs_t = torch.from_numpy(rgbs).to(device).float().permute(0, 3, 1, 2) / 255.0  # (N, 3, H, W)

    masks = np.zeros((N, H, W), dtype=bool)

    for i in range(N):
        votes = torch.zeros((H, W), device=device, dtype=torch.int32)
        # Trust mask: high texture AND not on a depth edge (residuals are
        # noisy on uniform sky/road or at depth discontinuities).
        trust = _texture_weight(rgbs_t[i], eps=texture_eps).bool()
        trust = trust & ~_depth_edge(depths_t[i], rtol=depth_edge_rtol)

        for off in neighbor_offsets:
            j = i + off
            if j < 0 or j >= N: continue
            warped, valid = warp_i_to_j(
                rgbs_t[j], depths_t[i],
                intrs_t[i], c2ws_t[i], c2ws_t[j],
            )
            resid = torch.abs(warped - rgbs_t[i]).mean(dim=0)         # (H, W)
            high = (resid > threshold) & valid & trust
            votes += high.int()

        dyn = (votes >= consensus)
        # Morphological clean-up: close small holes, open small specks.
        if morph_iters > 0:
            d = dyn.unsqueeze(0).unsqueeze(0).float()
            for _ in range(morph_iters):
                d = F.max_pool2d(d, 3, stride=1, padding=1)            # dilate
            for _ in range(morph_iters):
                d = -F.max_pool2d(-d, 3, stride=1, padding=1)          # erode
            dyn = (d.squeeze() > 0.5)

        masks[i] = dyn.cpu().numpy()

    return masks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, type=Path,
                   help="Pipeline output dir (with depth/, extrinsics/, intrinsics/, mask/, rgb/).")
    p.add_argument("--gt-rgb", required=True, type=Path,
                   help="StaDy4D dynamic rgb.mp4 (origin frames with cars).")
    p.add_argument("--out", required=True, type=Path,
                   help="Output PNG path showing teacher vs GroundedSAM mask.")
    p.add_argument("--threshold", type=float, default=0.04)
    p.add_argument("--consensus", type=int, default=2)
    p.add_argument("--show-frames", type=int, nargs="*", default=[5, 15, 25])
    args = p.parse_args()

    print(f"Loading Pi3X output from {args.pred}...")
    depths, c2ws, intrs = _load_pred_seq(args.pred)
    print(f"  {depths.shape[0]} frames @ {depths.shape[1]}x{depths.shape[2]}")

    print(f"Loading origin RGB from {args.gt_rgb}...")
    rgbs = _decode_video(args.gt_rgb)[:depths.shape[0]]
    if rgbs.shape[1:3] != depths.shape[1:3]:
        Hd, Wd = depths.shape[1:3]
        rgbs = np.stack([cv2.resize(r, (Wd, Hd)) for r in rgbs])
    print(f"  RGB: {rgbs.shape}")

    print(f"Computing photometric pseudo-mask (threshold={args.threshold}, consensus={args.consensus})...")
    teacher_masks = compute_pseudo_mask(depths, c2ws, intrs, rgbs,
                                        threshold=args.threshold, consensus=args.consensus)
    print(f"  teacher mask: {teacher_masks.mean()*100:.2f}% pixels flagged dynamic on average")

    # Load GroundedSAM masks for comparison.
    gsam_dir = args.pred / "mask"
    gsam_masks = []
    for i in range(depths.shape[0]):
        f = gsam_dir / f"mask_{i:04d}.png"
        if not f.exists():
            gsam_masks.append(np.zeros(depths.shape[1:], dtype=bool))
            continue
        m = np.array(imageio.imread(f))
        if m.ndim == 3: m = m[..., 0]
        if m.shape != depths.shape[1:]:
            m = cv2.resize(m, (depths.shape[2], depths.shape[1]), interpolation=cv2.INTER_NEAREST)
        gsam_masks.append(m > 127)
    gsam_masks = np.stack(gsam_masks)
    print(f"  GroundedSAM mask: {gsam_masks.mean()*100:.2f}% pixels flagged dynamic on average")

    # IoU between teacher and GroundedSAM
    inter = (teacher_masks & gsam_masks).sum()
    union = (teacher_masks | gsam_masks).sum()
    iou = inter / max(union, 1)
    print(f"  IoU(teacher, GroundedSAM) = {iou:.3f}")

    # Render comparison
    n = len(args.show_frames)
    fig, axes = plt.subplots(n, 4, figsize=(18, 4.0 * n))
    if n == 1:
        axes = axes[None, :]
    for row, idx in enumerate(args.show_frames):
        if idx >= len(rgbs): continue
        rgb = rgbs[idx]
        gsam = gsam_masks[idx]
        teach = teacher_masks[idx]
        # Photometric residual viz
        # (recompute residual for vis)
        agree = teach & gsam
        only_teach = teach & ~gsam
        only_gsam = gsam & ~teach
        overlay = rgb.copy()
        overlay[only_gsam] = (overlay[only_gsam] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
        overlay[only_teach] = (overlay[only_teach] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
        overlay[agree] = (overlay[agree] * 0.4 + np.array([0, 200, 0]) * 0.6).astype(np.uint8)

        axes[row, 0].imshow(rgb); axes[row, 0].set_title(f"frame {idx} RGB"); axes[row, 0].axis("off")
        axes[row, 1].imshow(rgb); axes[row, 1].imshow(gsam, alpha=0.5, cmap="Reds")
        axes[row, 1].set_title(f"GroundedSAM ({gsam.mean()*100:.1f}%)"); axes[row, 1].axis("off")
        axes[row, 2].imshow(rgb); axes[row, 2].imshow(teach, alpha=0.5, cmap="Blues")
        axes[row, 2].set_title(f"Photometric teacher ({teach.mean()*100:.1f}%)"); axes[row, 2].axis("off")
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title(f"green=agree red=GSAM-only blue=teach-only  IoU={(teach&gsam).sum()/max((teach|gsam).sum(),1):.2f}")
        axes[row, 3].axis("off")

    fig.suptitle(f"Photometric teacher vs GroundedSAM  •  {args.pred.name}  "
                 f"•  global IoU={iou:.3f}", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
