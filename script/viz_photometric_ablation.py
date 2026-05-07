#!/usr/bin/env python3
"""Sweep photometric-teacher hyperparameters on cached bundles and render
side-by-side visualisations.

Each row shows one scene; columns are different parameter settings.

The following knobs can be tuned:
  --threshold     residual cutoff (lower → more positives, more noise)
  --consensus     # neighbors that must agree (lower → more positives)
  --neighbors     frame offsets used (e.g. -3,-1,1,3 vs -5,-2,2,5)
  --texture-eps   gradient-magnitude floor (lower → trust uniform regions)
  --edge-rtol     depth-edge ratio (higher → trust depth discontinuities)

Computes IoU vs the cached GSAM mask for each setting so we can pick the
best config to use as a TTT teacher (or as an additional weighting signal).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from sigma.pipeline.motion.pi3_dyn_ttt import compute_photometric_labels


def _label(text: str, w: int, h: int = 24, bg: int = 30) -> np.ndarray:
    bar = np.full((h, w, 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return bar


def _derive_intrinsics_batch(local_pts: torch.Tensor, H: int, W: int) -> torch.Tensor:
    import math
    N = local_pts.shape[0]
    cx, cy = W / 2.0, H / 2.0
    device = local_pts.device
    u = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W).expand(N, H, W)
    v = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1).expand(N, H, W)
    x = local_pts[..., 0]; y = local_pts[..., 1]; z = local_pts[..., 2]
    valid = z > 0.01
    mfx = valid & ((u - cx).abs() > W * 0.1)
    mfy = valid & ((v - cy).abs() > H * 0.1)
    dfx = W / (2 * math.tan(math.radians(60)))
    dfy = H / (2 * math.tan(math.radians(60)))
    Ks = torch.zeros((N, 3, 3), device=device, dtype=torch.float32)
    for i in range(N):
        try:
            fx = float(torch.median((u[i][mfx[i]] - cx) / (x[i][mfx[i]] / z[i][mfx[i]])).item()) if mfx[i].sum() > 10 else dfx
            fy = float(torch.median((v[i][mfy[i]] - cy) / (y[i][mfy[i]] / z[i][mfy[i]])).item()) if mfy[i].sum() > 10 else dfy
        except Exception:
            fx, fy = dfx, dfy
        Ks[i, 0, 0] = float(np.clip(fx, W * 0.2, W * 5.0))
        Ks[i, 1, 1] = float(np.clip(fy, H * 0.2, H * 5.0))
        Ks[i, 0, 2] = cx; Ks[i, 1, 2] = cy; Ks[i, 2, 2] = 1.0
    return Ks


def _run_pi3(rgb_np: np.ndarray, pi3, device: str, pixel_limit: int):
    import math
    from PIL import Image
    from torchvision import transforms

    N, H_orig, W_orig = rgb_np.shape[:3]
    scale = math.sqrt(pixel_limit / max(W_orig * H_orig, 1))
    Wf, Hf = W_orig * scale, H_orig * scale
    k, m = round(Wf / 14), round(Hf / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > Wf / Hf:
            k -= 1
        else:
            m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14

    to_t = transforms.ToTensor()
    tensors = [to_t(Image.fromarray(rgb_np[i]).resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS))
               for i in range(N)]
    imgs = torch.stack(tensors, dim=0).unsqueeze(0).to(device)
    amp = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
           else torch.float16)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=amp):
            out = pi3(imgs)
    local_pts = out["local_points"][0].float()
    c2w = out["camera_poses"][0].float()
    depth = local_pts[..., 2].clamp_min(0.0)
    K = _derive_intrinsics_batch(local_pts, TARGET_H, TARGET_W)
    rgb_pi3 = imgs[0].to(local_pts.device, dtype=torch.float32)
    return rgb_pi3, depth, K, c2w, TARGET_H, TARGET_W


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.bool(), b.bool()
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / max(union, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=[
        "scene_T03_007/cam_05_orbit_crossroad",
        "scene_T03_002/cam_00_car_forward",
        "scene_T03_021/cam_00_car_forward",
    ])
    p.add_argument("--cache-dir", type=Path, default=Path("<DATA_ROOT>/sigma_cache/pi3_dyn_pool"))
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--out", type=Path, default=Path("outputs/pi3_dyn_photometric_ablation"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--pi3-checkpoint", default="yyfz233/Pi3X")
    p.add_argument("--pixel-limit", type=int, default=255000)
    p.add_argument("--frame", type=int, default=25)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Hyperparameter grid (each row = different setting)
    settings = [
        # name, threshold, consensus, neighbors, texture_eps, edge_rtol
        ("default",       0.10, 3, (-3, -1, 1, 3),     0.05, 0.10),
        ("loose-thr",     0.05, 3, (-3, -1, 1, 3),     0.05, 0.10),
        ("loose-cons",    0.10, 2, (-3, -1, 1, 3),     0.05, 0.10),
        ("wide-neigh",    0.10, 3, (-5, -2, 2, 5),     0.05, 0.10),
        ("trust-all",     0.10, 3, (-3, -1, 1, 3),     0.005, 0.30),
        ("aggressive",    0.05, 2, (-3, -2, -1, 1, 2, 3), 0.02, 0.20),
    ]

    # Need Pi3 to recompute photometric labels (depth+pose) for each scene
    from pi3.models.pi3x import Pi3X
    pi3 = Pi3X.from_pretrained(args.pi3_checkpoint).to(args.device).eval()
    for prm in pi3.parameters():
        prm.requires_grad_(False)

    for scene_name in args.scenes:
        bundle_name = scene_name.replace("/", "__")
        bp = args.cache_dir / f"{bundle_name}.pt"
        if not bp.exists():
            print(f"skip {scene_name}: no cache")
            continue

        b = torch.load(bp, map_location=args.device)
        gsam = b["gsam"].float() if "gsam" in b else None

        # Load RGB + run Pi3 to get depth/poses/K for photometric
        scene = scene_name.split("/")[0]
        cam = "/".join(scene_name.split("/")[1:])
        mp4 = args.root / scene / "dynamic" / cam / "rgb.mp4"
        r = imageio.get_reader(str(mp4))
        frames = np.stack([np.asarray(f) for f in r])
        r.close()

        rgb_pi3, depth, K, c2w, H, W = _run_pi3(frames, pi3, args.device, args.pixel_limit)

        fi = min(args.frame, frames.shape[0] - 1)
        rgb_view = cv2.resize(frames[fi], (W, H))

        # Build per-row panels: each row = [RGB | photometric overlay | GSAM]
        rows = []
        gv = None
        if gsam is not None:
            gv = (gsam[fi].cpu().numpy().astype(np.uint8) * 255)[..., None].repeat(3, axis=-1)

        for name, thr, cons, neigh, tex, edge in settings:
            with torch.no_grad():
                labels = compute_photometric_labels(
                    rgb=rgb_pi3, depth=depth, K=K, c2w=c2w,
                    neighbors=tuple(neigh),
                    residual_threshold=thr,
                    consensus=cons,
                    texture_eps=tex,
                    depth_edge_rtol=edge,
                )
            ps = labels.pseudo[fi].cpu().numpy().astype(np.uint8)
            tr = labels.trust[fi].cpu().numpy()

            ov = rgb_view.copy()
            no_trust = tr < 0.5
            ov[no_trust] = (ov[no_trust] * 0.7 + np.array([255, 100, 100]) * 0.3).astype(np.uint8)
            mk = ps > 0
            if mk.any():
                ov[mk] = (ov[mk] * 0.3 + np.array([0, 255, 0]) * 0.7).astype(np.uint8)

            iou = (_iou(labels.pseudo, gsam) if gsam is not None else float("nan"))
            stat = f"{name}  pos={ps.mean()*100:.2f}%  trust={(tr>0.5).mean()*100:.0f}%  IoU={iou:.3f}"

            row_panels = [rgb_view, ov]
            if gv is not None:
                row_panels.append(gv)
            row_img = np.concatenate(row_panels, axis=1)
            row_with_label = np.concatenate([_label(stat, row_img.shape[1], h=30), row_img], axis=0)
            rows.append(row_with_label)

        title_bar = np.full((40, rows[0].shape[1], 3), 60, dtype=np.uint8)
        cv2.putText(title_bar, f'{scene_name}  frame {fi}   columns: RGB | photometric overlay | GSAM target',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        grid = np.concatenate([title_bar] + rows, axis=0)
        out_path = args.out / f"{bundle_name}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
