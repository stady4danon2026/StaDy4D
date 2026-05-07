#!/usr/bin/env python3
"""Render side-by-side viz of: RGB | photometric pseudo | head prob | head mask | overlay.

Loads the trained DynHead + cached pool bundles and renders a 3-frame grid PNG
per scene to ``outputs/pi3_dyn_viz_head/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F


def _label_bar(text: str, w: int, h: int = 28, bg: int = 30) -> np.ndarray:
    bar = np.full((h, w, 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return bar


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path("<DATA_ROOT>/sigma_cache/pi3_dyn_pool"))
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--out", type=Path, default=Path("outputs/pi3_dyn_viz_head"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--scenes", nargs="+", default=[
        "scene_T03_000__cam_05_orbit_crossroad",
        "scene_T03_000__cam_00_car_forward",
        "scene_T03_000__cam_03_drone_forward",
        "scene_T07_016__cam_00_car_forward",
        "scene_T10_030__cam_05_orbit_crossroad",
        "scene_T10_030__cam_04_orbit_building",
    ])
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    from sigma.pipeline.motion.pi3_dyn_head import DynHead
    head = DynHead.load(args.checkpoint, map_location=args.device).to(args.device).eval()

    for name in args.scenes:
        bp = args.cache_dir / f"{name}.pt"
        if not bp.exists():
            print(f"skip (no cache): {name}")
            continue

        b = torch.load(bp, map_location=args.device)
        feats = b["features"].to(args.device, dtype=torch.float32)
        labels = b["labels"].cpu().numpy()
        trust = b["trust"].cpu().numpy()
        patch_h, patch_w = int(b["patch_h"]), int(b["patch_w"])
        N, H, W = labels.shape

        # Run head in chunks (avoid OOM)
        probs = []
        with torch.no_grad():
            for s in range(0, N, 8):
                logits = head(feats[s : s + 8], patch_h=patch_h, patch_w=patch_w)
                if logits.shape[-2:] != (H, W):
                    logits = F.interpolate(logits.unsqueeze(1), size=(H, W),
                                           mode="bilinear", align_corners=False).squeeze(1)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        prob = np.concatenate(probs, axis=0)        # (N, H, W) float32 in [0, 1]
        mask = (prob > args.threshold).astype(np.uint8)

        # Reload original RGB at bundle resolution
        scene = name.split("__")[0]
        cam = "__".join(name.split("__")[1:])
        mp4 = args.root / scene / "dynamic" / cam / "rgb.mp4"
        try:
            r = imageio.get_reader(str(mp4))
            rgb_orig = np.stack([np.asarray(f) for i, f in enumerate(r) if i < N])
            r.close()
            rgb = np.stack([cv2.resize(rgb_orig[i], (W, H)) for i in range(N)])
        except Exception:
            rgb = np.zeros((N, H, W, 3), dtype=np.uint8)

        frames_to_show = [N // 6, N // 2, 5 * N // 6]
        rows = []
        for f_idx in frames_to_show:
            rgb_f = rgb[f_idx]
            ps = labels[f_idx].astype(bool)
            tr = trust[f_idx].astype(bool)
            pr = prob[f_idx]
            mk = mask[f_idx].astype(bool)

            # Heatmap of prob
            heat = (np.clip(pr, 0, 1) * 255).astype(np.uint8)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

            ps_viz = (ps[..., None].astype(np.uint8) * 255).repeat(3, axis=-1)
            mk_viz = (mk[..., None].astype(np.uint8) * 255).repeat(3, axis=-1)

            ov = rgb_f.copy()
            no_trust = ~tr
            ov[no_trust] = (ov[no_trust] * 0.7 + np.array([255, 100, 100]) * 0.3).astype(np.uint8)
            if mk.any():
                ov[mk] = (ov[mk] * 0.3 + np.array([0, 255, 0]) * 0.7).astype(np.uint8)

            row = np.concatenate([rgb_f, ps_viz, heat, mk_viz, ov], axis=1)
            bar = _label_bar(
                f"frame {f_idx} | RGB | photometric pseudo | head prob (heat) | head mask thr={args.threshold:.2f} | overlay",
                row.shape[1],
            )
            rows.append(np.concatenate([bar, row], axis=0))

        grid = np.concatenate(rows, axis=0)
        title = (
            f"{name}   pseudo={labels.mean()*100:.2f}%  "
            f"head_pred>{args.threshold}={mask.mean()*100:.2f}%  "
            f"prob_mean={prob.mean():.3f}  prob_std={prob.std():.3f}"
        )
        title_bar = np.full((36, grid.shape[1], 3), 60, dtype=np.uint8)
        cv2.putText(title_bar, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        grid = np.concatenate([title_bar, grid], axis=0)

        out_path = args.out / f"{name}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"saved {out_path}  prob_mean={prob.mean():.3f}  prob_std={prob.std():.4f}")


if __name__ == "__main__":
    main()
