#!/usr/bin/env python3
"""Render side-by-side mask + reconstruction comparison between NEW_gsam and NEW_sam3.

For each scene, layout is 4 frames × 3 columns:
  RGB | NEW_gsam mask overlay | NEW_sam3 mask overlay
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2, numpy as np, imageio.v2 as imageio


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--methods-root", type=Path, default=Path("eval_results/sam3_toy_v2"))
    p.add_argument("--out", type=Path, default=Path("outputs/sam3_toy_compare_viz"))
    p.add_argument("--scenes", nargs="+", default=[
        "scene_T03_000/cam_00_car_forward",
        "scene_T03_021/cam_05_orbit_crossroad",
        "scene_T07_023/cam_05_orbit_crossroad",
        "scene_T07_007/cam_04_orbit_building",
        "scene_T10_028/cam_00_car_forward",
        "scene_T03_007/cam_03_drone_forward",
        "scene_T07_041/cam_07_pedestrian",
        "scene_T10_029/cam_01_car_forward",
        "scene_T07_034/cam_05_orbit_crossroad",
        "scene_T03_002/cam_00_car_forward",
    ])
    p.add_argument("--methods", nargs="+", default=["NEW_gsam", "NEW_sam3"])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for sc in args.scenes:
        scene, cam = sc.split("/")
        mp4 = args.root / scene / "dynamic" / cam / "rgb.mp4"
        if not mp4.exists():
            continue
        r = imageio.get_reader(str(mp4))
        frames = np.stack([np.asarray(f) for f in r])
        r.close()
        N, H, W = frames.shape[:3]
        rows = []
        frames_to_show = [N // 6, N // 3, N // 2, 2 * N // 3, 5 * N // 6]
        for fi in frames_to_show:
            rgb = frames[fi]
            cols = [rgb]
            for method in args.methods:
                md = args.methods_root / method / f"{scene}_{cam}" / "mask"
                m = cv2.imread(str(md / f"mask_{fi:04d}.png"), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    cols.append(rgb)
                    continue
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                ov = rgb.copy()
                hot = m > 127
                if hot.any():
                    ov[hot] = (ov[hot] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
                cols.append(ov)
            bar = np.full((22, sum(c.shape[1] for c in cols), 3), 30, dtype=np.uint8)
            label = f"frame {fi}  RGB | " + " | ".join(args.methods)
            cv2.putText(bar, label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            rows.append(np.concatenate([bar, np.concatenate(cols, axis=1)], axis=0))
        grid = np.concatenate(rows, axis=0)
        title = np.full((30, grid.shape[1], 3), 60, dtype=np.uint8)
        cv2.putText(title, sc, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        grid = np.concatenate([title, grid], axis=0)
        out_p = args.out / f"{scene}__{cam}.png"
        cv2.imwrite(str(out_p), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"saved {out_p}")


if __name__ == "__main__":
    main()
