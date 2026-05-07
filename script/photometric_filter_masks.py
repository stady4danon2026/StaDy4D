#!/usr/bin/env python3
"""Filter head-mask false positives using a scene-level photometric envelope.

For each scene with completed head masks:
  1. Run Pi3 (frozen) on the origin RGB → depth, pose, intrinsics.
  2. Compute photometric pseudo-labels with high-recall settings.
  3. Temporal-max across all frames → 2D ``envelope`` map (pixels flagged
     dynamic at ANY time during the video).
  4. Dilate by ``envelope_dilate`` to compensate for photometric low-recall.
  5. If the envelope covers ``> envelope_min_coverage`` of pixels, AND
     against the per-frame head mask. Otherwise leave the head mask alone
     (assumes photometric is dead for this camera, e.g. ego-moving).

Outputs filtered masks to ``<src_dir>_filtered/<scene>/<cam>/mask/``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger("photometric_filter")


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


def _derive_K(local_pts, H, W, device):
    N = local_pts.shape[0]
    cx, cy = W / 2.0, H / 2.0
    u = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W).expand(N, H, W)
    v = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1).expand(N, H, W)
    x, y, z = local_pts[..., 0], local_pts[..., 1], local_pts[..., 2]
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
        Ks[i, 0, 2] = cx
        Ks[i, 1, 2] = cy
        Ks[i, 2, 2] = 1.0
    return Ks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/pi3_dyn_keyframe_ttt"))
    p.add_argument("--dst", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/pi3_dyn_keyframe_ttt_filtered"))
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--pi3-checkpoint", default="yyfz233/Pi3X")
    p.add_argument("--device", default="cuda")
    p.add_argument("--pixel-limit", type=int, default=255000)
    # Photometric settings — chosen for HIGH RECALL on the camera-classes
    # where photometric works at all. False positives are tolerable because
    # we'll dilate aggressively.
    p.add_argument("--photo-threshold", type=float, default=0.05)
    p.add_argument("--photo-consensus", type=int, default=2)
    p.add_argument("--photo-neighbors", nargs="+", type=int, default=[-3, -1, 1, 3])
    p.add_argument("--envelope-dilate", type=int, default=31,
                   help="Dilation kernel size applied to the temporal-max envelope.")
    p.add_argument("--envelope-min-coverage", type=float, default=0.005,
                   help="If the dilated envelope covers less than this fraction of "
                        "pixels (i.e. photometric likely dead for this camera), pass "
                        "the head mask through unchanged.")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    # Discover scenes already processed (have at least one mask file).
    pairs: list[tuple[str, str]] = []
    for scene_dir in sorted(args.src.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene_"):
            continue
        for cam_dir in sorted(scene_dir.iterdir()):
            if not (cam_dir / "mask").is_dir():
                continue
            if not list((cam_dir / "mask").glob("mask_*.png")):
                continue
            pairs.append((scene_dir.name, cam_dir.name))
    if args.limit:
        pairs = pairs[: args.limit]
    LOGGER.info("Filtering %d (scene, cam) pairs", len(pairs))

    from pi3.models.pi3x import Pi3X
    from sigma.pipeline.motion.pi3_dyn_ttt import compute_photometric_labels

    LOGGER.info("Loading Pi3X (frozen)")
    pi3 = Pi3X.from_pretrained(args.pi3_checkpoint).to(args.device).eval()
    for prm in pi3.parameters():
        prm.requires_grad_(False)
    amp = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
           else torch.float16)

    summary = []
    t_start = time.time()
    to_t = transforms.ToTensor()

    for k, (scene, cam) in enumerate(pairs):
        src_dir = args.src / scene / cam / "mask"
        dst_dir = args.dst / scene / cam / "mask"
        if dst_dir.exists() and len(list(dst_dir.glob("mask_*.png"))) > 0:
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = args.root / scene / "dynamic" / cam / "rgb.mp4"
        if not rgb_path.exists():
            LOGGER.warning("[%d/%d] %s/%s — rgb.mp4 missing, skipping",
                           k + 1, len(pairs), scene, cam)
            continue

        # Load video
        try:
            r = imageio.get_reader(str(rgb_path))
            frames = np.stack([np.asarray(f) for f in r])
            r.close()
        except Exception as e:
            LOGGER.warning("[%d/%d] %s/%s — load fail: %s", k + 1, len(pairs), scene, cam, e)
            continue
        N, H_orig, W_orig = frames.shape[:3]

        TARGET_W, TARGET_H = _compute_target_size(H_orig, W_orig, args.pixel_limit)
        tensors = [to_t(Image.fromarray(frames[i]).resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS))
                   for i in range(N)]
        imgs = torch.stack(tensors, dim=0).unsqueeze(0).to(args.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=amp):
                out = pi3(imgs)
        local_pts = out["local_points"][0].float()
        c2w = out["camera_poses"][0].float()
        depth = local_pts[..., 2].clamp_min(0.0)
        K = _derive_K(local_pts, TARGET_H, TARGET_W, args.device)
        rgb_pi3 = imgs[0].to(local_pts.device, dtype=torch.float32)

        with torch.no_grad():
            labels = compute_photometric_labels(
                rgb=rgb_pi3, depth=depth, K=K, c2w=c2w,
                neighbors=tuple(args.photo_neighbors),
                residual_threshold=args.photo_threshold,
                consensus=args.photo_consensus,
            )
        env = (labels.pseudo.max(dim=0).values > 0).cpu().numpy().astype(np.uint8)  # (H_pi3, W_pi3)

        # Resize envelope to original res + dilate aggressively
        env_orig = cv2.resize(env, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
        if args.envelope_dilate > 0:
            kd = np.ones((args.envelope_dilate, args.envelope_dilate), np.uint8)
            env_orig = cv2.dilate(env_orig, kd)
        env_coverage = float(env_orig.mean())

        # Apply filter (or fallback)
        photo_alive = env_coverage > args.envelope_min_coverage
        head_pos_total = 0
        kept_pos_total = 0
        for i in range(N):
            f = src_dir / f"mask_{i:04d}.png"
            if not f.exists():
                continue
            m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = (m > 127).astype(np.uint8)
            head_pos_total += int(m.sum())
            if photo_alive:
                m = m & env_orig
            kept_pos_total += int(m.sum())
            cv2.imwrite(str(dst_dir / f"mask_{i:04d}.png"), m * 255)

        kept_frac = kept_pos_total / max(head_pos_total, 1)
        rec = {
            "scene": scene, "camera": cam,
            "envelope_coverage": env_coverage,
            "photo_alive": bool(photo_alive),
            "head_pos_total": head_pos_total,
            "kept_pos_total": kept_pos_total,
            "kept_frac": float(kept_frac),
        }
        summary.append(rec)
        if (k + 1) % 10 == 0 or (k + 1) == len(pairs):
            LOGGER.info(
                "[%d/%d] %s/%s  env=%.2f%%  alive=%s  kept=%.2f%% of head positives",
                k + 1, len(pairs), scene, cam,
                env_coverage * 100, photo_alive, kept_frac * 100,
            )

        with (args.dst / "summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2)

    elapsed = time.time() - t_start
    LOGGER.info("Done. %d scenes filtered in %.1fmin", len(summary), elapsed / 60)


if __name__ == "__main__":
    main()
