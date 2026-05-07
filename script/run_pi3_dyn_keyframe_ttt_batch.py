#!/usr/bin/env python3
"""Per-scene GSAM-keyframe TTT inference.

Pipeline per (scene, camera):
  1. Load N origin frames.
  2. Pi3X (frozen) forward → conf_decoder features captured.
  3. GroundedSAM on first ``--keyframes`` frames → dense pseudo-labels.
  4. TTT the DynHead (warm-started from a pooled checkpoint) for ``--ttt-steps``
     iterations on those K frames using focal BCE.
  5. Predict masks on all N frames with the adapted head; save to disk.

This is the "TTT with sparse oracle" story:
  - 5 GSAM forwards instead of 50 → 10× cheaper than full GSAM at deploy.
  - Per-scene adaptation closes the gap between pooled head and per-scene quality.

Outputs match the layout expected by downstream evaluation:
    out/<scene>/<camera>/mask/mask_NNNN.png
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("pi3_dyn_keyframe_ttt")


def _compute_target_size(H: int, W: int, pixel_limit: int) -> tuple[int, int]:
    scale = math.sqrt(pixel_limit / max(W * H, 1))
    W_f, H_f = W * scale, H * scale
    k, m = round(W_f / 14), round(H_f / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_f / H_f:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def _load_video(p: Path, max_frames: int | None) -> np.ndarray:
    r = imageio.get_reader(str(p))
    out = []
    for i, f in enumerate(r):
        if max_frames is not None and i >= max_frames:
            break
        out.append(np.asarray(f))
    r.close()
    return np.stack(out, axis=0)


def _focal_bce(logits, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    a = alpha * target + (1 - alpha) * (1 - target)
    return a * (1 - p_t).pow(gamma) * bce


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("StaDy4D"))
    p.add_argument("--duration", default="short")
    p.add_argument("--split", default="test")
    p.add_argument("--towns", nargs="+", default=["T03", "T07", "T10"])
    p.add_argument("--cameras", nargs="+", default=[
        "cam_00_car_forward", "cam_01_car_forward", "cam_02_car_forward",
        "cam_03_drone_forward", "cam_04_orbit_building", "cam_05_orbit_crossroad",
        "cam_06_cctv", "cam_07_pedestrian", "cam_08_pedestrian",
    ])
    p.add_argument("--out", type=Path,
                   default=Path("<DATA_ROOT>/sigma_outputs/pi3_dyn_keyframe_ttt"))
    p.add_argument("--pi3-checkpoint", default="yyfz233/Pi3X")
    p.add_argument("--init-checkpoint", type=Path, default=None,
                   help="Optional pooled DynHead checkpoint as warm start.")
    p.add_argument("--keyframes", type=int, default=5)
    p.add_argument("--keyframe-mode", choices=["first", "evenly"], default="evenly",
                   help="'first' = first K frames; 'evenly' = K frames spread across the video.")
    p.add_argument("--ttt-steps", type=int, default=50)
    p.add_argument("--ttt-lr", type=float, default=1e-3)
    p.add_argument("--ttt-batch-frames", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--morph-close", type=int, default=0,
                   help="Kernel size for morphological close (fills holes). 0 = off.")
    p.add_argument("--morph-dilate", type=int, default=0,
                   help="Kernel size for dilation (grows mask). 0 = off.")
    p.add_argument("--hull-fill-min-area", type=int, default=0,
                   help="If >0, replaces connected components >= this area with their "
                        "convex-hull-filled polygon (solves half-screen-object holes).")
    p.add_argument("--hull-fill-max-ratio", type=float, default=2.5,
                   help="Skip hull-fill if hull_area / component_area > this ratio. "
                        "Prevents runaway polygons from scattered specks.")
    p.add_argument("--pixel-limit", type=int, default=255000)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--min-blob-area", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save-overlays", action="store_true")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Discover scenes
    base = args.root / args.duration / args.split
    pairs: list[tuple[str, Path]] = []
    for sd in sorted(base.iterdir()):
        if not sd.is_dir() or not any(t in sd.name for t in args.towns):
            continue
        dyn = sd / "dynamic"
        if not dyn.is_dir():
            continue
        for cam in args.cameras:
            rgb = dyn / cam / "rgb.mp4"
            if rgb.exists():
                pairs.append((f"{sd.name}/{cam}", rgb))
    if args.limit:
        pairs = pairs[: args.limit]
    LOGGER.info("Discovered %d (scene, cam) pairs", len(pairs))

    # Load Pi3 and GSAM ONCE
    from pi3.models.pi3x import Pi3X
    from sigma.pipeline.motion.grounded_sam import GroundedSAMMotionEstimator
    from sigma.pipeline.motion.pi3_dyn_features import Pi3FeatureExtractor
    from sigma.pipeline.motion.pi3_dyn_head import DynHead
    from sigma.data.frame_record import FrameIORecord

    LOGGER.info("Loading Pi3X (frozen)")
    pi3 = Pi3X.from_pretrained(args.pi3_checkpoint).to(args.device).eval()
    for prm in pi3.parameters():
        prm.requires_grad_(False)

    LOGGER.info("Loading GroundedSAM teacher")
    gsam = GroundedSAMMotionEstimator(device=args.device)
    gsam.setup()

    if args.init_checkpoint and args.init_checkpoint.exists():
        LOGGER.info("Warm start: %s", args.init_checkpoint)
        init_state = DynHead.load(args.init_checkpoint, map_location="cpu").state_dict()
    else:
        LOGGER.info("Cold start (no warm-init checkpoint)")
        init_state = None

    summary = []
    t_start = time.time()
    to_tensor = transforms.ToTensor()
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    for k, (key, rgb_path) in enumerate(pairs):
        scene_name, cam_name = key.split("/")
        out_dir = args.out / scene_name / cam_name
        mask_dir = out_dir / "mask"
        if mask_dir.exists() and len(list(mask_dir.glob("mask_*.png"))) > 0:
            LOGGER.info("[%d/%d] %s — already processed, skipping", k + 1, len(pairs), key)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        try:
            frames = _load_video(rgb_path, args.max_frames)
        except Exception as e:
            LOGGER.warning("[%d/%d] %s — load fail: %s", k + 1, len(pairs), key, e)
            continue
        if frames.shape[0] < max(4, args.keyframes):
            LOGGER.warning("[%d/%d] %s — too few frames (%d)", k + 1, len(pairs), key, frames.shape[0])
            continue

        N, H_orig, W_orig = frames.shape[:3]
        TARGET_W, TARGET_H = _compute_target_size(H_orig, W_orig, args.pixel_limit)

        # Stack tensor for Pi3 input
        tensors = []
        for i in range(N):
            pil = Image.fromarray(frames[i]).resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            tensors.append(to_tensor(pil))
        imgs = torch.stack(tensors, dim=0).unsqueeze(0).to(args.device)

        t0 = time.time()
        # Pi3 forward + capture conf features
        extractor = Pi3FeatureExtractor(pi3)
        with extractor.capture():
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    _ = pi3(imgs)
        feats = extractor.features.to(torch.float32)                        # (N, P, 1024)
        patch_h, patch_w = extractor.patch_hw

        # Pick K keyframes either from the first K frames or evenly spread.
        K = min(args.keyframes, N)
        if args.keyframe_mode == "evenly":
            kf_indices = [int(round(i * (N - 1) / max(K - 1, 1))) for i in range(K)]
            kf_indices = sorted(set(i for i in kf_indices if 0 <= i < N))
            K = len(kf_indices)
        else:
            kf_indices = list(range(K))
        # GSAM's process_batch expects sequential frame indices (it indexes
        # frame_idx-1). Pass the K keyframes as 0..K-1 and map back ourselves.
        kf_records = {j: FrameIORecord(frame_idx=j, origin_image=frames[orig_i])
                       for j, orig_i in enumerate(kf_indices)}
        gsam_out = gsam.process_batch(kf_records)
        kf_masks_orig = np.stack(
            [gsam_out[j].data["motion"]["curr_mask"] for j in range(K)], axis=0
        )                                                                    # (K, H_orig, W_orig)
        kf_masks_t = torch.from_numpy(kf_masks_orig).float().unsqueeze(1).to(args.device)
        kf_labels = F.interpolate(kf_masks_t, size=(TARGET_H, TARGET_W),
                                   mode="nearest").squeeze(1)                # (K, H_pi3, W_pi3)

        # Build a fresh DynHead per scene (warm-init from pooled if available)
        head = DynHead().to(args.device)
        if init_state is not None:
            head.load_state_dict(init_state)

        # Only re-init bias when starting from cold (no warm start). Otherwise
        # keep the pooled head's learned bias — the keyframe sample alone is
        # too small to produce a reliable prior estimate.
        if init_state is None:
            prior = max(min(float(kf_labels.mean().item()), 0.5), 1e-3)
            with torch.no_grad():
                head.head.output_block[0][-1].bias.fill_(math.log(prior / (1 - prior)))

        # TTT — deterministic per-scene RNG so re-runs reproduce.
        head.train()
        opt = torch.optim.Adam(head.parameters(), lr=args.ttt_lr)
        kf_feats = feats[kf_indices]                                          # (K, P, 1024)
        ttt_gen = torch.Generator(device=args.device).manual_seed(0)
        for step in range(args.ttt_steps):
            idx = torch.randint(0, K, (min(args.ttt_batch_frames, K),), device=args.device, generator=ttt_gen)
            logits = head(kf_feats[idx], patch_h=patch_h, patch_w=patch_w)
            if logits.shape[-2:] != kf_labels.shape[-2:]:
                logits = F.interpolate(logits.unsqueeze(1),
                                        size=kf_labels.shape[-2:],
                                        mode="bilinear", align_corners=False).squeeze(1)
            loss = _focal_bce(logits, kf_labels[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()

        # Predict on all N frames in chunks
        with torch.no_grad():
            preds = []
            for s in range(0, N, 8):
                lg = head(feats[s : s + 8], patch_h=patch_h, patch_w=patch_w)
                if lg.shape[-2:] != (TARGET_H, TARGET_W):
                    lg = F.interpolate(lg.unsqueeze(1), size=(TARGET_H, TARGET_W),
                                        mode="bilinear", align_corners=False).squeeze(1)
                preds.append((torch.sigmoid(lg) > args.threshold).cpu().numpy().astype(np.uint8))
            masks_pi3 = np.concatenate(preds, axis=0)                        # (N, H_pi3, W_pi3)

        # Resize to origin + morph + blob filter
        out_masks = np.zeros((N, H_orig, W_orig), dtype=np.uint8)
        for i in range(N):
            m = cv2.resize(masks_pi3[i], (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
            if args.morph_close > 0 and m.any():
                kc = np.ones((args.morph_close, args.morph_close), np.uint8)
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
            if args.morph_dilate > 0 and m.any():
                kd = np.ones((args.morph_dilate, args.morph_dilate), np.uint8)
                m = cv2.dilate(m, kd)
            # Convex-hull fill: replace each large component with its filled hull,
            # so smooth-textured object bodies (white bus panel etc.) get covered.
            if args.hull_fill_min_area > 0 and m.any():
                n_lab2, lab2, stats2, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                m_filled = m.copy()
                for cc in range(1, n_lab2):
                    area = stats2[cc, cv2.CC_STAT_AREA]
                    if area < args.hull_fill_min_area:
                        continue
                    pts = np.column_stack(np.where(lab2 == cc))[:, [1, 0]]  # (x, y)
                    if len(pts) < 3:
                        continue
                    hull = cv2.convexHull(pts)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > area * args.hull_fill_max_ratio:
                        # Hull would balloon the component too much — skip.
                        continue
                    cv2.fillPoly(m_filled, [hull], 1)
                m = m_filled
            if args.min_blob_area > 0 and m.any():
                n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                keep = np.zeros_like(m)
                for cc in range(1, n_lab):
                    if stats[cc, cv2.CC_STAT_AREA] >= args.min_blob_area:
                        keep[lab == cc] = 1
                m = keep
            out_masks[i] = m
            cv2.imwrite(str(mask_dir / f"mask_{i:04d}.png"), m * 255)

        if args.save_overlays:
            ov_dir = out_dir / "overlay"
            ov_dir.mkdir(exist_ok=True)
            for i in range(N):
                ov = frames[i].copy()
                hot = out_masks[i] > 0
                if hot.any():
                    ov[hot] = (ov[hot] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
                cv2.imwrite(str(ov_dir / f"overlay_{i:04d}.png"),
                             cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

        elapsed = time.time() - t0
        rec = {
            "scene": scene_name,
            "camera": cam_name,
            "n_frames": int(N),
            "n_keyframes": int(K),
            "elapsed_s": round(elapsed, 2),
            "global_pos_frac": float(out_masks.mean() / 255 if out_masks.max() > 1 else out_masks.mean()),
        }
        summary.append(rec)

        eta = (time.time() - t_start) / (k + 1) * (len(pairs) - k - 1)
        LOGGER.info(
            "[%d/%d] %s done in %.1fs  pos=%.3f  ETA=%.1fmin",
            k + 1, len(pairs), key, elapsed, rec["global_pos_frac"], eta / 60,
        )

        with (args.out / "summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2)

    gsam.teardown()
    LOGGER.info("All done. Total: %.1fmin across %d pairs.", (time.time() - t_start) / 60, len(summary))


if __name__ == "__main__":
    main()
