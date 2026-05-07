#!/usr/bin/env python3
"""SAM3 + head-pt full sweep over T03/T07/T10 1458 videos.

Per scene:
  1. Pi3 forward → conf_decoder features captured.
  2. GroundedSAM on first K=5 evenly-spaced keyframes.
  3. TTT 30 steps on those keyframes (head warm-started from pooled checkpoint).
  4. DynHead predict on all N frames → ``head_mask``.
  5. SAM3 forward on all N frames with text="car. truck. bus. ...".
     Filter SAM3 candidates: keep candidate if it overlaps the (dilated) head
     mask by ≥ ``--min-overlap`` pixels. Union kept candidates → final mask.
  6. Save final masks per frame.

Output layout:
    out/<scene>/<cam>/mask/mask_NNNN.png
    out/summary.json
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
LOGGER = logging.getLogger("sam3_headpt")


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
    pt = p * target + (1 - p) * (1 - target)
    a = alpha * target + (1 - alpha) * (1 - target)
    return a * (1 - pt).pow(gamma) * bce


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--towns", nargs="+", default=["T03", "T07", "T10"])
    p.add_argument("--cameras", nargs="+", default=[
        "cam_00_car_forward", "cam_01_car_forward", "cam_02_car_forward",
        "cam_03_drone_forward", "cam_04_orbit_building", "cam_05_orbit_crossroad",
        "cam_06_cctv", "cam_07_pedestrian", "cam_08_pedestrian",
    ])
    p.add_argument("--out", type=Path, default=Path("<DATA_ROOT>/sigma_outputs/sam3_headpt"))
    p.add_argument("--init-checkpoint", type=Path,
                   default=Path("checkpoints/pi3_dyn_head_pooled.pt"))
    p.add_argument("--pi3-checkpoint", default="yyfz233/Pi3X")
    p.add_argument("--sam3-checkpoint", default="facebook/sam3")
    p.add_argument("--keyframes", type=int, default=5)
    p.add_argument("--ttt-steps", type=int, default=30)
    p.add_argument("--ttt-lr", type=float, default=5e-4)
    p.add_argument("--ttt-batch-frames", type=int, default=4)
    p.add_argument("--head-threshold", type=float, default=0.45)
    p.add_argument("--head-morph-close", type=int, default=5)
    p.add_argument("--head-morph-dilate", type=int, default=3)
    p.add_argument("--head-min-blob", type=int, default=100)
    p.add_argument("--text", default="car. truck. bus. motorcycle. bicycle. pedestrian.")
    p.add_argument("--sam3-score-thr", type=float, default=0.3)
    p.add_argument("--min-overlap", type=int, default=20)
    p.add_argument("--head-dilate-for-filter", type=int, default=11)
    p.add_argument("--pixel-limit", type=int, default=255000)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Discover scenes
    pairs: list[tuple[str, Path]] = []
    for sd in sorted(args.root.iterdir()):
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

    # Load models once
    from pi3.models.pi3x import Pi3X
    from sigma.pipeline.motion.grounded_sam import GroundedSAMMotionEstimator
    from sigma.pipeline.motion.pi3_dyn_features import Pi3FeatureExtractor
    from sigma.pipeline.motion.pi3_dyn_head import DynHead
    from sigma.data.frame_record import FrameIORecord
    from transformers import Sam3Processor, Sam3Model

    LOGGER.info("Loading Pi3X (frozen)")
    pi3 = Pi3X.from_pretrained(args.pi3_checkpoint).to(args.device).eval()
    for prm in pi3.parameters(): prm.requires_grad_(False)

    LOGGER.info("Loading GroundedSAM teacher")
    gsam = GroundedSAMMotionEstimator(device=args.device)
    gsam.setup()

    LOGGER.info("Loading SAM3 (%s)", args.sam3_checkpoint)
    sam3_processor = Sam3Processor.from_pretrained(args.sam3_checkpoint)
    sam3_model = Sam3Model.from_pretrained(args.sam3_checkpoint).to(args.device).eval()
    for prm in sam3_model.parameters(): prm.requires_grad_(False)

    if args.init_checkpoint and args.init_checkpoint.exists():
        LOGGER.info("Warm start: %s", args.init_checkpoint)
        init_state = DynHead.load(args.init_checkpoint, map_location="cpu").state_dict()
    else:
        init_state = None

    summary = []
    t_start = time.time()
    to_t = transforms.ToTensor()
    amp = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
           else torch.float16)

    for k_idx, (key, rgb_path) in enumerate(pairs):
        scene_name, cam_name = key.split("/")
        out_dir = args.out / scene_name / cam_name
        mask_dir = out_dir / "mask"
        if mask_dir.exists() and len(list(mask_dir.glob("mask_*.png"))) > 0:
            LOGGER.info("[%d/%d] %s — skip (cached)", k_idx + 1, len(pairs), key)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        try:
            frames = _load_video(rgb_path, args.max_frames)
        except Exception as e:
            LOGGER.warning("[%d/%d] %s — load fail: %s", k_idx + 1, len(pairs), key, e)
            continue
        if frames.shape[0] < max(4, args.keyframes):
            continue
        N, H_orig, W_orig = frames.shape[:3]
        TARGET_W, TARGET_H = _compute_target_size(H_orig, W_orig, args.pixel_limit)

        t0 = time.time()
        # ---- Pi3 forward + capture conf features ----
        tensors = [to_t(Image.fromarray(frames[i]).resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS))
                   for i in range(N)]
        imgs = torch.stack(tensors, dim=0).unsqueeze(0).to(args.device)
        extractor = Pi3FeatureExtractor(pi3)
        with extractor.capture():
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=amp):
                    _ = pi3(imgs)
        feats = extractor.features.to(torch.float32)
        ph, pw = extractor.patch_hw

        # ---- GSAM on K evenly-spaced keyframes ----
        K = min(args.keyframes, N)
        kf_indices = sorted(set(int(round(i * (N - 1) / max(K - 1, 1))) for i in range(K)))
        K = len(kf_indices)
        kf_records = {j: FrameIORecord(frame_idx=j, origin_image=frames[orig_i])
                       for j, orig_i in enumerate(kf_indices)}
        gsam_out = gsam.process_batch(kf_records)
        kf_masks_orig = np.stack([gsam_out[j].data["motion"]["curr_mask"] for j in range(K)], axis=0)
        kf_masks_t = torch.from_numpy(kf_masks_orig).float().unsqueeze(1).to(args.device)
        kf_labels = F.interpolate(kf_masks_t, size=(TARGET_H, TARGET_W), mode="nearest").squeeze(1)

        # ---- TTT ----
        head = DynHead().to(args.device)
        if init_state is not None:
            head.load_state_dict(init_state)
        head.train()
        opt = torch.optim.Adam(head.parameters(), lr=args.ttt_lr)
        kf_feats = feats[kf_indices]
        rng = torch.Generator(device=args.device).manual_seed(0)
        for _ in range(args.ttt_steps):
            idx = torch.randint(0, K, (min(args.ttt_batch_frames, K),), device=args.device, generator=rng)
            logits = head(kf_feats[idx], patch_h=ph, patch_w=pw)
            if logits.shape[-2:] != kf_labels.shape[-2:]:
                logits = F.interpolate(logits.unsqueeze(1), size=kf_labels.shape[-2:],
                                        mode="bilinear", align_corners=False).squeeze(1)
            loss = _focal_bce(logits, kf_labels[idx]).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        head.eval()

        # ---- DynHead predict on all frames ----
        with torch.no_grad():
            head_preds = []
            for s in range(0, N, 8):
                lg = head(feats[s:s + 8], patch_h=ph, patch_w=pw)
                if lg.shape[-2:] != (TARGET_H, TARGET_W):
                    lg = F.interpolate(lg.unsqueeze(1), size=(TARGET_H, TARGET_W),
                                        mode="bilinear", align_corners=False).squeeze(1)
                head_preds.append((torch.sigmoid(lg) > args.head_threshold).cpu().numpy().astype(np.uint8))
            head_masks_pi3 = np.concatenate(head_preds, axis=0)

        # Resize head masks to original + morph + min_blob
        head_masks = np.zeros((N, H_orig, W_orig), dtype=np.uint8)
        for i in range(N):
            m = cv2.resize(head_masks_pi3[i], (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
            if args.head_morph_close > 0 and m.any():
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((args.head_morph_close, args.head_morph_close), np.uint8))
            if args.head_morph_dilate > 0 and m.any():
                m = cv2.dilate(m, np.ones((args.head_morph_dilate, args.head_morph_dilate), np.uint8))
            if args.head_min_blob > 0 and m.any():
                n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                keep = np.zeros_like(m)
                for cc in range(1, n_lab):
                    if stats[cc, cv2.CC_STAT_AREA] >= args.head_min_blob:
                        keep[lab == cc] = 1
                m = keep
            head_masks[i] = m

        # ---- SAM3 + head-pt filter per frame ----
        out_masks = np.zeros((N, H_orig, W_orig), dtype=np.uint8)
        kept_total = 0; cand_total = 0
        for i in range(N):
            pil = Image.fromarray(frames[i])
            inp = sam3_processor(images=pil, text=args.text, return_tensors="pt").to(args.device)
            with torch.no_grad():
                so = sam3_model(**inp)
            scores = so.pred_logits[0].sigmoid().cpu().numpy()
            mks_lr = (so.pred_masks[0].sigmoid() > 0.5).cpu().numpy()
            keep_mask = scores > args.sam3_score_thr
            cand_total += int(keep_mask.sum())
            if keep_mask.sum() == 0:
                continue
            mks_lr = mks_lr[keep_mask]
            # Resize each candidate to (H_orig, W_orig)
            cand = np.zeros((mks_lr.shape[0], H_orig, W_orig), dtype=bool)
            for c in range(mks_lr.shape[0]):
                cand[c] = cv2.resize(mks_lr[c].astype(np.uint8), (W_orig, H_orig),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)

            # Filter by head overlap (dilated)
            head_dil = head_masks[i]
            if args.head_dilate_for_filter > 0:
                head_dil = cv2.dilate(head_dil, np.ones((args.head_dilate_for_filter,
                                                         args.head_dilate_for_filter), np.uint8))
            head_bool = head_dil.astype(bool)
            overlap = (cand & head_bool[None]).sum(axis=(1, 2))
            keep2 = overlap >= args.min_overlap
            kept_total += int(keep2.sum())
            if keep2.sum() == 0:
                continue
            final = cand[keep2].any(axis=0).astype(np.uint8)
            out_masks[i] = final
            cv2.imwrite(str(mask_dir / f"mask_{i:04d}.png"), final * 255)

        # Save zero-frames too (for completeness)
        for i in range(N):
            f = mask_dir / f"mask_{i:04d}.png"
            if not f.exists():
                cv2.imwrite(str(f), out_masks[i] * 255)

        elapsed = time.time() - t0
        rec = {
            "scene": scene_name, "camera": cam_name, "n_frames": int(N),
            "elapsed_s": round(elapsed, 2),
            "global_pos_frac": float(out_masks.mean() / 255 if out_masks.max() > 1 else out_masks.mean()),
            "candidates_total": int(cand_total), "kept_total": int(kept_total),
        }
        summary.append(rec)
        eta = (time.time() - t_start) / (k_idx + 1) * (len(pairs) - k_idx - 1)
        LOGGER.info(
            "[%d/%d] %s done in %.1fs  pos=%.4f  kept=%d/%d  ETA=%.1fmin",
            k_idx + 1, len(pairs), key, elapsed, rec["global_pos_frac"],
            kept_total, cand_total, eta / 60,
        )
        with (args.out / "summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2)

    gsam.teardown()
    LOGGER.info("All done. Total: %.1fmin across %d scenes.",
                (time.time() - t_start) / 60, len(summary))


if __name__ == "__main__":
    main()
