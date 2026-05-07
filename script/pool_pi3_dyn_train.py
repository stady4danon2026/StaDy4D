#!/usr/bin/env python3
"""Pooled photometric TTT training — Option A: cross-scene generalization.

Phase 1 (extract):
  For each (scene, camera) in the training pool:
    - Pi3 forward (frozen) → conf_decoder features captured.
    - Photometric pseudo-labels + trust mask from Pi3 depth/poses.
    - Cache (features, labels, trust, patch_h, patch_w) to disk.

Phase 2 (train):
  Train a single DynHead on the pooled bundles with masked weighted BCE.
  This produces ONE head adapted to the photometric teacher's signal across
  many scenes — no GSAM / oracle anywhere.

Output: a checkpoint usable by the existing Pi3DynMotionEstimator with
``ttt_steps=0`` (inference-only mode).

Usage:
    python script/pool_pi3_dyn_train.py \
        --root StaDy4D --duration short --split test \
        --towns T03 T07 T10 \
        --cameras cam_00_car_forward cam_03_drone_forward cam_06_cctv cam_07_pedestrian \
        --scenes-per-town 10 \
        --cache-dir cache/pi3_dyn_pool \
        --out checkpoints/pi3_dyn_head_pooled.pt \
        --epochs 8 --batch-frames 8
"""
from __future__ import annotations

import argparse
import logging
import math
import random
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

LOGGER = logging.getLogger("pool_pi3_dyn_train")


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


def _enumerate_pairs(root: Path, duration: str, split: str, towns: list[str],
                     cameras: list[str], scenes_per_town: int, seed: int) -> list[tuple[str, Path]]:
    base = root / duration / split
    rng = random.Random(seed)
    pairs: list[tuple[str, Path]] = []
    for town in towns:
        scenes = sorted(p for p in base.iterdir() if p.is_dir() and town in p.name)
        rng.shuffle(scenes)
        scenes = scenes[:scenes_per_town]
        for sd in scenes:
            for cam in cameras:
                rgb = sd / "dynamic" / cam / "rgb.mp4"
                if rgb.exists():
                    pairs.append((f"{sd.name}/{cam}", rgb))
    return pairs


def _run_gsam(estimator, rgb_np: np.ndarray) -> np.ndarray:
    """GroundedSAM motion estimator → (N, H_orig, W_orig) uint8 mask."""
    from sigma.data.frame_record import FrameIORecord
    records = {i: FrameIORecord(frame_idx=i, origin_image=rgb_np[i])
               for i in range(rgb_np.shape[0])}
    out = estimator.process_batch(records)
    return np.stack([out[i].data["motion"]["curr_mask"] for i in range(rgb_np.shape[0])], axis=0)


def _extract_one(rgb_np: np.ndarray, pi3_model, device, pixel_limit, photo_args,
                 gsam_estimator=None):
    from sigma.pipeline.motion.pi3_dyn_features import Pi3FeatureExtractor
    from sigma.pipeline.motion.pi3_dyn_ttt import compute_photometric_labels
    from sigma.pipeline.motion.pi3_dyn import _compute_target_size as _csz  # noqa: F401

    N, H_orig, W_orig = rgb_np.shape[:3]
    TARGET_W, TARGET_H = _compute_target_size(H_orig, W_orig, pixel_limit)
    to_tensor = transforms.ToTensor()
    tensors = [to_tensor(Image.fromarray(rgb_np[i]).resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS))
               for i in range(N)]
    imgs = torch.stack(tensors, dim=0).unsqueeze(0).to(device)

    extractor = Pi3FeatureExtractor(pi3_model)
    amp = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
           else torch.float16)
    with extractor.capture():
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=amp):
                out = pi3_model(imgs)
    feats = extractor.features.detach().cpu().float()
    patch_h, patch_w = extractor.patch_hw

    local_pts = out["local_points"][0].float()
    c2w = out["camera_poses"][0].float()
    depth = local_pts[..., 2].clamp_min(0.0)
    rgb_pi3 = imgs[0].to(local_pts.device, dtype=torch.float32)

    # Derive intrinsics (per frame median)
    K = torch.zeros((N, 3, 3), device=local_pts.device, dtype=torch.float32)
    cx, cy = TARGET_W / 2.0, TARGET_H / 2.0
    u = torch.arange(TARGET_W, device=local_pts.device, dtype=torch.float32).view(1, 1, TARGET_W).expand(N, TARGET_H, TARGET_W)
    v = torch.arange(TARGET_H, device=local_pts.device, dtype=torch.float32).view(1, TARGET_H, 1).expand(N, TARGET_H, TARGET_W)
    x, y, z = local_pts[..., 0], local_pts[..., 1], local_pts[..., 2]
    valid_z = z > 0.01
    mfx = valid_z & ((u - cx).abs() > TARGET_W * 0.1)
    mfy = valid_z & ((v - cy).abs() > TARGET_H * 0.1)
    dfx = TARGET_W / (2 * math.tan(math.radians(60)))
    dfy = TARGET_H / (2 * math.tan(math.radians(60)))
    for i in range(N):
        try:
            fx = float(torch.median((u[i][mfx[i]] - cx) / (x[i][mfx[i]] / z[i][mfx[i]])).item()) if mfx[i].sum() > 10 else dfx
            fy = float(torch.median((v[i][mfy[i]] - cy) / (y[i][mfy[i]] / z[i][mfy[i]])).item()) if mfy[i].sum() > 10 else dfy
        except Exception:
            fx, fy = dfx, dfy
        K[i, 0, 0] = float(np.clip(fx, TARGET_W * 0.2, TARGET_W * 5.0))
        K[i, 1, 1] = float(np.clip(fy, TARGET_H * 0.2, TARGET_H * 5.0))
        K[i, 0, 2] = cx
        K[i, 1, 2] = cy
        K[i, 2, 2] = 1.0

    with torch.no_grad():
        labels = compute_photometric_labels(
            rgb=rgb_pi3, depth=depth, K=K, c2w=c2w,
            residual_threshold=photo_args["threshold"],
            consensus=photo_args["consensus"],
            neighbors=tuple(photo_args["neighbors"]),
        )

    # Optional: GSAM masks at Pi3 input resolution.
    gsam_pi3 = None
    if gsam_estimator is not None:
        try:
            gsam_orig = _run_gsam(gsam_estimator, rgb_np)              # (N, H_orig, W_orig)
            gsam_t = torch.from_numpy(gsam_orig).float().unsqueeze(1)  # (N, 1, H_orig, W_orig)
            gsam_resized = F.interpolate(
                gsam_t, size=(TARGET_H, TARGET_W), mode="nearest"
            ).squeeze(1)
            gsam_pi3 = (gsam_resized > 0.5).to(torch.uint8).cpu()
        except Exception as e:
            LOGGER.warning("GSAM failed: %s — bundle saved without GSAM labels", e)


    # Compact storage: fp16 features, uint8 labels.
    bundle = {
        "features": feats.to(torch.float16),
        "labels": (labels.pseudo > 0.5).to(torch.uint8).cpu(),    # photometric pseudo
        "trust": (labels.trust > 0.5).to(torch.uint8).cpu(),
        "patch_h": patch_h,
        "patch_w": patch_w,
    }
    if gsam_pi3 is not None:
        bundle["gsam"] = gsam_pi3
    return bundle


def cmd_extract(args, pairs, pi3_model, device, gsam_estimator=None):
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    photo = {"threshold": args.photometric_threshold,
             "consensus": args.photometric_consensus,
             "neighbors": args.photometric_neighbors}
    for k, (key, rgb_path) in enumerate(pairs):
        safe = key.replace("/", "__")
        cache_file = args.cache_dir / f"{safe}.pt"

        # If a cache file exists but is missing GSAM and we have a GSAM
        # estimator now, augment it in place rather than re-running Pi3.
        if cache_file.exists():
            existing = torch.load(cache_file, map_location="cpu")
            if gsam_estimator is None or "gsam" in existing:
                LOGGER.info("[%d/%d] %s — cached, skip", k + 1, len(pairs), key)
                continue
            try:
                frames = _load_video(rgb_path, args.max_frames)
            except Exception as e:
                LOGGER.warning("[%d/%d] %s — load failed: %s", k + 1, len(pairs), key, e)
                continue
            try:
                gsam_orig = _run_gsam(gsam_estimator, frames)
                _, H_orig, W_orig = frames.shape[:3]
                TARGET_H, TARGET_W = existing["labels"].shape[-2:]
                gsam_t = torch.from_numpy(gsam_orig).float().unsqueeze(1)
                gsam_resized = F.interpolate(
                    gsam_t, size=(TARGET_H, TARGET_W), mode="nearest"
                ).squeeze(1)
                existing["gsam"] = (gsam_resized > 0.5).to(torch.uint8)
                torch.save(existing, cache_file)
                gsam_pos = float(existing["gsam"].float().mean().item())
                LOGGER.info("[%d/%d] %s — added GSAM (pos=%.3f)", k + 1, len(pairs), key, gsam_pos)
            except Exception as e:
                LOGGER.warning("[%d/%d] %s — GSAM augment failed: %s", k + 1, len(pairs), key, e)
            continue

        try:
            frames = _load_video(rgb_path, args.max_frames)
        except Exception as e:
            LOGGER.warning("[%d/%d] %s — load failed: %s", k + 1, len(pairs), key, e)
            continue
        if frames.shape[0] < 4:
            LOGGER.warning("[%d/%d] %s — too few frames", k + 1, len(pairs), key)
            continue
        t0 = time.time()
        bundle = _extract_one(frames, pi3_model, device, args.pixel_limit, photo,
                                gsam_estimator=gsam_estimator)
        torch.save(bundle, cache_file)
        pos = float(bundle["labels"].float().mean().item())
        gpos = float(bundle["gsam"].float().mean().item()) if "gsam" in bundle else None
        LOGGER.info("[%d/%d] %s extracted in %.1fs  photo=%.3f  gsam=%s",
                     k + 1, len(pairs), key, time.time() - t0, pos,
                     f"{gpos:.3f}" if gpos is not None else "n/a")


def _focal_bce(logits: torch.Tensor, target: torch.Tensor,
               alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Per-pixel focal BCE (Lin et al. 2017). Reduction='none'."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return alpha_t * (1 - p_t).pow(gamma) * bce


def cmd_train(args):
    from sigma.pipeline.motion.pi3_dyn_head import DynHead

    bundles = sorted(args.cache_dir.glob("*.pt"))
    if not bundles:
        raise RuntimeError(f"No bundles in {args.cache_dir}")
    LOGGER.info("Training on %d pooled bundles (focal alpha=%.2f gamma=%.1f, labels=%s)",
                 len(bundles), args.focal_alpha, args.focal_gamma, args.label_source)

    head = DynHead().to(args.device)

    # Bias-init the last conv to log(prior/(1-prior)) so the head doesn't
    # have to spend the first epochs un-fitting a uniform prior.
    prior = max(min(args.bias_prior, 0.99), 1e-4)
    bias_init = math.log(prior / (1 - prior))
    last_conv = head.head.output_block[0][-1]   # Conv2d(32, 1, 1, 1)
    last_conv.bias.data.fill_(bias_init)
    LOGGER.info("Bias-init last conv to logit(prior=%.4f) = %.3f", prior, bias_init)

    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    step = 0
    for epoch in range(args.epochs):
        random.shuffle(bundles)
        for bp in bundles:
            data = torch.load(bp, map_location=args.device)
            feats = data["features"].to(device=args.device, dtype=torch.float32)
            photo = data["labels"].to(device=args.device, dtype=torch.float32)
            trust_photo = data["trust"].to(device=args.device, dtype=torch.float32)
            gsam = data["gsam"].to(device=args.device, dtype=torch.float32) if "gsam" in data else None
            patch_h, patch_w = int(data["patch_h"]), int(data["patch_w"])
            N, H, W = photo.shape

            depthdiff = (data["depthdiff"].to(device=args.device, dtype=torch.float32)
                         if "depthdiff" in data else None)
            gt_clean = (data["gt_clean"].to(device=args.device, dtype=torch.float32)
                        if "gt_clean" in data else None)

            if args.label_source == "gsam":
                if gsam is None:
                    continue
                labels = gsam
                trust = torch.clamp(trust_photo + gsam, 0.0, 1.0)
            elif args.label_source == "depthdiff":
                if depthdiff is None:
                    continue
                labels = depthdiff
                trust = torch.ones_like(labels)
            elif args.label_source == "gt_clean":
                if gt_clean is None:
                    continue
                labels = gt_clean
                trust = torch.ones_like(labels)
            elif args.label_source == "union":
                if gsam is None:
                    labels = photo
                else:
                    labels = torch.clamp(photo + gsam, 0.0, 1.0)
                trust = torch.clamp(trust_photo + (gsam if gsam is not None else 0), 0.0, 1.0)
            else:  # photometric
                labels = photo
                trust = trust_photo

            if labels.sum() < 10:
                continue
            # Skip bundles where GT covers most of the image — usually
            # camera-pose mismatch in static-vs-dynamic depth pass.
            if args.label_source == "depthdiff" and labels.float().mean().item() > 0.30:
                continue

            pos_frac = float((labels * (trust > 0).float()).sum().item()) / max(float((trust > 0).float().sum().item()), 1.0)

            n_inner = max(1, N // max(1, args.batch_frames))
            for _ in range(n_inner):
                idx = torch.randint(0, N, (min(args.batch_frames, N),), device=args.device)
                logits = head(feats[idx], patch_h=patch_h, patch_w=patch_w)
                if logits.shape[-2:] != (H, W):
                    logits = F.interpolate(logits.unsqueeze(1), size=(H, W),
                                           mode="bilinear", align_corners=False).squeeze(1)
                fl = _focal_bce(logits, labels[idx],
                                 alpha=args.focal_alpha, gamma=args.focal_gamma)
                w = trust[idx]
                loss = (fl * w).sum() / w.sum().clamp_min(1.0)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step += 1
                if step % args.log_every == 0:
                    with torch.no_grad():
                        pmean = torch.sigmoid(logits).mean().item()
                    LOGGER.info("ep %d step %d  loss=%.4f  pos_frac=%.3f  pred_mean=%.3f",
                                 epoch, step, float(loss), pos_frac, pmean)

    head.eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    head.save(args.out, extra={"epochs": args.epochs, "n_bundles": len(bundles),
                                 "lr": args.lr, "loss": "focal",
                                 "focal_alpha": args.focal_alpha,
                                 "focal_gamma": args.focal_gamma})
    LOGGER.info("Saved pooled head → %s", args.out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("StaDy4D"))
    p.add_argument("--duration", default="short")
    p.add_argument("--split", default="test")
    p.add_argument("--towns", nargs="+", default=["T03", "T07", "T10"])
    p.add_argument("--cameras", nargs="+", default=[
        "cam_00_car_forward", "cam_03_drone_forward",
        "cam_06_cctv", "cam_07_pedestrian",
    ])
    p.add_argument("--scenes-per-town", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", type=Path, default=Path("cache/pi3_dyn_pool"))
    p.add_argument("--out", type=Path, default=Path("checkpoints/pi3_dyn_head_pooled.pt"))
    p.add_argument("--pi3-checkpoint", default="yyfz233/Pi3X")
    p.add_argument("--device", default="cuda")
    p.add_argument("--pixel-limit", type=int, default=255000)
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-frames", type=int, default=8)
    p.add_argument("--photometric-threshold", type=float, default=0.10)
    p.add_argument("--photometric-consensus", type=int, default=3)
    p.add_argument("--photometric-neighbors", nargs="+", type=int, default=[-3, -1, 1, 3])
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--focal-alpha", type=float, default=0.25)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--bias-prior", type=float, default=0.02,
                   help="Init the last conv bias to logit(prior). 0.02 ≈ avg pos frac.")
    p.add_argument("--use-gsam", action="store_true",
                   help="Run GroundedSAM during extract; store dense masks in bundles.")
    p.add_argument("--label-source", choices=["photometric", "gsam", "union", "depthdiff", "gt_clean"],
                   default="photometric",
                   help="Which label channel to optimize during training.")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    pairs = _enumerate_pairs(args.root, args.duration, args.split, args.towns,
                              args.cameras, args.scenes_per_town, args.seed)
    LOGGER.info("Pool size: %d videos (towns=%s, cams=%d, scenes/town=%d)",
                len(pairs), args.towns, len(args.cameras), args.scenes_per_town)

    if not args.skip_extract:
        from pi3.models.pi3x import Pi3X
        LOGGER.info("Loading Pi3X from %s", args.pi3_checkpoint)
        pi3 = Pi3X.from_pretrained(args.pi3_checkpoint).to(args.device).eval()
        for prm in pi3.parameters():
            prm.requires_grad_(False)

        gsam = None
        if args.use_gsam:
            from sigma.pipeline.motion.grounded_sam import GroundedSAMMotionEstimator
            LOGGER.info("Loading GroundedSAM teacher")
            gsam = GroundedSAMMotionEstimator(device=args.device)
            gsam.setup()

        cmd_extract(args, pairs, pi3, args.device, gsam_estimator=gsam)

        if gsam is not None:
            gsam.teardown()
        del pi3
        torch.cuda.empty_cache()

    if not args.skip_train:
        cmd_train(args)


if __name__ == "__main__":
    main()
