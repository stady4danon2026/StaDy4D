#!/usr/bin/env python3
"""Offline pretraining for the Pi3 DynHead, distilled from GroundedSAM masks.

Per scene:
  1. Load N origin (dynamic) frames from rgb.mp4.
  2. Forward through frozen Pi3X with a hook that captures conf_decoder
     features (the same features the DynHead consumes at inference).
  3. Run GroundedSAM on the same frames → binary dynamic-mask labels.
  4. Backprop weighted BCE through the DynHead only.

Caches per-scene (features, labels) to disk on first pass to amortize the
Pi3 + GSAM cost across epochs.

Usage:
    python script/train_pi3_dyn_head.py \
        --root StaDy4D --duration short --split train \
        --out checkpoints/pi3_dyn_head.pt \
        --cache-dir cache/pi3_dyn \
        --epochs 5 --max-frames 50
"""
from __future__ import annotations

import argparse
import logging
import math
import random
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from sigma.pipeline.motion.pi3_dyn_features import Pi3FeatureExtractor
from sigma.pipeline.motion.pi3_dyn_head import DynHead

LOGGER = logging.getLogger("train_pi3_dyn_head")


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

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


def _enumerate_scene_cams(root: Path, duration: str, split: str) -> list[tuple[str, Path]]:
    """Yield (scene/cam_name, rgb_path) pairs for the dynamic pass."""
    base = root / duration / split
    out: list[tuple[str, Path]] = []
    for scene_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        dyn_dir = scene_dir / "dynamic"
        if not dyn_dir.is_dir():
            continue
        for cam_dir in sorted(p for p in dyn_dir.iterdir() if p.is_dir()):
            rgb = cam_dir / "rgb.mp4"
            if rgb.exists():
                out.append((f"{scene_dir.name}/{cam_dir.name}", rgb))
    return out


def _load_video(p: Path, max_frames: int | None) -> np.ndarray:
    r = imageio.get_reader(str(p))
    out = []
    for i, f in enumerate(r):
        if max_frames is not None and i >= max_frames:
            break
        out.append(np.asarray(f))
    r.close()
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Feature + label extraction (cached to disk)
# ---------------------------------------------------------------------------

def extract_one_scene(
    rgb_np: np.ndarray,
    pi3_model: torch.nn.Module,
    gsam_estimator,
    device: str,
    pixel_limit: int,
) -> dict | None:
    """Returns dict(features, labels, patch_h, patch_w) or None on failure."""
    N, H_orig, W_orig = rgb_np.shape[:3]
    TARGET_W, TARGET_H = _compute_target_size(H_orig, W_orig, pixel_limit)

    # Stack to Pi3 input.
    to_tensor = transforms.ToTensor()
    tensors = []
    for i in range(N):
        pil = Image.fromarray(rgb_np[i]).resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        tensors.append(to_tensor(pil))
    imgs = torch.stack(tensors, dim=0).unsqueeze(0).to(device)

    extractor = Pi3FeatureExtractor(pi3_model)
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    with extractor.capture():
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _ = pi3_model(imgs)
    feats = extractor.features.detach().cpu().float()                # (N, P, 1024)
    patch_h, patch_w = extractor.patch_hw

    # GSAM masks at (TARGET_H, TARGET_W).
    masks_orig = _run_gsam(gsam_estimator, rgb_np)
    masks_t = torch.from_numpy(masks_orig).float().unsqueeze(1)      # (N, 1, H_orig, W_orig)
    masks_resized = F.interpolate(masks_t, size=(TARGET_H, TARGET_W), mode="nearest").squeeze(1)
    labels = (masks_resized > 0.5).float()                           # (N, H_pi3, W_pi3)

    return {
        "features": feats,
        "labels": labels,
        "patch_h": patch_h,
        "patch_w": patch_w,
    }


def _run_gsam(estimator, rgb_np: np.ndarray) -> np.ndarray:
    """Run a GroundedSAMMotionEstimator-style detector across N frames.

    Returns (N, H, W) uint8 binary masks.  Uses _detect_objects_batch +
    SAM segmentation if available; otherwise falls back to per-frame.
    """
    from sigma.data.frame_record import FrameIORecord

    records: dict[int, FrameIORecord] = {}
    for i in range(rgb_np.shape[0]):
        records[i] = FrameIORecord(frame_idx=i, origin_image=rgb_np[i])

    out_results = estimator.process_batch(records)
    masks = []
    for i in range(rgb_np.shape[0]):
        m = out_results[i].data["motion"]["curr_mask"]
        masks.append(m.astype(np.uint8))
    return np.stack(masks, axis=0)


def cached_extract(
    cache_dir: Path,
    scene_id: str,
    rgb_path: Path,
    pi3_model,
    gsam_estimator,
    device: str,
    pixel_limit: int,
    max_frames: int | None,
) -> dict | None:
    safe = scene_id.replace("/", "__")
    cache_file = cache_dir / f"{safe}.pt"
    if cache_file.exists():
        return torch.load(cache_file, map_location="cpu")

    try:
        rgb_np = _load_video(rgb_path, max_frames)
    except Exception as e:
        LOGGER.warning("Skipping %s: video read failed (%s)", scene_id, e)
        return None
    if rgb_np.shape[0] < 4:
        LOGGER.warning("Skipping %s: too few frames (%d)", scene_id, rgb_np.shape[0])
        return None

    bundle = extract_one_scene(rgb_np, pi3_model, gsam_estimator, device, pixel_limit)
    if bundle is None:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, cache_file)
    return bundle


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    head: DynHead,
    bundles: Iterable[Path],
    epochs: int,
    lr: float,
    device: str,
    batch_frames: int,
    log_every: int,
) -> None:
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    bundle_paths = list(bundles)
    if not bundle_paths:
        raise RuntimeError("No bundles to train on (cache empty).")

    step = 0
    for epoch in range(epochs):
        random.shuffle(bundle_paths)
        for bp in bundle_paths:
            data = torch.load(bp, map_location=device)
            feats = data["features"].to(device)            # (N, P, 1024)
            labels = data["labels"].to(device)             # (N, H, W)
            patch_h, patch_w = int(data["patch_h"]), int(data["patch_w"])
            N, H, W = labels.shape

            pos_frac = float(labels.mean().item()) if labels.numel() else 0.0
            pos_frac = max(min(pos_frac, 0.95), 0.05)
            pw = torch.tensor([(1 - pos_frac) / pos_frac], device=device)

            # Several batch-frame steps per scene.
            n_inner = max(1, N // max(1, batch_frames))
            for _ in range(n_inner):
                idx = torch.randint(0, N, (min(batch_frames, N),), device=device)
                logits = head(feats[idx], patch_h=patch_h, patch_w=patch_w)
                if logits.shape[-2:] != (H, W):
                    logits = F.interpolate(logits.unsqueeze(1), size=(H, W),
                                           mode="bilinear", align_corners=False).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits, labels[idx], pos_weight=pw)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step += 1
                if step % log_every == 0:
                    LOGGER.info("ep %d step %d  loss=%.4f  pos_frac=%.3f", epoch, step, float(loss), pos_frac)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("StaDy4D"))
    p.add_argument("--duration", default="short")
    p.add_argument("--split", default="train")
    p.add_argument("--out", type=Path, required=True, help="Path to save the trained head .pt")
    p.add_argument("--cache-dir", type=Path, default=Path("cache/pi3_dyn"))
    p.add_argument("--pi3-checkpoint", default="yyfz233/Pi3X")
    p.add_argument("--device", default="cuda")
    p.add_argument("--pixel-limit", type=int, default=255000)
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-frames", type=int, default=8)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--max-scenes", type=int, default=None,
                   help="Optional cap on the number of (scene, camera) bundles to use.")
    p.add_argument("--extract-only", action="store_true",
                   help="Run the GSAM + Pi3 extraction phase and exit (cache only).")
    p.add_argument("--no-extract", action="store_true",
                   help="Skip extraction, train only on existing cached bundles.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_extract:
        from pi3.models.pi3x import Pi3X
        from sigma.pipeline.motion.grounded_sam import GroundedSAMMotionEstimator

        LOGGER.info("Loading Pi3X (frozen) from %s", args.pi3_checkpoint)
        pi3 = Pi3X.from_pretrained(args.pi3_checkpoint).to(args.device).eval()
        for prm in pi3.parameters():
            prm.requires_grad_(False)

        LOGGER.info("Loading GroundedSAM teacher")
        gsam = GroundedSAMMotionEstimator(device=args.device)
        gsam.setup()

        scene_cams = _enumerate_scene_cams(args.root, args.duration, args.split)
        if args.max_scenes:
            scene_cams = scene_cams[: args.max_scenes]
        LOGGER.info("Found %d (scene, cam) pairs", len(scene_cams))

        for k, (sid, rgb_path) in enumerate(scene_cams):
            LOGGER.info("[%d/%d] extracting %s", k + 1, len(scene_cams), sid)
            cached_extract(args.cache_dir, sid, rgb_path, pi3, gsam,
                            args.device, args.pixel_limit, args.max_frames)

        gsam.teardown()
        del pi3
        torch.cuda.empty_cache()

    if args.extract_only:
        LOGGER.info("Extraction complete; --extract-only set, exiting.")
        return

    bundles = sorted(args.cache_dir.glob("*.pt"))
    if args.max_scenes:
        bundles = bundles[: args.max_scenes]
    LOGGER.info("Training DynHead on %d cached bundles", len(bundles))

    head = DynHead().to(args.device)
    train(head, bundles, epochs=args.epochs, lr=args.lr, device=args.device,
          batch_frames=args.batch_frames, log_every=args.log_every)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    head.save(args.out, extra={"epochs": args.epochs, "lr": args.lr})
    LOGGER.info("Saved DynHead to %s", args.out)


if __name__ == "__main__":
    main()
