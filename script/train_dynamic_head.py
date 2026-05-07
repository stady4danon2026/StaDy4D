#!/usr/bin/env python3
"""Train a small UNet student to mimic GroundedSAM's dynamic mask.

Input:  RGB image in [0, 1]
Target: binary mask (1 = dynamic from GroundedSAM)

Usage:
    python script/train_dynamic_head.py \
        --data-dir data/dynamic_train \
        --out checkpoints/student_unet.pt \
        --epochs 20 --batch-size 12 --img-size 320
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sigma.pipeline.motion.student_unet import StudentUNet  # noqa: E402


class DynamicMaskDataset(Dataset):
    """Reads pre-computed (rgb, mask) at fixed img_size from npz files.

    npz schema after precompute_train_rgb.py:
        rgb            uint8 (N, S, S, 3)
        mask_resized   uint8 (N, S, S)   — already at training resolution
    """

    def __init__(self, mask_files: list[Path], img_size: int = 320,
                 train: bool = True) -> None:
        self.files = mask_files
        self.img_size = img_size
        self.train = train
        self.samples: list[tuple[Path, int]] = []
        for f in mask_files:
            d = np.load(f)
            n = int(d["mask_resized"].shape[0]) if "mask_resized" in d.files \
                else int(d["masks"].shape[0])
            for j in range(n):
                self.samples.append((f, j))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        f, j = self.samples[i]
        d = np.load(f)
        if "rgb" in d.files and "mask_resized" in d.files:
            rgb = d["rgb"][j]               # (S, S, 3) uint8
            mask = d["mask_resized"][j]     # (S, S) uint8
            rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
            mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        else:
            raise RuntimeError(f"{f} missing precomputed rgb; run precompute_train_rgb.py first.")

        if self.train:
            if random.random() < 0.5:
                rgb_t = torch.flip(rgb_t, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])
            if random.random() < 0.5:
                rgb_t = torch.clamp(rgb_t * (0.85 + 0.3 * random.random()), 0, 1)

        return rgb_t, mask_t


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred_logits)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("StaDy4D"))
    p.add_argument("--out", type=Path, default=Path("checkpoints/student_unet.pt"))
    p.add_argument("--img-size", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(args.data_dir.glob("*.npz"))
    print(f"Found {len(files)} npz files")
    random.seed(0)
    random.shuffle(files)
    n_val = max(1, int(len(files) * args.val_fraction))
    val_files = files[:n_val]
    train_files = files[n_val:]
    train_ds = DynamicMaskDataset(train_files, args.img_size, train=True)
    val_ds = DynamicMaskDataset(val_files, args.img_size, train=False)
    print(f"Train: {len(train_ds)} samples ({len(train_files)} sequences)")
    print(f"Val:   {len(val_ds)} samples ({len(val_files)} sequences)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StudentUNet().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pos_weight = torch.tensor(15.0, device=device)  # masks are sparse

    best_val_iou = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.monotonic()
        losses = []
        for rgb, mask in train_loader:
            rgb, mask = rgb.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            logits = model(rgb)
            bce = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)
            dl = dice_loss(logits, mask)
            loss = bce + dl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        # Val
        model.eval()
        ious = []
        with torch.no_grad():
            for rgb, mask in val_loader:
                rgb, mask = rgb.to(device), mask.to(device)
                pred = (torch.sigmoid(model(rgb)) > 0.5).float()
                inter = (pred * mask).sum(dim=(2, 3))
                union = ((pred + mask) > 0).float().sum(dim=(2, 3))
                iou = (inter / (union + 1e-6)).cpu()
                ious.append(iou)
        ious_t = torch.cat(ious)
        val_iou = float(ious_t.mean())

        elapsed = time.monotonic() - t0
        train_loss = sum(losses) / len(losses)
        marker = ""
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), args.out)
            marker = " ★"
        print(f"Epoch {epoch+1:>3}/{args.epochs}  "
              f"loss={train_loss:.4f}  val_IoU={val_iou:.4f}  "
              f"({elapsed:.0f}s){marker}", flush=True)

    print(f"\nBest val IoU: {best_val_iou:.4f}  →  saved to {args.out}")


if __name__ == "__main__":
    main()
