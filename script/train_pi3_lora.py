#!/usr/bin/env python3
"""LoRA fine-tune Pi3X to predict static-only depth from origin (dynamic) RGB.

Supervises ``local_points[..., 2]`` (Pi3X's predicted depth in camera frame)
with the dataset's *static-pass* depth GT, on non-sky pixels only
(0.1 < gt < 999).  Dynamic-region pixels in the GT show the road/buildings
behind the actor — Pi3X learns to predict that even with cars in input.

Targets a small set of layers via PEFT LoRA (rank 8 default):
- BlockRope.attn.qkv (decoder)
- BlockRope.attn.proj
- BlockRope.mlp.fc1 / fc2
- point_decoder transformer blocks
- point_head conv heads (optional, off by default)

Usage:
    python script/train_pi3_lora.py \
        --data-dir data/lora_train \
        --out checkpoints/pi3_lora_static_depth.pt \
        --rank 8 --epochs 5 --batch-size 1 --window 5
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class LoRAStaDy4DDataset(Dataset):
    """Yields (rgb_window, depth_gt_window, K_window, c2w_window) clips."""

    def __init__(self, files: list[Path], window: int = 5) -> None:
        self.files = files
        self.window = window
        self.index: list[tuple[Path, int]] = []
        for f in files:
            try:
                n = int(np.load(f)["depth"].shape[0])
            except Exception:
                continue
            if n < window:
                continue
            for start in range(0, n - window + 1):
                self.index.append((f, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        f, start = self.index[i]
        d = np.load(f)
        end = start + self.window
        rgb = d["rgb"][start:end].astype(np.float32) / 255.0       # (W, H, W, 3) -> need permute
        depth = d["depth"][start:end].astype(np.float32)
        K = d["K"][start:end].astype(np.float32)
        c2w = d["c2w"][start:end].astype(np.float32)
        # (N, H, W, 3) -> (N, 3, H, W)
        rgb_t = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()
        depth_t = torch.from_numpy(depth)
        K_t = torch.from_numpy(K)
        c2w_t = torch.from_numpy(c2w)
        return rgb_t, depth_t, K_t, c2w_t


def attach_lora(model: nn.Module, rank: int, target_keywords: list[str],
                alpha: float = 16.0, dropout: float = 0.05) -> int:
    """Attach LoRA adapters to nn.Linear layers whose qualified name matches
    any keyword. Returns count of adapted layers.

    IMPORTANT: caller is responsible for freezing the rest of the model.
    """
    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, r: int, a: float, drop: float):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.r = r
            self.scale = a / r
            device = base.weight.device
            dtype = base.weight.dtype
            self.lora_A = nn.Linear(base.in_features, r, bias=False,
                                     device=device, dtype=dtype)
            self.lora_B = nn.Linear(r, base.out_features, bias=False,
                                     device=device, dtype=dtype)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.base(x) + self.scale * self.lora_B(self.lora_A(self.drop(x)))

    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(kw in name for kw in target_keywords):
            continue
        # Walk back to parent and replace
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(module, rank, alpha, dropout))
        count += 1
    return count


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("checkpoints/pi3_lora_static_depth.pt"))
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--max-train-clips", type=int, default=2000,
                   help="cap clips per epoch (smaller = more frequent eval)")
    p.add_argument("--depth-clip-min", type=float, default=0.1)
    p.add_argument("--depth-clip-max", type=float, default=999.0)
    # Which layers to LoRA — default is decoder + point head
    p.add_argument("--target-keywords", nargs="+",
                   default=["decoder", "point_decoder"])
    p.add_argument("--last-n-decoder-blocks", type=int, default=8,
                   help="apply LoRA only to the last-N BlockRope decoder blocks (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default="yyfz233/Pi3X")
    # ---- inline real-world eval ----
    p.add_argument("--eval-every-iters", type=int, default=500,
                   help="every N optimizer steps, run a quick depth-abs_rel eval on TUM")
    p.add_argument("--tum-eval-seq", default="data/realworld/tum_dynamics/rgbd_dataset_freiburg3_walking_xyz",
                   help="TUM sequence used for inline depth eval (RGB+depth GT)")
    p.add_argument("--tum-eval-frames", type=int, default=12,
                   help="frames sampled from the TUM sequence for inline eval")
    p.add_argument("--early-stop-patience", type=int, default=8,
                   help="stop if real-world abs_rel hasn't improved for this many evals")
    p.add_argument("--tum-train-share", type=float, default=0.5,
                   help="fraction of training clips drawn from TUM (oversampled if needed)")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    files = sorted(args.data_dir.glob("*.npz"))
    print(f"Found {len(files)} npz files")
    random.shuffle(files)
    n_val = max(1, int(len(files) * args.val_fraction))
    val_files = files[:n_val]; train_files = files[n_val:]

    train_ds = LoRAStaDy4DDataset(train_files, window=args.window)
    val_ds = LoRAStaDy4DDataset(val_files, window=args.window)
    print(f"Train clips: {len(train_ds)} ({len(train_files)} sequences)")
    print(f"Val   clips: {len(val_ds)} ({len(val_files)} sequences)")

    if len(train_ds) > args.max_train_clips:
        # Subsample for tractability while keeping coverage diverse.
        # IMPORTANT: oversample real-domain TUM clips so the gradient sees
        # them frequently (otherwise CARLA's 80%+ share dominates and
        # pulls Pi3X toward synthetic depth statistics).
        is_tum = lambda f: "TUM__" in f.name
        tum_idx = [s for s in train_ds.index if is_tum(s[0])]
        carla_idx = [s for s in train_ds.index if not is_tum(s[0])]
        tum_share = float(getattr(args, "tum_train_share", 0.5))
        n_tum = int(args.max_train_clips * tum_share)
        n_carla = args.max_train_clips - n_tum
        # Oversample TUM if not enough unique clips
        chosen_tum = (tum_idx * (n_tum // max(len(tum_idx), 1) + 1))[:n_tum] \
                     if tum_idx else []
        chosen_carla = random.sample(carla_idx, min(n_carla, len(carla_idx))) \
                        if carla_idx else []
        train_ds.index = chosen_tum + chosen_carla
        random.shuffle(train_ds.index)
        print(f"Subsampled train clips to {len(train_ds)} "
              f"({len(chosen_tum)} TUM + {len(chosen_carla)} CARLA, "
              f"target tum_share={tum_share*100:.0f}%)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Pi3X from {args.checkpoint}...")
    from pi3.models.pi3x import Pi3X
    # Keep model fp32; use autocast(bf16) during forward.  Mixed-dtype
    # model conversions caused brittle ConvHead crashes; this is cleaner.
    model = Pi3X.from_pretrained(args.checkpoint).to(device).eval()
    print("Model dtype: fp32 (autocast bf16 used at fwd time)")

    # Restrict decoder targets to last N blocks
    targets = list(args.target_keywords)
    if args.last_n_decoder_blocks > 0:
        # Blocks are indexed: model.decoder.<i>...
        # Replace generic "decoder" with explicit list
        targets = [k for k in targets if k != "decoder"]
        n_blocks = len(model.decoder)
        first = max(0, n_blocks - args.last_n_decoder_blocks)
        targets += [f"decoder.{i}." for i in range(first, n_blocks)]
        print(f"LoRA on decoder blocks [{first}..{n_blocks-1}] + extra: {targets}")

    # CRITICAL: freeze everything FIRST, then attach LoRA — otherwise base
    # parameters of non-LoRA modules stay trainable.
    for p in model.parameters():
        p.requires_grad_(False)

    n_lora = attach_lora(model, args.rank, targets, alpha=args.alpha)
    # Belt-and-suspenders: re-move whole model to ensure LoRA tensors land on the
    # same device as the base (covers any Python/dispatch quirk).
    model = model.to(device)

    # ---- Strip unused output heads ---------------------------------------
    # Our depth-supervision loss only needs local_points (from point_head).
    # Skip camera_head, metric_head, conf_head and their decoders to halve
    # forward memory and avoid OOM in their dense ConvHeads.
    import types
    import torch.nn.functional as F_
    def _slim_forward_head(self, hidden, pos, B, N, H, W, patch_h, patch_w):
        ret_point = self.point_decoder(hidden, xpos=pos)
        # Original Pi3X forces fp32 through point_head — preserve that since
        # ConvHead's UV table uses x.dtype to pick dtype.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            xy, z = self.point_head(ret_point[:, self.patch_start_idx:].float(),
                                      patch_h=patch_h, patch_w=patch_w)
            xy = xy.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
            z = z.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
            z = torch.exp(z.clamp(max=15.0))
            local_points = torch.cat([xy * z, z], dim=-1)
        return {"local_points": local_points}
    model.forward_head = types.MethodType(_slim_forward_head, model)
    print("Stripped forward_head to point_head only (skip camera/metric/conf)")

    # ---- Gradient checkpointing on the 36-layer decoder trunk ------------
    from torch.utils.checkpoint import checkpoint
    if hasattr(model, "decoder") and isinstance(model.decoder, nn.ModuleList):
        n_ckpt = 0
        for blk in model.decoder:
            orig_forward = blk.forward
            def make_ckpt_fwd(orig, blk_ref):
                def fwd(*a, **kw):
                    if blk_ref.training:
                        return checkpoint(orig, *a, **kw, use_reentrant=True)
                    return orig(*a, **kw)
                return fwd
            blk.forward = make_ckpt_fwd(orig_forward, blk)
            n_ckpt += 1
        print(f"Gradient checkpointing enabled on {n_ckpt} decoder blocks")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Attached LoRA to {n_lora} linear layers")
    print(f"Trainable params: {trainable_params/1e6:.2f}M / total {total_params/1e6:.2f}M "
          f"({100*trainable_params/total_params:.2f}%)")
    if trainable_params / total_params > 0.05:
        print(f"WARNING: trainable fraction is suspiciously high "
              f"({100*trainable_params/total_params:.1f}% > 5%) — check LoRA wiring")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-3,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    dtype = (torch.bfloat16 if device.startswith("cuda")
             and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)

    # ---- Build TUM eval batch once (used for inline real-world eval) ----
    tum_batch = None
    if args.eval_every_iters > 0:
        try:
            import imageio.v2 as imageio
            tum_path = Path(args.tum_eval_seq)
            rgb_lines = (tum_path / "rgb.txt").read_text().splitlines()
            depth_lines = (tum_path / "depth.txt").read_text().splitlines()
            def parse(lines):
                rows = []
                for ln in lines:
                    if not ln.strip() or ln.startswith("#"):
                        continue
                    parts = ln.split()
                    rows.append((float(parts[0]), parts[1]))
                return rows
            rgb_rows = parse(rgb_lines)[::5][: args.tum_eval_frames]
            depth_rows = parse(depth_lines)
            depth_t = np.array([r[0] for r in depth_rows])

            rgbs, gt_depths = [], []
            import math, cv2
            def pi3_size(H, W, pl=255000):
                s = math.sqrt(pl / max(H * W, 1))
                Wf, Hf = W * s, H * s
                k, m = round(Wf / 14), round(Hf / 14)
                while (k * 14) * (m * 14) > pl:
                    if k / m > Wf / Hf: k -= 1
                    else: m -= 1
                return max(1, k) * 14, max(1, m) * 14
            for t_rgb, rgb_path in rgb_rows:
                rgb = imageio.imread(tum_path / rgb_path)
                idx = int(np.argmin(np.abs(depth_t - t_rgb)))
                d_raw = imageio.imread(tum_path / depth_rows[idx][1])
                d = d_raw.astype(np.float32) / 5000.0
                d[d_raw == 0] = np.nan
                H, W = rgb.shape[:2]
                TW, TH = pi3_size(H, W)
                rgbs.append(cv2.resize(rgb, (TW, TH), interpolation=cv2.INTER_AREA))
                gt_depths.append(cv2.resize(d, (TW, TH), interpolation=cv2.INTER_NEAREST))
            tum_rgb = torch.from_numpy(np.stack(rgbs)).float().permute(0, 3, 1, 2) / 255.0
            tum_gt = torch.from_numpy(np.stack(gt_depths))
            tum_batch = (tum_rgb.unsqueeze(0).to(device), tum_gt.to(device))
            print(f"TUM eval batch ready: {tum_rgb.shape} (RGB) | {tum_gt.shape} (depth)",
                  flush=True)
        except Exception as e:
            print(f"WARNING: could not load TUM eval batch ({e}); inline eval disabled")
            tum_batch = None

    @torch.no_grad()
    def quick_tum_eval() -> float:
        if tum_batch is None:
            return float("nan")
        model.eval()
        rgb, gt = tum_batch                                 # rgb: (1,N,3,H,W), gt: (N,H,W)
        with torch.amp.autocast("cuda", dtype=dtype, enabled=device.startswith("cuda")):
            r = model(rgb)
            pd = r["local_points"][..., 2]                    # (B, N, H_pi, W_pi)
        # Drop batch dim so pd is (N, H_pi, W_pi)
        if pd.dim() == 4 and pd.shape[0] == 1:
            pd = pd[0]
        # Resize to GT shape
        if pd.shape[-2:] != gt.shape[-2:]:
            H, W = gt.shape[-2:]
            pd = F.interpolate(pd.unsqueeze(1), size=(H, W),
                                mode="nearest").squeeze(1)
        valid = (gt > 0.1) & (gt < 5.0) & torch.isfinite(pd) & (pd > 0)
        if valid.sum() < 100:
            model.train()
            for m in model.modules():
                if not any(p.requires_grad for p in m.parameters(recurse=False)):
                    m.eval()
            return float("nan")
        gt_med = torch.median(gt[valid]).clamp_min(1e-3)
        pd_med = torch.median(pd[valid]).clamp_min(1e-3)
        pdn = pd * (gt_med / pd_med)
        rel = (pdn[valid] - gt[valid]).abs() / gt[valid]
        model.train()
        for m in model.modules():
            if not any(p.requires_grad for p in m.parameters(recurse=False)):
                m.eval()
        return float(rel.mean().item())

    best_val = float("inf")
    best_real = float("inf")
    no_improve = 0
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        # Keep frozen base params in eval mode to avoid BN/dropout drift
        for m in model.modules():
            if not any(p.requires_grad for p in m.parameters(recurse=False)):
                m.eval()
        t0 = time.monotonic()
        total_loss = 0.0; n_steps = 0
        for rgb, depth_gt, K, c2w in train_loader:
            rgb = rgb.to(device)                                  # (B, N, 3, H, W)
            depth_gt = depth_gt.to(device)                        # (B, N, H, W)
            with torch.amp.autocast("cuda", dtype=dtype, enabled=device.startswith("cuda")):
                res = model(rgb)
                # local_points: (B, N, H, W, 3); depth = z-component
                local_pts = res["local_points"]
                pred_depth = local_pts[..., 2]                    # (B, N, H, W)
                # Match shapes (Pi3X resizes internally).
                if pred_depth.shape[-2:] != depth_gt.shape[-2:]:
                    H, W = depth_gt.shape[-2:]
                    pred_depth = F.interpolate(
                        pred_depth.flatten(0, 1).unsqueeze(1),
                        size=(H, W), mode="nearest",
                    ).squeeze(1).view(*depth_gt.shape)
                valid = (depth_gt > args.depth_clip_min) & (depth_gt < args.depth_clip_max) \
                         & torch.isfinite(pred_depth) & (pred_depth > 0)
                if valid.sum() < 100:
                    continue
                # FULLY scale-invariant log loss (lambda=1.0 → strict SI;
                # gradient cannot push pred toward CARLA's absolute magnitudes).
                log_diff = torch.log(pred_depth[valid].clamp_min(1e-3)) \
                            - torch.log(depth_gt[valid].clamp_min(1e-3))
                loss = torch.sqrt((log_diff ** 2).mean()
                                   - 1.0 * (log_diff.mean()) ** 2 + 1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item(); n_steps += 1
            global_step += 1

            # ---- inline real-world eval ----
            if (args.eval_every_iters > 0
                    and tum_batch is not None
                    and global_step % args.eval_every_iters == 0):
                t_eval = time.monotonic()
                tum_rel = quick_tum_eval()
                marker = ""
                if tum_rel < best_real:
                    best_real = tum_rel
                    no_improve = 0
                    lora_state = {k: v for k, v in model.state_dict().items()
                                   if "lora_" in k}
                    torch.save({
                        "lora_state": lora_state,
                        "config": {"rank": args.rank, "alpha": args.alpha,
                                    "targets": targets,
                                    "last_n": args.last_n_decoder_blocks,
                                    "global_step": global_step,
                                    "tum_abs_rel": tum_rel},
                    }, args.out)
                    marker = " ★ saved"
                else:
                    no_improve += 1
                print(f"  [step {global_step:>5}]  carla_loss={loss.item():.4f}  "
                      f"tum_abs_rel={tum_rel:.4f}  best={best_real:.4f}  "
                      f"({time.monotonic()-t_eval:.1f}s eval){marker}", flush=True)
                if no_improve >= args.early_stop_patience:
                    print(f"\n  early-stop: no improvement for {no_improve} evals "
                          f"(best tum_abs_rel = {best_real:.4f})", flush=True)
                    print(f"\nBest TUM abs_rel = {best_real:.4f}  →  saved to {args.out}",
                          flush=True)
                    return

        # Val
        model.eval()
        v_total = 0.0; v_steps = 0
        with torch.no_grad():
            for rgb, depth_gt, K, c2w in val_loader:
                rgb = rgb.to(device); depth_gt = depth_gt.to(device)
                with torch.amp.autocast("cuda", dtype=dtype, enabled=device.startswith("cuda")):
                    res = model(rgb)
                    pred_depth = res["local_points"][..., 2]
                    if pred_depth.shape[-2:] != depth_gt.shape[-2:]:
                        H, W = depth_gt.shape[-2:]
                        pred_depth = F.interpolate(
                            pred_depth.flatten(0, 1).unsqueeze(1),
                            size=(H, W), mode="nearest",
                        ).squeeze(1).view(*depth_gt.shape)
                    valid = (depth_gt > args.depth_clip_min) & (depth_gt < args.depth_clip_max) \
                             & torch.isfinite(pred_depth) & (pred_depth > 0)
                    if valid.sum() < 100:
                        continue
                    # val: same median-scale + abs_rel as before (more
                    # interpretable than silog for a val metric)
                    gt_med = torch.median(depth_gt[valid]).clamp_min(1e-3)
                    pr_med = torch.median(pred_depth[valid]).clamp_min(1e-3)
                    pred_scaled = pred_depth * (gt_med / pr_med)
                    rel = torch.abs(pred_scaled[valid] - depth_gt[valid]) / depth_gt[valid]
                    v_total += rel.mean().item(); v_steps += 1

        elapsed = time.monotonic() - t0
        train_loss = total_loss / max(n_steps, 1)
        val_abs_rel = v_total / max(v_steps, 1)
        marker = ""
        if val_abs_rel < best_val:
            best_val = val_abs_rel
            # Save only LoRA params (small)
            lora_state = {k: v for k, v in model.state_dict().items()
                          if "lora_" in k}
            torch.save({
                "lora_state": lora_state,
                "config": {"rank": args.rank, "alpha": args.alpha,
                            "targets": targets, "last_n": args.last_n_decoder_blocks},
            }, args.out)
            marker = " ★"
        print(f"Epoch {epoch+1:>2}/{args.epochs}  "
              f"loss={train_loss:.4f}  val_abs_rel={val_abs_rel:.4f}  "
              f"({elapsed:.0f}s){marker}", flush=True)

    if tum_batch is not None:
        print(f"\nBest TUM abs_rel = {best_real:.4f}  →  saved to {args.out}")
    else:
        print(f"\nBest val abs_rel = {best_val:.4f}  →  saved to {args.out}")


if __name__ == "__main__":
    main()
