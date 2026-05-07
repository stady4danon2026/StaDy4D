#!/usr/bin/env python3
"""Render head-pt vs SAM3 vs union breakdown for a few scenes.

Color code on the overlay column:
  - GREEN  = head ∩ SAM3   (both agree)
  - YELLOW = head only     (TTT-adapted DynHead fires; SAM3 doesn't)
  - RED    = SAM3 only     (SAM3 fires; head doesn't)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def _csz(H, W, lim=255000):
    s = math.sqrt(lim / max(W * H, 1))
    Wf, Hf = W * s, H * s
    k, m = round(Wf / 14), round(Hf / 14)
    while (k * 14) * (m * 14) > lim:
        if k / m > Wf / Hf:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def _focal_bce(lg, t, a=0.25, g=2.0):
    bce = F.binary_cross_entropy_with_logits(lg, t, reduction="none")
    p = torch.sigmoid(lg)
    pt = p * t + (1 - p) * (1 - t)
    aa = a * t + (1 - a) * (1 - t)
    return aa * (1 - pt).pow(g) * bce


def _bar(text, w, h=24):
    bar = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(bar, text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return bar


def _post(m, close=5, dilate=3, min_blob=100):
    if not m.any():
        return m
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for cc in range(1, n_lab):
        if stats[cc, cv2.CC_STAT_AREA] >= min_blob:
            keep[lab == cc] = 1
    return keep


def _sam3(processor, model, rgb, text, score_thr, device, per_query_cov_max=0.3,
          image_gate=0.5):
    pil = Image.fromarray(rgb)
    inp = processor(images=pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        o = model(**inp)
    scores = o.pred_logits[0].sigmoid().cpu().numpy()
    masks_lr = (o.pred_masks[0].sigmoid() > 0.5).cpu().numpy()
    H, W = rgb.shape[:2]
    if scores.max() < image_gate:
        return np.zeros((H, W), np.uint8)
    keep = scores > score_thr
    if keep.sum() == 0:
        return np.zeros((H, W), np.uint8)
    masks_lr = masks_lr[keep]
    out = np.zeros((H, W), bool)
    for c in range(masks_lr.shape[0]):
        m = cv2.resize(masks_lr[c].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        if m.mean() > per_query_cov_max:
            continue
        out |= m
    return out.astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=[
        "scene_T03_021/cam_05_orbit_crossroad",
        "scene_T07_023/cam_05_orbit_crossroad",
        "scene_T07_007/cam_04_orbit_building",
        "scene_T07_041/cam_07_pedestrian",
        "scene_T03_007/cam_03_drone_forward",
    ])
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--out", type=Path, default=Path("outputs/sam3_breakdown_viz"))
    p.add_argument("--head-checkpoint", default="checkpoints/pi3_dyn_head_pooled.pt")
    p.add_argument("--text", default="car. truck. bus. motorcycle. bicycle. pedestrian.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from pi3.models.pi3x import Pi3X
    from sigma.pipeline.motion.pi3_dyn_features import Pi3FeatureExtractor
    from sigma.pipeline.motion.pi3_dyn_head import DynHead
    from transformers import Sam3Processor, Sam3Model

    print("Loading Pi3X, SAM3, head...")
    pi3 = Pi3X.from_pretrained("yyfz233/Pi3X").to(args.device).eval()
    for x in pi3.parameters(): x.requires_grad_(False)
    sam3p = Sam3Processor.from_pretrained("facebook/sam3")
    sam3m = Sam3Model.from_pretrained("facebook/sam3").to(args.device).eval()
    for x in sam3m.parameters(): x.requires_grad_(False)
    head_init = DynHead.load(args.head_checkpoint, map_location="cpu").state_dict()
    to_t = transforms.ToTensor()
    amp = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
           else torch.float16)

    for scene_cam in args.scenes:
        scene, cam = scene_cam.split("/")
        mp4 = args.root / scene / "dynamic" / cam / "rgb.mp4"
        if not mp4.exists():
            continue
        r = imageio.get_reader(str(mp4))
        frames = np.stack([np.asarray(f) for f in r])
        r.close()
        N, H, W = frames.shape[:3]

        # Pi3 forward
        TW, TH_ = _csz(H, W)
        tensors = [to_t(Image.fromarray(frames[i]).resize((TW, TH_), Image.Resampling.LANCZOS)) for i in range(N)]
        imgs = torch.stack(tensors, 0).unsqueeze(0).to(args.device)
        ext = Pi3FeatureExtractor(pi3)
        with ext.capture():
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=amp):
                    _ = pi3(imgs)
        feats = ext.features.to(torch.float32)
        ph, pw = ext.patch_hw

        # TTT supervised by SAM3 keyframes
        K = 5
        kf_idx = sorted(set(int(round(i * (N - 1) / (K - 1))) for i in range(K)))
        K = len(kf_idx)
        kf_masks = np.stack([_sam3(sam3p, sam3m, frames[orig_i], args.text, 0.2, args.device)
                              for orig_i in kf_idx], 0)
        kf_t = torch.from_numpy(kf_masks).float().unsqueeze(1).to(args.device)
        kf_lbl = F.interpolate(kf_t, size=(TH_, TW), mode="nearest").squeeze(1)

        head = DynHead().to(args.device); head.load_state_dict(head_init); head.train()
        opt = torch.optim.Adam(head.parameters(), lr=5e-4)
        kf_feats = feats[kf_idx]
        gen = torch.Generator(device=args.device).manual_seed(0)
        for _ in range(30):
            idx = torch.randint(0, K, (min(4, K),), device=args.device, generator=gen)
            lg = head(kf_feats[idx], patch_h=ph, patch_w=pw)
            if lg.shape[-2:] != kf_lbl.shape[-2:]:
                lg = F.interpolate(lg.unsqueeze(1), size=kf_lbl.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            loss = _focal_bce(lg, kf_lbl[idx]).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        head.eval()

        rows = []
        for fi in [N // 6, N // 2, 5 * N // 6]:
            rgb = frames[fi]
            with torch.no_grad():
                lg = head(feats[fi:fi + 1], patch_h=ph, patch_w=pw)
                if lg.shape[-2:] != (TH_, TW):
                    lg = F.interpolate(lg.unsqueeze(1), size=(TH_, TW), mode="bilinear", align_corners=False).squeeze(1)
                head_pi3 = (torch.sigmoid(lg)[0] > 0.5).cpu().numpy().astype(np.uint8)
            head_orig = cv2.resize(head_pi3, (W, H), interpolation=cv2.INTER_NEAREST)
            head_mask = _post(head_orig).astype(bool)

            sam3_mask = _sam3(sam3p, sam3m, rgb, args.text, 0.2, args.device).astype(bool)

            # Color-coded overlay
            ov = rgb.copy()
            both = head_mask & sam3_mask
            head_only = head_mask & ~sam3_mask
            sam3_only = sam3_mask & ~head_mask
            if both.any():       ov[both]      = (ov[both]     *0.4 + np.array([0,255,0])*0.6).astype(np.uint8)  # green
            if head_only.any():  ov[head_only] = (ov[head_only]*0.4 + np.array([255,255,0])*0.6).astype(np.uint8)  # yellow
            if sam3_only.any():  ov[sam3_only] = (ov[sam3_only]*0.4 + np.array([255,0,0])*0.6).astype(np.uint8)   # red

            head_viz = (head_mask[..., None].astype(np.uint8)*255).repeat(3, axis=-1)
            sam3_viz = (sam3_mask[..., None].astype(np.uint8)*255).repeat(3, axis=-1)

            stat = (f"frame {fi}  head={head_mask.mean()*100:.1f}%  sam3={sam3_mask.mean()*100:.1f}%  "
                    f"agree={both.mean()*100:.1f}%   |  RGB | head only | SAM3 only | overlay (G=both, Y=head, R=SAM3)")
            row = np.concatenate([rgb, head_viz, sam3_viz, ov], axis=1)
            rows.append(np.concatenate([_bar(stat, row.shape[1], h=26), row], axis=0))

        grid = np.concatenate(rows, axis=0)
        title = np.full((30, grid.shape[1], 3), 60, np.uint8)
        cv2.putText(title, scene_cam, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        grid = np.concatenate([title, grid], axis=0)
        op = args.out / f"{scene}__{cam}.png"
        cv2.imwrite(str(op), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"saved {op}")


if __name__ == "__main__":
    main()
