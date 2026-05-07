#!/usr/bin/env python3
"""Toy ablation viz on worst-case scenes.

For each scene, runs Pi3 + GSAM + TTT once (head mask), runs SAM3 once,
then composites 3 final-mask variants:
  A. current default     (score=0.3, union, max_cov=0.5)
  B. SAM3 score=0.5      (no head, no cap)
  C. conditional union   (score=0.3, union only when head_cov<10%)

Outputs side-by-side: RGB | A | B | C  (plus a row showing component masks).
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import cv2, numpy as np, imageio.v2 as imageio, torch
import torch.nn.functional as F


def _csz(H, W, lim=255000):
    s = math.sqrt(lim / max(W*H, 1))
    Wf, Hf = W*s, H*s; k, m = round(Wf/14), round(Hf/14)
    while (k*14)*(m*14) > lim:
        if k/m > Wf/Hf: k -= 1
        else: m -= 1
    return max(1,k)*14, max(1,m)*14


def _focal_bce(lg, t, a=0.25, g=2.0):
    bce = F.binary_cross_entropy_with_logits(lg, t, reduction="none")
    p = torch.sigmoid(lg); pt = p*t + (1-p)*(1-t); aa = a*t + (1-a)*(1-t)
    return aa * (1-pt).pow(g) * bce


def _bar(text, w, h=26, bg=30):
    bar = np.full((h, w, 3), bg, np.uint8)
    cv2.putText(bar, text, (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    return bar


def _post(m, close=5, dilate=3, min_blob=100):
    if not m.any(): return m
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close,close), np.uint8))
    m = cv2.dilate(m, np.ones((dilate,dilate), np.uint8))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for cc in range(1, n_lab):
        if stats[cc, cv2.CC_STAT_AREA] >= min_blob:
            keep[lab == cc] = 1
    return keep


def _sam3_with_thr(processor, model, rgb, text, score_thr, device):
    from PIL import Image
    pil = Image.fromarray(rgb)
    inp = processor(images=pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        o = model(**inp)
    scores = o.pred_logits[0].sigmoid().cpu().numpy()
    masks_lr = (o.pred_masks[0].sigmoid() > 0.5).cpu().numpy()
    keep = scores > score_thr
    H, W = rgb.shape[:2]
    if keep.sum() == 0:
        return np.zeros((H, W), np.uint8)
    masks_lr = masks_lr[keep]
    out = np.zeros((H, W), bool)
    for c in range(masks_lr.shape[0]):
        m = cv2.resize(masks_lr[c].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        out |= m
    return out.astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=[
        "scene_T07_005/cam_05_orbit_crossroad",
        "scene_T07_034/cam_05_orbit_crossroad",
        "scene_T07_024/cam_06_cctv",
        "scene_T03_013/cam_05_orbit_crossroad",
        "scene_T10_037/cam_02_car_forward",
    ])
    p.add_argument("--frame", type=int, default=25)
    p.add_argument("--root", type=Path, default=Path("StaDy4D/short/test"))
    p.add_argument("--out", type=Path, default=Path("outputs/pi3_dyn_toy_ablation"))
    p.add_argument("--head-checkpoint", default="checkpoints/pi3_dyn_head_pooled.pt")
    p.add_argument("--text", default="car. truck. bus. motorcycle. bicycle. pedestrian.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from pi3.models.pi3x import Pi3X
    from sigma.pipeline.motion.grounded_sam import GroundedSAMMotionEstimator
    from sigma.pipeline.motion.pi3_dyn_features import Pi3FeatureExtractor
    from sigma.pipeline.motion.pi3_dyn_head import DynHead
    from sigma.data.frame_record import FrameIORecord
    from transformers import Sam3Processor, Sam3Model
    from PIL import Image
    from torchvision import transforms

    print("Loading models...")
    pi3 = Pi3X.from_pretrained("yyfz233/Pi3X").to(args.device).eval()
    for x in pi3.parameters(): x.requires_grad_(False)
    gsam = GroundedSAMMotionEstimator(device=args.device); gsam.setup()
    sam3p = Sam3Processor.from_pretrained("facebook/sam3")
    sam3m = Sam3Model.from_pretrained("facebook/sam3").to(args.device).eval()
    head_init = DynHead.load(args.head_checkpoint, map_location="cpu").state_dict()
    to_t = transforms.ToTensor()
    amp = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0]>=8 else torch.float16

    for scene_cam in args.scenes:
        scene, cam = scene_cam.split("/")
        mp4 = args.root / scene / "dynamic" / cam / "rgb.mp4"
        if not mp4.exists():
            print(f"skip {scene_cam}"); continue
        r = imageio.get_reader(str(mp4))
        frames = np.stack([np.asarray(f) for f in r]); r.close()
        N, H, W = frames.shape[:3]
        fi = min(args.frame, N - 1)
        rgb = frames[fi]

        # Pi3 forward to get conf features
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

        # GSAM 5 keyframes + TTT
        K = 5
        kf_idx = sorted(set(int(round(i*(N-1)/(K-1))) for i in range(K)))
        K = len(kf_idx)
        kf_recs = {j: FrameIORecord(frame_idx=j, origin_image=frames[orig]) for j, orig in enumerate(kf_idx)}
        gout = gsam.process_batch(kf_recs)
        kf_masks = np.stack([gout[j].data["motion"]["curr_mask"] for j in range(K)], 0)
        kf_t = torch.from_numpy(kf_masks).float().unsqueeze(1).to(args.device)
        kf_lbl = F.interpolate(kf_t, size=(TH_, TW), mode="nearest").squeeze(1)

        head = DynHead().to(args.device); head.load_state_dict(head_init); head.train()
        opt = torch.optim.Adam(head.parameters(), lr=5e-4)
        kf_feats = feats[kf_idx]; gen = torch.Generator(device=args.device).manual_seed(0)
        for _ in range(30):
            idx = torch.randint(0, K, (min(4, K),), device=args.device, generator=gen)
            lg = head(kf_feats[idx], patch_h=ph, patch_w=pw)
            if lg.shape[-2:] != kf_lbl.shape[-2:]:
                lg = F.interpolate(lg.unsqueeze(1), size=kf_lbl.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            loss = _focal_bce(lg, kf_lbl[idx]).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        head.eval()

        # Head mask for this frame
        with torch.no_grad():
            lg = head(feats[fi:fi+1], patch_h=ph, patch_w=pw)
            if lg.shape[-2:] != (TH_, TW):
                lg = F.interpolate(lg.unsqueeze(1), size=(TH_, TW), mode="bilinear", align_corners=False).squeeze(1)
            head_pi3 = (torch.sigmoid(lg)[0] > 0.45).cpu().numpy().astype(np.uint8)
        head_orig = cv2.resize(head_pi3, (W, H), interpolation=cv2.INTER_NEAREST)
        head_mask = _post(head_orig)

        # SAM3 at 0.3 and 0.5
        sam3_03 = _sam3_with_thr(sam3p, sam3m, rgb, args.text, 0.3, args.device)
        sam3_05 = _sam3_with_thr(sam3p, sam3m, rgb, args.text, 0.5, args.device)

        # 3 variants
        # A. current method: score 0.3, union, max-cov cap (drop SAM3 if >50%)
        sam3_capped = sam3_03 if sam3_03.mean() <= 0.50 else np.zeros_like(sam3_03)
        var_A = sam3_capped | head_mask

        # B. SAM3 score 0.5 only
        var_B = sam3_05

        # C. conditional union: union only when head_cov<10%
        if head_mask.mean() < 0.10:
            var_C = sam3_capped | head_mask
        else:
            var_C = head_mask

        # Render: RGB | A | B | C
        def overlay(rgb, m, color=(0,255,0)):
            o = rgb.copy()
            if m.any(): o[m.astype(bool)] = (o[m.astype(bool)]*0.4 + np.array(color)*0.6).astype(np.uint8)
            return o

        ov_A = overlay(rgb, var_A); ov_B = overlay(rgb, var_B); ov_C = overlay(rgb, var_C)
        bar = _bar(f"frame {fi}  head_cov={head_mask.mean()*100:.1f}%  sam3_03={sam3_03.mean()*100:.1f}%  sam3_05={sam3_05.mean()*100:.1f}%",
                   W*4, h=28)
        labels_bar = _bar(f"RGB | A current ({var_A.mean()*100:.1f}%) | B sam3_thr0.5 ({var_B.mean()*100:.1f}%) | C condUnion ({var_C.mean()*100:.1f}%)",
                          W*4, h=24)
        row = np.concatenate([rgb, ov_A, ov_B, ov_C], axis=1)
        title = np.full((36, W*4, 3), 60, np.uint8)
        cv2.putText(title, scene_cam, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        grid = np.concatenate([title, bar, labels_bar, row], axis=0)
        out_path = args.out / f"{scene}__{cam}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"saved {out_path}")

    gsam.teardown()


if __name__ == "__main__":
    main()
