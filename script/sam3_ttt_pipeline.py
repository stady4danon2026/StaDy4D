#!/usr/bin/env python3
"""Flat end-to-end SAM3+TTT pipeline (no stage/cache abstractions).

Steps:
  1. Pi3X forward (once) on origin frames -> depth, poses, world points, conf features
  2. SAM3 per frame (with image_gate / score_thr / per_query_cov_max filters)
  3. DynHead TTT (30 steps) supervised by SAM3 keyframe masks
  4. Per-frame head inference -> head mask
  5. Union mask = SAM3 mask | head mask
  6. Build static point cloud (PLY) from Pi3 world points where union mask = 0
     + composited frames (RGB with dynamic region highlighted)

No FrameIORecord, no BaseStage, no cache machinery. Reads StaDy4D scene, writes:
  out_dir/
    masks/mask_NNNN.png          # binary union mask
    composited/comp_NNNN.png     # rgb + green dynamic overlay
    static_cloud.ply             # accumulated static cloud
    metadata.json                # poses, intrinsics, frame count
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ─────────────────────────────  helpers  ─────────────────────────────
def _csz(H, W, lim=255000):
    s = math.sqrt(lim / max(W * H, 1))
    Wf, Hf = W * s, H * s
    k, m = round(Wf / 14), round(Hf / 14)
    while (k * 14) * (m * 14) > lim:
        if k / m > Wf / Hf: k -= 1
        else: m -= 1
    return max(1, k) * 14, max(1, m) * 14


def _focal_bce(lg, t, a=0.25, g=2.0):
    bce = F.binary_cross_entropy_with_logits(lg, t, reduction="none")
    p = torch.sigmoid(lg); pt = p * t + (1 - p) * (1 - t); aa = a * t + (1 - a) * (1 - t)
    return aa * (1 - pt).pow(g) * bce


def _post(m, close=5, dilate=3, min_blob=100):
    if not m.any(): return m
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for cc in range(1, n_lab):
        if stats[cc, cv2.CC_STAT_AREA] >= min_blob:
            keep[lab == cc] = 1
    return keep


def _sam3(processor, model, rgb, text, score_thr, device,
          per_query_cov_max=0.3, image_gate=0.5):
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


# ─────────────────────────  Pi3 forward  ─────────────────────────────
def run_pi3(pi3, frames, device):
    """Single Pi3 forward; capture conf_decoder features via hook.
    Returns dict with depth, world_points, camera_poses, conf_features, patch_hw, target_hw.
    """
    from sigma.pipeline.motion.pi3_dyn_features import Pi3FeatureExtractor

    N, H, W = frames.shape[:3]
    TW, TH_ = _csz(H, W)
    to_t = transforms.ToTensor()
    tensors = [to_t(Image.fromarray(frames[i]).resize((TW, TH_), Image.Resampling.LANCZOS)) for i in range(N)]
    imgs = torch.stack(tensors, 0).unsqueeze(0).to(device)

    ext = Pi3FeatureExtractor(pi3)
    amp = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
           else torch.float16)
    with ext.capture():
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=amp):
                out = pi3(imgs)
    return {
        "local_points": out["local_points"][0].float(),    # (N, TH, TW, 3) cam-frame
        "world_points": out["points"][0].float(),          # (N, TH, TW, 3) world
        "camera_poses": out["camera_poses"][0].float(),    # (N, 4, 4) c2w
        "conf_features": ext.features.to(torch.float32),   # (N, P, 1024)
        "patch_hw": ext.patch_hw,
        "target_hw": (TH_, TW),
    }


# ───────────────────────────  TTT head  ──────────────────────────────
def ttt_head(head_init_sd, conf_feats, kf_masks, kf_idx, patch_hw, target_hw,
             device, steps=30, lr=5e-4):
    from sigma.pipeline.motion.pi3_dyn_head import DynHead

    ph, pw = patch_hw
    TH_, TW = target_hw
    head = DynHead().to(device); head.load_state_dict(head_init_sd); head.train()
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    kf_t = torch.from_numpy(kf_masks).float().unsqueeze(1).to(device)
    kf_lbl = F.interpolate(kf_t, size=(TH_, TW), mode="nearest").squeeze(1)
    kf_feats = conf_feats[kf_idx]
    K = len(kf_idx)
    gen = torch.Generator(device=device).manual_seed(0)
    for _ in range(steps):
        idx = torch.randint(0, K, (min(4, K),), device=device, generator=gen)
        lg = head(kf_feats[idx], patch_h=ph, patch_w=pw)
        if lg.shape[-2:] != kf_lbl.shape[-2:]:
            lg = F.interpolate(lg.unsqueeze(1), size=kf_lbl.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        loss = _focal_bce(lg, kf_lbl[idx]).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    head.eval()
    return head


def head_infer(head, conf_feats, fi, patch_hw, target_hw, orig_hw, thr=0.5):
    ph, pw = patch_hw
    TH_, TW = target_hw
    H, W = orig_hw
    with torch.no_grad():
        lg = head(conf_feats[fi:fi+1], patch_h=ph, patch_w=pw)
        if lg.shape[-2:] != (TH_, TW):
            lg = F.interpolate(lg.unsqueeze(1), size=(TH_, TW), mode="bilinear", align_corners=False).squeeze(1)
        m_low = (torch.sigmoid(lg)[0] > thr).cpu().numpy().astype(np.uint8)
    m_orig = cv2.resize(m_low, (W, H), interpolation=cv2.INTER_NEAREST)
    return _post(m_orig)


# ──────────────────────  static cloud assembly  ──────────────────────
def build_static_cloud(world_pts, frames, masks_orig, target_hw, orig_hw, stride=4):
    """Accumulate world_points where union mask = 0 (static), with RGB color.
    world_pts:    (N, TH, TW, 3) torch
    masks_orig:   (N, H, W) numpy uint8 — union mask on original resolution
    """
    TH_, TW = target_hw
    H, W = orig_hw
    N = world_pts.shape[0]
    pts_all, col_all = [], []
    for i in range(N):
        # downsample mask to Pi3 input resolution
        m = cv2.resize(masks_orig[i], (TW, TH_), interpolation=cv2.INTER_NEAREST)
        static = (m == 0)
        if not static.any(): continue
        # color from RGB resized to Pi3 input
        rgb_lr = cv2.resize(frames[i], (TW, TH_), interpolation=cv2.INTER_AREA)
        # stride sample
        ys = np.arange(0, TH_, stride); xs = np.arange(0, TW, stride)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        sel = static[yy, xx]
        if not sel.any(): continue
        p = world_pts[i, yy[sel], xx[sel]].cpu().numpy()
        c = rgb_lr[yy[sel], xx[sel]]
        pts_all.append(p); col_all.append(c)
    if not pts_all: return np.empty((0, 3)), np.empty((0, 3), np.uint8)
    return np.concatenate(pts_all, 0), np.concatenate(col_all, 0)


def save_ply(path, pts, cols):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


# ──────────────────────────────  main  ───────────────────────────────
def load_models(args):
    from pi3.models.pi3x import Pi3X
    from sigma.pipeline.motion.pi3_dyn_head import DynHead
    from transformers import Sam3Processor, Sam3Model
    print("[load] Pi3X, SAM3, head init")
    pi3 = Pi3X.from_pretrained("yyfz233/Pi3X").to(args.device).eval()
    for p in pi3.parameters(): p.requires_grad_(False)
    sam3p = Sam3Processor.from_pretrained("facebook/sam3")
    sam3m = Sam3Model.from_pretrained("facebook/sam3").to(args.device).eval()
    for p in sam3m.parameters(): p.requires_grad_(False)
    head_init_sd = DynHead.load(args.head_checkpoint, map_location="cpu").state_dict()
    return pi3, sam3p, sam3m, head_init_sd


def run_scene(scene_path: Path, out_dir: Path, args, models=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)
    (out_dir / "composited").mkdir(exist_ok=True)
    if args.save_eval:
        for d in ["depth", "extrinsics", "intrinsics", "rgb", "inpainted", "mask"]:
            (out_dir / d).mkdir(exist_ok=True)

    mp4 = scene_path / "rgb.mp4"
    r = imageio.get_reader(str(mp4))
    frames = np.stack([np.asarray(f) for f in r]); r.close()
    N, H, W = frames.shape[:3]
    print(f"[scene] {scene_path}  N={N}  HxW={H}x{W}")

    if models is None:
        pi3, sam3p, sam3m, head_init_sd = load_models(args)
    else:
        pi3, sam3p, sam3m, head_init_sd = models

    # 1. Pi3 forward (once)
    print("[step1] Pi3X forward")
    pi3_out = run_pi3(pi3, frames, args.device)

    # 2. SAM3 keyframes (TTT supervision) + per-frame masks
    print(f"[step2] SAM3 (score_thr={args.score_thr}, image_gate={args.image_gate}, cap={args.cov_max})")
    K = args.kf_count
    kf_idx = sorted(set(int(round(i * (N - 1) / (K - 1))) for i in range(K)))
    K = len(kf_idx)
    sam3_masks = np.zeros((N, H, W), np.uint8)
    for i in range(N):
        sam3_masks[i] = _sam3(sam3p, sam3m, frames[i], args.text,
                              args.score_thr, args.device,
                              per_query_cov_max=args.cov_max, image_gate=args.image_gate)
    kf_masks = sam3_masks[kf_idx]
    print(f"  SAM3 mean coverage: {sam3_masks.mean()*100:.2f}%   keyframes: {kf_idx}")

    # 3. TTT
    print(f"[step3] DynHead TTT  steps={args.ttt_steps}")
    head = ttt_head(head_init_sd, pi3_out["conf_features"], kf_masks, kf_idx,
                    pi3_out["patch_hw"], pi3_out["target_hw"], args.device,
                    steps=args.ttt_steps, lr=args.ttt_lr)

    # 4. Per-frame head + union
    print("[step4] head inference + union")
    union_masks = np.zeros((N, H, W), np.uint8)
    for i in range(N):
        head_m = head_infer(head, pi3_out["conf_features"], i,
                            pi3_out["patch_hw"], pi3_out["target_hw"],
                            (H, W), thr=args.head_thr)
        union_masks[i] = ((sam3_masks[i].astype(bool)) | (head_m.astype(bool))).astype(np.uint8)
        cv2.imwrite(str(out_dir / "masks" / f"mask_{i:04d}.png"), union_masks[i] * 255)
        # composited: green overlay where union=1
        comp = frames[i].copy()
        if union_masks[i].any():
            mb = union_masks[i].astype(bool)
            comp[mb] = (comp[mb] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
        cv2.imwrite(str(out_dir / "composited" / f"comp_{i:04d}.png"), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        if args.save_eval:
            cv2.imwrite(str(out_dir / "mask" / f"mask_{i:04d}.png"), union_masks[i] * 255)
            cv2.imwrite(str(out_dir / "rgb" / f"rgb_{i:04d}.png"), cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / "inpainted" / f"inpainted_{i:04d}.png"), cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))

    print(f"  union mean coverage: {union_masks.mean()*100:.2f}%")

    # 4b. eval-format depth/extrinsics/intrinsics
    if args.save_eval:
        TH_, TW = pi3_out["target_hw"]
        local = pi3_out["local_points"]   # (N, TH, TW, 3) torch
        depth_lr = local[..., 2].abs()    # (N, TH, TW)  Pi3 z is metric depth
        depth_lr = depth_lr.unsqueeze(1)  # (N, 1, TH, TW)
        depth_hr = F.interpolate(depth_lr, size=(H, W), mode="bilinear", align_corners=False).squeeze(1).cpu().numpy().astype(np.float16)
        # derive K from local_points: fx = median((u-cx)/(x/z)) at z>1
        u = np.arange(W); v = np.arange(H)
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        # use first frame for K (all frames share)
        l0 = local[0].cpu().numpy()                                 # (TH, TW, 3)
        l0_hr = cv2.resize(l0, (W, H), interpolation=cv2.INTER_LINEAR)  # (H, W, 3)
        x, y, z = l0_hr[..., 0], l0_hr[..., 1], l0_hr[..., 2]
        d = np.abs(z)
        valid = (d > 1) & (d < 200)
        uu, vv = np.meshgrid(u, v)
        fx_arr = (uu - cx) * d / x
        fy_arr = (vv - cy) * d / y    # OpenCV Y-down (matches NEW_gsam intrinsics)
        fx_med = float(np.median(fx_arr[valid & (np.abs(x) > 1e-3)]))
        fy_med = float(np.median(fy_arr[valid & (np.abs(y) > 1e-3)]))
        K_mat = np.array([[fx_med, 0, cx], [0, fy_med, cy], [0, 0, 1]], np.float64)
        c2w = pi3_out["camera_poses"].cpu().numpy()                 # (N, 4, 4)
        for i in range(N):
            np.save(out_dir / "depth" / f"depth_{i:04d}.npy", depth_hr[i])
            np.save(out_dir / "extrinsics" / f"extrinsic_{i:04d}.npy", c2w[i, :3, :4].astype(np.float64))
            np.save(out_dir / "intrinsics" / f"intrinsic_{i:04d}.npy", K_mat)

    # 5. Static cloud
    print("[step5] static point cloud")
    pts, cols = build_static_cloud(pi3_out["world_points"], frames, union_masks,
                                    pi3_out["target_hw"], (H, W), stride=args.cloud_stride)
    save_ply(out_dir / "static_cloud.ply", pts, cols)
    print(f"  saved {len(pts)} points → static_cloud.ply")

    # 6. Metadata
    meta = {
        "scene": str(scene_path),
        "n_frames": int(N),
        "resolution": [int(H), int(W)],
        "patch_hw": list(pi3_out["patch_hw"]),
        "target_hw": list(pi3_out["target_hw"]),
        "camera_poses_c2w": pi3_out["camera_poses"].cpu().numpy().tolist(),
        "config": {
            "score_thr": args.score_thr, "image_gate": args.image_gate,
            "cov_max": args.cov_max, "head_thr": args.head_thr,
            "ttt_steps": args.ttt_steps, "kf_count": args.kf_count,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {out_dir}")


def discover_scenes(root: Path, scene_filter: str | None, cam_filter: str | None,
                    scene_type: str = "dynamic"):
    """Return list of (scene_name, cam_name, scene_cam_path) under root."""
    out = []
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene_"): continue
        if scene_filter and scene_filter not in scene_dir.name: continue
        type_dir = scene_dir / scene_type
        if not type_dir.is_dir(): continue
        for cam_dir in sorted(type_dir.iterdir()):
            if not cam_dir.is_dir() or not cam_dir.name.startswith("cam_"): continue
            if cam_filter and cam_filter not in cam_dir.name: continue
            if not (cam_dir / "rgb.mp4").exists(): continue
            out.append((scene_dir.name, cam_dir.name, cam_dir))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, default=None,
                   help="single scene/camera path; mutually exclusive with --batch-root")
    p.add_argument("--batch-root", type=Path, default=None,
                   help="e.g. StaDy4D/short/test — runs all scene/cam combos")
    p.add_argument("--scene-filter", default=None, help="substring filter on scene name")
    p.add_argument("--cam-filter", default=None, help="substring filter on cam name")
    p.add_argument("--scene-cam-list", default=None,
                   help="comma-separated 'scene/cam' list, e.g. 'scene_T03_000/cam_00,scene_T07_007/cam_04'")
    p.add_argument("--scene-type", default="dynamic", choices=["dynamic", "static"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--head-checkpoint", default="checkpoints/pi3_dyn_head_pooled.pt")
    p.add_argument("--text", default="car. truck. bus. motorcycle. bicycle. pedestrian.")
    p.add_argument("--device", default="cuda")
    # SAM3
    p.add_argument("--score-thr", type=float, default=0.2)
    p.add_argument("--image-gate", type=float, default=0.5)
    p.add_argument("--cov-max", type=float, default=0.3)
    # head
    p.add_argument("--head-thr", type=float, default=0.5)
    # TTT
    p.add_argument("--kf-count", type=int, default=5)
    p.add_argument("--ttt-steps", type=int, default=30)
    p.add_argument("--ttt-lr", type=float, default=5e-4)
    # cloud
    p.add_argument("--cloud-stride", type=int, default=4)
    p.add_argument("--save-eval", action="store_true",
                   help="also save depth/extrinsics/intrinsics/rgb/inpainted/mask in evaluator format")
    p.add_argument("--inline-eval", action="store_true",
                   help="(batch only) run sigma evaluator after each scene → metrics JSON")
    p.add_argument("--cleanup-after-eval", action="store_true",
                   help="(batch only) delete scene output dir after metrics saved (keep _metrics/*.json only)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip scene-cam if metadata.json already exists")
    args = p.parse_args()

    if (args.scene is None) == (args.batch_root is None):
        raise SystemExit("Provide exactly one of --scene or --batch-root")

    if args.scene is not None:
        run_scene(args.scene, args.out, args)
        return

    # batch mode
    import shutil, subprocess, time
    if args.scene_cam_list:
        todo = []
        for entry in args.scene_cam_list.split(","):
            scene, cam_prefix = entry.strip().split("/")
            cam_dir_root = args.batch_root / scene / args.scene_type
            for cd in sorted(cam_dir_root.iterdir()):
                if cd.is_dir() and cd.name.startswith(cam_prefix) and (cd / "rgb.mp4").exists():
                    todo.append((scene, cd.name, cd))
                    break
    else:
        todo = discover_scenes(args.batch_root, args.scene_filter, args.cam_filter, args.scene_type)
    print(f"[batch] {len(todo)} scene-camera combos")
    models = load_models(args)
    metrics_dir = args.out / "_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, (scene, cam, path) in enumerate(todo):
        out_d = args.out / scene / cam
        metrics_json = metrics_dir / f"{scene}__{cam}.json"
        if args.skip_existing and metrics_json.exists():
            print(f"[{k+1}/{len(todo)}] skip (metrics exist) {scene}/{cam}")
            continue
        ts = time.time()
        try:
            run_scene(path, out_d, args, models=models)
        except Exception as e:
            print(f"  ERROR scene {scene}/{cam}: {e}")
            continue

        # inline eval (if save-eval was on)
        if args.save_eval and args.inline_eval:
            gt = args.batch_root / scene / args.scene_type / cam
            try:
                eval_cmd = ["python", "-m", "sigma.evaluation.evaluate",
                            "--gt", str(gt), "--pred", str(out_d),
                            "--metrics", "pose", "depth", "pointcloud",
                            "--output", str(metrics_json)]
                r = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=300)
                if r.returncode != 0:
                    print(f"  EVAL FAILED {scene}/{cam}: {r.stderr[-200:]}")
            except Exception as e:
                print(f"  EVAL ERROR {scene}/{cam}: {e}")

        # cleanup
        if args.cleanup_after_eval and metrics_json.exists():
            shutil.rmtree(out_d, ignore_errors=True)

        elapsed = time.time() - t0
        per = elapsed / (k + 1)
        eta = per * (len(todo) - k - 1)
        print(f"[{k+1}/{len(todo)}] {scene}/{cam}  took {time.time()-ts:.1f}s  "
              f"avg {per:.1f}s/scene  ETA {eta/3600:.2f}h")


if __name__ == "__main__":
    main()
