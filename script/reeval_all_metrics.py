#!/usr/bin/env python3
"""Re-run sigma evaluator on all SIGMA scenes (after pose-alignment fix)."""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_one(scene_cam, gt_root, pred_root, out_dir):
    scene, cam = scene_cam
    pred = pred_root / scene / cam
    if not pred.exists(): return scene_cam, "no pred"
    gt = gt_root / scene / "dynamic" / cam
    out = out_dir / f"{scene}__{cam}.json"
    cmd = ["python", "-m", "sigma.evaluation.evaluate",
           "--gt", str(gt), "--pred", str(pred),
           "--metrics", "pose", "depth", "pointcloud",
           "--output", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0: return scene_cam, f"err: {r.stderr[-100:]}"
    return scene_cam, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", type=Path, default=Path("outputs/sam3_full_run"))
    ap.add_argument("--gt-root", type=Path, default=Path("StaDy4D/short/test"))
    ap.add_argument("--out", type=Path, default=Path("outputs/sam3_full_run/_metrics_v2"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Use existing _metrics dir to enumerate scene-cams (cleanup deleted scene dirs)
    todo = []
    for f in sorted((args.pred_root / "_metrics").glob("*.json")):
        if "__" not in f.stem: continue
        scene, cam = f.stem.split("__", 1)
        if (args.out / f.name).exists(): continue
        todo.append((scene, cam))

    print(f"to re-eval: {len(todo)}")
    if not todo: return
    t0 = time.time()
    done = 0
    failed = 0
    # Note: pred dir was cleaned up — need to regenerate. Skip for now if not exist.
    todo = [(s,c) for s,c in todo if (args.pred_root / s / c).exists()]
    print(f"with pred available: {len(todo)}")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, sc, args.gt_root, args.pred_root, args.out): sc for sc in todo}
        for fut in as_completed(futs):
            sc, status = fut.result()
            done += 1
            if status != "ok":
                failed += 1
                if failed < 5: print(f"  FAIL {sc}: {status}")
            if done % 50 == 0:
                el = time.time() - t0
                eta = el / done * (len(todo) - done)
                print(f"  [{done}/{len(todo)}]  failed={failed}  ETA {eta/60:.1f}min")
    print(f"done: {done - failed}/{len(todo)} ok, {failed} failed")


if __name__ == "__main__":
    main()
