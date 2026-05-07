#!/usr/bin/env python3
"""Scan StaDy4D dataset and check that every camera folder has all required files."""

import sys
from pathlib import Path

REQUIRED_CAM_FILES = {
    "depth.safetensors",
    "extrinsics.safetensors",
    "intrinsics.safetensors",
    "rgb.mp4",
}

REQUIRED_SCENE_FILES = {"metadata.json", "actors.json"}
DATA_FORMATS = ["dynamic", "static"]

DURATIONS = ["short", "mid"]
SPLITS = ["test", "train"]


def check_dataset(root: Path):
    issues = 0
    cam_checked = 0
    scene_checked = 0

    for duration in DURATIONS:
        for split in SPLITS:
            split_dir = root / duration / split
            if not split_dir.is_dir():
                continue

            scenes = sorted(p for p in split_dir.iterdir() if p.is_dir())
            for scene in scenes:
                scene_checked += 1
                rel_scene = scene.relative_to(root)

                # Check scene-level files
                for f in sorted(REQUIRED_SCENE_FILES):
                    if not (scene / f).is_file():
                        print(f"MISSING  {rel_scene}/{f}")
                        issues += 1

                # Check dynamic/static folders exist
                for fmt in DATA_FORMATS:
                    fmt_dir = scene / fmt
                    if not fmt_dir.is_dir():
                        print(f"MISSING  {rel_scene}/{fmt}/  (entire folder)")
                        issues += 1
                        continue

                    cams = sorted(p for p in fmt_dir.iterdir() if p.is_dir())
                    if not cams:
                        print(f"EMPTY    {rel_scene}/{fmt}/  (no camera folders)")
                        issues += 1
                        continue

                    for cam in cams:
                        cam_checked += 1
                        existing = {f.name for f in cam.iterdir() if f.is_file()}
                        missing = REQUIRED_CAM_FILES - existing
                        extra = existing - REQUIRED_CAM_FILES
                        rel = cam.relative_to(root)
                        if missing:
                            issues += len(missing)
                            print(f"MISSING  {rel}  ->  {', '.join(sorted(missing))}")
                        if extra:
                            print(f"EXTRA    {rel}  ->  {', '.join(sorted(extra))}")

                # Check camera consistency between dynamic and static
                dyn_dir = scene / "dynamic"
                sta_dir = scene / "static"
                if dyn_dir.is_dir() and sta_dir.is_dir():
                    dyn_cams = {p.name for p in dyn_dir.iterdir() if p.is_dir()}
                    sta_cams = {p.name for p in sta_dir.iterdir() if p.is_dir()}
                    only_dyn = dyn_cams - sta_cams
                    only_sta = sta_cams - dyn_cams
                    if only_dyn:
                        print(f"MISMATCH {rel_scene}  dynamic only: {', '.join(sorted(only_dyn))}")
                        issues += len(only_dyn)
                    if only_sta:
                        print(f"MISMATCH {rel_scene}  static only: {', '.join(sorted(only_sta))}")
                        issues += len(only_sta)

    print(f"\nScanned {scene_checked} scenes, {cam_checked} camera folders, {issues} issues found.")
    return issues


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("StaDy4D")
    if not root.is_dir():
        sys.exit(f"Error: {root} is not a directory")
    check_dataset(root)
