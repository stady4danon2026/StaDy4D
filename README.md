# SIGMA — Self-supervised In-context Grounded Map Aggregation

This repository contains the code release for **SIGMA**, accompanying the **StaDy4D** benchmark.
SIGMA is a flat, single-pass pipeline that produces a clean static reconstruction of
dynamic driving / urban scenes by combining (i) a single Pi3X forward, (ii) open-vocabulary
text-prompted segmentation (SAM3), and (iii) test-time training (TTT) of a lightweight
dynamic head supervised by SAM3 keyframes.

---

## 1. Repository layout

```
sigma/                         Core library
├── data/                      Frame records + StaDy4D safetensors loader
├── pipeline/
│   ├── motion/                DynHead, Pi3 feature extractor, SAM3+TTT stage,
│   │                          GroundedSAM baseline, photometric teacher
│   ├── inpainting/            Geometric / blank / SDXL inpainters
│   ├── reconstruction/        VGGT and Pi3X reconstructors
│   ├── _pi3_cache.py          Shared per-scene Pi3 forward cache
│   └── _model_registry.py     Lazy-loaded model singleton
├── evaluation/                Pose / depth / point-cloud metrics + CLI
├── visualization/             Rerun viewer, multi-camera world builder
├── runners/                   Phase-based pipeline runner
├── configs/                   Hydra configs
└── utils/

script/                        End-to-end scripts (training, eval, viz, baselines)
├── sam3_ttt_pipeline.py       MAIN: flat SIGMA pipeline (single + batch)
├── train_pi3_dyn_head.py      Train DynHead on dynamic-pseudo-labels
├── run_pi3_dyn_ttt_batch.py   TTT-only batch driver
├── run_pi3_dyn_keyframe_ttt_batch.py
├── aggregate_sam3_vs_baselines.py   Aggregate SIGMA vs baseline summaries
├── sigma_breakdown_table.py         Per-scene / per-weather / per-town tables
├── compare_sam3_vs_gsam.py          Per-scene comparison (CLI)
├── viz_worst_nc.py                  Depth / mask diagnostic viz
├── sam3_5col_compare.py             5-column mask comparison viz
├── sam3_ttt_breakdown_viz.py        Color-coded SAM3 vs head breakdown
├── eval_ablation_*.py               Baselines (DA3, CUT3R, MonST3R, Easi3R,
│                                    VGGT4D, MapAny, LoGeR, MonST3R, Pi3, VGGT)
└── ...

requirements.txt
pyproject.toml
```

---

## 2. Setup

### Python environment

```bash
conda create -n sigma python=3.10 -y && conda activate sigma
pip install -r requirements.txt
pip install -e .                    # install sigma/ as editable package
```

### Pi3X (third-party) and SAM3 (HuggingFace)

- **Pi3X** is the reconstruction backbone (`yyfz233/Pi3X` on HuggingFace). Either:
  - `pip install git+https://github.com/<UPSTREAM>/Pi3X.git`, **or**
  - Place a copy of the Pi3X repo under `third_party/Pi3` and install in editable mode.
- **SAM3** comes from `transformers >= 4.45`. The first run will auto-download
  `facebook/sam3` weights (~2 GB).

### Dataset

This release **does not include the dataset**. Download **StaDy4D** from
the project page and arrange it as follows:

```
<your-data-root>/StaDy4D/short/test/
  scene_T03_000/
    metadata.json
    actors.json
    dynamic/
      cam_00_car_forward/
        rgb.mp4
        depth.safetensors
        extrinsics.safetensors
        intrinsics.safetensors
      cam_05_orbit_crossroad/
      ...
    static/
      ...
  scene_T03_001/
  ...
```

In the commands below, `--batch-root <your-data-root>/StaDy4D/short/test`.

### Pre-trained DynHead weights

The DynHead used in our experiments is initialized from a small
ConvHead trained on auto-generated dynamic-mask pseudo-labels
(`script/train_pi3_dyn_head.py`). To reproduce SIGMA from scratch:

```bash
# 1. Collect training data (DynHead pseudo-labels from Pi3 features)
python script/collect_dynamic_training_data.py \
    --root <your-data-root>/StaDy4D/short/train \
    --out  <cache>/dyn_head_train

# 2. Train head
python script/train_pi3_dyn_head.py \
    --cache <cache>/dyn_head_train \
    --out   checkpoints/pi3_dyn_head_pooled.pt
```

We provide hyper-parameters used in the paper inside the script defaults.

---

## 3. Running SIGMA

### Single scene / camera

```bash
python script/sam3_ttt_pipeline.py \
    --scene <data>/StaDy4D/short/test/scene_T03_000/dynamic/cam_00_car_forward \
    --out   outputs/sigma/scene_T03_000_cam_00 \
    --head-checkpoint checkpoints/pi3_dyn_head_pooled.pt
```

Outputs:

```
outputs/sigma/.../
  masks/mask_NNNN.png            # binary union mask (SAM3 ∪ DynHead)
  composited/comp_NNNN.png       # RGB with green dynamic overlay
  static_cloud.ply               # accumulated static point cloud
  metadata.json                  # poses, intrinsics, configuration
```

Add `--save-eval` to also write `depth/`, `extrinsics/`, `intrinsics/`, `rgb/`,
`inpainted/`, `mask/` in the format consumed by the evaluator.

### Batch (full split with inline evaluation + cleanup)

```bash
python script/sam3_ttt_pipeline.py \
    --batch-root <data>/StaDy4D/short/test \
    --out outputs/sigma_full_run \
    --save-eval --inline-eval --cleanup-after-eval --skip-existing
```

This loops every `scene_*/dynamic/cam_*` combination, evaluates immediately,
keeps only `_metrics/{scene}__{cam}.json`, and deletes per-scene artefacts.

Selective subsets:

- `--scene-filter scene_T03_`
- `--cam-filter cam_00`
- `--scene-cam-list scene_T03_000/cam_00,scene_T07_007/cam_04`

### SIGMA hyper-parameters (CLI defaults)

| Flag | Default | Meaning |
|---|---|---|
| `--score-thr`     | 0.20  | SAM3 query confidence threshold |
| `--image-gate`    | 0.50  | Image-level early-exit (max query score) |
| `--cov-max`       | 0.30  | Per-query mask coverage cap |
| `--head-thr`      | 0.50  | DynHead pixel threshold |
| `--kf-count`      | 5     | TTT keyframe count |
| `--ttt-steps`     | 30    | TTT iterations |
| `--ttt-lr`        | 5e-4  | TTT Adam learning rate |
| `--text` | `"car. truck. bus. motorcycle. bicycle. pedestrian."` | SAM3 prompt |

---

## 4. Evaluation

```bash
python -m sigma.evaluation.evaluate \
    --gt   <data>/StaDy4D/short/test/scene_T03_000/dynamic/cam_00_car_forward \
    --pred outputs/sigma/scene_T03_000_cam_00 \
    --metrics pose depth pointcloud \
    --output  outputs/sigma/scene_T03_000_cam_00.eval.json
```

> **Note on alignment:** When the camera trajectory is nearly static (e.g. a
> stopped car or fixed observer), trajectory-Procrustes is unreliable. The
> evaluator falls back to depth-median scale + rotation-based alignment when
> the two scales disagree by more than 2×. See
> `sigma/evaluation/evaluator.py:222` and `sigma/evaluation/metrics.py:42`.

---

## 5. Aggregation and breakdowns

Once the batch run has produced `outputs/sigma_full_run/_metrics/`:

```bash
# Aggregate vs baselines (place baseline summaries under eval_results/)
python script/aggregate_sam3_vs_baselines.py \
    --sam3-metrics outputs/sigma_full_run/_metrics \
    --baselines    eval_results/NEW_gsam_summary.json \
                   eval_results/NEW_student_summary.json

# Per-scene-type / per-weather / per-town breakdown
python script/sigma_breakdown_table.py
```

---

## 6. Baselines

`script/eval_ablation_<method>.py` provide drop-in evaluators for: DA3, CUT3R,
MonST3R, Easi3R, VGGT, VGGT4D, MapAny / MapAnything, LoGeR, Pi3X (raw).
Each produces a per-scene metrics JSON in the same schema as SIGMA so they can
be aggregated together.

---

## 7. Visualization (optional)

```bash
# Side-by-side mask comparison: RGB | GSAM | SAM3-only | head-only | union
python script/sam3_5col_compare.py

# Color-coded breakdown (head ∩ SAM3 vs head-only vs SAM3-only)
python script/sam3_ttt_breakdown_viz.py

# Depth / point-cloud diagnostic on worst-NC scenes
python script/viz_worst_nc.py
```

---

## 8. Reproducibility

| Item | Value |
|---|---|
| Pi3X checkpoint    | `yyfz233/Pi3X` (HuggingFace) |
| SAM3 checkpoint    | `facebook/sam3` (HuggingFace) |
| DynHead init       | `checkpoints/pi3_dyn_head_pooled.pt` (trained per §2) |
| Hardware           | Single GPU with ≥ 16 GB VRAM (tested on Ampere-class) |
| Per scene-cam time | ~22 s (Pi3X + SAM3 + 30-step TTT + cloud assembly + inline eval) |
| Full ShortVid run  | ~9 h on 1×GPU (1456 scene-camera pairs) |

---

## 9. Citation

```bibtex
@inproceedings{sigma2026,
  title     = {SIGMA: Self-supervised In-context Grounded Map Aggregation},
  author    = {Anonymous},
  booktitle = {NeurIPS 2026 (under review)},
  year      = {2026}
}
```

---

## License

This code is released under the terms attached in `LICENSE` (to be added in
final version). Third-party components retain their own licenses.
