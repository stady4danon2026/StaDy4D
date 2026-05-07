#!/usr/bin/env python3
"""Render the current SIGMA pipeline diagram (NEW + student head)."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, xy, w, h, label, color, fontsize=10, weight="bold"):
    rect = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                          linewidth=1.4, edgecolor="#333", facecolor=color)
    ax.add_patch(rect)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            weight=weight, wrap=True)


def arrow(ax, p1, p2, label=None, color="#333", style="-|>"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, lw=1.4, color=color)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.12, label, ha="center", fontsize=8.5, style="italic", color="#555")


# =========================================================================
# Figure 1: Inference pipeline (current state)
# =========================================================================
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("SIGMA — current inference pipeline\n"
             "(no learned inpainter; student head replaces GroundedSAM at inference)",
             fontsize=13.5, weight="bold", pad=14)

# Row A — Input + dynamic detection
box(ax, (0.3, 7.8), 2.4, 1.0, "StaDy4D\nrgb.mp4 (origin frames)", "#e6f0ff")
box(ax, (3.4, 7.8), 2.6, 1.0, "★ Student head\n(3.7M-param U-Net,\nGSAM-distilled)", "#ffe0a3")
box(ax, (3.4, 6.4), 2.6, 1.0, "GroundedSAM\n(slower fallback,\n~3.9 s/scene)", "#fff2cc")
box(ax, (6.6, 7.1), 1.8, 0.9, "dynamic\nmask", "#fff2cc")

arrow(ax, (2.7, 8.3), (3.4, 8.3))
arrow(ax, (2.7, 7.95), (3.4, 6.9))
arrow(ax, (6.0, 8.3), (6.6, 7.7), label="< 1 ms / frame")
arrow(ax, (6.0, 6.9), (6.6, 7.3), label="3.9 s / scene")

# Row B — Pi3X reconstruction
box(ax, (9.0, 7.5), 3.0, 1.2, "Pi3X foundation model\n(use_origin_frames=true)\n→ depth, conf, c2w, K, points",
    "#ffe0b3", fontsize=10)
arrow(ax, (2.7, 8.3), (9.0, 8.1), label="origin RGB")

# Row C — post-recon: exclude + aggregate + fill
box(ax, (0.5, 4.3), 3.0, 1.0, "exclude_moving_objects\n→ zero depth/conf in mask",
    "#ffe0e0", fontsize=10)
box(ax, (4.0, 4.3), 3.0, 1.0, "Per-frame static cloud\n→ SceneAggregator", "#fff2cc")
box(ax, (7.5, 4.3), 4.5, 1.2, "★ Multi-view depth fill ★\nfor each frame: project global cloud\nthrough inv(c2w)+K,  z-buffer,\nfill mask-region depth",
    "#ffd6d6", fontsize=10)

arrow(ax, (8.4, 7.5), (8.4, 5.5))
arrow(ax, (7.4, 7.1), (2.0, 5.3), color="#888")
arrow(ax, (3.5, 4.8), (4.0, 4.8))
arrow(ax, (7.0, 4.8), (7.5, 4.8))

# Row D — outputs
box(ax, (0.5, 1.9), 3.6, 1.0, "Per-frame depth\n(filled in mask region)", "#e6f0ff")
box(ax, (4.7, 1.9), 3.6, 1.0, "Global static\npoint cloud + colors", "#e6f0ff")
box(ax, (8.9, 1.9), 3.6, 1.0, "Camera trajectory\n(c2w per frame)", "#e6f0ff")

arrow(ax, (9.5, 4.3), (2.3, 2.9))
arrow(ax, (9.5, 4.3), (6.5, 2.9))
arrow(ax, (9.5, 4.3), (10.7, 2.9))

# Row E — eval
box(ax, (2.0, 0.3), 11.0, 1.0,
    "Evaluator: pose ATE/RPE/AUC  •  depth abs_rel/rmse/a1  •  cloud Acc/Comp/NC",
    "#fffbcc", fontsize=10.5)
arrow(ax, (2.3, 1.9), (4.0, 1.3))
arrow(ax, (6.5, 1.9), (7.5, 1.3))
arrow(ax, (10.7, 1.9), (10.0, 1.3))

# Annotations
ax.text(8.0, 9.5, "Phase: detect dynamics → reconstruct static → fill mask region geometrically",
        ha="center", fontsize=10, style="italic", color="#444")
ax.text(13.5, 7.8, "removed: no\nSDXL/learned\ninpainter",
        ha="center", fontsize=9.5, color="#aa3a3a", weight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff", edgecolor="#aa3a3a"))

# Legend
legend = [
    mpatches.Patch(color="#fff2cc", label="motion / segmentation"),
    mpatches.Patch(color="#ffe0a3", label="★ NEW: student head (this work)"),
    mpatches.Patch(color="#ffe0b3", label="Pi3X foundation model"),
    mpatches.Patch(color="#ffd6d6", label="★ NEW: multi-view geometric fill"),
    mpatches.Patch(color="#ffe0e0", label="post-process"),
    mpatches.Patch(color="#e6f0ff", label="data / output"),
]
ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.05),
          ncol=6, fontsize=9, frameon=False)

fig.tight_layout()
fig.savefig("comparison_vis/pipeline_current.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("Saved comparison_vis/pipeline_current.png")


# =========================================================================
# Figure 2: Training pipeline for the student head (offline, one-time)
# =========================================================================
fig, ax = plt.subplots(figsize=(15, 5.5))
ax.set_xlim(0, 15); ax.set_ylim(0, 6); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Student head training (offline, distilled from GroundedSAM)",
             fontsize=13, weight="bold", pad=12)

box(ax, (0.3, 3.5), 2.6, 1.2, "Train scenes\nTown07 + Town10\n972 (scene, cam) pairs", "#e6f0ff")
box(ax, (3.5, 3.5), 2.6, 1.2, "GroundedSAM\nteacher (frozen)\n→ binary masks", "#fff2cc")
box(ax, (6.7, 3.5), 2.4, 1.2, "Cache\n(rgb_320, mask_320)\n× 26,250 samples", "#e8f5e9")
box(ax, (9.7, 3.5), 2.6, 1.2, "Student U-Net\n(3.7M params)\nrandom init", "#ffe0a3")
box(ax, (12.7, 3.5), 2.0, 1.2, "checkpoints/\nstudent_unet.pt", "#e6f0ff")

arrow(ax, (2.9, 4.1), (3.5, 4.1), label="origin RGB")
arrow(ax, (6.1, 4.1), (6.7, 4.1), label="(rgb, mask)")
arrow(ax, (9.1, 4.1), (9.7, 4.1), label="train batches")
arrow(ax, (12.3, 4.1), (12.7, 4.1), label="best val ckpt")

# Loss + epochs
box(ax, (3.5, 1.0), 8.0, 1.5,
    "Loss = BCE(pos_weight=15) + Dice    Optimizer = AdamW(lr=1e-3)\n"
    "20 epochs × 26,250 samples × bs=16   →   best val IoU = 0.55",
    "#ffd6d6", fontsize=10.5)
arrow(ax, (10.7, 3.5), (10.7, 2.5), label="loss = teacher mimic", color="#aa3a3a")
arrow(ax, (10.7, 1.0), (10.7, 0.4), color="#aa3a3a")
ax.text(10.7, 0.2, "back-prop into student weights only", ha="center",
        fontsize=9.5, color="#aa3a3a", style="italic")

fig.tight_layout()
fig.savefig("comparison_vis/student_training.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("Saved comparison_vis/student_training.png")
