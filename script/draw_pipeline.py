#!/usr/bin/env python3
"""Render the SIGMA pipeline diagram (NEW config) as a PNG."""

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


def arrow(ax, p1, p2, label=None, color="#333"):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                        lw=1.4, color=color)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.10, label, ha="center", fontsize=8.5,
                style="italic", color="#555")


fig, ax = plt.subplots(figsize=(15, 7.2))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8)
ax.set_aspect("equal")
ax.axis("off")

ax.set_title("SIGMA Pipeline — NEW config (multi-view depth fill, no learned inpainter)",
             fontsize=13, weight="bold", pad=14)

# Row 1: input
box(ax, (0.3, 6.0), 2.4, 0.9, "StaDy4D\nrgb.mp4 + masks GT", "#e6f0ff")

# Row 1: motion
box(ax, (3.4, 6.0), 2.4, 0.9, "Motion\nGroundedSAM\n(per-frame mask)", "#fff2cc")

# Row 1: inpainting (now trivial)
box(ax, (6.5, 6.0), 2.4, 0.9, "Inpainting\nBlankInpainter\n(zeros mask region)", "#e8f5e9")

# Row 1: Pi3X model
box(ax, (9.6, 6.0), 2.4, 0.9, "Pi3X reconstructor\n(deferred / batch)", "#ffe0b3")

# Row 1: Per-frame outputs
box(ax, (12.7, 6.0), 2.1, 0.9, "Per-frame\ndepth, K, c2w,\npoint cloud", "#ffd6d6")

arrow(ax, (2.7, 6.45), (3.4, 6.45))
arrow(ax, (5.8, 6.45), (6.5, 6.45), label="origin RGB + mask")
arrow(ax, (8.9, 6.45), (9.6, 6.45), label="use_origin_frames=true")
arrow(ax, (12.0, 6.45), (12.7, 6.45))

# Row 2: post-processing inside Pi3 finalize()
box(ax, (1.8, 4.0), 2.6, 1.0, "exclude_moving_objects\n→ zero depth/conf\nin mask region", "#ffe0e0")
box(ax, (5.0, 4.0), 2.4, 1.0, "Per-frame static\npoint cloud\n(non-mask only)", "#fff2cc")
box(ax, (8.0, 4.0), 2.6, 1.0, "SceneAggregator\nmerge to global cloud", "#e8f5e9")
box(ax, (11.2, 4.0), 3.0, 1.0,
    "★ Multi-view depth fill ★\nproject global cloud through\ninv(c2w) + K of each frame,\nz-buffer fill mask-region depth",
    "#ffd6d6", fontsize=9)

# Connect row 1 → row 2
arrow(ax, (13.7, 6.0), (12.7, 5.0), color="#888")
arrow(ax, (13.7, 5.0), (10.6, 4.5))
arrow(ax, (8.0, 4.5), (7.4, 4.5))
arrow(ax, (5.0, 4.5), (4.4, 4.5))
arrow(ax, (1.8, 4.5), (1.8, 4.5))  # placeholder

arrow(ax, (4.4, 4.5), (5.0, 4.5))
arrow(ax, (7.4, 4.5), (8.0, 4.5))
arrow(ax, (10.6, 4.5), (11.2, 4.5))

# Row 3: outputs
box(ax, (3.5, 1.6), 3.0, 0.9, "Per-frame depth maps\n(filled in mask region)", "#e6f0ff")
box(ax, (7.5, 1.6), 3.0, 0.9, "Global static\npoint cloud + colors", "#e6f0ff")
box(ax, (11.5, 1.6), 3.0, 0.9, "Camera trajectory\n(c2w per frame)", "#e6f0ff")

arrow(ax, (12.7, 4.0), (5.0, 2.5))
arrow(ax, (9.5, 4.0), (9.0, 2.5))
arrow(ax, (12.7, 4.0), (13.0, 2.5))

# Row 4: eval
box(ax, (3.0, 0.1), 9.0, 0.9,
    "Evaluator: pose ATE/RPE  •  depth abs_rel/rmse/a1  •  cloud Acc/Comp/NC",
    "#fffbcc", fontsize=10)
arrow(ax, (5.0, 1.6), (5.0, 1.0))
arrow(ax, (9.0, 1.6), (9.0, 1.0))
arrow(ax, (13.0, 1.6), (10.0, 1.0))

# Annotations
ax.text(7.5, 5.55, "Phase 1-4: feed-forward, no inpainting model",
        ha="center", fontsize=9, style="italic", color="#444")
ax.text(7.5, 3.45, "Phase 5: Pi3X output → aggregate → fill mask-region depth from global cloud",
        ha="center", fontsize=9, style="italic", color="#444")

# Legend
legend_items = [
    mpatches.Patch(color="#fff2cc", label="motion / segmentation"),
    mpatches.Patch(color="#e8f5e9", label="model-free / cheap"),
    mpatches.Patch(color="#ffe0b3", label="Pi3X foundation model"),
    mpatches.Patch(color="#ffd6d6", label="NEW geometric fill"),
    mpatches.Patch(color="#e6f0ff", label="data / output"),
]
ax.legend(handles=legend_items, loc="lower left", bbox_to_anchor=(0.0, -0.05),
          ncol=5, fontsize=9, frameon=False)

fig.tight_layout()
fig.savefig("comparison_vis/pipeline_diagram.png", dpi=130, bbox_inches="tight")
print("Saved comparison_vis/pipeline_diagram.png")
