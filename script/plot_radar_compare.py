"""Radar chart comparing StaDy vs SOTA across Pose / Depth / Point Cloud metrics.

Layout: 8 axes grouped into three sectored arcs (Point Cloud on top, Depth on
the right, Camera Pose on the bottom-left). Both group arc labels and metric
axis labels follow the curve. Each method uses a single hue (line + fill).
Worst-performing methods drawn first so the best (Ours) sits on top.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# --- data ------------------------------------------------------------------
# (group, label, direction, theoretical_worst, theoretical_best)
# Per-metric (worst, best) bounds. Tightened around the actual data so the
# spread between methods is visible without making everyone look "perfect".
METRICS = [
    ("Point Cloud", "Accuracy",          "down", 5.0,   2.4),    # top-left
    ("Point Cloud", "Completeness",      "down", 4.2,   1.5),    # top
    ("Point Cloud", "Normal Cons.",      "up",   0.45,  0.80),   # top-right
    ("Depth",       "Abs Rel",           "down", 0.20,  0.10),   # right
    ("Depth",       r"$\delta_{1.25}$",  "up",   0.80,  0.96),   # bottom-right
    ("Pose",        "RRA",               "up",   99.7, 100.0),   # bottom
    ("Pose",        "RTA",               "up",   92.0,  97.0),   # bottom-left
    ("Pose",        "AUC",               "up",   78.0,  92.0),   # left
]

# Per-method values, ordered to match METRICS above
# (Acc, Comp, NC, AbsRel, d1.25, RRA@30, RTA@30, AUC)
DATA = {
    "DA3":         [3.413, 3.151, 0.786, 0.168, 0.851, 100.0, 96.8, 91.2],
    "VGGT4D":      [4.452, 3.843, 0.758, 0.168, 0.832, 100.0, 95.0, 87.1],
    "MapAnything": [2.945, 3.331, 0.751, 0.174, 0.844, 100.0, 93.1, 80.1],
    "VGGT":        [4.924, 3.551, 0.629, 0.163, 0.829, 100.0, 94.6, 85.0],
    r"$\pi^3$":    [3.885, 3.342, 0.495, 0.157, 0.867,  99.9, 93.6, 84.7],
    "Ours":        [2.520, 2.183, 0.790, 0.112, 0.915, 100.0, 96.8, 91.1],
}

COLORS = {
    "DA3":         "#4F94B0",   # mid blue
    "VGGT4D":      "#3FB5C2",   # bright cyan-teal
    "MapAnything": "#4A7050",   # deep green
    "VGGT":        "#7BC0C2",   # light teal-blue
    r"$\pi^3$":    "#1B5A75",   # deep blue
    "Ours":        "#A85F40",   # rust (distinct)
}

# Continuous gradient running along the y = x diagonal: light blue at the
# lower-left end of that line, light green at the upper-right end. Each ring
# segment samples the gradient at its (x, y) midpoint.
# Palette gradient — uses the project's standard light-blue -> mint tones
GRADIENT_LO = "#68BED9"   # palette light blue (lower-left of y=x)
GRADIENT_HI = "#BFDFD2"   # palette mint        (upper-right of y=x)


# --- normalisation: per-metric absolute bounds, mapped to [R_MIN, R_MAX] ---
R_MAX = 0.85
R_MIN = 0.18

methods = list(DATA.keys())
K = len(METRICS)
raw = np.array([DATA[m] for m in methods], dtype=float)        # [M, K]

# t in [0, 1] using each metric's (worst, best) bounds; "down" handled by
# bounds being (worst_high, best_low).
t = np.zeros_like(raw)
for k, (_, _, direction, worst, best) in enumerate(METRICS):
    span_k = best - worst
    if abs(span_k) < 1e-9:
        t[:, k] = 0.5
    else:
        t[:, k] = np.clip((raw[:, k] - worst) / span_k, 0.0, 1.0)

unit = R_MIN + t * (R_MAX - R_MIN)
scores = t.mean(axis=1)
draw_order = np.argsort(-scores)   # best drawn first, worst on top


# --- angle layout ----------------------------------------------------------
# clockwise from top-left so PC sits on top (left -> right)
angles = 3 * np.pi / 4 - np.arange(K) * (2 * np.pi / K)
# normalise to [-pi, pi] just for sanity (not strictly required by polar)
closed_angles = np.concatenate([angles, angles[:1]])

R_DATA = 1.0
R_LABEL = 1.07
R_ARC_INNER = 1.20
R_ARC_OUTER = 1.36
R_ARC_TEXT = (R_ARC_INNER + R_ARC_OUTER) / 2
R_LIM = 1.44

fig = plt.figure(figsize=(7.6, 7.6), dpi=200)
ax = fig.add_subplot(111, projection="polar")
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)
ax.set_ylim(0, R_LIM)
ax.set_yticks([])
ax.set_xticks([])
ax.spines["polar"].set_visible(False)
ax.set_facecolor("none")
fig.patch.set_alpha(0.0)

# spokes only (polygon gridlines removed)
for ang in angles:
    ax.plot([ang, ang], [0, R_DATA],
            color="#888888", lw=1.2, ls=(0, (2.5, 2.5)), zorder=1)

# Ours drawn first (underneath); baselines overlay best -> worst on top.
ours_mi = methods.index("Ours")
ours_vals = np.concatenate([unit[ours_mi], unit[ours_mi][:1]])
ours_color = COLORS["Ours"]
ax.plot(closed_angles, ours_vals, color=ours_color, lw=3.4, zorder=2)
ax.fill(closed_angles, ours_vals, color=ours_color, alpha=0.18, zorder=1.8)

for rank, mi in enumerate(draw_order):
    m = methods[mi]
    if m == "Ours":
        continue
    vals = np.concatenate([unit[mi], unit[mi][:1]])
    color = COLORS[m]
    zo = 3 + rank * 0.1
    ax.plot(closed_angles, vals, color=color, lw=2.4, alpha=0.95, zorder=zo + 0.5)
    ax.fill(closed_angles, vals, color=color, alpha=0.08, zorder=zo)


# --- curved-text helper ----------------------------------------------------
def is_bottom(angle):
    a = (angle + np.pi) % (2 * np.pi) - np.pi
    return np.sin(a) < -0.25


_FONT_CACHE: dict = {}


def _measure_widths(text, fontsize, weight):
    """Return per-character widths (in TextPath font units) for proportional
    spacing. A space character collapses to zero, so substitute a small
    positive width."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    key = (fontsize, weight)
    fp = _FONT_CACHE.get(key)
    if fp is None:
        fp = FontProperties(size=fontsize, weight=weight)
        _FONT_CACHE[key] = fp
    widths = []
    for ch in text:
        if ch == " ":
            widths.append(fontsize * 0.32)
            continue
        tp = TextPath((0, 0), ch, prop=fp)
        ext = tp.get_extents()
        w = ext.width
        if w <= 0:
            w = fontsize * 0.3
        widths.append(w + fontsize * 0.04)   # small kerning padding
    return np.array(widths)


def draw_curved_text(ax, text, mid_angle, r, *, fontsize=11, color="white",
                     weight="bold", scale=None, zorder=8):
    """Render `text` along an arc centred on `mid_angle`, with per-character
    angular widths proportional to each glyph's actual width — so 'C' takes
    more arc length than 'i'."""
    n = len(text)
    widths = _measure_widths(text, fontsize, weight)
    if scale is None:
        # angular units per font-unit; tuned so a 12-pt label sits comfortably
        scale = 0.0062 / max(r, 0.5)
    ang_w = widths * scale
    total = ang_w.sum()
    cum = np.concatenate([[0.0], np.cumsum(ang_w)])
    centers = (cum[:-1] + cum[1:]) / 2.0  # in [0, total]

    on_bot = is_bottom(mid_angle)
    if on_bot:
        chars_a = mid_angle - total / 2 + centers
        rot_offset = +90.0
    else:
        # banner style: place chars from high angle to low angle so the word
        # reads naturally L-to-R when viewed from outside the circle
        chars_a = mid_angle + total / 2 - centers
        rot_offset = -90.0

    for ch, ang in zip(text, chars_a):
        rot = np.degrees(ang) + rot_offset
        ax.text(ang, r, ch, ha="center", va="center",
                fontsize=fontsize, color=color, weight=weight,
                rotation=rot, rotation_mode="anchor",
                zorder=zorder)


# --- group arcs (segmented + diagonal gradient) ---------------------------
import matplotlib.colors as mcolors

GROUPS = [
    ("Point Cloud Estimation", [0, 1, 2]),
    ("Depth Estimation",       [3, 4]),
    ("Camera Pose Estimation", [5, 6, 7]),
]
HALF_STEP = np.pi / K
SEG_GAP = 0.025          # angular gap between arc segments (rad)
N_SEGS = 80              # gradient resolution per arc


def lerp_color(c0, c1, t):
    r0, g0, b0 = mcolors.to_rgb(c0)
    r1, g1, b1 = mcolors.to_rgb(c1)
    return (r0 + (r1 - r0) * t,
            g0 + (g1 - g0) * t,
            b0 + (b1 - b0) * t)


def diag_color(theta, r):
    """Sample the y=x gradient at polar (theta, r). Project the cartesian
    point onto the y=x diagonal and remap to t in [0, 1]."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    proj = (x + y) / np.sqrt(2.0)            # signed distance along y=x
    # arc radius is the same for every sample (same ring), so the projection
    # ranges over [-r, +r]. Normalise to [0, 1].
    t = (proj + r) / (2.0 * r)
    t = float(np.clip(t, 0.0, 1.0))
    return lerp_color(GRADIENT_LO, GRADIENT_HI, t)


for label, idxs in GROUPS:
    a_lo = angles[idxs[-1]] - HALF_STEP + SEG_GAP / 2
    a_hi = angles[idxs[0]]  + HALF_STEP - SEG_GAP / 2
    width = a_hi - a_lo
    center = (a_lo + a_hi) / 2

    seg_w = width / N_SEGS
    centers = a_lo + (np.arange(N_SEGS) + 0.5) * seg_w
    colors = [diag_color(c, R_ARC_TEXT) for c in centers]
    ax.bar(centers, R_ARC_OUTER - R_ARC_INNER,
           width=seg_w * 1.02, bottom=R_ARC_INNER,
           color=colors, edgecolor="none",
           align="center", zorder=6)

    # clean outer/inner edges
    arc_thetas = np.linspace(a_lo, a_hi, 100)
    ax.plot(arc_thetas, [R_ARC_OUTER] * len(arc_thetas),
            color="white", lw=0.8, zorder=6.5)
    ax.plot(arc_thetas, [R_ARC_INNER] * len(arc_thetas),
            color="white", lw=0.8, zorder=6.5)

    draw_curved_text(ax, label, center, R_ARC_TEXT,
                     fontsize=15, color="#1F4F60", weight="bold")


# --- metric (axis) labels — rotated tangent to the circle ------------------
# Single rotated text per label (preserves LaTeX like $\delta_{1.25}$).
for ang, (_, name, *_rest) in zip(angles, METRICS):
    if is_bottom(ang):
        rot = np.degrees(ang) + 90.0
    else:
        rot = np.degrees(ang) - 90.0
    # keep text upright across the side regions
    if rot > 90:
        rot -= 180
    elif rot < -90:
        rot += 180
    ax.text(ang, R_LABEL, name, ha="center", va="center",
            fontsize=13, color="#222222",
            rotation=rot, rotation_mode="anchor", zorder=12)


# --- legend ----------------------------------------------------------------
legend_order = list(np.argsort(-scores))
legend_handles = []
for mi in legend_order:
    m = methods[mi]
    color = COLORS[m]
    lw = 5.5 if m == "Ours" else 4.5
    legend_handles.append(plt.Line2D([0], [0], color=color, lw=lw,
                                     solid_capstyle="round", label=m))
fig.legend(handles=legend_handles, loc="lower center",
           bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False,
           fontsize=14, handlelength=2.0, columnspacing=1.5)

out_pdf = Path("<REPO_ROOT>/_NeurIPS__StaDy___Sigma/src/radar_compare.pdf")
out_png = Path("<REPO_ROOT>/outputs/radar_compare.png")
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.subplots_adjust(left=0.04, right=0.96, top=0.98, bottom=0.10)
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05, transparent=True)
fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, transparent=True)
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")
