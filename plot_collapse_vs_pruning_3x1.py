"""3x1 (stacked, shared x-axis) version of collapse_vs_pruning.png: accuracy /
embedding collapse / discriminative gap, all vs. % of decoder layers pruned
(evenly-spaced/alternating drop pattern, Qwen3-VL-2B, best checkpoint).

Data provenance -- the original collapse_vs_pruning.png was committed as an
image with no surviving generation script (only ephemeral /tmp output), so
values here come from three different confidence levels. Filled markers =
exact, recomputed from raw source files still in this repo. Hollow markers =
approximated by pixel-calibrated reading of the original PNG (matplotlib
gridlines / exact-anchor-point calibration, not OCR or eyeballing -- see
below -- but still not a re-measurement).

- Accuracy: exact for all points except 0% pruned. Source:
  logs/depth_pruning_other_strategies/train_acc_{7,14,18,19,20,21}layer*best*.log
  ("Final train top-1 accuracy", in-batch 4-way proxy metric, best checkpoint
  per depth). The 0% (unpruned, 28-layer) point has no surviving train_acc
  log -- approximated by calibrated pixel-read of pruning_accuracy_v2.png.
- Discriminative gap: exact at 0%/28.6%/32.1%, recomputed directly from the
  raw qry-vs-pos cosine-similarity matrices at
  logs/depth_pruning_other_strategies/sim_matrix_{0,28.6,32.1}pct.npy
  (gap = diag.mean() - offdiag.mean()). The other 4 points (25/35.7/50/75%)
  have no surviving raw matrix -- approximated by pixel-calibrated reading of
  collapse_vs_pruning.png, with the pixel->value fit anchored to the 3 exact
  points above (fit residual < 0.0035).
- Embedding collapse (qry-vs-qry cosine sim): no raw matrix survives for any
  point (only the qry-vs-pos matrices above were preserved). All 7 points are
  approximated by pixel-calibrated reading of collapse_vs_pruning.png, with
  the pixel->value fit anchored to the plot's own gridlines (0.3-0.7 axis
  ticks, detected directly from the image, fit residual < 0.0004).

Bottom line: treat the hollow-marker points as indicative, not re-measured.
If exact numbers are ever needed, only inspect_similarity.py re-runs (which
require retraining the missing per-depth LoRA checkpoints -- ~2h47m/depth on
a single V100, per train_19layer.log) can recover them.
"""
import matplotlib.pyplot as plt

x = [0, 25, 28.6, 32.1, 35.7, 50, 75]

# accuracy: best-checkpoint, in-batch 4-way proxy accuracy (%)
acc = [84.0, 76.0, 71.5, 47.0, 41.5, 40.5, 33.0]
acc_exact = [False, True, True, True, True, True, True]

# embedding collapse: avg cosine sim between different queries' embeddings
collapse = [0.277, 0.359, 0.442, 0.716, 0.773, 0.742, 0.702]
collapse_exact = [False] * 7

# discriminative gap: pos-pair minus neg-pair mean cosine sim
gap = [0.1128, 0.0624, 0.0491, 0.0048, 0.0030, 0.0032, -0.0009]
gap_exact = [True, False, True, True, False, False, False]

fig, axes = plt.subplots(3, 1, figsize=(7.5, 9), sharex=True)

panels = [
    (axes[0], acc, acc_exact, "#1baf7a", "In-batch accuracy\n(4-way, %)", "o"),
    (axes[1], collapse, collapse_exact, "#d62728", "Embedding collapse\n(qry-vs-qry cos. sim)", "o"),
    (axes[2], gap, gap_exact, "#1f77b4", "Discriminative gap\n(pos $-$ neg cos. sim)", "s"),
]

for ax, y, exact, color, ylabel, marker in panels:
    ax.plot(x, y, color=color, linewidth=1.8, zorder=1)
    xs_exact = [xi for xi, e in zip(x, exact) if e]
    ys_exact = [yi for yi, e in zip(y, exact) if e]
    xs_approx = [xi for xi, e in zip(x, exact) if not e]
    ys_approx = [yi for yi, e in zip(y, exact) if not e]
    ax.scatter(xs_exact, ys_exact, marker=marker, s=55, color=color,
               edgecolor="black", linewidth=0.6, zorder=3, label="measured")
    ax.scatter(xs_approx, ys_approx, marker=marker, s=55, facecolor="white",
               edgecolor=color, linewidth=1.6, zorder=3, label="approximated")
    ax.axvspan(28.6, 32.1, color="0.85", zorder=0)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)

axes[0].text((28.6 + 32.1) / 2, axes[0].get_ylim()[1] * 0.9, "cliff",
             ha="center", fontsize=9, color="0.4")
axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
axes[-1].set_xlabel("% of decoder layers pruned (evenly spaced)")
axes[0].set_title(
    "Embedding geometry across evenly-spaced Qwen3-VL-2B depth pruning",
    fontsize=12,
)

fig.text(0.01, 0.005,
         "Hollow markers = approximated (pixel-calibrated reading of the original figure; see script docstring for exact provenance per point).",
         fontsize=7.5, color="0.4", va="bottom")

fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig("figures/depth_pruning/collapse_vs_pruning_3x1.png", dpi=150, bbox_inches="tight")
print("saved figures/depth_pruning/collapse_vs_pruning_3x1.png")
