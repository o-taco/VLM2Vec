"""Same as plot_collapse_vs_pruning_3x1.py but with no legend and uniform
(filled) markers throughout -- see that script's docstring for full data
provenance (which points are exact vs. pixel-calibrated approximations).
"""
import matplotlib.pyplot as plt

x = [0, 25, 28.6, 32.1, 35.7, 50, 75]

acc = [84.0, 76.0, 71.5, 47.0, 41.5, 40.5, 33.0]
collapse = [0.277, 0.359, 0.442, 0.716, 0.773, 0.742, 0.702]
gap = [0.1128, 0.0624, 0.0491, 0.0048, 0.0030, 0.0032, -0.0009]

fig, axes = plt.subplots(3, 1, figsize=(7.5, 9), sharex=True)

panels = [
    (axes[0], acc, "#1baf7a", "In-batch accuracy\n(4-way, %)", "o"),
    (axes[1], collapse, "#d62728", "Embedding collapse\n(qry-vs-qry cos. sim)", "o"),
    (axes[2], gap, "#1f77b4", "Discriminative gap\n(pos $-$ neg cos. sim)", "s"),
]

for ax, y, color, ylabel, marker in panels:
    ax.plot(x, y, color=color, linewidth=1.8, zorder=1)
    ax.scatter(x, y, marker=marker, s=55, color=color,
               edgecolor="black", linewidth=0.6, zorder=3)
    ax.axvspan(28.6, 32.1, color="0.85", zorder=0)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)

axes[0].text((28.6 + 32.1) / 2, axes[0].get_ylim()[1] * 0.9, "cliff",
             ha="center", fontsize=9, color="0.4")
axes[-1].set_xlabel("% of decoder layers pruned (evenly spaced)")
axes[0].set_title(
    "Embedding geometry across evenly-spaced Qwen3-VL-2B depth pruning",
    fontsize=12,
)

fig.tight_layout()
fig.savefig("figures/depth_pruning/collapse_vs_pruning_3x1_clean.png", dpi=150, bbox_inches="tight")
print("saved figures/depth_pruning/collapse_vs_pruning_3x1_clean.png")
