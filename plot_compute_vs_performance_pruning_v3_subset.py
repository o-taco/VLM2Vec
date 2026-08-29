"""Same compute-vs-performance framing as plot_compute_vs_performance_with_probe.py,
but subset to ONLY the series that actually appear in the original
figures/depth_pruning/pruning_accuracy_v3_with_probe.png (plot_pruning_accuracy_v3_with_probe.py):
evenly-spaced/alternating, keep-first-N, keep-last-N (front-drop, which folds in
the keep-mid single-seam ablation at its 19-layer point per that script's own
note), the keep-26 single-seam ablation, and the linear probe. Everything else
this session added (8B runs, spread pruning, adapter-freeze sweep, FFN-head-only,
LoRA+FFN-head) is deliberately left out here -- this is a re-axis of that one
figure (%-pruned -> estimated compute), not a new sweep.

Only "best checkpoint" values are used (matching this chart family's earlier
simplification), not the original figure's separate best/latest lines.

The linear probe's 0-kept-layers point is dropped per prior feedback on this
chart (it's off in its own decade and isn't informative) even though the
source figure includes it.

Values are copied verbatim from plot_pruning_accuracy_v3_with_probe.py's own
arrays (alt_best, kf_best, kl_best, keep26_best, probe_x/probe_y), converting
its x-axis (% of decoder layers pruned) back to kept_layers = 28 * (1 - pct/100).
"""
import matplotlib.pyplot as plt
import numpy as np

# colors match the original figure's own palette, for visual continuity
PURPLE, BLUE, BROWN, GREEN, CYAN = "#9467bd", "#1f77b4", "#8c564b", "#2ca02c", "#17becf"

SERIES_META = {
    "keepfirst-2B":     ("Keep-first-N (LoRA)",                 PURPLE, "D", False),
    "evenly_spaced-2B": ("Evenly-spaced / alternating (LoRA)",  BLUE,   "o", False),
    "keeplast-2B":      ("Keep-last-N / front-drop (LoRA)",     BROWN,  "s", False),
    "keep26_ablation-2B": ("Single-seam keep-26 ablation",      GREEN,  "*", True),
    "linear_probe-2B":  ("Linear probe, keep-first-N (real acc.)", CYAN, "^", False),
}

# rows: series -> list of (kept_layers, adapted_layers, head_extra_B, best_acc)
ROWS = {
    "keepfirst-2B": [
        (7, 7, 0.0, 69.0), (14, 14, 0.0, 77.5), (18, 18, 0.0, 75.0),
        (20, 20, 0.0, 77.5), (21, 21, 0.0, 80.0), (28, 28, 0.0, 84.0),
    ],
    "evenly_spaced-2B": [
        (7, 7, 0.0, 33.0), (14, 14, 0.0, 40.5), (18, 18, 0.0, 41.5),
        (19, 19, 0.0, 47.0), (20, 20, 0.0, 71.5), (21, 21, 0.0, 76.0), (28, 28, 0.0, 84.0),
    ],
    "keeplast-2B": [
        (7, 7, 0.0, 32.0), (14, 14, 0.0, 37.5), (18, 18, 0.0, 39.5),
        (19, 19, 0.0, 36.0),  # the keep-mid single-seam ablation, folded in here per the source figure
        (20, 20, 0.0, 42.5), (21, 21, 0.0, 50.5), (23, 23, 0.0, 51.5),
        (25, 25, 0.0, 64.5), (26, 26, 0.0, 72.0), (27, 27, 0.0, 79.0), (28, 28, 0.0, 84.0),
    ],
    "keep26_ablation-2B": [(19, 19, 0.0, 50.5)],
    "linear_probe-2B": [
        (2, 0, 0.0000082, 32.0), (4, 0, 0.0000082, 44.0), (6, 0, 0.0000082, 46.67),
        (8, 0, 0.0000082, 63.33), (10, 0, 0.0000082, 66.67), (12, 0, 0.0000082, 79.33),
        (14, 0, 0.0000082, 83.33), (16, 0, 0.0000082, 83.33), (18, 0, 0.0000082, 82.0),
        (20, 0, 0.0000082, 80.0), (22, 0, 0.0000082, 81.33), (24, 0, 0.0000082, 80.0),
        (26, 0, 0.0000082, 81.33), (28, 0, 0.0000082, 79.33),
    ],
}

MODEL_SIZE_B, TOTAL_LAYERS = 2, 28


def compute_units(kept, adapted, head_extra_B):
    per_layer = MODEL_SIZE_B / TOTAL_LAYERS
    return per_layer * (kept + 2 * adapted) + head_extra_B


fig, ax = plt.subplots(figsize=(5.5, 6.5))

for series, rows in ROWS.items():
    label, color, marker, single = SERIES_META[series]
    xs = [compute_units(k, a, h) for k, a, h, _ in rows]
    ys = [b for *_, b in rows]
    order = np.argsort(xs)
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    if single:
        ax.scatter(xs, ys, marker=marker, s=220, color=color, edgecolors="white",
                   linewidths=1.2, zorder=6, label=label)
    else:
        lw = 2.4 if series == "linear_probe-2B" else 2.0
        ax.plot(xs, ys, marker=marker, color=color, linewidth=lw, markersize=8,
                alpha=0.95, label=label, zorder=5 if series == "linear_probe-2B" else 3)

ax.axhline(25, color="0.5", linestyle=":", linewidth=1.4, zorder=1)
ax.text(0.11, 27, "chance (25%)", fontsize=8.5, color="0.4", va="bottom")

ax.set_xscale("log")
ax.set_xlim(0.1, 10)
ax.set_ylim(0, 100)
ax.set_xlabel(
    "Relative training compute, log scale\n"
    "(~ backbone size x [layers run forward +\n"
    "2x layers backpropped] + trained-head params)",
    fontsize=8.5,
)
ax.set_ylabel("")
ax.tick_params(axis="y", labelleft=False)
ax.grid(True, alpha=0.3, which="both")
legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=8.2,
                    framealpha=0.9, columnspacing=1.1, handletextpad=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.subplots_adjust(left=0.05, right=0.97, top=0.96, bottom=0.40)
fig.savefig("figures/depth_pruning/compute_vs_performance_pruning_v3_subset.png", dpi=150)
fig.savefig("figures/depth_pruning/compute_vs_performance_pruning_v3_subset.pdf")
print("saved figures/depth_pruning/compute_vs_performance_pruning_v3_subset.{png,pdf}")
