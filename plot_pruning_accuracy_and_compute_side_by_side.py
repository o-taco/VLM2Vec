"""Side-by-side combination of the two panels meant to sit together in the
paper: figures/depth_pruning/pruning_accuracy_v3_with_probe.png (left) and
figures/depth_pruning/compute_vs_performance_pruning_v3_subset.png (right).

This is a single matplotlib figure with two subplots sharing a y-axis (both
plot accuracy on 0-100), rather than two separately-rendered images pasted
together -- so it's a real vector-quality figure (the .pdf is native, not a
raster paste-up) and the shared axis is enforced by matplotlib (sharey=True)
rather than by matching margins by hand across two scripts.

All data/formatting is copied verbatim from the two source scripts:
- plot_pruning_accuracy_v3_with_probe.py (left panel)
- plot_compute_vs_performance_pruning_v3_subset.py (right panel)
Edit those scripts (and their own standalone output files) first if the
underlying data changes, then mirror the change here.
"""
import matplotlib.pyplot as plt
import numpy as np

# ============================== left panel data ==============================
x0_best, x0_latest = 84.0, 85.0

alt_x = [0, 25, 28.6, 32.1, 35.7, 50, 75]
alt_best = [x0_best, 76.0, 71.5, 47.0, 41.5, 40.5, 33.0]
alt_latest = [x0_latest, 76.0, 72.0, 48.0, 41.0, 39.0, 36.0]

kf_x = [0, 25, 28.6, 35.7, 50, 75]
kf_best = [x0_best, 80.0, 77.5, 75.0, 77.5, 69.0]
kf_latest = [x0_latest, 77.5, 75.5, 75.0, 76.0, 68.5]

kl_x = [0, 3.6, 7.1, 10.7, 17.9, 25, 28.6, 32.1, 35.7, 50, 75]
kl_best = [x0_best, 79.0, 72.0, 64.5, 51.5, 50.5, 42.5, 36.0, 39.5, 37.5, 32.0]
kl_latest = [x0_latest, 76.5, 72.0, 64.5, 51.5, 50.5, 42.0, 32.0, 38.0, 37.0, 30.5]

probe_depths = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
probe_accs = [0.32666666666666666, 0.32, 0.44, 0.4666666666666667, 0.6333333333333333,
              0.6666666666666666, 0.7933333333333333, 0.8333333333333334, 0.8333333333333334,
              0.82, 0.8, 0.8133333333333334, 0.8, 0.8133333333333334, 0.7933333333333333]
probe_x = [(28 - d) / 28 * 100 for d in probe_depths]
probe_y = [a * 100 for a in probe_accs]
probe_x, probe_y = zip(*sorted(zip(probe_x, probe_y)))

# ============================== right panel data ==============================
PURPLE, BLUE, BROWN, GREEN, CYAN = "#9467bd", "#1f77b4", "#8c564b", "#2ca02c", "#17becf"

SERIES_META = {
    "keepfirst-2B":     ("Keep-first-N (LoRA)",                 PURPLE, "D", False),
    "evenly_spaced-2B": ("Evenly-spaced / alternating (LoRA)",  BLUE,   "o", False),
    "keeplast-2B":      ("Keep-last-N / front-drop (LoRA)",     BROWN,  "s", False),
    "linear_probe-2B":  ("Linear probe, keep-first-N (real acc.)", CYAN, "^", False),
}

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
        (19, 19, 0.0, 36.0),
        (20, 20, 0.0, 42.5), (21, 21, 0.0, 50.5), (23, 23, 0.0, 51.5),
        (25, 25, 0.0, 64.5), (26, 26, 0.0, 72.0), (27, 27, 0.0, 79.0), (28, 28, 0.0, 84.0),
    ],
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


# ============================== figure ==============================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6.5), sharey=True)

# --- left panel: accuracy vs. % pruned ---
ax1.plot(alt_x, alt_best, marker="o", color="#1f77b4", linewidth=2, markersize=7,
         label="Evenly spaced, best ckpt.")
ax1.plot(alt_x, alt_latest, marker="o", color="#ff7f0e", linewidth=2, markersize=7,
         label="Evenly spaced, latest ckpt.")
ax1.plot(kf_x, kf_best, marker="D", color="#9467bd", linewidth=2, markersize=8,
          label="Keep-first-N, best ckpt.")
ax1.plot(kf_x, kf_latest, marker="D", color="#9467bd", linewidth=2, markersize=7,
         linestyle="--", alpha=0.85, label="Keep-first-N, latest ckpt.")
ax1.plot(probe_x, probe_y, marker="^", color="#17becf", linewidth=2, markersize=7,
         label="Keep-first-N, linear probe (real acc.)")
ax1.plot(kl_x, kl_best, marker="s", color="#8c564b", linewidth=2, markersize=8,
         label="Keep-last-N (front-drop), best ckpt.")
ax1.plot(kl_x, kl_latest, marker="s", color="#8c564b", linewidth=2, markersize=7,
         linestyle="--", alpha=0.85, label="Keep-last-N (front-drop), latest ckpt.")

ax1.axhline(25, color="0.5", linestyle=":", linewidth=1.5, label="Chance (25%)")

ax1.annotate(f"{x0_best:.1f}%", (0, x0_best), textcoords="offset points", xytext=(-8, 14),
             ha="right", fontsize=9, color="#1f77b4", fontweight="bold")
ax1.annotate(f"{x0_latest:.1f}%", (0, x0_latest), textcoords="offset points", xytext=(-8, -4),
             ha="right", fontsize=9, color="#ff7f0e", fontweight="bold")

probe_peak_i = max(range(len(probe_y)), key=lambda i: probe_y[i])
ax1.annotate(f"{probe_y[probe_peak_i]:.1f}%", (probe_x[probe_peak_i], probe_y[probe_peak_i]),
             textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9,
             color="#17becf", fontweight="bold")

ax1.annotate(f"{kl_best[-1]:.1f}%", (kl_x[-1], kl_best[-1]), textcoords="offset points",
             xytext=(8, 6), ha="left", fontsize=9, color="#8c564b", fontweight="bold")
ax1.annotate(f"{kl_latest[-1]:.1f}%", (kl_x[-1], kl_latest[-1]), textcoords="offset points",
             xytext=(8, -14), ha="left", fontsize=9, color="#8c564b", fontweight="bold")

ax1.set_xlim(-3, 83)
ax1.set_ylim(0, 100)
ax1.set_xlabel("% of decoder layers pruned")
ax1.set_ylabel("Accuracy (%)")
legend1 = ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=8.2,
                      framealpha=0.9, columnspacing=1.1, handletextpad=0.5)
ax1.grid(True, alpha=0.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# --- right panel: accuracy vs. relative training compute ---
for series, rows in ROWS.items():
    label, color, marker, single = SERIES_META[series]
    xs = [compute_units(k, a, h) for k, a, h, _ in rows]
    ys = [b for *_, b in rows]
    order = np.argsort(xs)
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    if single:
        ax2.scatter(xs, ys, marker=marker, s=220, color=color, edgecolors="white",
                    linewidths=1.2, zorder=6, label=label)
    else:
        lw = 2.4 if series == "linear_probe-2B" else 2.0
        ax2.plot(xs, ys, marker=marker, color=color, linewidth=lw, markersize=8,
                  alpha=0.95, label=label, zorder=5 if series == "linear_probe-2B" else 3)

ax2.axhline(25, color="0.5", linestyle=":", linewidth=1.4, zorder=1)
ax2.text(0.11, 27, "chance (25%)", fontsize=8.5, color="0.4", va="bottom")

ax2.set_xscale("log")
ax2.set_xlim(0.1, 10)
ax2.set_xlabel(
    "Relative training compute, log scale\n"
    "(~ backbone size x [layers run forward +\n"
    "2x layers backpropped] + trained-head params)",
    fontsize=8.5,
)
ax2.tick_params(axis="y", labelleft=False)
ax2.grid(True, alpha=0.3, which="both")
legend2 = ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=8.2,
                      framealpha=0.9, columnspacing=1.1, handletextpad=0.5)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.40, wspace=0.06)

fig.savefig("figures/depth_pruning/pruning_accuracy_and_compute_side_by_side.png", dpi=150)
fig.savefig("figures/depth_pruning/pruning_accuracy_and_compute_side_by_side.pdf")
print("saved figures/depth_pruning/pruning_accuracy_and_compute_side_by_side.{png,pdf}")
