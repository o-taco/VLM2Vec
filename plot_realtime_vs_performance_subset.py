"""REAL wall-clock version of plot_compute_vs_performance_pruning_v3_subset.py.
That earlier chart used an estimated compute proxy (backbone size x layers
forward/backprop) because no one had checked whether real timing survived in
the raw logs. It does, for most rows -- this chart uses actual elapsed
wall-clock time to the best checkpoint, pulled from tqdm progress lines in the
raw training logs (not the *_results.txt summaries, which have no timing).

Two things are NOT here, and can't be added without new data:
  - The shared kept_layers=28 (full, unpruned) point for keepfirst,
    evenly-spaced, and keeplast. No raw log survives for this run in any of
    the three series -- plot_pruning_accuracy_v3_with_probe.py's own docstring
    already flags this ("no raw train_acc log for the full/unpruned...run
    survived on this (recycled) container"), and a fresh grep across every
    log directory this session confirms it's still gone. So each of those
    three lines stops at kept_layers=21 (keepfirst/evenly-spaced) or 27
    (keeplast) instead of reaching the 84-85% full-depth point.
  - The linear probe. It's not gradient-trained (no tqdm loop to time), and
    the purpose-built timing scripts in eval_probe/ (probe_prune_final.py,
    time_linear_probe_qwen3.py) were apparently never run to a saved log --
    no trace of their output anywhere in the repo. Even if they had been run,
    they only cover 2 of the 14 kept_layers conditions used in the probe
    curve elsewhere, not a full sweep. So there is no honest way to put the
    probe on a REAL-timing version of this chart; it's simply left off.

Elapsed time is time-to-BEST-checkpoint (not time-to-step-500) -- i.e. how
long you'd actually have needed to run training to get the reported accuracy,
including the (real) benefit of a run that happened to peak early. This is
why e.g. keeplast's kept_layers=19 point (the keep-mid ablation, which peaked
at step 320 instead of ~400-480 like its neighbors) shows LESS elapsed time
than the 18- and 20-layer points on either side of it -- that's a genuine
early-stopping effect, not noise.

GPU identity is never logged anywhere in this repo (no device name,
CUDA_VISIBLE_DEVICES, or nvidia-smi output survives), so cross-run hardware
consistency can't be directly confirmed. Per-step throughput (s/it) for
matching layer counts agrees within ~1-4% across three independently-launched
sweeps run weeks apart, which is suggestive of the same GPU class throughout
(a GPU swap would typically show a much larger 2-3x gap) -- but this is
inferred, not confirmed, and is noted here rather than asserted as fact.
"""
import matplotlib.pyplot as plt
import numpy as np

PURPLE, BLUE, BROWN, GREEN = "#9467bd", "#1f77b4", "#8c564b", "#2ca02c"

SERIES_META = {
    "keepfirst-2B":       ("Keep-first-N (LoRA)",                PURPLE, "D", False),
    "evenly_spaced-2B":   ("Evenly-spaced / alternating (LoRA)", BLUE,   "o", False),
    "keeplast-2B":        ("Keep-last-N / front-drop (LoRA)",    BROWN,  "s", False),
    "keep26_ablation-2B": ("Single-seam keep-26 ablation",       GREEN,  "*", True),
}

# rows: series -> list of (kept_layers, elapsed_seconds_at_best_step, best_acc)
ROWS = {
    "keepfirst-2B": [
        (7, 5364, 69.0), (14, 5388, 77.5), (18, 6291, 75.0),
        (20, 7936, 77.5), (21, 7841, 80.0),
    ],
    "evenly_spaced-2B": [
        (7, 4988, 33.0), (14, 7643, 40.5), (18, 7436, 41.5),
        (19, 9249, 47.0), (20, 9545, 71.5), (21, 11394, 76.0),
    ],
    "keeplast-2B": [
        (7, 4896, 32.0), (14, 7025, 37.5), (18, 8929, 39.5),
        (19, 6151, 36.0),  # keep-mid ablation, best_step=320 -- peaked early, hence the dip in elapsed time
        (20, 9160, 42.5), (21, 10314, 50.5), (23, 8346, 51.5),
        (25, 9851, 64.5), (26, 9124, 72.0), (27, 9882, 79.0),
    ],
    "keep26_ablation-2B": [(19, 9397, 50.5)],
}

fig, ax = plt.subplots(figsize=(10, 6.5))

for series, rows in ROWS.items():
    label, color, marker, single = SERIES_META[series]
    xs = [s / 3600 for _, s, _ in rows]
    ys = [a for *_, a in rows]
    order = np.argsort(xs)
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    if single:
        ax.scatter(xs, ys, marker=marker, s=220, color=color, edgecolors="white",
                   linewidths=1.2, zorder=6, label=label)
    else:
        ax.plot(xs, ys, marker=marker, color=color, linewidth=2.0, markersize=8,
                alpha=0.95, label=label, zorder=3)

ax.axhline(25, color="0.5", linestyle=":", linewidth=1.4, zorder=1)
ax.text(1.05, 27, "chance (25%)", fontsize=8.5, color="0.4", va="bottom")

ax.set_xlim(1.0, 3.4)
ax.set_ylim(0, 100)
ax.set_xlabel("Real wall-clock time to best checkpoint (hours)")
ax.set_ylabel("Best-checkpoint accuracy (%)")
ax.set_title(
    "Qwen3-VL-2B on A-OKVQA: MEASURED training time vs. accuracy\n"
    "(same series as the compute-proxy subset chart; full-depth points and the linear probe omitted -- no surviving timing data)",
    fontsize=11.5,
)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

fig.tight_layout()
fig.savefig("figures/depth_pruning/realtime_vs_performance_subset.png", dpi=150, bbox_inches="tight")
fig.savefig("figures/depth_pruning/realtime_vs_performance_subset.pdf", bbox_inches="tight")
print("saved figures/depth_pruning/realtime_vs_performance_subset.{png,pdf}")
