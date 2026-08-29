"""Same as plot_compute_vs_performance.py, plus the linear-probe series (kept out
of the base chart per the original request scope, added here as a separate
comparison figure).

Probe data (kept_layers -> real A-OKVQA question_accuracy) is the exact series
from plot_pruning_accuracy_v3_with_probe.py's `probe_depths`/`probe_accs`
(keep-first-N depths, 2B). Two things make the probe's inclusion here an
apples-to-oranges comparison, both called out again in the figure's own note:

  1. Metric mismatch: the probe's accuracy is the real multiple-choice
     `question_accuracy` metric. Every other point on this chart uses the
     in-batch 4-way accuracy proxy. Both happen to have chance=25%, but they
     are not the same measurement -- treat the probe's vertical position as
     indicative, not a precise apples-to-apples y-value.
  2. Compute UNDER-statement: the probe's x here reuses this chart's shared
     formula (forward-only, adapted=0, same as FFN-head-only) so its layer
     term is directly comparable to FFN-head-only at the same kept_layers.
     But unlike every other series, the probe isn't 500 iterative training
     steps at all -- it's ~1 forward pass over the training set followed by a
     closed-form/few-iteration linear fit, with zero gradient steps through
     the backbone. This formula has no notion of "iteration count" (it's a
     shared constant for every other series, since they're all 500 steps), so
     it cannot capture that saving. The probe's true compute is almost
     certainly even lower, relative to everything else, than plotted here --
     treat its x-position as a conservative (upper-bound) estimate.
"""
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
PROBE_COLOR = "#17becf"  # matches this repo's existing probe convention (pruning_accuracy_v3_with_probe.py)

SERIES_META = {
    "keepfirst-2B":              (2, 28, "Keep-first-N (LoRA), 2B",                    BLUE,    "o", "-",  False),
    "keepfirst-8B":               (8, 36, "Keep-first-N (LoRA), 8B",                    BLUE,    "s", "-",  False),
    "keeplast-2B":                (2, 28, "Keep-last-N / front-drop (LoRA), 2B",        YELLOW,  "o", "-",  False),
    "evenly_spaced-2B":           (2, 28, "Evenly-spaced / alternating (LoRA), 2B",      VIOLET,  "o", "-",  False),
    "spread-2B":                  (2, 28, "Spread, non-contiguous (LoRA), 2B",          MAGENTA, "o", "-",  False),
    "spread-8B":                  (8, 36, "Spread, non-contiguous (LoRA), 8B",          MAGENTA, "s", "-",  False),
    "adapter_coverage_freeze-2B": (2, 28, "Adapter-coverage freeze sweep (LoRA), 2B",    GREEN,   "o", "-",  False),
    "ffnhead_ablation-2B":        (2, 28, "FFN-head-only, frozen backbone, 2B",         ORANGE,  "o", "-",  False),
    "ffnhead_ablation-8B":        (8, 36, "FFN-head-only, frozen backbone, 8B",         ORANGE,  "s", "-",  False),
    "ffnhead_capacity_control-8B": (8, 36, "FFN-head-only, param-matched head, 8B",     ORANGE,  "s", "--", False),
    "lora_full-2B":               (2, 28, "Full-depth LoRA baseline, 2B",               RED,     "*", "",   True),
    "keepmid_ablation-2B":        (2, 28, "Single-seam keep-mid ablation, 2B",          RED,     "*", "",   True),
    "keep26_ablation-2B":         (2, 28, "Single-seam keep-26 ablation, 2B",           RED,     "*", "",   True),
    "ffnhead_full-2B":            (2, 28, "FFN-head-only, full depth, 2B",              ORANGE,  "*", "",   True),
    "lora_ffnhead_full-2B":       (2, 28, "LoRA + FFN-head, full depth, 2B",            AQUA,    "*", "",   True),
    "linear_probe-2B":            (2, 28, "Linear probe, keep-first-N, 2B (real question_accuracy)", PROBE_COLOR, "^", "-", False),
}

ROWS = {
    "lora_full-2B": [(28, 28, 0.0, 84.0, 85.0)],
    "keepfirst-2B": [
        (4, 4, 0.0, 34.0, 33.5), (5, 5, 0.0, 39.0, 37.5), (6, 6, 0.0, 59.5, 58.0),
        (7, 7, 0.0, 69.0, 68.5), (14, 14, 0.0, 77.5, 76.0), (18, 18, 0.0, 75.0, 75.0),
        (20, 20, 0.0, 77.5, 75.5), (21, 21, 0.0, 80.0, 77.5), (28, 28, 0.0, 84.0, 85.0),
    ],
    "keepfirst-8B": [
        (3, 3, 0.0, 29.5, 29.0), (4, 4, 0.0, 33.0, 31.5), (5, 5, 0.0, 42.5, 42.5),
        (6, 6, 0.0, 51.5, 51.0), (7, 7, 0.0, 62.5, 61.5), (8, 8, 0.0, 65.0, 65.0),
        (9, 9, 0.0, 70.5, 70.5),
    ],
    "keeplast-2B": [
        (7, 7, 0.0, 32.0, 30.5), (14, 14, 0.0, 37.5, 37.0), (18, 18, 0.0, 39.5, 38.0),
        (20, 20, 0.0, 42.5, 42.0), (21, 21, 0.0, 50.5, 50.5), (23, 23, 0.0, 51.5, 51.5),
        (25, 25, 0.0, 64.5, 64.5), (26, 26, 0.0, 72.0, 72.0), (27, 27, 0.0, 79.0, 76.5),
    ],
    "evenly_spaced-2B": [
        (7, 7, 0.0, 33.0, 36.0), (14, 14, 0.0, 40.5, 39.0), (18, 18, 0.0, 41.5, 41.0),
        (19, 19, 0.0, 47.0, 48.0), (20, 20, 0.0, 71.5, 72.0), (21, 21, 0.0, 76.0, 76.0),
    ],
    "spread-2B": [(4, 4, 0.0, 37.5, 35.0), (6, 6, 0.0, 28.0, 27.0), (7, 7, 0.0, 37.0, 36.0)],
    "spread-8B": [(4, 4, 0.0, 29.5, 26.5), (6, 6, 0.0, 35.0, 30.5), (9, 9, 0.0, 31.0, 30.5)],
    "adapter_coverage_freeze-2B": [
        (28, 7, 0.0, 65.0, 64.0), (28, 14, 0.0, 77.0, 76.0), (28, 21, 0.0, 82.0, 81.5),
        (28, 28, 0.0, 86.0, 86.0),
    ],
    "ffnhead_ablation-2B": [
        (4, 0, 0.0083927, 49.0, 47.0), (5, 0, 0.0083927, 47.0, 45.0),
        (6, 0, 0.0083927, 55.5, 55.5), (7, 0, 0.0083927, 58.5, 54.5),
    ],
    "ffnhead_ablation-8B": [
        (3, 0, 0.033562624, 49.0, 48.5), (4, 0, 0.033562624, 55.0, 54.5),
        (6, 0, 0.033562624, 54.0, 52.5), (7, 0, 0.033562624, 64.0, 63.0),
        (9, 0, 0.033562624, 73.5, 72.0),
    ],
    "ffnhead_capacity_control-8B": [
        (3, 0, 0.008393728, 36.0, 28.5), (6, 0, 0.008393728, 33.0, 32.5),
        (7, 0, 0.008393728, 55.5, 54.0), (9, 0, 0.008393728, 62.0, 61.5),
    ],
    "keepmid_ablation-2B": [(19, 19, 0.0, 36.0, 32.0)],
    "keep26_ablation-2B": [(19, 19, 0.0, 50.5, 52.0)],
    "ffnhead_full-2B": [(28, 0, 0.0083927, 68.5, 67.0)],
    "lora_ffnhead_full-2B": [(28, 28, 0.0083927, 88.0, 84.5)],
    # linear probe: forward-only (adapted=0), tiny classifier head (hidden 2048 x 4 classes ~ 8,192 params).
    # best_acc == latest_acc since a probe is a single closed-form fit, not a checkpoint sweep.
    "linear_probe-2B": [
        (2, 0, 0.0000082, 32.0, 32.0),
        (4, 0, 0.0000082, 44.0, 44.0), (6, 0, 0.0000082, 46.67, 46.67),
        (8, 0, 0.0000082, 63.33, 63.33), (10, 0, 0.0000082, 66.67, 66.67),
        (12, 0, 0.0000082, 79.33, 79.33), (14, 0, 0.0000082, 83.33, 83.33),
        (16, 0, 0.0000082, 83.33, 83.33), (18, 0, 0.0000082, 82.0, 82.0),
        (20, 0, 0.0000082, 80.0, 80.0), (22, 0, 0.0000082, 81.33, 81.33),
        (24, 0, 0.0000082, 80.0, 80.0), (26, 0, 0.0000082, 81.33, 81.33),
        (28, 0, 0.0000082, 79.33, 79.33),
    ],
}


def compute_units(model_size_B, total_layers, kept, adapted, head_extra_B):
    per_layer = model_size_B / total_layers
    return per_layer * (kept + 2 * adapted) + head_extra_B


fig, ax = plt.subplots(figsize=(12, 7.5))

for series, rows in ROWS.items():
    model_size_B, total_layers, label, color, marker, ls, single = SERIES_META[series]
    xs = [compute_units(model_size_B, total_layers, k, a, h) for k, a, h, _, _ in rows]
    ys = [b for *_, b, _ in rows]
    order = np.argsort(xs)
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    if single:
        ax.scatter(xs, ys, marker=marker, s=220, color=color, edgecolors="white",
                   linewidths=1.2, zorder=6, label=label)
    elif series == "linear_probe-2B":
        ax.plot(xs, ys, marker=marker, color=color, linewidth=2.4, markersize=8,
                linestyle=ls, alpha=0.95, zorder=7, label=label)
    else:
        ax.plot(xs, ys, marker=marker, color=color, linewidth=1.8, markersize=7,
                linestyle=ls, alpha=0.9, label=label)

ax.axhline(25, color="0.5", linestyle=":", linewidth=1.4, zorder=1)
ax.text(0.012, 27, "chance (25%)", fontsize=8.5, color="0.4", va="bottom")

ax.set_xscale("log")
ax.set_xlim(0.01, 30)
ax.set_ylim(0, 100)
ax.set_xlabel(
    "Est. relative training compute, log scale\n"
    "(~ backbone size x [layers run forward + 2x layers backpropped] + trained-head params)",
    fontsize=10,
)
ax.set_ylabel("Best-checkpoint accuracy (%, see note on metric mismatch)")
ax.set_title(
    "Qwen3-VL 2B & 8B on A-OKVQA: estimated training compute vs. accuracy,\n"
    "with the linear-probe floor added",
    fontsize=12,
)
ax.grid(True, alpha=0.3, which="both")

legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.3,
                    framealpha=0.9, columnspacing=1.1, handletextpad=0.5)

fig.text(0.01, -0.14,
         "Note: compute is an estimate, not a measurement (see docstring). All gradient-trained runs use\n"
         "identical steps (500) and batch size (64). The linear probe (teal) is NOT one of them -- it's a\n"
         "single forward pass over the training set plus a closed-form/few-iteration linear fit, zero\n"
         "backward passes through the backbone, so its plotted x is a conservative (upper-bound) estimate;\n"
         "its true compute cost relative to everything else is almost certainly even lower than shown.\n"
         "The probe's accuracy is also a different metric -- real question_accuracy vs. the in-batch 4-way\n"
         "proxy every other point uses (both chance=25%, but not the same measurement) -- so read its\n"
         "vertical position as indicative, not a precise apples-to-apples comparison to the other curves.\n"
         "Excludes eval_probe/'s other configs (this is the keep-first-N probe sweep only) and\n"
         "collapse_pretraining_check/ (an untrained/random baseline).",
         fontsize=7.8, color="0.4", va="top")

fig.tight_layout()
fig.savefig("figures/depth_pruning/compute_vs_performance_with_probe.png", dpi=150, bbox_inches="tight")
fig.savefig("figures/depth_pruning/compute_vs_performance_with_probe.pdf", bbox_inches="tight")
print("saved figures/depth_pruning/compute_vs_performance_with_probe.{png,pdf}")
