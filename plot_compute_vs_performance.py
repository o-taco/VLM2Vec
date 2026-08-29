"""Compute needed vs. performance, across every NON-linear-probe training run in
logs/ (linear probe lives separately in eval_probe/ and is excluded per request;
the untrained collapse_pretraining_check/ baseline is also excluded -- it's not a
trained model).

Compute proxy (this repo has no per-run FLOP counters or wall-clock timers left
in the logs, so this is a principled estimate, not a measurement):

    compute ~= (model_size_B / total_layers) * (kept_layers + 2 * adapted_layers)
               + head_extra_B

  - kept_layers: decoder layers actually present in the forward pass (depth
    pruning removes layers outright; adapter_coverage_freeze keeps all 28 but
    freezes most of them, so kept_layers=28 there always).
  - adapted_layers: of those, how many get a backward pass (LoRA-trained).
    Standard forward:backward FLOP accounting is ~1:2, so a layer that's
    forward-only costs 1 unit and a layer that's also backpropped costs 3
    (1 forward + 2 backward) -- hence `kept + 2*adapted`.
  - head_extra_B: the small FFN+pooling head's own trainable-param count
    (from "Number of trainable parameters" in the logs), for methods that
    train one (FFN-head-only, LoRA+FFN-head). Negligible at full depth,
    non-negligible at N=3-4 layers where it's most of the trained compute.
  - All 44 runs use identical steps (500) and batch size (64) -- confirmed by
    grepping every sweep driver log -- so this proxy is directly proportional
    to actual relative training compute, up to one shared constant.

Every (series, kept_layers, best_acc/latest_acc) triple below was extracted
from logs/**/*_results.txt and logs/**/*.log by a dedicated pass this session,
cross-checked against this repo's existing plot_*.py scripts for the runs they
already covered. Two things were deliberately left out during that pass:
  - plot_freeze_vs_pruning.py's keepfirst-2B point at 19L/77.0%: no
    corroborating raw log or results.txt row exists for it, and
    plot_pruning_accuracy_v3_with_probe.py's own docstring says this series
    has "NO measured point at 32.1%" -- looks like an undocumented
    interpolation, so it's not reused here.
  - The p1_*.log retrieval-style P@1 accuracy track (chance ~=0.11%, full
    model ~21%): a real but incompatible metric vs. the in-batch 4-way proxy
    (chance=25%) every point below uses. Mixing them would be misleading.
"""
import matplotlib.pyplot as plt
import numpy as np

# --- palette: dataviz skill's validated 8-slot categorical set, one slot per method family ---
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)

# (series, model_size_B, total_layers, label, family_color, marker, linestyle, is_single_point)
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
}

# rows: series -> list of (kept_layers, adapted_layers, head_extra_B, best_acc, latest_acc)
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
    else:
        ax.plot(xs, ys, marker=marker, color=color, linewidth=1.8, markersize=7,
                linestyle=ls, alpha=0.9, label=label)

ax.axhline(25, color="0.5", linestyle=":", linewidth=1.4, zorder=1)
ax.text(0.011, 27, "chance (25%, in-batch 4-way)", fontsize=8.5, color="0.4", va="bottom")

ax.set_xscale("log")
ax.set_xlim(0.006, 30)
ax.set_ylim(0, 100)
ax.set_xlabel(
    "Est. relative training compute, log scale\n"
    "(~ backbone size x [layers run forward + 2x layers backpropped] + trained-head params)",
    fontsize=10,
)
ax.set_ylabel("Best-checkpoint accuracy (in-batch 4-way, %)")
ax.set_title(
    "Qwen3-VL 2B & 8B on A-OKVQA: estimated training compute vs. accuracy\n"
    "every non-linear-probe method family in logs/ (depth pruning x 3 patterns, FFN-head-only,\n"
    "LoRA+FFN-head, adapter-freeze sweep, single-seam ablations)",
    fontsize=12,
)
ax.grid(True, alpha=0.3, which="both")

legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.3,
                    framealpha=0.9, columnspacing=1.1, handletextpad=0.5)

fig.text(0.01, -0.14,
         "Note: compute is an estimate (this repo has no surviving per-run FLOP/wall-clock counters), not a\n"
         "measurement -- see docstring for the exact formula and its assumptions. All 44 runs use identical\n"
         "steps (500) and batch size (64), confirmed across every sweep driver log, so the proxy is directly\n"
         "proportional to relative training compute up to one shared constant. Metric is the in-batch 4-way\n"
         "accuracy proxy used throughout this repo's other figures (chance=25%), not the harder full-pool P@1\n"
         "retrieval metric (chance~0.11%, full 2B model ~21%) computed separately in some p1_*.log files.\n"
         "Excludes eval_probe/ (linear probe, not a fine-tuning run) and collapse_pretraining_check/ (an\n"
         "untrained/random baseline, ~chance with a negative discriminative gap) per the request scope.",
         fontsize=7.8, color="0.4", va="top")

fig.tight_layout()
fig.savefig("figures/depth_pruning/compute_vs_performance.png", dpi=150, bbox_inches="tight")
fig.savefig("figures/depth_pruning/compute_vs_performance.pdf", bbox_inches="tight")
print("saved figures/depth_pruning/compute_vs_performance.{png,pdf}")
