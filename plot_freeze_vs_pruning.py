"""Compares the adapter-coverage (freeze-fraction) sweep against the existing
depth-pruning curves. Freeze-fraction data, the keep-first-N + LoRA curve, and the
keep-first-N + linear-probe curve are all exact:
- logs/adapter_coverage_freeze/freeze_sweep_results.txt
- logs/depth_pruning_keepfirst/keepfirst_sweep_results.txt
- teammate's ffn_aokvqa_history.json-derived depths/accs (A-OKVQA only), keyed by
  *layers kept* (depth) -> converted to % pruned = (28 - depth) / 28 * 100 here.

Two distinct pruning strategies exist in this repo and must not be conflated:
- prune_model.py: keeps layers 0,1,2 fixed (deepstack alignment), evenly subsamples
  the rest -> an evenly-spaced/alternating drop pattern. Collapses sharply ~32-36%.
  Only a LoRA-finetuned version of this one is plotted (no linear-probe data for it).
- prune_model_generic.py --keep_idx range(N): keeps the first N contiguous layers,
  drops the rest -> no cliff, stays flat out to 75% pruned (pruning_accuracy_keepfirst_only.png).
  The linear-probe curve below uses this same keep-first-N truncation, so it, the
  keep-first-N + LoRA curve, and the adapter-coverage curve are all directly comparable
  (same architecture truncation / freeze-from-front, varying only how much gets adapted).

Caveat: the linear-probe curve's accuracy is real A-OKVQA multiple-choice
question_accuracy (a proper eval), while the other curves use the in-batch 4-way
training-accuracy proxy metric (compute_train_accuracy.py). Both are chance=25%
4-way tasks, but the eval methodology differs, so treat cross-curve comparisons as
indicative rather than exactly equivalent.
"""
import matplotlib.pyplot as plt

freeze = {"x": [0, 25, 50, 75], "y": [86.0, 82.0, 77.0, 65.0]}
prune_evenly_spaced_lora = {"x": [0, 25, 35.7, 50, 75], "y": [84.0, 76.0, 41.0, 39.0, 33.0]}
prune_keepfirst_lora = {"x": [0, 25, 28.6, 32.1, 35.7, 50, 75], "y": [84.0, 80.0, 77.5, 77.0, 75.0, 77.5, 69.0]}

_probe_depths = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
_probe_accs = [0.32666666666666666, 0.32, 0.44, 0.4666666666666667, 0.6333333333333333,
               0.6666666666666666, 0.7933333333333333, 0.8333333333333334, 0.8333333333333334,
               0.82, 0.8, 0.8133333333333334, 0.8, 0.8133333333333334, 0.7933333333333333]
# chart is chopped at 75% pruned to match the other curves' range; this drops
# depths 0/2/4 (85.7-100% pruned), where the real data actually keeps declining
# further (down to ~32-45%) -- see git history for the uncropped version.
prune_keepfirst_probe = {
    "x": [(28 - d) / 28 * 100 for d in _probe_depths if (28 - d) / 28 * 100 <= 75],
    "y": [a * 100 for d, a in zip(_probe_depths, _probe_accs) if (28 - d) / 28 * 100 <= 75],
}

fig, ax = plt.subplots(figsize=(9.5, 6))

ax.plot(freeze["x"], freeze["y"], marker="o", color="#2a78d6", linewidth=2,
        markersize=6, label="Adapter-coverage sweep (new, full depth, last-k% adapted)")
ax.plot(prune_keepfirst_lora["x"], prune_keepfirst_lora["y"], marker="D", color="#7a4fb5",
        linewidth=2, markersize=7,
        label="Depth pruning, keep-first-N + LoRA (existing, contiguous drop)")
ax.plot(prune_evenly_spaced_lora["x"], prune_evenly_spaced_lora["y"], marker="o", color="#1baf7a", linewidth=2,
        markersize=6, label="Depth pruning, evenly-spaced + LoRA (existing)")
ax.plot(prune_keepfirst_probe["x"], prune_keepfirst_probe["y"], marker="o", color="#e34948",
        linewidth=2, markersize=4.5,
        label="Depth pruning, keep-first-N + linear probe (real, A-OKVQA question_accuracy)")

ax.axhline(25, color="0.6", linewidth=1, linestyle=":", zorder=0)
ax.text(1, 27.5, "chance (25%)", va="bottom", ha="left", fontsize=9, color="0.4")

for x, y in zip(freeze["x"], freeze["y"]):
    ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, color="#2a78d6", fontweight="bold")
for x, y in zip(prune_keepfirst_lora["x"], prune_keepfirst_lora["y"]):
    ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(13, -2),
                ha="left", fontsize=8, color="#7a4fb5", fontweight="bold")
for x, y in zip(prune_evenly_spaced_lora["x"], prune_evenly_spaced_lora["y"]):
    ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=9, color="#1baf7a", fontweight="bold")
# label only the endpoints + peak of the (cropped) probe curve, to avoid clutter
_probe_y = prune_keepfirst_probe["y"]
_probe_label_idxs = [0, _probe_y.index(max(_probe_y)), len(_probe_y) - 1]
for i in _probe_label_idxs:
    x, y = prune_keepfirst_probe["x"][i], prune_keepfirst_probe["y"][i]
    ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8, color="#e34948", fontweight="bold")

ax.set_xlim(-3, 83)
ax.set_ylim(0, 100)
ax.set_xlabel("% of decoder layers pruned (existing curves) / frozen (new curve)")
ax.set_ylabel("In-batch accuracy (4-way, chance = 25%)")
ax.set_title("Qwen3-VL-2B: freezing vs. pruning layers, two pruning\npatterns, and the real linear-probe floor", fontsize=12)
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)

fig.text(0.01, -0.04,
         "Note: x-axis means different things per series -- frozen layers still run forward (full 28-layer depth); pruned\n"
         "layers are physically removed. Blue/purple/red share the same keep-first-N truncation (directly comparable,\n"
         "varying only how much gets adapted); green uses a different evenly-spaced pruning pattern (see docstring).\n"
         "Red uses real A-OKVQA question_accuracy, a different eval methodology than the in-batch 4-way proxy metric\n"
         "the other three curves use -- both are chance=25% 4-way, but treat cross-curve deltas as indicative.",
         fontsize=8, color="0.4", va="top")

fig.tight_layout()
fig.savefig("figures/depth_pruning/freeze_vs_pruning_comparison.png", dpi=150, bbox_inches="tight")
print("saved figures/depth_pruning/freeze_vs_pruning_comparison.png")
