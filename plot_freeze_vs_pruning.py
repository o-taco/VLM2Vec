"""Compares the adapter-coverage (freeze-fraction) sweep against the existing
depth-pruning curves. Freeze-fraction data and the keep-first-N curve are exact
(logs/adapter_coverage_freeze/, logs/depth_pruning_keepfirst/keepfirst_sweep_results.txt).
The evenly-spaced-pruning + linear-probe curve is traced by eye off the source chart
(no logged numbers survive for it) and is marked as approximate.

Two distinct pruning strategies exist in this repo and must not be conflated:
- prune_model.py: keeps layers 0,1,2 fixed (deepstack alignment), evenly subsamples
  the rest -> an evenly-spaced/alternating drop pattern. Collapses sharply ~32-36%.
- prune_model_generic.py --keep_idx range(N): keeps the first N contiguous layers,
  drops the rest -> no cliff, stays flat out to 75% pruned (pruning_accuracy_keepfirst_only.png).
"""
import matplotlib.pyplot as plt

freeze = {"x": [0, 25, 50, 75], "y": [86.0, 82.0, 77.0, 65.0]}
prune_evenly_spaced_lora = {"x": [0, 25, 35.7, 50, 75], "y": [84.0, 76.0, 41.0, 39.0, 33.0]}
prune_evenly_spaced_probe_approx = {"x": [0, 25, 35.7, 50, 75], "y": [74.0, 57.0, 33.0, 28.0, 25.0]}
prune_keepfirst_lora = {"x": [0, 25, 28.6, 32.1, 35.7, 50, 75], "y": [84.0, 80.0, 77.5, 77.0, 75.0, 77.5, 69.0]}

fig, ax = plt.subplots(figsize=(9.5, 6))

ax.plot(freeze["x"], freeze["y"], marker="o", color="#2a78d6", linewidth=2,
        markersize=6, label="Adapter-coverage sweep (new, full depth, last-k% adapted)")
ax.plot(prune_keepfirst_lora["x"], prune_keepfirst_lora["y"], marker="D", color="#7a4fb5",
        linewidth=2, markersize=7,
        label="Depth pruning, keep-first-N + LoRA (existing, contiguous drop)")
ax.plot(prune_evenly_spaced_lora["x"], prune_evenly_spaced_lora["y"], marker="o", color="#1baf7a", linewidth=2,
        markersize=6, label="Depth pruning, evenly-spaced + LoRA (existing)")
ax.plot(prune_evenly_spaced_probe_approx["x"], prune_evenly_spaced_probe_approx["y"], marker="o", color="#e34948",
        linewidth=2, markersize=6, linestyle="--",
        label="Depth pruning, evenly-spaced + linear probe (existing, traced -- approximate)")

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

ax.set_xlim(-3, 83)
ax.set_ylim(0, 100)
ax.set_xlabel("% of decoder layers pruned (existing curves) / frozen (new curve)")
ax.set_ylabel("In-batch accuracy (4-way, chance = 25%)")
ax.set_title("Qwen3-VL-2B: freezing vs. pruning layers, two pruning\npatterns, and the linear-probe floor", fontsize=12)
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)

fig.text(0.01, -0.03,
         "Note: x-axis means different things per series -- frozen layers still run forward (full 28-layer depth); pruned\n"
         "layers are physically removed, either evenly-spaced across the depth or as a contiguous keep-first-N block\n"
         "(see docstring). Linear-probe curve is approximate (traced, not logged).",
         fontsize=8, color="0.4", va="top")

fig.tight_layout()
fig.savefig("figures/depth_pruning/freeze_vs_pruning_comparison.png", dpi=150, bbox_inches="tight")
print("saved figures/depth_pruning/freeze_vs_pruning_comparison.png")
