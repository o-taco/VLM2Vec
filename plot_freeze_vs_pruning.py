"""Compares the adapter-coverage (freeze-fraction) sweep against the existing
depth-pruning curves. Freeze-fraction data is exact (logs/adapter_coverage_freeze/);
the linear-probe curve is traced by eye off figures/depth_pruning/pruning_accuracy_keepfirst_only.png
(no logged numbers survive for it) and is marked as approximate.
"""
import matplotlib.pyplot as plt

freeze = {"x": [0, 25, 50, 75], "y": [86.0, 82.0, 77.0, 65.0]}
prune_lora = {"x": [0, 25, 35.7, 50, 75], "y": [84.0, 76.0, 41.0, 39.0, 33.0]}
prune_probe_approx = {"x": [0, 25, 35.7, 50, 75], "y": [74.0, 57.0, 33.0, 28.0, 25.0]}

fig, ax = plt.subplots(figsize=(8.5, 5.5))

ax.plot(freeze["x"], freeze["y"], marker="o", color="#2a78d6", linewidth=2,
        markersize=6, label="Adapter-coverage sweep (new, full depth, last-k% adapted)")
ax.plot(prune_lora["x"], prune_lora["y"], marker="o", color="#1baf7a", linewidth=2,
        markersize=6, label="Depth pruning + LoRA (existing)")
ax.plot(prune_probe_approx["x"], prune_probe_approx["y"], marker="o", color="#e34948",
        linewidth=2, markersize=6, linestyle="--",
        label="Depth pruning + linear probe (existing, traced -- approximate)")

ax.axhline(25, color="0.6", linewidth=1, linestyle=":", zorder=0)
ax.text(1, 27.5, "chance (25%)", va="bottom", ha="left", fontsize=9, color="0.4")

for x, y in zip(freeze["x"], freeze["y"]):
    ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, color="#2a78d6", fontweight="bold")
for x, y in zip(prune_lora["x"], prune_lora["y"]):
    ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=9, color="#1baf7a", fontweight="bold")

ax.set_xlim(-3, 83)
ax.set_ylim(0, 100)
ax.set_xlabel("% of decoder layers pruned (existing curves) / frozen (new curve)")
ax.set_ylabel("In-batch accuracy (4-way, chance = 25%)")
ax.set_title("Qwen3-VL-2B: does freezing (not removing) layers bridge\nfull LoRA and the linear-probe floor?", fontsize=12)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax.grid(True, alpha=0.3)

fig.text(0.01, -0.02,
         "Note: x-axis means different things per series -- frozen layers still run forward (full 28-layer depth);\n"
         "pruned layers are physically removed. Linear-probe curve is approximate (traced, not logged).",
         fontsize=8, color="0.4", va="top")

fig.tight_layout()
fig.savefig("figures/depth_pruning/freeze_vs_pruning_comparison.png", dpi=150, bbox_inches="tight")
print("saved figures/depth_pruning/freeze_vs_pruning_comparison.png")
