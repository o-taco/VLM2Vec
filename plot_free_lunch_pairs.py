"""Extends the original single-N (N=6 only) "contiguous vs. spread" bar chart
(figures/depth_pruning/combined_figure_preview.png panel (c), no surviving
generation script) into a proper swept comparison across N, per model size.

Data: logs/depth_pruning_keepfirst/{sweep_2b_floor_results.txt,
keepfirst_sweep_results.txt, sweep_8b_boundary_results.txt,
sweep_8b_results.txt} (contiguous, pre-existing) and
logs/spread_pruning/sweep_spread_results.txt (spread, includes this
session's N=4/7 (2B) and N=4/9 (8B) additions alongside the original N=6
pair). All best-checkpoint in-batch 4-way accuracy.
"""
import matplotlib.pyplot as plt

# --- Qwen3-VL-2B (28 layers total) ---
n_2b = [4, 6, 7]
contig_2b = [34.0, 59.5, 69.0]
spread_2b = [37.5, 28.0, 37.0]

# --- Qwen3-VL-8B (36 layers total) ---
n_8b = [4, 6, 9]
contig_8b = [33.0, 51.5, 70.5]
spread_8b = [29.5, 35.0, 31.0]

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

panels = [
    (axes[0], n_2b, contig_2b, spread_2b, "Qwen3-VL-2B (28 layers total)"),
    (axes[1], n_8b, contig_8b, spread_8b, "Qwen3-VL-8B (36 layers total)"),
]

for ax, n, contig, spread, title in panels:
    ax.plot(n, contig, marker="o", color="#9467bd", linewidth=2, markersize=9,
             label="Contiguous (keep-first-N)")
    ax.plot(n, spread, marker="s", color="#d62728", linewidth=2, markersize=9,
             linestyle="--", label="Spread across depth")
    ax.axhline(25, color="0.5", linestyle=":", linewidth=1.5, label="Chance (25%)")

    for xi, yi in zip(n, contig):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points", xytext=(-12, 8),
                    ha="right", fontsize=9, color="#9467bd", fontweight="bold")
    for xi, yi in zip(n, spread):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points", xytext=(12, -10),
                    ha="left", fontsize=9, color="#d62728", fontweight="bold")

    ax.set_xlabel("Layers retained (N)")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(n)
    ax.set_ylim(0, 85)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("In-batch accuracy\n(4-way, best ckpt., %)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=9.5,
           framealpha=0.9)
fig.suptitle("Contiguous vs. spread-across-depth pruning, at matched layer count", fontsize=13)

fig.tight_layout(rect=(0, 0.06, 1, 0.96))
fig.savefig("figures/depth_pruning/spread_vs_contiguous.png", dpi=150, bbox_inches="tight")
fig.savefig("figures/depth_pruning/spread_vs_contiguous.pdf", bbox_inches="tight")
print("saved figures/depth_pruning/spread_vs_contiguous.{png,pdf}")
