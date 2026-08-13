"""Front-drop ("keeplast": keep 0,1,2 + everything from some start layer to
27, drop a contiguous block right after the deepstack prefix) vs. the
existing back-drop ("keepfirst": keep 0..N-1, drop the tail) dose-response
curve, both on Qwen3-VL-2B / A-OKVQA in-batch accuracy (batch=4, chance=25%).

Data sources:
- keepfirst: logs/depth_pruning_keepfirst/keepfirst_sweep_results.txt
  (x0 shared baseline 84.0/85.0 best/latest reused from
  plot_pruning_accuracy_v3_with_probe.py, per that script's own note that
  it's read off the original figure's data-labels, no surviving raw log).
- keeplast: logs/depth_pruning_keeplast/keeplast_sweep_results.txt (this
  session's Phase A sweep, N=27/26/25/23) plus the x=32.1 point, which is
  exactly the keeplast recipe at N=19 (keep 0,1,2 + 12..27) already run
  under the name "keepmid" -- same keep_idx, reused from
  plot_pruning_accuracy_v3_with_probe.py's keepmid_best/latest.
"""
import matplotlib.pyplot as plt

# --- shared 0%-pruned point (unpruned model) ---
x0_best, x0_latest = 84.0, 85.0

# --- keep-first-N (drop the tail) ---
kf_x = [0, 25, 28.6, 35.7, 50, 75]
kf_best = [x0_best, 80.0, 77.5, 75.0, 77.5, 69.0]
kf_latest = [x0_latest, 77.5, 75.5, 75.0, 76.0, 68.5]

# --- keep-last-N (drop a block right after the deepstack prefix) ---
kl_x = [0, 3.6, 7.1, 10.7, 17.9, 32.1]
kl_best = [x0_best, 79.0, 72.0, 64.5, 51.5, 36.0]
kl_latest = [x0_latest, 76.5, 72.0, 64.5, 51.5, 32.0]

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(kf_x, kf_best, marker="D", color="#9467bd", linewidth=2, markersize=8,
        label="Keep-first-N (drop tail), best ckpt.")
ax.plot(kf_x, kf_latest, marker="D", color="#9467bd", linewidth=2, markersize=7,
        linestyle="--", alpha=0.85, label="Keep-first-N (drop tail), latest ckpt.")
ax.plot(kl_x, kl_best, marker="s", color="#8c564b", linewidth=2, markersize=8,
        label="Keep-last-N (drop front), best ckpt.")
ax.plot(kl_x, kl_latest, marker="s", color="#8c564b", linewidth=2, markersize=7,
        linestyle="--", alpha=0.85, label="Keep-last-N (drop front), latest ckpt.")

ax.axvline(32.1, color="0.65", linestyle=":", linewidth=1, zorder=1)
ax.annotate("19L (=keepmid)", (32.1, 99), ha="center", va="top", fontsize=7.5, color="0.45")

ax.axhline(25, color="0.5", linestyle=":", linewidth=1.5, label="Chance (25%)")

# --- annotate the shared start and each curve's last point ---
ax.annotate(f"{x0_best:.1f}%", (0, x0_best), textcoords="offset points", xytext=(-8, 14),
            ha="right", fontsize=9, color="0.25", fontweight="bold")
ax.annotate(f"{x0_latest:.1f}%", (0, x0_latest), textcoords="offset points", xytext=(-8, -12),
            ha="right", fontsize=9, color="0.25", fontweight="bold")

ax.annotate(f"{kf_best[-1]:.1f}%", (kf_x[-1], kf_best[-1]), textcoords="offset points",
            xytext=(8, 8), ha="left", fontsize=9, color="#9467bd", fontweight="bold")
ax.annotate(f"{kl_best[-1]:.1f}%", (kl_x[-1], kl_best[-1]), textcoords="offset points",
            xytext=(8, 6), ha="left", fontsize=9, color="#8c564b", fontweight="bold")
ax.annotate(f"{kl_latest[-1]:.1f}%", (kl_x[-1], kl_latest[-1]), textcoords="offset points",
            xytext=(8, -14), ha="left", fontsize=9, color="#8c564b", fontweight="bold")

ax.set_xlim(-3, 83)
ax.set_ylim(0, 100)
ax.set_xlabel("% of decoder layers pruned")
ax.set_ylabel("Accuracy (%)")
ax.set_title(
    "Qwen3-VL-2B depth pruning vs A-OKVQA accuracy\n"
    "dropping the tail (keep-first-N) vs. dropping right after the deepstack prefix (keep-last-N)",
    fontsize=12,
)
legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8.5,
                    framealpha=0.9, columnspacing=1.1, handletextpad=0.5)
ax.grid(True, alpha=0.3)

fig.tight_layout()

fig.savefig("figures/depth_pruning/keeplast_vs_keepfirst.png", dpi=150, bbox_inches="tight")
fig.savefig("figures/depth_pruning/keeplast_vs_keepfirst.pdf", bbox_inches="tight")
print("saved figures/depth_pruning/keeplast_vs_keepfirst.{png,pdf}")
