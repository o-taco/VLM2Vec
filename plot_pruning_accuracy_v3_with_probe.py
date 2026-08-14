"""Updated version of pruning_accuracy_v2.png (the source for the paper's
fig:qwen3vl-2b-depth-pruning) with the real linear-probe question_accuracy
curve added, per the figure's \\todo{replace with most recent curves
including linear probe}.

All non-probe data points are exact, recomputed/re-read from source rather
than reused from the old image:
- Evenly-spaced (alternating) best/latest: logs/depth_pruning_other_strategies/
  train_acc_{7,14,18,19,20,21}layer*.log ("Final train top-1 accuracy").
- Keep-first-N best/latest: logs/depth_pruning_keepfirst/keepfirst_sweep_results.txt.
  Note: keep-first-N has NO measured point at 32.1% (19-layer) -- confirmed
  by zooming into the original PNG's cliff region, where the line passes
  through that x with no diamond marker. Do not interpolate one in.
- keep-26 ablation: train_acc_19layer_keep26_*.log (best = max across all
  logged checkpoints, latest = step-500 value).
- Keep-last-N (front-drop): logs/depth_pruning_keeplast/keeplast_sweep_results.txt,
  Phase A+B (this repo's most recent sweep) -- full 0-75pct range,
  ratio-matched to keep-first-N's own x values. Its 32.1pct point IS the old
  "keep-mid" single-seam ablation (same keep_idx, keepmid_best/latest =
  36.0/32.0) -- now folded into the curve instead of a standalone scatter
  point, per plot_keeplast_vs_keepfirst.py's own note on that equivalence.
- x=0 (shared by both patterns, since 0% pruned is the same unpruned model):
  84.0/85.0 best/latest -- read directly off the original figure's own
  printed data-labels (not from any surviving raw log for this point).
- Linear probe: prune_keepfirst_probe series from plot_freeze_vs_pruning.py
  (real A-OKVQA question_accuracy, a properly-evaluated multiple-choice
  metric -- NOT the in-batch 4-way proxy every other curve here uses; both
  happen to have chance=25%, but treat cross-curve deltas as indicative
  only, per that script's own caveat).
"""
import matplotlib.pyplot as plt

# --- shared 0%-pruned point ---
x0_best, x0_latest = 84.0, 85.0

# --- evenly-spaced / alternating pattern (cliff) ---
alt_x = [0, 25, 28.6, 32.1, 35.7, 50, 75]
alt_best = [x0_best, 76.0, 71.5, 47.0, 41.5, 40.5, 33.0]
alt_latest = [x0_latest, 76.0, 72.0, 48.0, 41.0, 39.0, 36.0]

# --- keep-first-N (contiguous, no cliff) -- no measured point at 32.1% ---
kf_x = [0, 25, 28.6, 35.7, 50, 75]
kf_best = [x0_best, 80.0, 77.5, 75.0, 77.5, 69.0]
kf_latest = [x0_latest, 77.5, 75.5, 75.0, 76.0, 68.5]

# --- keep-26 ablation, single point at 32.1% (19-layer) ---
keep26_best, keep26_latest = 50.5, 52.0

# --- keep-last-N (drop right after the deepstack prefix, front-drop) ---
kl_x = [0, 3.6, 7.1, 10.7, 17.9, 25, 28.6, 32.1, 35.7, 50, 75]
kl_best = [x0_best, 79.0, 72.0, 64.5, 51.5, 50.5, 42.5, 36.0, 39.5, 37.5, 32.0]
kl_latest = [x0_latest, 76.5, 72.0, 64.5, 51.5, 50.5, 42.0, 32.0, 38.0, 37.0, 30.5]

# --- linear probe (real A-OKVQA question_accuracy), keep-first-N depths ---
probe_depths = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
probe_accs = [0.32666666666666666, 0.32, 0.44, 0.4666666666666667, 0.6333333333333333,
              0.6666666666666666, 0.7933333333333333, 0.8333333333333334, 0.8333333333333334,
              0.82, 0.8, 0.8133333333333334, 0.8, 0.8133333333333334, 0.7933333333333333]
probe_x = [(28 - d) / 28 * 100 for d in probe_depths]
probe_y = [a * 100 for a in probe_accs]
# sort by x ascending for a clean line
probe_x, probe_y = zip(*sorted(zip(probe_x, probe_y)))

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(alt_x, alt_best, marker="o", color="#1f77b4", linewidth=2, markersize=7,
        label="Evenly spaced, best ckpt.")
ax.plot(alt_x, alt_latest, marker="o", color="#ff7f0e", linewidth=2, markersize=7,
        label="Evenly spaced, latest ckpt.")
ax.plot(kf_x, kf_best, marker="D", color="#9467bd", linewidth=2, markersize=8,
        label="Keep-first-N, best ckpt.")
ax.plot(kf_x, kf_latest, marker="D", color="#9467bd", linewidth=2, markersize=7,
        linestyle="--", alpha=0.85, label="Keep-first-N, latest ckpt.")
ax.plot(probe_x, probe_y, marker="^", color="#17becf", linewidth=2, markersize=7,
        label="Keep-first-N, linear probe (real acc.)")
ax.plot(kl_x, kl_best, marker="s", color="#8c564b", linewidth=2, markersize=8,
        label="Keep-last-N (front-drop), best ckpt.")
ax.plot(kl_x, kl_latest, marker="s", color="#8c564b", linewidth=2, markersize=7,
        linestyle="--", alpha=0.85, label="Keep-last-N (front-drop), latest ckpt.")

# The remaining 19L single-seam ablation (keep-26) sits at x=32.1 and would
# land right on top of the cliff/keep-last-N curves at that size, so: nudge
# it apart slightly in x, and add a white edge so it stays visually
# separable. A dotted guide line + "19L" tag ties it back to its true x
# position. (The old "keep-mid" ablation used to get the same treatment, but
# it's exactly the keep-last-N curve's own x=32.1 point -- see docstring --
# so it's just part of that curve now instead of a duplicate scatter point.)
ax.axvline(32.1, color="0.65", linestyle=":", linewidth=1, zorder=1)
ax.annotate("19L", (32.1, 99), ha="center", va="top", fontsize=7.5, color="0.45")

ax.scatter([31.5], [keep26_best], marker="*", s=130, color="#2ca02c",
           edgecolors="white", linewidths=1.2, zorder=6,
           label="19L keep-26, best")
ax.scatter([32.7], [keep26_latest], marker="*", s=130, color="#d62728",
           edgecolors="white", linewidths=1.2, zorder=6,
           label="19L keep-26, latest")

ax.axhline(25, color="0.5", linestyle=":", linewidth=1.5, label="Chance (25%)")

# --- peak annotations: the highest point on each curve/series ---
# alt_best and kf_best share the same peak (x=0, both start at the unpruned
# model), likewise alt_latest/kf_latest -- label each shared peak once.
ax.annotate(f"{x0_best:.1f}%", (0, x0_best), textcoords="offset points", xytext=(-8, 14),
            ha="right", fontsize=9, color="#1f77b4", fontweight="bold")
ax.annotate(f"{x0_latest:.1f}%", (0, x0_latest), textcoords="offset points", xytext=(-8, -4),
            ha="right", fontsize=9, color="#ff7f0e", fontweight="bold")

probe_peak_i = max(range(len(probe_y)), key=lambda i: probe_y[i])
ax.annotate(f"{probe_y[probe_peak_i]:.1f}%", (probe_x[probe_peak_i], probe_y[probe_peak_i]),
            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9,
            color="#17becf", fontweight="bold")

ax.annotate(f"{keep26_latest:.1f}%", (32.7, keep26_latest), textcoords="offset points",
            xytext=(10, 14), ha="left", fontsize=8.5, color="#d62728", fontweight="bold")
ax.annotate(f"{keep26_best:.1f}%", (31.5, keep26_best), textcoords="offset points",
            xytext=(-12, 16), ha="right", fontsize=8.5, color="#2ca02c", fontweight="bold")

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
    "alternating drop (cliff) vs. keep-first-N (back-drop) vs. keep-last-N (front-drop) vs. the real linear-probe floor",
    fontsize=12,
)
legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=8.2,
                    framealpha=0.9, columnspacing=1.1, handletextpad=0.5)
ax.grid(True, alpha=0.3)

fig.tight_layout()

fig.savefig("figures/depth_pruning/pruning_accuracy_v3_with_probe.png", dpi=150, bbox_inches="tight")
fig.savefig("figures/depth_pruning/pruning_accuracy_v3_with_probe.pdf", bbox_inches="tight")
print("saved figures/depth_pruning/pruning_accuracy_v3_with_probe.{png,pdf}")
