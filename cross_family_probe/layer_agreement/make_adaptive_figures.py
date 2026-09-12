import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)

results = json.loads(Path("outputs/adaptive_depth/results.json").read_text())
plt.rcParams.update({"font.size": 11})

fixed = results["fixed_depth"]
fixed_depth = [r["depth"] for r in fixed]
fixed_acc = [r["accuracy"] * 100 for r in fixed]

fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.plot(fixed_depth, fixed_acc, color="#4a5568", lw=2, label="fixed depth (all examples, same N)")

wa = results["window_agreement"]
ax.plot([r["avg_depth"] for r in wa], [r["accuracy"] * 100 for r in wa],
        marker="o", ms=5, color="#2b6cb0", label="agreement-gated (window agreement)")

mt = results["margin_threshold"]
mt = [r for r in mt if r["avg_depth"] <= 28]
ax.plot([r["avg_depth"] for r in mt], [r["accuracy"] * 100 for r in mt],
        marker="s", ms=5, color="#c05621", label="confidence-gated (margin threshold)")

# best AND-combined per window=3 series
and3 = [r for r in results["combined_and"] if r["window"] == 3]
ax.plot([r["avg_depth"] for r in and3], [r["accuracy"] * 100 for r in and3],
        marker="^", ms=5, ls="--", color="#805ad5", label="agreement AND confidence (W=3)")

ax.set_xlabel("average number of decoder layers executed (of 28)")
ax.set_ylabel("A-OKVQA accuracy (%)")
ax.set_title("Per-example adaptive depth vs. fixed uniform pruning")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_xlim(-1, 29)
fig.tight_layout()
fig.savefig(out_dir / "adaptive_depth_frontier.png", dpi=300)
plt.close(fig)

# Zoomed low-to-mid depth region where the gap is largest
fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.plot(fixed_depth, fixed_acc, color="#4a5568", lw=2, label="fixed depth (all examples, same N)")
ax.plot([r["avg_depth"] for r in wa], [r["accuracy"] * 100 for r in wa],
        marker="o", ms=5, color="#2b6cb0", label="agreement-gated (window agreement)")
ax.plot([r["avg_depth"] for r in mt], [r["accuracy"] * 100 for r in mt],
        marker="s", ms=5, color="#c05621", label="confidence-gated (margin threshold)")
ax.set_xlabel("average number of decoder layers executed (of 28)")
ax.set_ylabel("A-OKVQA accuracy (%)")
ax.set_title("Zoomed: low-to-mid average depth")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_xlim(0, 18)
ax.set_ylim(20, 90)
fig.tight_layout()
fig.savefig(out_dir / "adaptive_depth_frontier_zoom.png", dpi=300)
plt.close(fig)

print("wrote", out_dir / "adaptive_depth_frontier.png")
print("wrote", out_dir / "adaptive_depth_frontier_zoom.png")
