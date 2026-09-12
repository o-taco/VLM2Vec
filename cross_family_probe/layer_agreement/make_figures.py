import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out_dir = Path("outputs/figures")
out_dir.mkdir(parents=True, exist_ok=True)

results = json.loads(Path("outputs/agreement_analysis/results.json").read_text())
combined = json.loads(Path("outputs/agreement_analysis/combined_signal_results.json").read_text())
raw = np.load("outputs/agreement_analysis/raw_votes.npz")

plt.rcParams.update({"font.size": 11})

# 1. Layer-wise probe accuracy curve
fig, ax = plt.subplots(figsize=(5.2, 3.6))
acc = results["layer_accuracy_curve"]
ax.plot(range(len(acc)), [a * 100 for a in acc], marker="o", ms=3, color="#2b6cb0")
ax.axhline(25, ls=":", color="gray", lw=1, label="chance (25%)")
ax.set_xlabel("decoder layer index (0 = embedding output)")
ax.set_ylabel("A-OKVQA probe accuracy (%)")
ax.set_title("Per-layer frozen probe accuracy\n(Qwen3-VL-2B, 4000-question training subset)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out_dir / "layer_accuracy_curve.png", dpi=300)
plt.close(fig)

# 2. Accuracy conditioned on plurality-agreement bin
bins = results["accuracy_by_plurality_bin"]
labels = list(bins.keys())
accs = [bins[k]["accuracy"] * 100 for k in labels]
ns = [bins[k]["n"] for k in labels]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
bars = ax.bar(labels, accs, color="#2f855a")
for bar, n in zip(bars, ns):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"n={n}",
            ha="center", va="bottom", fontsize=9)
ax.set_xlabel("fraction of 29 layer-probes agreeing on the plurality answer")
ax.set_ylabel("final-layer prediction accuracy (%)")
ax.set_title("When layer probes agree,\nthe final answer is more often right")
ax.set_ylim(0, 108)
fig.tight_layout()
fig.savefig(out_dir / "accuracy_by_agreement_bin.png", dpi=300)
plt.close(fig)

# 3. AUROC comparison bar chart
auroc = results["auroc_correctness"]
order = ["final_layer_softmax_margin", "agree_with_final_all_layers", "plurality_frac_all_layers",
         "vote_entropy_all_layers", "plurality_frac_late_half_layers"]
nice_names = {
    "final_layer_softmax_margin": "final-layer\nsoftmax margin\n(baseline)",
    "agree_with_final_all_layers": "agree-with-final\n(all layers)",
    "plurality_frac_all_layers": "plurality frac.\n(all layers)",
    "vote_entropy_all_layers": "vote entropy\n(all layers,\ninverted)",
    "plurality_frac_late_half_layers": "plurality frac.\n(late-half\nlayers only)",
}
vals = [auroc[k] for k in order]
colors = ["#c05621"] + ["#2b6cb0"] * (len(order) - 1)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.bar([nice_names[k] for k in order], vals, color=colors)
ax.axhline(0.5, ls=":", color="gray", lw=1, label="chance (AUROC=0.5)")
ax.set_ylabel("AUROC (predicting final-layer correctness)")
ax.set_ylim(0.45, 0.9)
ax.tick_params(axis="x", labelsize=9.5)
ax.legend(frameon=False, loc="lower right")
ax.set_title("Cross-layer agreement predicts correctness, but less\nstrongly than the final layer's own confidence", fontsize=12)
fig.tight_layout()
fig.savefig(out_dir / "auroc_comparison.png", dpi=300)
plt.close(fig)

# 4. Risk-coverage (selective prediction) curves
fig, ax = plt.subplots(figsize=(5.6, 4.2))
rc = results["risk_coverage"]
style = {
    "final_softmax_margin": ("final-layer softmax margin", "#c05621", "-"),
    "plurality_frac_all_layers": ("plurality frac. (all layers)", "#2b6cb0", "-"),
    "plurality_frac_late_half": ("plurality frac. (late-half layers)", "#805ad5", "--"),
}
for key, (label, color, ls) in style.items():
    cov = [c * 100 for c in rc[key]["coverage"]]
    acc = [a * 100 for a in rc[key]["accuracy"]]
    ax.plot(cov, acc, label=label, color=color, ls=ls, marker="o", ms=3)
ax.set_xlabel("coverage (% of questions answered)")
ax.set_ylabel("accuracy on answered questions (%)")
ax.set_title("Selective prediction:\nconfidence-ranked risk-coverage")
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(out_dir / "risk_coverage.png", dpi=300)
plt.close(fig)

# 5. Combined-signal AUROC (does agreement add to margin?)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
names = list(combined.keys())
means = [combined[n]["mean_auroc"] for n in names]
stds = [combined[n]["std_auroc"] for n in names]
colors = ["#c05621" if n == "margin_only" else "#2b6cb0" for n in names]
ax.barh(names, means, xerr=stds, color=colors)
ax.set_xlabel("5-fold CV AUROC (predicting final-layer correctness)")
ax.set_xlim(0.6, 0.9)
ax.set_title("Does layer-agreement add to the\nsoftmax-margin baseline?", fontsize=13)
fig.tight_layout()
fig.savefig(out_dir / "combined_signal_auroc.png", dpi=300)
plt.close(fig)

print("wrote figures to", out_dir)
for f in sorted(out_dir.glob("*.png")):
    print(" -", f)
