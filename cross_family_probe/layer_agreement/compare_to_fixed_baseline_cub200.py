import json
from pathlib import Path

import numpy as np

results = json.loads(Path("outputs/cub200/adaptive_depth/results.json").read_text())
fixed = results["fixed_depth"]
fixed_depths = np.array([r["depth"] for r in fixed])
fixed_acc = np.array([r["accuracy"] for r in fixed])


def gap_vs_fixed(avg_depth, accuracy):
    interp_acc = float(np.interp(avg_depth, fixed_depths, fixed_acc))
    return interp_acc, accuracy - interp_acc


comparison = {}
for policy_name in ["window_agreement", "margin_threshold", "combined_or_W3"]:
    rows = []
    for r in results[policy_name]:
        interp_acc, gap = gap_vs_fixed(r["avg_depth"], r["accuracy"])
        rows.append({**r, "fixed_interp_accuracy": interp_acc, "gap": gap})
    comparison[policy_name] = rows

rows = []
for r in results["combined_and"]:
    interp_acc, gap = gap_vs_fixed(r["avg_depth"], r["accuracy"])
    rows.append({**r, "fixed_interp_accuracy": interp_acc, "gap": gap})
comparison["combined_and"] = rows

Path("outputs/cub200/adaptive_depth/comparison.json").write_text(json.dumps(comparison, indent=2))

for name, rows in comparison.items():
    print(f"\n=== {name} vs. interpolated fixed-depth baseline (CUB-200) ===")
    for r in rows:
        extra = f"W={r['window']} " if "window" in r and name != "window_agreement" else ""
        tau = f"tau={r['tau']:.2f} " if "tau" in r else ""
        w = f"W={r['window']} " if name == "window_agreement" else ""
        print(f"  {w}{tau}{extra}avg_depth={r['avg_depth']:6.2f}  "
              f"accuracy={r['accuracy']:.4f}  fixed@same_depth={r['fixed_interp_accuracy']:.4f}  "
              f"gap={r['gap']:+.4f}")
