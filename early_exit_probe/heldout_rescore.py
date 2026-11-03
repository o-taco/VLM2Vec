"""Re-scores a baseline on the paper's 800 held-out rows per MMEB dataset (Sec. 4 TODO), so
baseline-vs-fitted comparisons are row matched like final-vs-routed already are.

Input: JSON {dataset: [per-row correctness, 0/1 or bool]} over the dataset's fixed 1,000 rows
in dataset order (row i = the i-th row of the source pool), one file per baseline. The
calibration split is the paper's: rank `prefix|dataset|row_index` by SHA-256 and hold out the
first 200 per dataset. Prints the 1,000-row and 800-row accuracies side by side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from early_exit_probe.routing_baselines import CLASSIFICATION, SPLIT_PREFIX, VQA, sha_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baselines", nargs="+", help="per-row correctness JSON files, one per baseline")
    ap.add_argument("--n_cal", type=int, default=200)
    ap.add_argument("--split_prefix", default=SPLIT_PREFIX)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {}
    for path in args.baselines:
        per_row = json.load(open(path))
        rows = {}
        for name, vals in per_row.items():
            v = np.asarray(vals, dtype=bool)
            held = ~sha_split(name, len(v), args.n_cal, args.split_prefix)
            rows[name] = {"all_rows": 100 * v.mean(), "heldout": 100 * v[held].mean(), "n_heldout": int(held.sum())}
        groups = {"classification": CLASSIFICATION, "vqa": VQA, "all": CLASSIFICATION + VQA}
        macro = {g: {k: float(np.mean([rows[n][k] for n in names]))
                     for k in ("all_rows", "heldout")}
                 for g, names in groups.items() if all(n in rows for n in names)}
        report[Path(path).stem] = {"per_dataset": rows, "macro": macro}
        print(Path(path).stem)
        for n, r in rows.items():
            print(f"  {n:13s} 1000-row {r['all_rows']:6.2f}   held-out({r['n_heldout']}) {r['heldout']:6.2f}")
        for g, m in macro.items():
            print(f"  [{g}] 1000-row {m['all_rows']:.2f}   held-out {m['heldout']:.2f}")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
