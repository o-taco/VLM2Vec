"""Offline routing baselines and diagnostics for the early-exit paper (cvpr2027_keep_it_simple).

Everything here replays saved per-depth rows; no GPU pass. Input rows have the shape that
diagnose_routing.py saves: {dataset: {depth: {field: 1-D tensor over rows}}} with fields
`pred`, `correct`, `agreement_prob` and optionally `answer_maxprob` / `answer_margin`
(softmax top-1 probability and top-1 minus top-2 probability of the answer head). Any set of
depths works: the 4-tap selected-depth model or the every-layer model.

Policies, all fitted with the same sequential calibration rule as the paper (Sec. 3.4):
  agreement   - our agreement head
  maxprob     - answer-head max softmax (only if saved)
  margin      - answer-head top-1/top-2 margin (only if saved)
Baselines that need no fitting:
  static@d    - every row exits at depth d with the answer head's prediction
  random      - per dataset, the same exit-depth distribution as the agreement policy but rows
                assigned at random; reported as its exact expectation sum_d p_d * acc_d. If the
                agreement head beats this, it is choosing *which* rows exit, not just how many.
Diagnostics for the agreement policy at the main target:
  early-exit outcome breakdown (final correct -> early wrong, final wrong -> early correct, ...)

Calibration rows come either from a separate file (--calibration_rows) or, as in the paper,
from a per-dataset SHA-256 split of the evaluation rows (--sha_split, 200 calibration rows).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

CLASSIFICATION = ("N24News", "HatefulMemes", "VOC2007", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211")
VQA = ("A-OKVQA", "Visual7W", "ScienceQA")
SPLIT_PREFIX = "260924_78_kis_every_layer_shared_mlp_probe"
SCORES = ("agreement_prob", "answer_maxprob", "answer_margin")
POLICY_NAME = {"agreement_prob": "agreement", "answer_maxprob": "maxprob", "answer_margin": "margin"}


def to_numpy(rows):
    return {name: {int(d): {f: t.numpy() for f, t in fs.items()} for d, fs in per_depth.items()}
            for name, per_depth in rows.items()}


def sha_split(name, n, n_cal, prefix=SPLIT_PREFIX):
    """Paper's split: rank `prefix|dataset|row_index` by SHA-256, first n_cal rows calibrate."""
    keys = [hashlib.sha256(f"{prefix}|{name}|{i}".encode()).hexdigest() for i in range(n)]
    order = np.argsort(keys)
    cal = np.zeros(n, dtype=bool)
    cal[order[:n_cal]] = True
    return cal


def subset(per_depth, mask):
    return {d: {f: v[mask] for f, v in fs.items()} for d, fs in per_depth.items()}


def concat(datasets):
    depths = sorted(datasets[0])
    return {d: {f: np.concatenate([ds[d][f] for ds in datasets]) for f in datasets[0][d]} for d in depths}


def fit_sequential(cal, score, target):
    """Sec. 3.4: process depths in order; among rows not yet assigned, pick the lowest cutoff
    whose cohort (score >= cutoff) has agreement with the final depth >= target; remove that
    cohort before fitting the next depth. Returns {depth: cutoff or None (exit disabled)}."""
    depths = sorted(cal)
    final = depths[-1]
    remaining = np.ones(len(cal[final]["pred"]), dtype=bool)
    cutoffs = {}
    for d in depths[:-1]:
        s = cal[d][score][remaining]
        agree = (cal[d]["pred"] == cal[final]["pred"])[remaining]
        if len(s) == 0:
            cutoffs[d] = None
            continue
        order = np.argsort(-s, kind="stable")
        s_sorted, a_sorted = s[order], agree[order].astype(float)
        precision = np.cumsum(a_sorted) / np.arange(1, len(s) + 1)
        # A cutoff admits every row tied at its score, so only evaluate at the end of each tie run.
        run_end = np.r_[s_sorted[1:] != s_sorted[:-1], True]
        ok = np.nonzero((precision >= target) & run_end)[0]
        if len(ok) == 0:
            cutoffs[d] = None
            continue
        cutoffs[d] = float(s_sorted[ok.max()])
        remaining &= ~(cal[d][score] >= cutoffs[d])
    return cutoffs


def route(rows, score, cutoffs):
    """Returns (exit depth per row, routed prediction correctness per row)."""
    depths = sorted(rows)
    final = depths[-1]
    n = len(rows[final]["pred"])
    exit_depth = np.full(n, final)
    remaining = np.ones(n, dtype=bool)
    for d in depths[:-1]:
        if cutoffs.get(d) is None:
            continue
        go = remaining & (rows[d][score] >= cutoffs[d])
        exit_depth[go] = d
        remaining &= ~go
    correct = np.zeros(n, dtype=bool)
    for d in depths:
        m = exit_depth == d
        correct[m] = rows[d]["correct"][m]
    return exit_depth, correct


def outcome_breakdown(rows, exit_depth):
    """Splits early exits by how the early answer relates to the final one."""
    depths = sorted(rows)
    final = depths[-1]
    early = exit_depth < final
    pred_e = np.zeros_like(rows[final]["pred"])
    corr_e = np.zeros_like(rows[final]["correct"])
    for d in depths[:-1]:
        m = exit_depth == d
        pred_e[m], corr_e[m] = rows[d]["pred"][m], rows[d]["correct"][m]
    agree = pred_e == rows[final]["pred"]
    corr_f = rows[final]["correct"]
    n = len(exit_depth)
    counts = {
        "agree_both_correct": early & agree & corr_f,
        "agree_both_wrong": early & agree & ~corr_f,
        "final_correct_early_wrong": early & ~agree & corr_f,
        "final_wrong_early_correct": early & ~agree & ~corr_f & corr_e,
        "final_wrong_early_wrong_different": early & ~agree & ~corr_f & ~corr_e,
    }
    out = {k: int(v.sum()) for k, v in counts.items()}
    out["n_rows"], out["n_early"] = n, int(early.sum())
    out["net_accuracy_change_points"] = 100 * (out["final_wrong_early_correct"] - out["final_correct_early_wrong"]) / n
    return out


def dataset_metrics(rows, exit_depth, correct):
    depths = sorted(rows)
    final = depths[-1]
    early = exit_depth < final
    agree = np.zeros(len(exit_depth), dtype=bool)
    for d in depths:
        m = exit_depth == d
        agree[m] = rows[d]["pred"][m] == rows[final]["pred"][m]
    exit_share = {d: float((exit_depth == d).mean()) for d in depths}
    return {
        "final_acc": 100 * float(rows[final]["correct"].mean()),
        "routed_acc": 100 * float(correct.mean()),
        "mean_depth": float(exit_depth.mean()),
        "coverage": float(early.mean()),
        "early_agreement": float(agree[early].mean()) if early.any() else None,
        "n_early": int(early.sum()),
        # Same exit-depth distribution, rows assigned at random (exact expectation).
        "random_matched_acc": 100 * float(sum(p * rows[d]["correct"].mean() for d, p in exit_share.items())),
    }


def macro(per_dataset, names, key):
    vals = [per_dataset[n][key] for n in names if n in per_dataset and per_dataset[n][key] is not None]
    return float(np.mean(vals)) if vals else None


def pooled_early_agreement(per_dataset, names):
    num = sum(per_dataset[n]["early_agreement"] * per_dataset[n]["n_early"]
              for n in names if n in per_dataset and per_dataset[n]["n_early"])
    den = sum(per_dataset[n]["n_early"] for n in names if n in per_dataset)
    return num / den if den else None


def groups_present(names):
    groups = {"classification": [n for n in CLASSIFICATION if n in names],
              "vqa": [n for n in VQA if n in names],
              "all": list(names)}
    return {g: v for g, v in groups.items() if v}


def summarize(per_dataset, names):
    out = {}
    for g, members in groups_present(names).items():
        out[g] = {k: macro(per_dataset, members, k)
                  for k in ("final_acc", "routed_acc", "mean_depth", "coverage", "random_matched_acc")}
        out[g]["early_agreement_pooled"] = pooled_early_agreement(per_dataset, members)
    return out


def run_policy(evals, cal_pool, score, target):
    cutoffs = fit_sequential(cal_pool, score, target)
    per_dataset, breakdown = {}, {}
    for name, rows in evals.items():
        exit_depth, correct = route(rows, score, cutoffs)
        per_dataset[name] = dataset_metrics(rows, exit_depth, correct)
        breakdown[name] = outcome_breakdown(rows, exit_depth)
    return cutoffs, per_dataset, breakdown


def static_curve(evals, names):
    depths = sorted(next(iter(evals.values())))
    out = {}
    for g, members in groups_present(names).items():
        out[g] = [{"depth": d, "acc": float(np.mean([100 * evals[n][d]["correct"].mean() for n in members]))}
                  for d in depths]
    return out


def plot(results, max_depth, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(results["static"])
    colors = {"agreement": "#2a78d6", "maxprob": "#eb6834", "margin": "#1baf7a"}
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.4), squeeze=False)
    for ax, g in zip(axes[0], groups):
        st = results["static"][g]
        ax.plot([p["depth"] / max_depth for p in st], [p["acc"] for p in st], color="#8a8a85",
                lw=1.5, ls="--", marker="o", ms=4, label="static exit")
        for policy, sweep in results["sweeps"].items():
            pts = [(s["summary"][g]["mean_depth"] / max_depth, s["summary"][g]["routed_acc"]) for s in sweep]
            ax.plot(*zip(*pts), color=colors[policy], lw=2, marker="o", ms=3, label=f"{policy} routing")
        main = results["main"]["summary"][g]
        ax.scatter([main["mean_depth"] / max_depth], [main["random_matched_acc"]], marker="x", s=60,
                   color="#3a3a38", zorder=5, label="random, matched depths")
        ax.scatter([main["mean_depth"] / max_depth], [main["routed_acc"]], marker="*", s=140,
                   color=colors["agreement"], edgecolor="white", zorder=6, label=f"agreement @ {results['target']}")
        ax.set_title({"classification": "Classification", "vqa": "VQA", "all": "All tasks"}[g])
        ax.set_xlabel("mean selected depth / full depth")
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("macro accuracy (%)")
    axes[0][-1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(Path(path).with_suffix(".pdf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="evaluation rows .pt (diagnose_routing.py format)")
    ap.add_argument("--calibration_rows", default=None, help="separate calibration rows .pt, pooled")
    ap.add_argument("--sha_split", type=int, default=None,
                    help="instead: hold out this many rows per eval dataset for calibration (paper: 200)")
    ap.add_argument("--split_prefix", default=SPLIT_PREFIX)
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--sweep", default="0.70:0.995:0.005", help="start:stop:step of calibration targets")
    ap.add_argument("--out", required=True, help="output prefix (.json and .png/.pdf are written)")
    args = ap.parse_args()
    assert (args.calibration_rows is None) != (args.sha_split is None), "pick one calibration source"

    rows = to_numpy(torch.load(args.rows, weights_only=True))
    names = [n for n in (*CLASSIFICATION, *VQA) if n in rows] + [n for n in rows if n not in CLASSIFICATION + VQA]
    if args.sha_split:
        evals, cals = {}, []
        for n in names:
            m = sha_split(n, len(rows[n][max(rows[n])]["pred"]), args.sha_split, args.split_prefix)
            cals.append(subset(rows[n], m))
            evals[n] = subset(rows[n], ~m)
        cal_pool = concat(cals)
    else:
        evals = {n: rows[n] for n in names}
        cal_rows = to_numpy(torch.load(args.calibration_rows, weights_only=True))
        cal_pool = concat(list(cal_rows.values()))

    some = next(iter(evals.values()))
    max_depth = max(some)
    scores = [s for s in SCORES if s in some[min(some)] and s in cal_pool[min(cal_pool)]]
    missing = [POLICY_NAME[s] for s in SCORES if s not in scores]
    if missing:
        print(f"[baselines] rows lack fields for {missing} routing; re-dump with answer confidences to add them")

    start, stop, step = map(float, args.sweep.split(":"))
    targets = [round(t, 4) for t in np.arange(start, stop + 1e-9, step)]
    results = {"rows": args.rows, "depths": sorted(some), "target": args.target, "datasets": names,
               "static": static_curve(evals, names), "sweeps": {}}
    for s in scores:
        sweep = []
        for t in targets:
            cutoffs, per_dataset, _ = run_policy(evals, cal_pool, s, t)
            sweep.append({"target": t, "cutoffs": cutoffs, "summary": summarize(per_dataset, names)})
        results["sweeps"][POLICY_NAME[s]] = sweep
        cutoffs, per_dataset, breakdown = run_policy(evals, cal_pool, s, args.target)
        entry = {"cutoffs": cutoffs, "per_dataset": per_dataset, "summary": summarize(per_dataset, names),
                 "breakdown": breakdown}
        if s == "agreement_prob":
            results["main"] = entry
        else:
            results[f"main_{POLICY_NAME[s]}"] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out.with_suffix(".json"), "w"), indent=2, default=str)
    plot(results, max_depth, out.with_suffix(".png"))

    print(f"depths {results['depths']}, target {args.target}")
    for g, m in results["main"]["summary"].items():
        print(f"  [{g}] final {m['final_acc']:.2f}  agreement-routed {m['routed_acc']:.2f}  "
              f"random-matched {m['random_matched_acc']:.2f}  depth {m['mean_depth']:.2f}  "
              f"coverage {m['coverage']:.3f}  early-agree {m['early_agreement_pooled']:.3f}")
        st = results["static"][g]
        below = [p for p in st if p["depth"] <= m["mean_depth"]]
        if below:
            p = below[-1]
            print(f"        static exit @ depth {p['depth']} (<= routed mean depth): {p['acc']:.2f}")
    tot = {k: sum(b[k] for b in results["main"]["breakdown"].values())
           for k in next(iter(results["main"]["breakdown"].values()))}
    print("  early-exit outcomes (all datasets pooled):",
          {k: v for k, v in tot.items() if k != "net_accuracy_change_points"})
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
