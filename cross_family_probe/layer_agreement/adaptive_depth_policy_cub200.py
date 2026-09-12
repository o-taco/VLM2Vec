"""Agreement-gated adaptive-depth inference -- CUB-200 variant.

Identical method to ``adaptive_depth_policy.py`` (see that file's docstring
for the full rationale); only the input/output paths and n_choices differ
(CUB-200 uses n_choices=4 via extract_cub200_layers.py's sampled-choices
protocol, so the vote/margin machinery is unchanged).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

d = np.load("outputs/cub200/agreement_analysis/raw_votes.npz")
votes = d["votes"]
margins = d["margins"]
gold = d["gold"]
n_q, n_layers = votes.shape
final_layer = n_layers - 1
fixed_depth_accuracy = [(votes[:, L] == gold).mean() for L in range(n_layers)]


def window_agreement_policy(window: int, min_depth: int = 0):
    stop_layer = np.full(n_q, final_layer, dtype=np.int64)
    stopped = np.zeros(n_q, dtype=bool)
    for L in range(max(window - 1, min_depth), n_layers):
        if stopped.all():
            break
        lo = L - window + 1
        if lo < 0:
            continue
        window_votes = votes[:, lo:L + 1]
        all_agree = (window_votes == window_votes[:, [0]]).all(axis=1)
        newly_stopped = all_agree & ~stopped
        stop_layer[newly_stopped] = L
        stopped |= newly_stopped
    pred = votes[np.arange(n_q), stop_layer]
    return stop_layer, pred


def margin_threshold_policy(tau: float, min_depth: int = 0):
    stop_layer = np.full(n_q, final_layer, dtype=np.int64)
    stopped = np.zeros(n_q, dtype=bool)
    for L in range(min_depth, n_layers):
        if stopped.all():
            break
        newly_stopped = (margins[:, L] >= tau) & ~stopped
        stop_layer[newly_stopped] = L
        stopped |= newly_stopped
    pred = votes[np.arange(n_q), stop_layer]
    return stop_layer, pred


def combined_or_policy(window: int, tau: float, min_depth: int = 0):
    stop_layer = np.full(n_q, final_layer, dtype=np.int64)
    stopped = np.zeros(n_q, dtype=bool)
    for L in range(min_depth, n_layers):
        if stopped.all():
            break
        margin_hit = margins[:, L] >= tau
        agree_hit = np.zeros(n_q, dtype=bool)
        lo = L - window + 1
        if lo >= 0:
            window_votes = votes[:, lo:L + 1]
            agree_hit = (window_votes == window_votes[:, [0]]).all(axis=1)
        newly_stopped = (margin_hit | agree_hit) & ~stopped
        stop_layer[newly_stopped] = L
        stopped |= newly_stopped
    pred = votes[np.arange(n_q), stop_layer]
    return stop_layer, pred


def combined_and_policy(window: int, tau: float, min_depth: int = 0):
    stop_layer = np.full(n_q, final_layer, dtype=np.int64)
    stopped = np.zeros(n_q, dtype=bool)
    for L in range(min_depth, n_layers):
        if stopped.all():
            break
        margin_hit = margins[:, L] >= tau
        agree_hit = np.zeros(n_q, dtype=bool)
        lo = L - window + 1
        if lo >= 0:
            window_votes = votes[:, lo:L + 1]
            agree_hit = (window_votes == window_votes[:, [0]]).all(axis=1)
        newly_stopped = (margin_hit & agree_hit) & ~stopped
        stop_layer[newly_stopped] = L
        stopped |= newly_stopped
    pred = votes[np.arange(n_q), stop_layer]
    return stop_layer, pred


def evaluate(stop_layer, pred):
    return {"avg_depth": float(stop_layer.mean()), "accuracy": float((pred == gold).mean())}


results = {"fixed_depth": [{"depth": L, "accuracy": fixed_depth_accuracy[L]} for L in range(n_layers)]}

results["window_agreement"] = []
for W in [1, 2, 3, 4, 5, 6, 8, 10]:
    sl, pred = window_agreement_policy(W)
    r = evaluate(sl, pred); r["window"] = W
    results["window_agreement"].append(r)

results["margin_threshold"] = []
for tau in [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9]:
    sl, pred = margin_threshold_policy(tau)
    r = evaluate(sl, pred); r["tau"] = tau
    results["margin_threshold"].append(r)

results["combined_or_W3"] = []
for tau in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    sl, pred = combined_or_policy(window=3, tau=tau)
    r = evaluate(sl, pred); r["tau"] = tau
    results["combined_or_W3"].append(r)

results["combined_and"] = []
for window in [2, 3, 4]:
    for tau in [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3]:
        sl, pred = combined_and_policy(window=window, tau=tau)
        r = evaluate(sl, pred); r["tau"] = tau; r["window"] = window
        results["combined_and"].append(r)

target_depth = 14
results["headline"] = {
    "target_depth": target_depth,
    "fixed_depth_accuracy_at_target": fixed_depth_accuracy[target_depth],
    "final_layer_accuracy": fixed_depth_accuracy[final_layer],
}

Path("outputs/cub200/adaptive_depth").mkdir(parents=True, exist_ok=True)
Path("outputs/cub200/adaptive_depth/results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results["headline"], indent=2))
print("\nwindow_agreement:")
for r in results["window_agreement"]:
    print(f"  W={r['window']:2d}  avg_depth={r['avg_depth']:5.2f}  accuracy={r['accuracy']:.4f}")
print("\nmargin_threshold:")
for r in results["margin_threshold"]:
    print(f"  tau={r['tau']:.2f}  avg_depth={r['avg_depth']:5.2f}  accuracy={r['accuracy']:.4f}")
print("\ncombined_or_W3:")
for r in results["combined_or_W3"]:
    print(f"  tau={r['tau']:.2f}  avg_depth={r['avg_depth']:5.2f}  accuracy={r['accuracy']:.4f}")
print("\ncombined_and:")
for r in results["combined_and"]:
    print(f"  W={r['window']} tau={r['tau']:.2f}  avg_depth={r['avg_depth']:5.2f}  accuracy={r['accuracy']:.4f}")
