"""Agreement-gated adaptive-depth inference.

Instead of pruning every example to the same fixed decoder depth (the paper's
existing keep-first-N contiguous pruning), stop the forward pass *per example*
as soon as a stopping rule fires, using the layer-wise probes as cheap,
causal, intermediate exit classifiers. Because decoder layers are causal, the
per-layer hidden state read from a full-depth forward pass is identical to
what an actually-truncated model at that depth would produce (this is exactly
the paper's own half-depth-backbone argument), so this simulation over cached
per-layer votes/margins is a faithful estimate of real truncated-depth
deployment -- no new backbone passes are needed.

Policies (all causal: layer L's decision only uses votes/margins at layers
<= L):
  - window_agreement(W): stop at the first layer L >= W-1 such that the last
    W layer-probe votes are all identical to each other.
  - margin_threshold(tau): stop at the first layer L with softmax margin
    (top-1 minus top-2 candidate probability) >= tau.
  - combined(W, tau): stop at the first layer satisfying either rule.
Any example that never satisfies its rule falls back to the final layer
(depth 28), matching what a real deployed system would do.

Cost is reported as the average number of decoder layers executed per
example (directly comparable to the paper's Figure 1 x-axis, "% of decoder
layers pruned" / retained depth, since compute scales ~linearly with
executed layers for a fixed backbone).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

d = np.load("outputs/agreement_analysis/raw_votes.npz")
votes = d["votes"]            # (n_q, 29) -- votes[:, L] is layer-L probe's predicted choice
margins = d["margins"]        # (n_q, 29) -- softmax margin of layer-L probe
gold = d["gold"]              # (n_q,)
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
    """Require BOTH window agreement and a margin floor before stopping --
    more conservative than either rule alone."""
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
    return {
        "avg_depth": float(stop_layer.mean()),
        "accuracy": float((pred == gold).mean()),
    }


results = {"fixed_depth": [{"depth": L, "accuracy": fixed_depth_accuracy[L]} for L in range(n_layers)]}

results["window_agreement"] = []
for W in [1, 2, 3, 4, 5, 6, 8, 10]:
    sl, pred = window_agreement_policy(W)
    r = evaluate(sl, pred)
    r["window"] = W
    results["window_agreement"].append(r)

results["margin_threshold"] = []
for tau in [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9]:
    sl, pred = margin_threshold_policy(tau)
    r = evaluate(sl, pred)
    r["tau"] = tau
    results["margin_threshold"].append(r)

results["combined_or_W3"] = []
for tau in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    sl, pred = combined_or_policy(window=3, tau=tau)
    r = evaluate(sl, pred)
    r["tau"] = tau
    results["combined_or_W3"].append(r)

results["combined_and"] = []
for window in [2, 3, 4]:
    for tau in [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3]:
        sl, pred = combined_and_policy(window=window, tau=tau)
        r = evaluate(sl, pred)
        r["tau"] = tau
        r["window"] = window
        results["combined_and"].append(r)

# headline comparison: at matched avg depth ~14 (the paper's chosen half-depth
# point), what does each adaptive policy achieve vs. the fixed-depth curve?
target_depth = 14
fixed_at_target = fixed_depth_accuracy[target_depth]
results["headline"] = {
    "target_depth": target_depth,
    "fixed_depth_accuracy_at_target": fixed_at_target,
    "final_layer_accuracy": fixed_depth_accuracy[final_layer],
}

Path("outputs/adaptive_depth").mkdir(parents=True, exist_ok=True)
Path("outputs/adaptive_depth/results.json").write_text(json.dumps(results, indent=2))
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
