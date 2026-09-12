"""Fit one frozen logistic-regression probe per decoder layer, then test
whether cross-layer prediction agreement is a useful confidence signal.

Hypothesis under test (from the paper's existing layer-wise probe analysis,
Appendix Figs 11-12): if independent per-layer probes mostly agree on the same
candidate, the backbone has converged on that answer early and is more likely
correct; if per-layer probes disagree, the representation is still unsettled
and the final answer is less trustworthy.

For each validation question and each layer L we get a predicted candidate
from a probe trained only on layer-L features. We then define, per question:
  - plurality_frac: fraction of layers whose vote equals the plurality vote
  - agree_with_final: fraction of layers whose vote equals the last layer's vote
  - vote_entropy: entropy of the per-layer vote distribution (in bits)
and test each as a predictor of whether the FINAL layer's probe is correct,
compared against the final layer's own softmax margin (the standard
single-layer confidence baseline).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_layer(x_train, y_train, x_val, c=1e-3):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, solver="lbfgs", max_iter=500, class_weight="balanced"),
    )
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_val)[:, 1]


def per_question_argmax(scores, qids, cids, n_choices=4):
    """Return (n_questions,) predicted choice id and (n_questions,) softmax margin."""
    uniq = np.unique(qids)
    pred = np.zeros(len(uniq), dtype=np.int64)
    margin = np.zeros(len(uniq), dtype=np.float64)
    for i, qid in enumerate(uniq):
        mask = qids == qid
        s = scores[mask]
        c = cids[mask]
        order = np.argsort(-s)
        pred[i] = c[order[0]]
        exp = np.exp(s - s.max())
        p = exp / exp.sum()
        p_sorted = np.sort(p)[::-1]
        margin[i] = p_sorted[0] - (p_sorted[1] if len(p_sorted) > 1 else 0.0)
    return uniq, pred, margin


def vote_entropy_bits(votes_row, n_choices):
    counts = np.bincount(votes_row, minlength=n_choices).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def risk_coverage(confidence, correct, n_points=20):
    order = np.argsort(-confidence)
    correct_sorted = correct[order]
    n = len(correct)
    coverages, accuracies = [], []
    for frac in np.linspace(1.0 / n, 1.0, n_points):
        k = max(1, int(round(frac * n)))
        coverages.append(k / n)
        accuracies.append(float(correct_sorted[:k].mean()))
    return coverages, accuracies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--c", type=float, default=1e-3)
    ap.add_argument("--n-choices", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = np.load(args.train_cache)
    val = np.load(args.val_cache)

    x_train_all = train["features"]  # (N, n_layers+1, hidden)
    y_train = train["labels"]
    x_val_all = val["features"]
    val_qids_full = val["question_ids"]
    val_cids_full = val["choice_ids"]
    val_labels_full = val["labels"]

    n_layers_plus1 = x_train_all.shape[1]
    print(f"n_layers+1 = {n_layers_plus1}, train candidates = {len(y_train)}, "
          f"val candidates = {len(val_qids_full)}")

    uniq_qids = np.unique(val_qids_full)
    n_q = len(uniq_qids)
    votes = np.zeros((n_q, n_layers_plus1), dtype=np.int64)
    margins = np.zeros((n_q, n_layers_plus1), dtype=np.float64)
    layer_accuracy = np.zeros(n_layers_plus1, dtype=np.float64)

    # gold answer per question (fixed across layers)
    gold = np.zeros(n_q, dtype=np.int64)
    for i, qid in enumerate(uniq_qids):
        mask = val_qids_full == qid
        gold[i] = val_cids_full[mask][np.argmax(val_labels_full[mask])]

    for layer in range(n_layers_plus1):
        x_train = x_train_all[:, layer, :].astype(np.float32)
        x_val = x_val_all[:, layer, :].astype(np.float32)
        scores = fit_layer(x_train, y_train, x_val, c=args.c)
        q_order, pred, margin = per_question_argmax(scores, val_qids_full, val_cids_full, args.n_choices)
        assert np.array_equal(q_order, uniq_qids)
        votes[:, layer] = pred
        margins[:, layer] = margin
        layer_accuracy[layer] = float((pred == gold).mean())
        print(f"layer {layer:2d}/{n_layers_plus1 - 1}: val accuracy = {layer_accuracy[layer]:.4f}")

    final_layer = n_layers_plus1 - 1
    final_pred = votes[:, final_layer]
    final_correct = (final_pred == gold).astype(np.int64)
    final_margin = margins[:, final_layer]

    plurality_frac = np.zeros(n_q)
    agree_with_final = np.zeros(n_q)
    vote_entropy = np.zeros(n_q)
    for i in range(n_q):
        row = votes[i]
        counts = np.bincount(row, minlength=args.n_choices)
        plurality_frac[i] = counts.max() / n_layers_plus1
        agree_with_final[i] = float((row == final_pred[i]).mean())
        vote_entropy[i] = vote_entropy_bits(row, args.n_choices)

    def auroc(signal, higher_is_more_confident=True):
        s = signal if higher_is_more_confident else -signal
        return float(roc_auc_score(final_correct, s))

    results = {
        "n_layers_plus1": n_layers_plus1,
        "n_val_questions": int(n_q),
        "final_layer_accuracy": float(final_correct.mean()),
        "layer_accuracy_curve": layer_accuracy.tolist(),
        "auroc_correctness": {
            "final_layer_softmax_margin": auroc(final_margin, True),
            "plurality_frac_all_layers": auroc(plurality_frac, True),
            "agree_with_final_all_layers": auroc(agree_with_final, True),
            "vote_entropy_all_layers": auroc(vote_entropy, False),
        },
        "corr_with_final_margin": {
            "plurality_frac": float(np.corrcoef(plurality_frac, final_margin)[0, 1]),
            "agree_with_final": float(np.corrcoef(agree_with_final, final_margin)[0, 1]),
        },
        "accuracy_by_plurality_bin": {},
    }

    # accuracy conditioned on agreement level (the paper's own framing: "when
    # layers agree, is the model more likely right?")
    bins = [0.0, 0.4, 0.6, 0.8, 1.0001]
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (plurality_frac >= lo) & (plurality_frac < hi)
        if mask.sum() > 0:
            results["accuracy_by_plurality_bin"][f"[{lo:.2f},{hi:.2f})"] = {
                "n": int(mask.sum()), "accuracy": float(final_correct[mask].mean()),
            }

    # late-layers-only variant (layers past the accuracy plateau, matching
    # Appendix Fig 12's observation that A-OKVQA plateaus around layer 14-16)
    late_start = n_layers_plus1 // 2
    late_votes = votes[:, late_start:]
    late_plurality_frac = np.zeros(n_q)
    for i in range(n_q):
        counts = np.bincount(late_votes[i], minlength=args.n_choices)
        late_plurality_frac[i] = counts.max() / late_votes.shape[1]
    results["auroc_correctness"]["plurality_frac_late_half_layers"] = auroc(late_plurality_frac, True)
    results["late_layers_start_index"] = late_start

    # risk-coverage (selective prediction) curves
    for name, sig, higher in [
        ("final_softmax_margin", final_margin, True),
        ("plurality_frac_all_layers", plurality_frac, True),
        ("plurality_frac_late_half", late_plurality_frac, True),
    ]:
        cov, acc = risk_coverage(sig if higher else -sig, final_correct)
        results.setdefault("risk_coverage", {})[name] = {"coverage": cov, "accuracy": acc}

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(
        out_dir / "raw_votes.npz",
        votes=votes, margins=margins, gold=gold, question_ids=uniq_qids,
        final_correct=final_correct, plurality_frac=plurality_frac,
        agree_with_final=agree_with_final, vote_entropy=vote_entropy,
        late_plurality_frac=late_plurality_frac,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
