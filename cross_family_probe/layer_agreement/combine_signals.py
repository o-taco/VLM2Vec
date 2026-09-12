"""Does cross-layer agreement add information beyond the final layer's own
softmax margin? 5-fold CV logistic regression predicting final-layer
correctness from each signal alone and combined with the margin baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

d = np.load("outputs/agreement_analysis/raw_votes.npz")
margin = d["margins"][:, -1]
plurality = d["plurality_frac"]
agree_final = d["agree_with_final"]
entropy = d["vote_entropy"]
late_plurality = d["late_plurality_frac"]
y = d["final_correct"]

feature_sets = {
    "margin_only": np.stack([margin], axis=1),
    "plurality_only": np.stack([plurality], axis=1),
    "agree_with_final_only": np.stack([agree_final], axis=1),
    "margin + plurality": np.stack([margin, plurality], axis=1),
    "margin + agree_with_final": np.stack([margin, agree_final], axis=1),
    "margin + entropy": np.stack([margin, -entropy], axis=1),
    "margin + late_plurality": np.stack([margin, late_plurality], axis=1),
    "margin + plurality + entropy": np.stack([margin, plurality, -entropy], axis=1),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
results = {}
for name, X in feature_sets.items():
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    results[name] = {"mean_auroc": float(scores.mean()), "std_auroc": float(scores.std())}
    print(f"{name:32s} AUROC = {scores.mean():.4f} +/- {scores.std():.4f}")

Path("outputs/agreement_analysis/combined_signal_results.json").write_text(json.dumps(results, indent=2))
