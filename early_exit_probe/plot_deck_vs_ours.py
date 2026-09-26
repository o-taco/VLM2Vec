"""Side-by-side bars: deck (260914_keep_it_simple.pdf, page 7) vs. our reproduction on the
10 evaluation sets (7 classification + 3 VQA). Ours is read from eval_mmeb_results.json (and
eval_results.json for A-OKVQA); the deck numbers are transcribed from the page-7 tables.


with mean exit depth under each group. Ours is the A-OKVQA-only checkpoint, routed with
A-OKVQA-calibrated thresholds.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Deck page 7: (Joint 2-head @28, 90% agreement acc, 90% agreement mean depth)
DECK = {
    "N24News": (52.70, 52.75, 18.74),
    "HatefulMemes": (59.60, 59.13, 20.00),
    "VOC2007": (74.70, 75.75, 17.16),
    "ImageNet-A": (72.10, 71.63, 19.34),
    "ImageNet-R": (87.70, 89.13, 16.32),
    "ObjectNet": (75.30, 74.25, 15.30),
    "Country211": (16.30, 17.00, 19.96),
    "A-OKVQA": (79.30, 80.38, 18.18),
    "Visual7W": (88.50, 92.50, 20.00),
    "ScienceQA": (86.40, 84.75, 18.27),
}
CLS = ["N24News", "HatefulMemes", "VOC2007", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"]
VQA = ["A-OKVQA", "Visual7W", "ScienceQA"]
DECK_AVG = {"cls": (62.63, 62.80, 18.11), "vqa": (84.73, 85.88, 18.82)}
WAYS = {"N24News": "24-way", "HatefulMemes": "2-way", "VOC2007": "20-way", "ImageNet-A": "200-way",
        "ImageNet-R": "200-way", "ObjectNet": "113-way", "Country211": "211-way",
        "A-OKVQA": "4-way", "Visual7W": "4-way", "ScienceQA": "2-5-way"}

# Categorical slots 1-3 of the reference palette (validated all-pairs: CVD dE 9.2, normal-vision dE 24.0;
# aqua is 2.74:1 against the surface, so every bar carries a visible value label)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

ROOT = Path(__file__).resolve().parent
RUNS = [  # (legend label, short depth-caption tag, color, checkpoint dir)
    ("P1: ours, A-OKVQA only (500 steps)", "P1", ORANGE, ROOT / "outputs/phase1_run1/checkpoint-500"),
    ("P3: ours, A-OKVQA + ScienceQA (1000 steps)", "P3", AQUA, ROOT / "outputs/phase3_pool_run1/checkpoint-1000"),
]


def load_run(ckpt: Path):
    raw = json.load(open(ckpt / "eval_mmeb_results.json"))
    raw["A-OKVQA"] = json.load(open(ckpt / "eval_results.json"))
    return {n: (raw[n]["full_depth28_accuracy"] * 100, raw[n]["routed_accuracy"] * 100,
                raw[n]["routed_mean_depth"]) for n in DECK}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/deck_vs_ours"))
    args = ap.parse_args()

    runs = [(lab, tag, col, load_run(ck)) for lab, tag, col, ck in RUNS]

    def avg(res, group):
        return tuple(np.mean([res[n][i] for n in group]) for i in range(3))

    names_all = CLS + ["Avg. (cls)"] + VQA + ["Avg. (VQA)"]
    deck_rows = [DECK[n] for n in CLS] + [DECK_AVG["cls"]] + [DECK[n] for n in VQA] + [DECK_AVG["vqa"]]
    run_rows = [[res[n] for n in CLS] + [avg(res, CLS)] + [res[n] for n in VQA] + [avg(res, VQA)]
                for _, _, _, res in runs]
    label_rows = [(n, WAYS[n]) for n in CLS] + [("Average", "7 sets")] + [(n, WAYS[n]) for n in VQA] \
        + [("Average", "3 sets")]

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": GRID})
    fig, axes = plt.subplots(2, 1, figsize=(18, 9), facecolor=SURFACE)
    x = np.arange(len(names_all))
    n_series = 1 + len(runs)
    w = 0.26
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * (w + 0.01)

    panels = [
        (0, "Full depth 28 (\"Joint 2-head\")", "Top-1 accuracy (%)"),
        (1, "Early exit at 90% agreement (routed)", "Top-1 accuracy (%)"),
    ]
    for ax, (col, title, ylabel) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        series = [("Deck (5-dataset training pool)", BLUE, [r[col] for r in deck_rows])] + \
                 [(lab, c, [r[col] for r in rows]) for (lab, _, c, _), rows in zip(runs, run_rows)]
        for off, (lab, c, vals) in zip(offsets, series):
            bars = ax.bar(x + off, vals, w, color=c, label=lab, zorder=3)
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}",
                        ha="center", va="bottom", fontsize=6.8, color=INK2)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0, colors=INK2)
        ax.tick_params(axis="x", length=0, colors=INK)
        ax.set_xlim(-0.6, len(names_all) - 0.4)
        # separators around the classification average
        for cut in (len(CLS) - 0.5, len(CLS) + 0.5):
            ax.axvline(cut, color=GRID, linewidth=1.2, zorder=1)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{n}\n{w_}" for n, w_ in label_rows], fontsize=9)
    depth_cols = [[DECK[n][2] for n in CLS] + [DECK_AVG["cls"][2]] + [DECK[n][2] for n in VQA] + [DECK_AVG["vqa"][2]]] \
        + [[r[2] for r in rows] for rows in run_rows]
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f"{n}\n" + " | ".join(f"{col[i]:.1f}" for col in depth_cols) for i, (n, _) in enumerate(label_rows)],
        fontsize=9)
    axes[1].text(1.0, -0.27, "mean exit depth: deck | " + " | ".join(t for _, t, _, _ in runs), transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=8.5, color=INK2)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.006, 0.935), ncol=3, frameon=False, fontsize=10.5)
    fig.suptitle("Early Exit Probing on the 10 MMEB evaluation sets: deck vs. our reproduction",
                 x=0.012, y=0.985, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.012, 0.952, "Native candidate sets. Ours never trains on classification data; thresholds are calibrated on held-out "
             "rows of each run's own training pool. A-OKVQA is scored on its 1,145-row val split.",
             fontsize=9.5, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.885), h_pad=3.0)

    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
