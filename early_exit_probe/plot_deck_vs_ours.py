"""Side-by-side bars: deck (260914_keep_it_simple.pdf, page 7) vs. our reproduction on the
10 evaluation sets (7 classification + 3 VQA). Ours is read from eval_mmeb_results.json (and
eval_results.json for A-OKVQA); the deck numbers are transcribed from the page-7 tables.

Top panel: "Joint 2-head" (full depth 28). Bottom panel: "90% agreement" routed accuracy,
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

# Categorical slots 1 and 2 of the reference palette (validated: CVD dE 24.7, contrast >= 3:1)
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument("--results", default=str(root / "outputs/phase1_run1/checkpoint-500/eval_mmeb_results.json"))
    ap.add_argument("--out", default=str(root / "outputs/deck_vs_ours"))
    args = ap.parse_args()

    ours_raw = json.load(open(args.results))
    aokvqa = json.load(open(Path(args.results).with_name("eval_results.json")))
    ours_raw["A-OKVQA"] = aokvqa
    ours = {n: (ours_raw[n]["full_depth28_accuracy"] * 100, ours_raw[n]["routed_accuracy"] * 100,
                ours_raw[n]["routed_mean_depth"]) for n in DECK}

    def avg(group):
        return tuple(np.mean([ours[n][i] for n in group]) for i in range(3))

    ours_cls, ours_vqa = avg(CLS), avg(VQA)
    names_all = CLS + ["Avg. (cls)"] + VQA + ["Avg. (VQA)"]
    deck_rows = [DECK[n] for n in CLS] + [DECK_AVG["cls"]] + [DECK[n] for n in VQA] + [DECK_AVG["vqa"]]
    ours_rows = [ours[n] for n in CLS] + [ours_cls] + [ours[n] for n in VQA] + [ours_vqa]
    label_rows = [(n, WAYS[n]) for n in CLS] + [("Average", "7 sets")] + [(n, WAYS[n]) for n in VQA] \
        + [("Average", "3 sets")]

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": GRID})
    fig, axes = plt.subplots(2, 1, figsize=(16, 8.6), facecolor=SURFACE)
    x = np.arange(len(names_all))
    w = 0.3

    panels = [
        (0, "Full depth 28 (\"Joint 2-head\")", "Top-1 accuracy (%)"),
        (1, "Early exit at 90% agreement (routed)", "Top-1 accuracy (%)"),
    ]
    for ax, (col, title, ylabel) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        d = [r[col] for r in deck_rows]
        o = [r[col] for r in ours_rows]
        b1 = ax.bar(x - w / 2 - 0.01, d, w, color=BLUE, label="Deck (5-dataset training pool)", zorder=3)
        b2 = ax.bar(x + w / 2 + 0.01, o, w, color=ORANGE, label="Ours (A-OKVQA-only, 500 steps)", zorder=3)
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}",
                        ha="center", va="bottom", fontsize=7.5, color=INK2)
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
    depth_pairs = [(DECK[n][2], ours[n][2]) for n in CLS] + [(DECK_AVG["cls"][2], ours_cls[2])] \
        + [(DECK[n][2], ours[n][2]) for n in VQA] + [(DECK_AVG["vqa"][2], ours_vqa[2])]
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{n}\ndepth {d:.1f} | {o:.1f}" for (n, _), (d, o) in zip(label_rows, depth_pairs)],
                            fontsize=9)
    axes[1].text(1.0, -0.27, "mean exit depth: deck | ours", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=8.5, color=INK2)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.006, 0.935), ncol=2, frameon=False, fontsize=10.5)
    fig.suptitle("Early Exit Probing on the 10 MMEB evaluation sets: deck vs. our reproduction",
                 x=0.012, y=0.985, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.012, 0.952, "Native candidate sets. Ours trained on A-OKVQA only (evaluated on its 1,145-row val split); every other set is "
             "zero-transfer, and routing thresholds were calibrated on A-OKVQA.", fontsize=9.5, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.885), h_pad=3.0)

    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
