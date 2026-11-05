"""Paper figures and tables for the 2B reproduction, seeds and ablations.

First run (needs outputs/ on this box) extracts the curves into figure_data.json; later runs
rebuild every figure from that file alone, so the figures survive losing outputs/.

    python early_exit_probe/results/make_figures.py            # 3-domain pool
    python early_exit_probe/results/make_figures.py --pool 5d  # 5-domain pool (no latency figure)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent / "outputs"
FIG = HERE / "figures"
D = 28
MATCHED = (11.2, 14.0, 15.7)  # 40%, 50%, 56% of layers (56% = paper's 2B routed depth)

# Validated categorical slots (dataviz reference palette, light): blue, orange, aqua; neutrals.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
POLICY = {"agreement": ("Agreement head (ours)", BLUE, "o"),
          "margin": ("Top-2 margin", ORANGE, "s"),
          "maxprob": ("Max prob.", AQUA, "^")}
RUNS = {  # display name -> output dir
    "Main (seed 1)": "repro2b_el_s26091615", "Main (seed 2)": "repro2b_el_s26091616",
    "Main (seed 3)": "repro2b_el_s26091617",
    "No FiLM (depth-blind trunk)": "repro2b_el_no_film_s26091615",
    "Independent per-depth trunks": "repro2b_el_independent_s26091615",
    "No last-option context": "repro2b_el_no_last_s26091615",
    "Selected depths (8/12/20/28)": "repro2b_el_selected_depth_s26091615",
    "No LoRA (frozen backbone)": "repro2b_el_no_lora_s26091615",
}
SEEDS = [k for k in RUNS if k.startswith("Main")]
ABLATIONS = [k for k in RUNS if not k.startswith("Main")]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "pdf.fonttype": 42, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def extract(prefix, latency=True):
    data = {"runs": {}}
    for name, d in RUNS.items():
        d = d.replace("repro2b_el_", prefix, 1)
        f = sorted(glob.glob(str(OUTPUTS / d / "checkpoint-*" / "baselines" / "sha200.json")))[-1]
        r = json.load(open(f))
        run = {"main": r["main"]["summary"]["all"],
               "main_groups": {g: r["main"]["summary"][g] for g in ("classification", "vqa")},
               "static": {str(x["depth"]): x["acc"] for x in r["static"]["all"]},
               "sweeps": {p: [{"target": x["target"], **{k: x["summary"]["all"][k] for k in
                                ("mean_depth", "routed_acc", "random_matched_acc")}}
                              for x in r["sweeps"][p]] for p in POLICY}}
        data["runs"][name] = run
    if not latency:
        return data
    lat = OUTPUTS / RUNS["Main (seed 1)"] / "checkpoint-939" / "latency"
    data["paper_latency"] = json.load(open(lat / "paper_2b_estimate.json"))
    data["halting"] = json.load(open(lat / "halting_bs1.json"))
    return data


def curve(run, policy):
    pts = sorted((x["mean_depth"], x["routed_acc"]) for x in run["sweeps"][policy])
    return np.array(pts).T


def at_depth(run, policy, d):
    x, y = curve(run, policy)
    return float(np.interp(d, x, y, left=np.nan, right=np.nan))


def static_at(run, d):
    s = run["static"]
    return float(np.interp(d, [int(k) for k in s], list(s.values())))


SUFFIX = ""  # "_5d" for the 5-domain pool


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}{SUFFIX}.{ext}")
    plt.close(fig)


def fig_tradeoff(data):
    """Accuracy vs mean exit depth, mean over 3 seeds with min-max band."""
    seeds = [data["runs"][s] for s in SEEDS]
    grid = np.linspace(7, 24, 120)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for p, (label, color, marker) in POLICY.items():
        ys = np.array([np.interp(grid, *curve(r, p), left=np.nan, right=np.nan) for r in seeds])
        ok = ~np.isnan(ys).any(0)
        ax.fill_between(grid[ok], ys.min(0)[ok], ys.max(0)[ok], color=color, alpha=0.18, lw=0)
        ax.plot(grid[ok], ys.mean(0)[ok], color=color, label=label, marker=marker,
                markevery=15, ms=4, mec="white", mew=0.6)
    st = np.array([[static_at(r, d) for d in grid] for r in seeds]).mean(0)
    ax.plot(grid, st, color=INK2, ls="--", lw=1.3, label="Static exit (all rows at one depth)")
    rnd = np.array([np.interp(grid, *np.array(sorted((x["mean_depth"], x["random_matched_acc"])
                    for x in r["sweeps"]["agreement"])).T, left=np.nan, right=np.nan) for r in seeds])
    ok = ~np.isnan(rnd).any(0)
    ax.plot(grid[ok], rnd.mean(0)[ok], color=MUTED, ls=":", lw=1.3, label="Random, depth-matched")
    final = np.mean([r["main"]["final_acc"] for r in seeds])
    ax.axhline(final, color=INK, lw=0.8)
    ax.text(24, final + 0.25, f"full depth {final:.1f}", ha="right", va="bottom", fontsize=6.5, color=INK)
    ax.axvline(15.7, color=MUTED, lw=0.7, ls="-.")
    ax.text(15.55, 68.3, "paper's routed depth (56%)", fontsize=6, color=INK2, ha="right", va="top")
    ax.set_xlabel("Mean exit depth (layers of 28)")
    ax.set_ylabel("Accuracy, all 10 datasets (%)")
    ax.set_xlim(7, 24)
    ax.set_ylim(56, 68.5)
    sec = ax.secondary_xaxis("top", functions=(lambda d: 100 * d / D, lambda p: p * D / 100))
    sec.set_xlabel("% of layers", fontsize=7, color=INK2)
    ax.legend(loc="lower right", frameon=False, handlelength=2.2)
    save(fig, "fig1_accuracy_vs_depth")


def fig_ablations(data):
    """Dot plot at matched depth: each ablation vs the 3-seed band of the main recipe."""
    runs = data["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    names = ABLATIONS[::-1]
    vals = [at_depth(runs[n], p, d) for n in names for p in ("agreement", "margin") for d in MATCHED]
    xlo = min(59, np.floor(np.nanmin(vals)) - 1)  # 59 unless a point would fall off the axis
    for ax, d in zip(axes, MATCHED):
        seed_vals = [at_depth(runs[s], "agreement", d) for s in SEEDS]
        lo, hi, mu = min(seed_vals), max(seed_vals), np.mean(seed_vals)
        ax.axvspan(lo, hi, color=BLUE, alpha=0.15, lw=0)
        ax.axvline(mu, color=BLUE, lw=1.2)
        for i, n in enumerate(names):
            a = at_depth(runs[n], "agreement", d)
            m = at_depth(runs[n], "margin", d)
            if not np.isnan(a):
                ax.plot(a, i, "o", color=BLUE, ms=5, mec="white", mew=0.6, zorder=3)
                ax.text(a, i + 0.28, f"{a - mu:+.1f}", ha="center", fontsize=6, color=INK2)
            else:
                ax.text(xlo + 0.5 if np.isnan(m) else m + 0.6, i, "agreement never\nexits this early", fontsize=6, color=MUTED, va="center")
            if not np.isnan(m):
                ax.plot(m, i, "s", mfc="none", mec=ORANGE, ms=4.5, mew=1.1, zorder=3)
        ax.set_title(f"{d:g} layers ({100 * d / D:.0f}%)")
        ax.set_xlim(xlo, 68.5 if xlo < 59 else 68)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Accuracy (%)")
    axes[0].set_yticks(range(len(names)), names)
    axes[0].set_ylim(-0.6, len(names) - 0.2)
    handles = [plt.Line2D([], [], marker="o", color=BLUE, ls="", label="Agreement routing (label: Δ vs main)"),
               plt.Line2D([], [], marker="s", mfc="none", mec=ORANGE, ls="", label="Top-2 margin routing"),
               plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.25, label="Main recipe, 3-seed range")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.6, 1.08))
    save(fig, "fig2_ablations_matched_depth")


def fig_layers(data):
    """Static-exit accuracy by layer: every-layer training plateaus by ~16; LoRA is what makes it."""
    runs = data["runs"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    depths = np.arange(1, D + 1)
    main = np.array([[runs[s]["static"][str(d)] for d in depths] for s in SEEDS])
    ax.fill_between(depths, main.min(0), main.max(0), color=BLUE, alpha=0.18, lw=0)
    ax.plot(depths, main.mean(0), color=BLUE, label="Every-layer joint + LoRA (3 seeds)")
    nl = [runs["No LoRA (frozen backbone)"]["static"][str(d)] for d in depths]
    ax.plot(depths, nl, color=ORANGE, marker="s", markevery=3, ms=3.5, mec="white", mew=0.5,
            label="No LoRA")
    sd = runs["Selected depths (8/12/20/28)"]["static"]
    ax.plot([int(k) for k in sd], list(sd.values()), color=AQUA, marker="^", ms=5, mec="white",
            mew=0.6, ls="none", label="Selected depths only")
    peak = int(depths[main.mean(0).argmax()])
    ax.annotate(f"plateau from ~16\n(peak @ {peak}: {main.mean(0).max():.1f})", xy=(peak, main.mean(0).max()),
                xytext=(19.5, 50), fontsize=6.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set_xlabel("Exit layer (all rows exit here)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(1, D)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "fig3_accuracy_by_layer")


def fig_latency(data):
    """Measured wall-clock speedup vs the layer-count proxy at the paper's 2B exit depths."""
    pl = data["paper_latency"]
    rows = pl.get("per_dataset", pl.get("rows", pl))
    rows = {k: v for k, v in rows.items() if isinstance(v, dict) and "speedup" in v}
    names = sorted(rows, key=lambda n: rows[n]["n_choices"])
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for i, n in enumerate(names):
        r = rows[n]
        ax.plot([r["speedup"], r["depth_proxy_speedup"]], [i, i], color=GRID, lw=2.5, zorder=1)
        ax.plot(r["depth_proxy_speedup"], i, "o", mfc="white", mec=INK2, ms=5, mew=1, zorder=2)
        ax.plot(r["speedup"], i, "o", color=BLUE, ms=5.5, mec="white", mew=0.6, zorder=3)
        ax.text(r["speedup"] - 0.04, i, f"{r['speedup']:.2f}×", ha="right", va="center", fontsize=6, color=INK2)
    ax.set_yticks(range(len(names)), [f"{n} (K={rows[n]['n_choices']})" for n in names])
    ax.set_xlabel("Speedup over full depth (×), V100, batch 1")
    ax.set_xlim(0.9, 2.7)
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", color=BLUE, label="Measured (real decoder halting)")
    ax.plot([], [], "o", mfc="white", mec=INK2, label="Layer-count proxy (28 / mean depth)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, frameon=False, fontsize=6.5)
    save(fig, "fig4_latency")


def tables(data):
    runs = data["runs"]
    cols = []
    for n in RUNS:
        r = runs[n]
        cols.append((n, r["main"]["final_acc"], r["main"]["routed_acc"], r["main"]["mean_depth"],
                     [at_depth(r, "agreement", d) for d in MATCHED], [static_at(r, d) for d in MATCHED],
                     [at_depth(r, "margin", d) for d in MATCHED]))
    fmt = lambda v: "–" if np.isnan(v) else f"{v:.1f}"
    md = ["| Run | Final | Routed@0.9 | Depth | Agreement @ " + " / ".join(f"{d:g}" for d in MATCHED)
          + " | Static @ same | Margin @ same |", "|" + "---|" * 7]
    tex = [r"\begin{tabular}{lccc|ccc|ccc|ccc}", r"\toprule",
           r"& & & & \multicolumn{3}{c|}{Agreement} & \multicolumn{3}{c|}{Static exit} & \multicolumn{3}{c}{Top-2 margin} \\",
           r"Run & Final & Routed & $\bar d$ & " + " & ".join([f"{d:g}" for d in MATCHED] * 3) + r" \\", r"\midrule"]
    seed_rows = [c for c in cols if c[0] in SEEDS]
    mean = ("Main, mean of 3 seeds", *[np.mean([c[i] for c in seed_rows]) for i in (1, 2, 3)],
            *[list(np.mean([c[i] for c in seed_rows], 0)) for i in (4, 5, 6)])
    for c in [mean] + [c for c in cols if c[0] not in SEEDS]:
        md.append(f"| {c[0]} | {c[1]:.2f} | {c[2]:.2f} | {c[3]:.1f} | " + " / ".join(map(fmt, c[4]))
                  + " | " + " / ".join(map(fmt, c[5])) + " | " + " / ".join(map(fmt, c[6])) + " |")
        tex.append(f"{c[0]} & {c[1]:.1f} & {c[2]:.1f} & {c[3]:.1f} & " + " & ".join(map(fmt, c[4] + c[5] + c[6])) + r" \\")
    spread = [max(c[i] for c in seed_rows) - min(c[i] for c in seed_rows) for i in (1, 2)]
    md.append(f"\nSeed range (max−min): final {spread[0]:.2f}, routed {spread[1]:.2f}. "
              f"Matched depths {', '.join(f'{d:g}' for d in MATCHED)} = "
              f"{', '.join(f'{100 * d / D:.0f}%' for d in MATCHED)} of 28 layers.")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (HERE / f"table_ablations{SUFFIX}.md").write_text("\n".join(md) + "\n")
    (HERE / f"table_ablations{SUFFIX}.tex").write_text("\n".join(tex) + "\n")


def main():
    global SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=("3d", "5d"), default="3d")
    five = ap.parse_args().pool == "5d"
    SUFFIX, prefix = ("_5d", "repro2b_el5d_") if five else ("", "repro2b_el_")
    data_file = HERE / f"figure_data{SUFFIX}.json"
    if OUTPUTS.exists() and (OUTPUTS / RUNS["Main (seed 1)"].replace("repro2b_el_", prefix, 1)).exists():
        data = extract(prefix, latency=not five)
        json.dump(data, open(data_file, "w"), indent=1)
    else:
        data = json.load(open(data_file))
    fig_tradeoff(data)
    fig_ablations(data)
    fig_layers(data)
    if not five:
        fig_latency(data)
    tables(data)
    print(f"wrote {FIG}/fig1-{3 if five else 4}{SUFFIX} (.pdf/.png), table_ablations{SUFFIX}.md/.tex, {data_file.name}")


if __name__ == "__main__":
    main()
