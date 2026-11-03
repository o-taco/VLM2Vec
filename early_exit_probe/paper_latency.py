"""Measured-latency estimate for the paper's reported Qwen3-VL-2B every-layer results, without
their per-row scores: wall clock depends on where rows exit, not on the trained weights, so
the every-layer profile from `latency.py --profile_every_layer` (agreement checked at every
layer) is evaluated at each dataset's reported mean selected depth (paper Tables 6-7).
Using the mean is exact when time is affine in depth; the profile's fit residual bounds the
error and is printed alongside.
"""
from __future__ import annotations

import argparse
import json
import statistics

import numpy as np

# Paper Tables 6-7, "Ours every-layer routed": mean selected depth of 28.
PAPER_2B_ROUTED_DEPTH = {
    "N24News": 14.21, "HatefulMemes": 22.56, "VOC2007": 12.59, "ImageNet-A": 17.51,
    "ImageNet-R": 12.96, "ObjectNet": 12.08, "Country211": 18.76,
    "A-OKVQA": 14.53, "Visual7W": 15.20, "ScienceQA": 16.51,
}
CLASSIFICATION = ("N24News", "HatefulMemes", "VOC2007", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211")
VQA = ("A-OKVQA", "Visual7W", "ScienceQA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="latency.py --profile_every_layer output")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    prof = json.load(open(args.profile))
    depths = prof["depths"]

    rows = {}
    print(f"{prof['gpu']}, batch size {prof['batch_size']}")
    print(f"{'dataset':13s}{'K':>4s} {'depth':>6s} {'full ms':>8s} {'routed':>7s} {'gen':>6s} "
          f"{'gen/routed':>10s} {'speedup':>8s} {'proxy':>6s} {'fit resid':>9s}")
    for name, d_bar in PAPER_2B_ROUTED_DEPTH.items():
        s = prof["per_dataset"].get(name)
        if s is None:
            continue
        t = np.asarray(s["ms_by_depth"])
        full, routed = float(t[-1]), float(np.interp(d_bar, depths, t))
        r = {"n_choices": s["n_choices"], "mean_depth": d_bar, "full_ms": full, "routed_ms": routed,
             "speedup": full / routed, "depth_proxy_speedup": depths[-1] / d_bar,
             "fit_residual_ms": s["affine_fit"]["max_abs_residual_ms"]}
        if "gen_ms" in s:
            r["gen_ms"], r["gen_over_routed"] = s["gen_ms"], s["gen_ms"] / routed
        rows[name] = r
        g = f"{r['gen_ms']:6.0f} {r['gen_over_routed']:9.2f}x" if "gen_ms" in r else f"{'-':>6s} {'-':>10s}"
        print(f"{name:13s}{r['n_choices']:4d} {d_bar:6.2f} {full:8.0f} {routed:7.0f} {g} "
              f"{r['speedup']:7.2f}x {r['depth_proxy_speedup']:5.2f}x {r['fit_residual_ms']:8.1f}")
    for g, names in (("classification", CLASSIFICATION), ("vqa", VQA), ("all", CLASSIFICATION + VQA)):
        present = [n for n in names if n in rows]
        if present:
            m = lambda k: statistics.fmean(rows[n][k] for n in present)
            print(f"[{g}] mean speedup {m('speedup'):.2f}x vs depth proxy {m('depth_proxy_speedup'):.2f}x"
                  + (f", generation {m('gen_over_routed'):.2f}x slower than routed" if all("gen_ms" in rows[n] for n in present) else ""))
    if args.out:
        json.dump({"gpu": prof["gpu"], "per_dataset": rows}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
