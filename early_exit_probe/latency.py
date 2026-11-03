"""Measured early-halting latency (paper Sec. 3.4 TODO), as opposed to the offline replay.

The decoder really stops: a forward hook on each tapped decoder layer gathers the candidate
end-token states, runs the probe, and raises once the agreement head clears that depth's
threshold, so the remaining layers never execute. Hook outputs equal
`output_hidden_states[d]` exactly for this backbone (checked for every depth incl. the
last), so halted predictions match the offline replay.

Per evaluation row, at batch size 1, with LoRA merged into the weights:
  full     - all D layers + probe at D
  routed   - halting at the first tapped depth whose agreement clears its threshold
  fixed@d  - halting unconditionally at each tapped depth d (latency-vs-depth profile)
  prefix   - processor/vision/embedding time up to decoder layer 1 (the part no exit saves)
  gen      - optional: frozen base model, greedy numbered-choice generation, 64-token cap
Each timing is a CUDA-synchronized wall clock of the model call; CPU preprocessing
(processor + tokenization) is timed separately and reported, not folded in.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from early_exit_probe.batching import collate, collate_for_generation
from early_exit_probe.data_aokvqa import AOKVQADataset, CHOICE_LETTERS
from early_exit_probe.data_mmeb_eval import ALL_DATASETS, MMEBClassificationDataset
from early_exit_probe.inference import load_checkpoint, probe_for_checkpoint
from early_exit_probe.model import load_backbone, load_processor

AOKVQA = "A-OKVQA"


class Halt(Exception):
    def __init__(self, depth, pred):
        self.depth, self.pred = depth, pred


class HaltingRunner:
    """Installs one hook per tapped layer. mode: 'route' (thresholds), 'fixed' (stop at
    self.stop_depth), 'full' (score at the last depth, stop there)."""

    def __init__(self, model, probe, depths, thresholds):
        self.model, self.probe, self.depths = model, probe, sorted(depths)
        self.thresholds = thresholds
        self.final = self.depths[-1]
        self.layers = model.model.language_model.layers
        self.mode, self.stop_depth, self.pos = "full", None, None
        self.first_layer_t = None
        self.layers[0].register_forward_pre_hook(self._mark_prefix)
        for d in self.depths:
            self.layers[d - 1].register_forward_hook(self._make_hook(d))

    def _mark_prefix(self, module, args):
        torch.cuda.synchronize()
        self.first_layer_t = time.perf_counter()

    def _make_hook(self, d):
        def hook(module, args, output):
            if self.mode == "profile" and d > self.stop_depth:
                return
            if self.mode == "fixed" and d != self.stop_depth:
                return
            if self.mode == "full" and d != self.final:
                return
            h = output[0] if isinstance(output, tuple) else output
            h_choices = h[0, self.pos]                       # (K, H)
            x = torch.cat([h_choices, h_choices[-1:].expand_as(h_choices)], dim=-1).unsqueeze(0)
            logits, z = self.probe(x.float(), d)
            pred = logits.argmax(dim=-1)
            if self.mode == "route" and d != self.final:
                thr = self.thresholds.get(d)
                if thr is None:
                    return
                agree = torch.sigmoid(self.probe.agreement_logits_for_predicted(z, pred))
                if agree.item() < thr:                       # the per-layer host sync is real cost
                    return
            if self.mode == "profile" and d < self.stop_depth:
                # Every-layer routing cost without exiting: the agreement check and its host sync.
                torch.sigmoid(self.probe.agreement_logits_for_predicted(z, pred)).item()
                return
            raise Halt(d, int(pred.item()))
        return hook

    @torch.no_grad()
    def __call__(self, batch, pos, mode, stop_depth=None):
        self.mode, self.stop_depth, self.pos = mode, stop_depth, torch.tensor(pos[0], device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            self.model(**batch, use_cache=False, return_dict=True, logits_to_keep=1)
            raise RuntimeError("forward finished without reaching a scoring hook")
        except Halt as h:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            return {"ms": 1e3 * (t1 - t0), "prefix_ms": 1e3 * (self.first_layer_t - t0),
                    "depth": h.depth, "pred": h.pred}


def load_dataset(name, limit):
    if name == AOKVQA:
        ds = AOKVQADataset("validation", split_filter=None)
        return [ds[i] for i in range(min(limit, len(ds)))]
    ds = MMEBClassificationDataset(name, limit=limit)
    return [ds[i] for i in range(len(ds))]


@torch.no_grad()
def time_generation(model, processor, ex):
    n = len(ex.choices)
    suffix = " Respond with only the letter." if n <= len(CHOICE_LETTERS) else \
        " Respond with only the number of the correct option."
    batch, _ = collate_for_generation([ex], processor, suffix=suffix)
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**batch, max_new_tokens=64, do_sample=False)
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0), int(out.shape[1] - batch["input_ids"].shape[1])


def summarize(xs):
    return {"mean": statistics.fmean(xs), "median": statistics.median(xs)} if xs else None


def profile_every_layer(model, probe, gen_model, processor, args):
    """Per dataset, mean wall clock to halt at each depth d (agreement checked at layers < d),
    plus full depth and generation. Combine with any reported mean exit depth: time is
    affine in d, so E[t(d*)] = t(E[d*]) up to the fit residual reported here."""
    depth_max = len(model.model.language_model.layers)
    depths = list(range(1, depth_max + 1))
    runner = HaltingRunner(model, probe, depths, {})
    results = {"gpu": torch.cuda.get_device_name(), "batch_size": 1, "depths": depths, "per_dataset": {}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        examples = load_dataset(name, args.limit + args.warmup)
        per_depth = {d: [] for d in depths}
        gen_ms, gen_tokens = [], []
        for i, ex in enumerate(examples):
            batch, pos, _ = collate([ex], processor)
            batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
            times = {d: runner(batch, pos, "profile", d)["ms"] for d in depths}
            gen = time_generation(gen_model, processor, ex) if gen_model is not None else None
            if i < args.warmup:
                continue
            for d in depths:
                per_depth[d].append(times[d])
            if gen is not None:
                gen_ms.append(gen[0])
                gen_tokens.append(gen[1])
        mean_t = [statistics.fmean(per_depth[d]) for d in depths]
        slope, intercept = statistics.linear_regression(depths, mean_t)
        resid = max(abs(t - (intercept + slope * d)) for d, t in zip(depths, mean_t))
        s = {"n": len(per_depth[1]), "n_choices": len(examples[0].choices), "ms_by_depth": mean_t,
             "affine_fit": {"ms_at_0": intercept, "ms_per_layer": slope, "max_abs_residual_ms": resid}}
        if gen_ms:
            s["gen_ms"], s["gen_tokens"] = statistics.fmean(gen_ms), statistics.fmean(gen_tokens)
        results["per_dataset"][name] = s
        json.dump(results, open(out_path, "w"), indent=2)
        g = f"  gen {s['gen_ms']:.0f}ms" if gen_ms else ""
        print(f"[{name}] K={s['n_choices']} t(1) {mean_t[0]:.0f}ms  t({depth_max}) {mean_t[-1]:.0f}ms  "
              f"fit {intercept:.0f} + {slope:.1f}*d (max resid {resid:.1f}ms){g}", flush=True)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--thresholds", default=None, help="default: <checkpoint>/thresholds.json")
    ap.add_argument("--datasets", nargs="+", default=list(ALL_DATASETS) + [AOKVQA])
    ap.add_argument("--limit", type=int, default=50, help="rows per dataset")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--generation", action="store_true", help="also time the generation baseline")
    ap.add_argument("--profile_every_layer", action="store_true",
                    help="time halting at every depth 1..D with the agreement check run at every layer "
                         "(the every-layer method's cost); exit decisions are forced, so weights don't matter")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    thr_path = Path(args.thresholds) if args.thresholds else ckpt / "thresholds.json"
    thr_raw = json.load(open(thr_path))["thresholds"] if thr_path.exists() else {}
    # None marks a disabled exit (sequential calibration found no qualifying cutoff).
    thresholds = {int(d): (v["threshold"] if isinstance(v, dict) else v) for d, v in thr_raw.items()}
    thresholds = {d: t for d, t in thresholds.items() if t is not None}

    processor = load_processor()
    processor.tokenizer.padding_side = "left"
    backbone = load_backbone(lora=True).cuda().eval()
    probe, DEPTH_TAPS = probe_for_checkpoint(args.checkpoint)
    probe = probe.cuda().eval()
    load_checkpoint(backbone, probe, args.checkpoint)
    model = backbone.merge_and_unload().eval()
    gen_model = load_backbone(lora=False).cuda().eval() if args.generation else None
    if args.profile_every_layer:
        return profile_every_layer(model, probe, gen_model, processor, args)
    assert thresholds or args.profile_every_layer, f"no thresholds at {thr_path}"
    runner = HaltingRunner(model, probe, DEPTH_TAPS, thresholds)

    results = {"checkpoint": str(ckpt), "depths": list(DEPTH_TAPS), "thresholds": thresholds,
               "gpu": torch.cuda.get_device_name(), "batch_size": 1, "per_dataset": {}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        examples = load_dataset(name, args.limit + args.warmup)
        rec = {k: [] for k in ("cpu_ms", "full_ms", "routed_ms", "prefix_ms", "routed_depth", "gen_ms", "gen_tokens")}
        fixed_depths = [d for d in DEPTH_TAPS[:-1] if d in thresholds] if len(DEPTH_TAPS) <= 8 else []
        rec.update({f"fixed{d}_ms": [] for d in fixed_depths})
        agree_full = 0
        for i, ex in enumerate(examples):
            t0 = time.perf_counter()
            batch, pos, _ = collate([ex], processor)
            cpu_ms = 1e3 * (time.perf_counter() - t0)
            batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
            full = runner(batch, pos, "full")
            routed = runner(batch, pos, "route")
            fixed = {d: runner(batch, pos, "fixed", d) for d in fixed_depths}
            gen = time_generation(gen_model, processor, ex) if gen_model is not None else None
            if i < args.warmup:
                continue
            rec["cpu_ms"].append(cpu_ms)
            rec["full_ms"].append(full["ms"])
            rec["prefix_ms"].append(full["prefix_ms"])
            rec["routed_ms"].append(routed["ms"])
            rec["routed_depth"].append(routed["depth"])
            for d, r in fixed.items():
                rec[f"fixed{d}_ms"].append(r["ms"])
            if gen is not None:
                rec["gen_ms"].append(gen[0])
                rec["gen_tokens"].append(gen[1])
            agree_full += routed["pred"] == full["pred"] if routed["depth"] < max(DEPTH_TAPS) else 1
        s = {k: summarize(v) for k, v in rec.items() if v}
        s["n"] = len(rec["full_ms"])
        s["n_choices"] = len(examples[0].choices)
        s["routed_speedup_mean"] = s["full_ms"]["mean"] / s["routed_ms"]["mean"]
        s["depth_proxy_speedup"] = max(DEPTH_TAPS) / s["routed_depth"]["mean"]
        s["routed_agreement_with_full"] = agree_full / s["n"]
        if gen_model is not None:
            s["gen_over_full"] = s["gen_ms"]["mean"] / s["full_ms"]["mean"]
        results["per_dataset"][name] = {"summary": s, "raw": rec}
        json.dump(results, open(out_path, "w"), indent=2)
        g = f"  gen {s['gen_ms']['mean']:.0f}ms" if gen_model is not None else ""
        print(f"[{name}] K={s['n_choices']} full {s['full_ms']['mean']:.0f}ms  routed {s['routed_ms']['mean']:.0f}ms "
              f"(depth {s['routed_depth']['mean']:.1f})  prefix {s['prefix_ms']['mean']:.0f}ms  "
              f"speedup {s['routed_speedup_mean']:.2f}x vs depth-proxy {s['depth_proxy_speedup']:.2f}x{g}", flush=True)
    results["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
