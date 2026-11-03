"""The paper's every-layer joint recipe (Secs. 3.2-3.3, 4 "Training"; Appendix A), for
reproducing the Qwen3-VL-2B row on this box.

- FiLMTrunkProbe (input LN, FiLM inside both 512-wide hidden layers) on decoder LoRA
  (q/k/v/o, rank 64, alpha 128, dropout 0.05).
- One fixed, ordered pool with Table 3's per-domain role counts. Answer-fit rows minimize
  mean_d CE(s_d, y) over all D depths. Agreement-fit rows never see their label: they minimize
  mean_{d<D} BCE(a_d, 1[argmax s_d == argmax s_D]) with both argmaxes detached and the target
  from the current model. Each row uses only its own role's loss.
- One pass, batch of 1 row, gradients accumulated over 8 rows (1,024 updates for 8,192 rows),
  AdamW lr 2e-5 (LoRA) / 2e-4 (heads), weight decay 0.01, clip 1.0, constant LR (the paper
  gives no schedule). Images 336x336 bicubic (set here via EARLY_EXIT_IMAGE_SIZE).

Answer-fit rows come from each source's answer_fit split and agreement-fit rows from its
confidence_fit split, so the two roles are image-disjoint. Domains whose data is not on this
machine can be dropped with --domains; the manifest records exactly which rows were used.
"""
from __future__ import annotations

import os

os.environ.setdefault("EARLY_EXIT_IMAGE_SIZE", "336")  # before batching is imported

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from early_exit_probe.batching import collate, extract_depth_concats
from early_exit_probe.data_aokvqa import AOKVQADataset, load_or_build_split_index
from early_exit_probe.model import build_probe, load_backbone, load_processor

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_INDEX_PATH = REPO_ROOT / "early_exit_probe" / "outputs" / "aokvqa_split_index.json"

# Paper Table 3: (answer-fit, agreement-fit) rows per training domain.
TABLE3 = {"A-OKVQA": (3363, 987), "CUB-200": (1256, 337), "ScienceQA": (1236, 332),
          "ImageNet": (303, 46), "iNaturalist": (242, 90)}
POOL_SEED = 26091615


def load_source(name: str, split: str):
    if name == "A-OKVQA":
        return AOKVQADataset("train", split, load_or_build_split_index(SPLIT_INDEX_PATH))
    if name == "ScienceQA":
        from early_exit_probe.data_scienceqa import ScienceQATrainDataset
        return ScienceQATrainDataset(split)
    from early_exit_probe.data_classification import SOURCES, ClassificationTrainDataset
    if name not in SOURCES:
        raise SystemExit(f"no loader for {name} on this machine; drop it with --domains")
    return ClassificationTrainDataset(name, split)


def build_pool(domains, seed):
    """Fixed ordered list of (domain, role, split, index), plus its SHA-256 identity."""
    rng = random.Random(seed)
    datasets, rows = {}, []
    for name in domains:
        for role, split, n in (("answer", "answer_fit", TABLE3[name][0]),
                               ("agreement", "confidence_fit", TABLE3[name][1])):
            ds = datasets[(name, split)] = load_source(name, split)
            assert len(ds) >= n, f"{name}/{split} has {len(ds)} rows, Table 3 needs {n}"
            rows += [(name, role, split, i) for i in sorted(rng.sample(range(len(ds)), n))]
    rng.shuffle(rows)
    digest = hashlib.sha256("\n".join(f"{d}|{r}|{s}|{i}" for d, r, s, i in rows).encode()).hexdigest()
    return rows, datasets, digest


def row_loss(backbone, probe, depths, processor, device, example, role):
    batch, pos, correct = collate([example], processor)
    x = extract_depth_concats(backbone, batch, pos, device, depths=depths)
    out = {d: probe(x[d], d) for d in depths}
    final = depths[-1]
    if role == "answer":
        target = correct.to(device)
        return torch.stack([F.cross_entropy(out[d][0], target) for d in depths]).mean()
    pred_final = out[final][0].argmax(dim=-1).detach()
    losses = []
    for d in depths[:-1]:
        logits, z = out[d]
        pred_d = logits.argmax(dim=-1).detach()
        a = probe.agreement_logits_for_predicted(z, pred_d)
        losses.append(F.binary_cross_entropy_with_logits(a, (pred_d == pred_final).float()))
    return torch.stack(losses).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=list(TABLE3))
    ap.add_argument("--pool_seed", type=int, default=POOL_SEED)
    ap.add_argument("--seed", type=int, default=POOL_SEED, help="init/torch seed (vary for repeat runs)")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr_lora", type=float, default=2e-5)
    ap.add_argument("--lr_heads", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max_rows", type=int, default=None, help="truncate the pool (smoke tests)")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"

    rows, datasets, digest = build_pool(args.domains, args.pool_seed)
    if args.max_rows:
        rows = rows[:args.max_rows]
    counts = {f"{d}/{r}": sum(1 for x in rows if x[0] == d and x[1] == r) for d in args.domains for r in ("answer", "agreement")}
    json.dump({"pool_sha256": digest, "pool_seed": args.pool_seed, "counts": counts,
               "rows": [list(r) for r in rows], "args": vars(args),
               "image_size": os.environ["EARLY_EXIT_IMAGE_SIZE"]}, open(out_dir / "pool_manifest.json", "w"))
    print(f"[train] pool {len(rows)} rows sha256={digest}\n{counts}", flush=True)

    processor = load_processor()
    backbone = load_backbone(lora=True, lora_r=64, lora_alpha=128, lora_dropout=0.05).to(device).train()
    probe, depths = build_probe("film_hidden")
    probe = probe.to(device).train()
    lora_params = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": lora_params, "lr": args.lr_lora},
                                   {"params": list(probe.parameters()), "lr": args.lr_heads}],
                                  weight_decay=args.weight_decay)
    print(f"[train] trainable: LoRA {sum(p.numel() for p in lora_params):,}  "
          f"heads {sum(p.numel() for p in probe.parameters()):,}", flush=True)

    log = open(out_dir / "train_log.jsonl", "w")
    started, window = time.perf_counter(), {"answer": [], "agreement": []}
    optimizer.zero_grad(set_to_none=True)
    n_updates = 0
    for i, (name, role, split, idx) in enumerate(rows, 1):
        loss = row_loss(backbone, probe, depths, processor, device, datasets[(name, split)][idx], role)
        (loss / args.accum).backward()
        window[role].append(loss.item())
        if i % args.accum == 0 or i == len(rows):
            torch.nn.utils.clip_grad_norm_(lora_params + list(probe.parameters()), args.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_updates += 1
            if n_updates % 16 == 0 or n_updates == 1:
                rec = {"update": n_updates, "rows": i, "elapsed_s": time.perf_counter() - started,
                       **{f"{k}_loss": sum(v) / len(v) for k, v in window.items() if v}}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"[train] update {n_updates} rows {i}/{len(rows)} " +
                      " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k.endswith("_loss")) +
                      f" ({rec['elapsed_s']:.0f}s)", flush=True)
                window = {"answer": [], "agreement": []}
            if n_updates % 256 == 0 or i == len(rows):
                save(backbone, probe, out_dir / f"checkpoint-{n_updates}")
    print(f"[train] done: {n_updates} updates in {time.perf_counter() - started:.0f}s", flush=True)


def save(backbone, probe, ckpt):
    ckpt.mkdir(parents=True, exist_ok=True)
    backbone.save_pretrained(str(ckpt / "lora_adapter"))
    torch.save(probe.state_dict(), ckpt / "probe.pt")
    json.dump({"probe": "film_hidden"}, open(ckpt / "probe_config.json", "w"))
    print(f"[train] saved {ckpt}", flush=True)


if __name__ == "__main__":
    main()
