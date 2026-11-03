"""Shared inference helpers for calibrate.py and eval.py: run the LoRA backbone + probe
over a batch and return, per example and per tapped depth, the predicted choice and the
probe's predicted agreement-with-depth-28 probability.
"""
from __future__ import annotations

import torch

from .model import DEPTH_TAPS
from .batching import collate, extract_depth_concats


@torch.no_grad()
def run_batch_all_depths(backbone, probe, processor, examples, device, depths=DEPTH_TAPS):
    batch, choice_pos, correct_idx = collate(examples, processor)
    correct_idx = correct_idx.to(device)
    depth_concats = extract_depth_concats(backbone, batch, choice_pos, device, depths=depths)

    per_depth = {}
    for d in depths:
        logits, z = probe(depth_concats[d], d)
        pred = logits.argmax(dim=-1)
        agreement_logit = probe.agreement_logits_for_predicted(z, pred)
        agreement_prob = torch.sigmoid(agreement_logit)
        # Answer-head confidences, for the answer-confidence routing baselines.
        top2 = logits.float().softmax(dim=-1).topk(min(2, logits.size(-1)), dim=-1).values
        per_depth[d] = {"pred": pred, "agreement_prob": agreement_prob, "answer_maxprob": top2[:, 0],
                        "answer_margin": top2[:, 0] - (top2[:, 1] if top2.size(-1) > 1 else 0)}

    pred_28 = per_depth[max(depths)]["pred"]
    for d in depths:
        per_depth[d]["agrees_with_28"] = (per_depth[d]["pred"] == pred_28)
        per_depth[d]["correct"] = (per_depth[d]["pred"] == correct_idx)
    return per_depth, correct_idx


def load_checkpoint(backbone, probe, ckpt_dir):
    """backbone must already be the LoRA-wrapped PeftModel; loads adapter weights in place."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    import os
    adapter_path = os.path.join(ckpt_dir, "lora_adapter")
    state = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
    set_peft_model_state_dict(backbone, state)
    probe.load_state_dict(torch.load(
        os.path.join(ckpt_dir, "probe.pt"), map_location=probe.answer_head.weight.device, weights_only=True,
    ))
    return backbone, probe


def probe_for_checkpoint(ckpt_dir):
    """(probe, depths) for a checkpoint: probe_config.json names the probe kind; checkpoints
    without one are the phase 1-3 4-tap DepthProbe."""
    import json
    import os
    from .model import build_probe
    cfg_path = os.path.join(ckpt_dir, "probe_config.json")
    kind = json.load(open(cfg_path))["probe"] if os.path.exists(cfg_path) else "film_input"
    return build_probe(kind)
