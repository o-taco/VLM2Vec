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
        per_depth[d] = {"pred": pred, "agreement_prob": agreement_prob}

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
