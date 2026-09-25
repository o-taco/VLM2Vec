"""Turns a list of `Example`s into a left-padded multimodal batch, and runs the backbone
to extract per-depth, per-choice concat vectors ready for the DepthProbe.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .data_aokvqa import Example, build_prompt
from .model import DEPTH_TAPS, locate_choice_token_positions


def collate(examples: list[Example], processor):
    tokenizer = processor.tokenizer
    per_example = []
    max_len = 0
    for ex in examples:
        prompt = build_prompt(ex.question, ex.choices)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = processor(text=[text], images=[ex.image], return_tensors="pt")
        final_len = enc["input_ids"].shape[1]
        choice_positions = locate_choice_token_positions(processor, text, len(ex.choices), final_len)
        per_example.append((enc, choice_positions, ex))
        max_len = max(max_len, enc["input_ids"].shape[1])

    input_ids, attention_mask, choice_pos_batch = [], [], []
    pixel_values, image_grid_thw = [], []
    correct_idx, n_choices_list = [], []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for enc, positions, ex in per_example:
        seq_len = enc["input_ids"].shape[1]
        pad_len = max_len - seq_len
        ids = enc["input_ids"][0]
        mask = enc["attention_mask"][0]
        if pad_len > 0:
            ids = F.pad(ids, (pad_len, 0), value=pad_id)  # left pad
            mask = F.pad(mask, (pad_len, 0), value=0)
        input_ids.append(ids)
        attention_mask.append(mask)
        choice_pos_batch.append([p + pad_len for p in positions])
        # Images have variable patch counts (dynamic resolution -> different grid_thw per
        # image), so pixel_values/image_grid_thw are kept as one-tensor-per-example PYTHON
        # LISTS, not pre-stacked/concatenated -- this repo's Qwen3VLForConditionalGeneration
        # wrapper (src/vlm_backbone/qwen3_vl/modeling_qwen3_vl.py) does the final
        # torch.cat(dim=0) itself, indexing pixel_values/image_grid_thw as per-example
        # entries. Pre-concatenating here would make its `pixel_values[i]` index into a
        # single patch row instead of a whole example's block.
        pixel_values.append(enc["pixel_values"])
        image_grid_thw.append(enc["image_grid_thw"])
        correct_idx.append(ex.correct_choice_idx)
        n_choices_list.append(len(ex.choices))

    assert len(set(n_choices_list)) == 1, "mixed choice counts in one batch not supported"
    batch = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    return batch, choice_pos_batch, torch.tensor(correct_idx, dtype=torch.long)


def collate_for_generation(examples: list[Example], processor, suffix: str = ""):
    """Like collate(), but opens an assistant turn (add_generation_prompt=True) instead of
    ending mid-template. collate()'s prompt is built for the *probe*, which reads hidden
    states from fixed token positions inside the prompt and never generates, so it
    deliberately does NOT open an assistant turn. A raw logit argmax against a prompt that
    never opens one is off-distribution (verified empirically: it scores at chance) --
    generation must end with the actual generation-prompt tokens for next-token logits to
    mean anything.
    """
    tokenizer = processor.tokenizer
    per_example = []
    max_len = 0
    for ex in examples:
        prompt = build_prompt(ex.question, ex.choices) + suffix
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[text], images=[ex.image], return_tensors="pt")
        per_example.append((enc, ex))
        max_len = max(max_len, enc["input_ids"].shape[1])

    input_ids, attention_mask = [], []
    pixel_values, image_grid_thw, correct_idx = [], [], []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for enc, ex in per_example:
        seq_len = enc["input_ids"].shape[1]
        pad_len = max_len - seq_len
        ids = enc["input_ids"][0]
        mask = enc["attention_mask"][0]
        if pad_len > 0:
            ids = F.pad(ids, (pad_len, 0), value=pad_id)
            mask = F.pad(mask, (pad_len, 0), value=0)
        input_ids.append(ids)
        attention_mask.append(mask)
        pixel_values.append(enc["pixel_values"])
        image_grid_thw.append(enc["image_grid_thw"])
        correct_idx.append(ex.correct_choice_idx)

    batch = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    return batch, torch.tensor(correct_idx, dtype=torch.long)


def extract_depth_concats(backbone, batch, choice_pos_batch, device, depths=DEPTH_TAPS):
    """Runs the backbone once with output_hidden_states=True and returns
    {depth: x} where x is (batch, n_choices, 2*hidden), built from that depth's
    hidden_states via concat(h_i, h_last).
    """
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = backbone(**batch, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden_states = out.hidden_states  # tuple length num_layers+1; hidden_states[d] = output of layer d

    bsz = batch["input_ids"].shape[0]
    n_choices = len(choice_pos_batch[0])
    batch_idx = torch.arange(bsz, device=device)
    pos_tensor = torch.tensor(choice_pos_batch, device=device, dtype=torch.long)  # (bsz, n_choices)

    result = {}
    for d in depths:
        h = hidden_states[d]  # (bsz, seq, hidden)
        # gather per-choice hidden states: (bsz, n_choices, hidden)
        h_choices = h[batch_idx.unsqueeze(1), pos_tensor]
        h_last = h_choices[:, -1, :]  # (bsz, hidden) -- highest-index choice's own state
        h_last_expanded = h_last.unsqueeze(1).expand(-1, n_choices, -1)
        x = torch.cat([h_choices, h_last_expanded], dim=-1)  # (bsz, n_choices, 2*hidden)
        result[d] = x
    return result
