"""Backbone + depth-conditioned FiLM probe for the early-exit method in
260914_keep_it_simple.pdf (pages 3-6, 18-19).

Architecture recap:
  - Qwen3-VL-2B-Instruct, frozen base + frozen vision tower.
  - LoRA r=64 on the language decoder's q/k/v/o projections, all 28 layers.
  - At each tapped depth d, per choice i: x_i = concat(h_i^d, h_last^d) in R^4096, where
    h_i is choice i's own last-content-token hidden state at depth d, and h_last is the
    same kind of vector for the highest-index choice (causal attention => it has seen
    every option).
  - One shared trunk, FiLM-conditioned on d: FiLM -> LayerNorm -> 2x512 GELU MLP -> z.
  - answer_head: Linear(512,1) per choice -> logits -> argmax. CE loss (ground truth).
  - agreement_head: Linear(512,1) on the predicted choice's z -> BCE against
    detached 1[yhat_d == yhat_28].
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vlm_backbone.qwen3_vl import Qwen3VLForConditionalGeneration, patch_vision_patch_embed_for_volta

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEPTH_TAPS = (8, 12, 20, 28)
HIDDEN = 2048
CONCAT_DIM = 2 * HIDDEN
DEPTH_EMB_DIM = 32
TRUNK_HIDDEN = 512
MAX_DEPTH = 28  # number of decoder layers; depth embedding indexed 1..MAX_DEPTH


def load_processor():
    return AutoProcessor.from_pretrained(MODEL_ID)


def load_backbone(lora: bool = True, lora_r: int = 64, lora_alpha: int = 128, lora_dropout: float = 0.0):
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Volta (V100, sm_70) has no FlashAttention-2 support; sdpa runs everywhere (same
    # pattern as src/model.py's QWEN3_VL branch).
    config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    config.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, config=config, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    patch_vision_patch_embed_for_volta(model)

    for p in model.parameters():
        p.requires_grad_(False)

    if lora:
        # Confirmed empirically (see plan): Qwen3-VL's vision tower uses fused `qkv`/`proj`
        # module names, not q_proj/k_proj/v_proj/o_proj, so this target list only ever
        # matches the 28 language decoder self-attention layers -- vision stays frozen.
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            init_lora_weights="gaussian",
            use_dora=False,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
    return model


class DepthProbe(nn.Module):
    def __init__(self, hidden=HIDDEN, depth_emb_dim=DEPTH_EMB_DIM, trunk_hidden=TRUNK_HIDDEN, max_depth=MAX_DEPTH):
        super().__init__()
        concat_dim = 2 * hidden
        self.max_depth = max_depth
        self.depth_embed = nn.Embedding(max_depth, depth_emb_dim)
        self.film_gen = nn.Linear(depth_emb_dim, 2 * concat_dim)
        # Start FiLM as identity (gamma=1, beta=0) so training begins at a sane init.
        nn.init.zeros_(self.film_gen.weight)
        with torch.no_grad():
            self.film_gen.bias[:concat_dim] = 1.0
            self.film_gen.bias[concat_dim:] = 0.0
        self.ln = nn.LayerNorm(concat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(),
        )
        self.answer_head = nn.Linear(trunk_hidden, 1)
        self.agreement_head = nn.Linear(trunk_hidden, 1)

    def forward(self, x: torch.Tensor, depth: int):
        """x: (batch, n_choices, concat_dim) -> answer_logits (batch, n_choices), z (batch, n_choices, trunk_hidden)"""
        device = x.device
        d_idx = torch.full((1,), depth - 1, device=device, dtype=torch.long)
        e = self.depth_embed(d_idx)  # (1, depth_emb_dim)
        gamma_beta = self.film_gen(e)  # (1, 2*concat_dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each (1, concat_dim)
        x = gamma.unsqueeze(1) * x + beta.unsqueeze(1)  # broadcast over n_choices
        x = self.ln(x)
        z = self.mlp(x)
        answer_logits = self.answer_head(z).squeeze(-1)
        return answer_logits, z

    def agreement_logits_for_predicted(self, z: torch.Tensor, pred_idx: torch.Tensor):
        """z: (batch, n_choices, trunk_hidden); pred_idx: (batch,) long -> (batch,) logits"""
        batch_idx = torch.arange(z.size(0), device=z.device)
        z_pred = z[batch_idx, pred_idx]
        return self.agreement_head(z_pred).squeeze(-1)


def locate_choice_token_positions(processor, text: str, n_choices: int, final_len: int) -> list[int]:
    """Find, for one example, the final (post image-token-expansion) sequence position of
    each choice's last content token: "A. toothpick" -> the token for "pick".

    Re-tokenizing marker substrings in isolation (e.g. "\\nA.") does not reliably match
    their tokenization *in context* (BPE merges depend on what precedes them), so instead
    we use the RAW (pre image-expansion) tokenizer offset mapping to find each choice's
    character boundary, then shift every resulting token index by the number of tokens
    the single `<|image_pad|>` placeholder expanded into (`final_len - raw_len`) -- valid
    because every choice appears strictly after the image block in our prompt template.
    """
    from .data_aokvqa import CHOICE_LETTERS

    raw = processor.tokenizer(text, return_offsets_mapping=True)
    offsets = raw["offset_mapping"]
    shift = final_len - len(raw["input_ids"])

    boundaries_char = [text.index(f"\n{CHOICE_LETTERS[i]}.") for i in range(1, n_choices)]
    boundaries_char.append(text.index("\nAnswer"))

    positions = []
    for boundary in boundaries_char:
        tok_idx = None
        for i, (s, e) in enumerate(offsets):
            if e <= boundary and e > 0:
                tok_idx = i
            elif s >= boundary:
                break
        if tok_idx is None:
            raise ValueError(f"could not locate a choice token before char {boundary} in: {text!r}")
        positions.append(tok_idx + shift)
    return positions
