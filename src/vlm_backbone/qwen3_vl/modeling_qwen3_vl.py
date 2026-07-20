"""
Thin wrapper around HF's native Qwen3VLForConditionalGeneration.

Unlike the qwen2_vl backbone (vendored as a full copy in this repo because it
predates transformers factoring the image-merge logic into reusable methods),
Qwen3-VL ships natively in transformers>=4.57 with `get_image_features` /
`get_placeholder_mask` already factored out, so no reimplementation is needed.

The only gap: VLM2Vec's collator (src/model_utils.py's *_process_fn functions)
builds `pixel_values` as one slot per batch row (dummy zero-filled rows for
text-only examples) and `image_grid_thw` as a plain Python list with `None`
entries for those rows -- not the native HF convention where pixel_values/
image_grid_thw contain only the real images, concatenated, with no per-row
padding. This wrapper converts between the two conventions and delegates
everything else (deepstack visual-feature injection, RoPE index, etc.) to the
upstream forward implementation.
"""
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as _Qwen3VLForConditionalGeneration,
)


def patch_vision_patch_embed_for_volta(model):
    """
    Volta (sm_70, e.g. V100) has no bf16 tensor cores, and this torch/cuDNN build's
    bf16 Conv3d kernel for the vision patch_embed hits a hard CUDA launch failure
    ("too many resources requested for launch") -- confirmed empirically fp32/fp16
    Conv3d both work fine, only bf16 fails. Run just this one conv in fp32.
    """
    patch_embed = model.model.visual.patch_embed
    proj = patch_embed.proj

    def forward(hidden_states):
        # cast weight/bias to fp32 inline on every call, rather than upcasting proj
        # once and relying on it staying fp32 -- callers (e.g. compute_val_loss.py's
        # model.to(dtype=torch.bfloat16) after load()) can re-cast the whole model
        # afterward, silently reverting a one-time upcast and reintroducing the crash.
        compute_dtype = proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, patch_embed.in_channels, patch_embed.temporal_patch_size,
            patch_embed.patch_size, patch_embed.patch_size,
        )
        # training runs under torch.autocast(bf16), which silently re-casts Conv3d
        # inputs back to bf16 regardless of an explicit .float() call -- must disable
        # autocast for this op, not just cast the tensor, or the crash comes right back.
        with torch.autocast(device_type="cuda", enabled=False):
            weight = proj.weight.float()
            bias = proj.bias.float() if proj.bias is not None else None
            hidden_states = torch.nn.functional.conv3d(
                hidden_states.float(), weight, bias,
                stride=proj.stride, padding=proj.padding, dilation=proj.dilation, groups=proj.groups,
            )
        hidden_states = hidden_states.to(compute_dtype).view(-1, patch_embed.embed_dim)
        return hidden_states

    patch_embed.forward = forward


class Qwen3VLForConditionalGeneration(_Qwen3VLForConditionalGeneration):
    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                image_grid_thw=None, **kwargs):
        if pixel_values is not None:
            # some callers (e.g. compute_val_loss.py's process_vlm_inputs) omit
            # image_grid_thw entirely for text-only batches rather than filling it
            # with Nones -- treat "absent" the same as "all None" rather than
            # letting a dummy pixel_values tensor reach the vision tower unchecked.
            if image_grid_thw is None:
                image_grid_thw = [None] * pixel_values.shape[0]
            idx_w_image = [i for i, g in enumerate(image_grid_thw) if g is not None]
            if len(idx_w_image) > 0:
                pixel_values = torch.cat(
                    [pixel_values[i] if isinstance(pixel_values[i], torch.Tensor)
                     else torch.from_numpy(pixel_values[i]) for i in idx_w_image],
                    dim=0,
                ).to(input_ids.device)
                image_grid_thw = torch.cat(
                    [image_grid_thw[i] if isinstance(image_grid_thw[i], torch.Tensor)
                     else torch.from_numpy(image_grid_thw[i]) for i in idx_w_image],
                    dim=0,
                ).to(input_ids.device)
            else:
                pixel_values = None
                image_grid_thw = None

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )
