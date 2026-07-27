"""
Depth-prune Qwen3-VL-2B-Instruct's text decoder from 28 -> 14 layers (~half the
decoder layers, ~67% of total params -- the vision tower + tied embeddings are
fixed overhead that layer-dropping can't touch, see conversation notes).

Layers 0,1,2 are preserved in-place: Qwen3-VL's deepstack mechanism injects the
vision tower's 3 intermediate feature maps into text decoder layers by *position*
(range(len(deepstack_visual_embeds)) == range(3)), so naive every-other-layer
dropping would scramble which deepstack feature lands in which layer. Keeping
0,1,2 fixed and evenly subsampling the remaining 25 layers for the other 11
slots preserves that alignment exactly.
"""
import argparse
import os
import torch
from transformers import AutoConfig, AutoProcessor

from src.model_utils import backbone2model, QWEN3_VL

BACKBONE = "Qwen/Qwen3-VL-2B-Instruct"

parser = argparse.ArgumentParser()
parser.add_argument("--n_keep", type=int, default=14, help="number of decoder layers to keep (out of 28)")
parser.add_argument("--out_dir", default=None, help="defaults to pruned_models/qwen3vl-2b-<n_keep>layer")
parser.add_argument("--keep_idx", default=None, help="comma-separated explicit layer indices to keep, overriding the even-subsample heuristic (must include 0,1,2)")
args = parser.parse_args()

OUT_DIR = args.out_dir or f"pruned_models/qwen3vl-2b-{args.n_keep}layer"

os.makedirs(OUT_DIR, exist_ok=True)

print(f"loading {BACKBONE}")
config = AutoConfig.from_pretrained(BACKBONE, trust_remote_code=True)
cls = backbone2model[QWEN3_VL]
model = cls.from_pretrained(BACKBONE, config=config, dtype=torch.bfloat16, low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(BACKBONE, trust_remote_code=True)

lm = model.model.language_model
layers = lm.layers
n_layers = len(layers)
print(f"original decoder layers: {n_layers}")

# keep 0,1,2 (deepstack recipients) in place; evenly subsample the remaining
# slots from the tail (3..n_layers-1) to land on n_keep total.
n_keep = args.n_keep
deepstack_prefix = [0, 1, 2]
if args.keep_idx:
    keep_idx = sorted(int(x) for x in args.keep_idx.split(","))
    n_keep = len(keep_idx)
else:
    tail_candidates = list(range(len(deepstack_prefix), n_layers))
    tail_keep_count = n_keep - len(deepstack_prefix)
    tail_positions = torch.linspace(tail_candidates[0], tail_candidates[-1], tail_keep_count)
    tail_idx = sorted(set(int(round(x.item())) for x in tail_positions))
    keep_idx = deepstack_prefix + tail_idx
assert len(keep_idx) == n_keep, f"expected {n_keep} unique indices, got {keep_idx}"
print(f"keeping layer indices: {keep_idx} ({len(keep_idx)} layers)")

new_layers = torch.nn.ModuleList([layers[i] for i in keep_idx])
for new_i, layer in enumerate(new_layers):
    layer.self_attn.layer_idx = new_i
lm.layers = new_layers
lm.config.num_hidden_layers = len(keep_idx)
model.config.text_config.num_hidden_layers = len(keep_idx)

total = sum(p.numel() for p in model.parameters())
print(f"new decoder layers: {len(lm.layers)}")
print(f"new total params: {total:,}")

print(f"saving to {OUT_DIR}")
model.save_pretrained(OUT_DIR)
processor.save_pretrained(OUT_DIR)

# save_pretrained on the processor can silently miss files the original snapshot
# ships (tokenizer.model, chat_template.jinja, etc, per past experience pruning
# llava-next) -- copy anything from the original cached snapshot that isn't
# already present in the pruned output dir.
from huggingface_hub import snapshot_download
orig_dir = snapshot_download(BACKBONE)
for fname in os.listdir(orig_dir):
    dst = os.path.join(OUT_DIR, fname)
    src = os.path.join(orig_dir, fname)
    if os.path.isfile(src) and not os.path.exists(dst) and fname not in ("model.safetensors", "pytorch_model.bin"):
        import shutil
        shutil.copy2(src, dst)
        print(f"copied missing file: {fname}")

print("done:", sorted(os.listdir(OUT_DIR)))
