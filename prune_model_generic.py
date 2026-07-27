"""
Backbone-agnostic depth-pruning: same recipe as prune_model.py (keep layers
0,1,2 in place for Qwen3-VL's deepstack injection, drop the rest per
--keep_idx), but parameterized over --backbone so it works for any Qwen3-VL
size (2B/8B/32B/...), and disk-safe for large backbones via --free_cache:
after the full model is loaded into RAM and cloned (fully decoupled from any
mmap'd view into the downloaded safetensors files), the on-disk HF cache for
--backbone is deleted BEFORE the (smaller) pruned model is written out, so the
full-size download and the pruned copy never have to coexist on disk at once.
"""
import argparse
import os
import shutil
import torch
from transformers import AutoConfig, AutoProcessor
from huggingface_hub import snapshot_download

from src.model_utils import backbone2model, QWEN3_VL

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", required=True, help="HF repo id, e.g. Qwen/Qwen3-VL-8B-Instruct")
parser.add_argument("--keep_idx", required=True, help="comma-separated explicit layer indices to keep, must start with 0,1,2")
parser.add_argument("--out_dir", required=True)
parser.add_argument("--free_cache", action="store_true", help="delete the HF cache for --backbone after loading, before writing the pruned model")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

print(f"loading {args.backbone}")
config = AutoConfig.from_pretrained(args.backbone, trust_remote_code=True)
cls = backbone2model[QWEN3_VL]
model = cls.from_pretrained(args.backbone, config=config, dtype=torch.bfloat16, low_cpu_mem_usage=False)
processor = AutoProcessor.from_pretrained(args.backbone, trust_remote_code=True)

# force every tensor to own independent memory (not a view into a memory-mapped
# checkpoint file) so it's safe to delete the on-disk cache next.
for p in model.parameters():
    p.data = p.data.clone()
for b in model.buffers():
    b.data = b.data.clone()

lm = model.model.language_model
layers = lm.layers
n_layers = len(layers)
print(f"original decoder layers: {n_layers}")

keep_idx = sorted(int(x) for x in args.keep_idx.split(","))
assert keep_idx[:3] == [0, 1, 2], f"deepstack requires layers 0,1,2 kept in place, got {keep_idx[:3]}"

new_layers = torch.nn.ModuleList([layers[i] for i in keep_idx])
for new_i, layer in enumerate(new_layers):
    layer.self_attn.layer_idx = new_i
lm.layers = new_layers
lm.config.num_hidden_layers = len(keep_idx)
model.config.text_config.num_hidden_layers = len(keep_idx)

total = sum(p.numel() for p in model.parameters())
print(f"new decoder layers: {len(lm.layers)}")
print(f"new total params: {total:,}")

# copy auxiliary files (tokenizer, chat template, etc) save_pretrained can miss,
# from the original cached snapshot -- do this BEFORE freeing the cache.
orig_dir = snapshot_download(args.backbone)
SKIP_SUFFIXES = (".safetensors", ".bin")
for fname in os.listdir(orig_dir):
    src = os.path.join(orig_dir, fname)
    dst = os.path.join(args.out_dir, fname)
    if os.path.isfile(src) and not fname.endswith(SKIP_SUFFIXES) and "index.json" not in fname and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"copied aux file: {fname}")

if args.free_cache:
    cache_dir = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub",
                              f"models--{args.backbone.replace('/', '--')}")
    print(f"freeing base-model cache before writing pruned save: {cache_dir}")
    shutil.rmtree(cache_dir, ignore_errors=True)

print(f"saving to {args.out_dir}")
model.save_pretrained(args.out_dir)
processor.save_pretrained(args.out_dir)

print("done:", sorted(os.listdir(args.out_dir)))
