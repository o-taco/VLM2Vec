# How queries and targets are built (A-OKVQA, MMEB format)

This is a dual-encoder retrieval setup (same family as CLIP / dense retrieval),
**not** transformer attention Q/K/V. Two independent things get embedded into
vectors, then compared by cosine similarity:

- **query** = image + instruction-wrapped question
- **target** ("pos_text"/"tgt_text") = a plain answer string, text-only

There is no "key" in the attention sense here — when people loosely say
"keys," in this context they mean the target/candidate answer embeddings that
the query embedding is compared against.

## 1. Training pairs (`data/A-OKVQA/original-00000-of-00001.parquet`)

Columns: `qry`, `qry_image_path`, `pos_text`, `pos_image_path`, `neg_text`, `neg_image_path`.

Real row from the file:

```
qry:            <|image_1|>
                Represent the given image with the following question: What is the man by the bags awaiting?
qry_image_path: images/A-OKVQA/Train/A-OKVQA_image_0.jpg
pos_text:       cab
pos_image_path: (empty)
neg_text:       train           <- present in the file but NEVER used (see below)
neg_image_path: (empty)
```

**Query** = `qry` (text) + the image at `qry_image_path`. The text is always:

```
<|image_1|>
Represent the given image with the following question: {question}
```

- `<|image_1|>` is a placeholder token from how the dataset was authored
  (Phi-3-Vision convention). At load time it's swapped for whatever the
  actual backbone needs — for Qwen3-VL that's `<|image_pad|>`
  (`vlm_image_tokens` in `src/model_utils.py`), then the processor expands it
  into the real number of image patch tokens for that image.
- `"Represent the given image with the following question: "` is a fixed
  **instruction prefix** — it's not part of the A-OKVQA question itself, it's
  prepended so the model knows "this is an embedding/retrieval query," a
  convention MMEB borrows from instruction-tuned text embedding models (e5,
  etc).
- The image itself is resized to a fixed square (`--image_resolution 336` in
  every training script here → 336×336) before being handed to the processor.

**Target/positive** = `pos_text` alone — e.g. `"cab"` — **no image**
(`pos_image_path` is empty; `process_vlm_inputs` is called with
`"image": [None]*len(pos_texts)`). It is NOT wrapped in any instruction
template — just the raw answer string.

**`neg_text` is dead weight.** The collator (`TrainTextImageDataCollator` in
`src/collator.py`) loads `neg_text` from the parquet row but never returns it
to the model. The actual negatives during training are simply *every other
example's `pos_text` in the same batch* (in-batch negatives) — for a batch of
64, each query is scored against all 64 positives, and the diagonal (its own
positive) is the correct target. So `neg_text`/`neg_image_path` in the raw
data file are unused legacy columns.

## 2. Eval candidates (`TIGER-Lab/MMEB-eval`, subset `A-OKVQA`, split `test`)

Columns: `qry_text`, `qry_img_path`, `tgt_text`, `tgt_img_path`.

Real row:

```
qry_text:  <|image_1|>
           Represent the given image with the following question: What is in the motorcyclist's mouth?
qry_img_path: A-OKVQA/image_0.jpg
tgt_text:  ['cigarette', '1000', 'eleven', 'i do', 'malaysia', ...]   (894 candidates total)
tgt_img_path: ['', '', '', '', '', ...]                              (all empty -- text-only, same as training)
```

Same query construction as training (image + instruction-wrapped question).
The difference is the target side: instead of one positive text, each query
comes with a **fixed pool of 894 candidate answer strings**, and **the ground
truth is always at index 0** (`tgt_text[0]`). `compute_p1_accuracy.py` embeds
the query and all 894 candidates, and checks whether index 0 gets the highest
cosine similarity (P@1). `compute_val_loss.py` subsamples down to
`--num_candidates` (default 50) of those 894 to keep the eval cheap, always
keeping index 0 (the true answer) in the subsample.

## 3. What about A-OKVQA's original multiple-choice options?

**They aren't used at all.** A-OKVQA (the original dataset) ships 4
multiple-choice options per question, but MMEB reformulates every VQA-style
dataset as **open-ended text retrieval**: the "choices" field is discarded
entirely. The model never sees the 4 options — it just has to embed the
question+image close to the free-text correct answer string, and far from
whatever other answer strings end up as negatives (in-batch during training,
or the 894-candidate pool during eval). The "candidates" a query is being
compared against are other examples' ground-truth answers pulled into the
same batch/pool, not the original dataset's multiple-choice distractors.

## Summary table

| | query | target (positive) |
|---|---|---|
| **contains image?** | yes (the actual photo) | no, text-only |
| **contains instruction text?** | yes, fixed template + question | no, just the raw answer |
| **source field (train)** | `qry` + `qry_image_path` | `pos_text` |
| **source field (eval)** | `qry_text` + `qry_img_path` | `tgt_text[i]` |
| **negatives** | — | other examples' positives (train) / candidate pool minus index 0 (eval) |

## 4. The three metrics used across the pruning ablation -- and why they're comparable

Every pruning level (0/25/28.6/32.1/35.7/50/75%) is trained with the identical
recipe (LoRA r=8 alpha=16, batch 64, 500 steps, lr 1e-5, `--image_resolution
336`) and then scored with all three of these, each applied **identically
across every pruning level** (verified by grepping every checkpoint's log,
not assumed):

| Metric | Script | Data split | Pool size | Confirmed uniform via |
|---|---|---|---|---|
| 4-way train accuracy/loss | `compute_train_accuracy.py` | training parquet (`data/A-OKVQA`) | 4, in-batch | every log: `"Loaded 200 train rows -> 50 batches of 4"` |
| Val loss | `compute_val_loss.py` | held-out `MMEB-eval` test split | 50 candidates | every log: `"Loaded 200 eval rows"` + `"num_candidates=50"` |
| P@1 | `compute_p1_accuracy.py` | held-out `MMEB-eval` test split | 894 candidates | every log: `"Loaded 1000 eval rows"` |

None of the three ever use A-OKVQA's original multiple-choice options (see
section 3) -- that's uniform across all three metrics and all seven pruning
levels, not a per-level inconsistency.

**Why 3 metrics instead of 1:** they trade off eval cost against how hard the
discrimination task is. 4-way is cheap and forgiving (chance=25%); P@1 is
expensive but is the "real" generalization number (chance≈0.11%, 894
candidates); val loss sits in between and is cheap enough to run at every
saved checkpoint for best-checkpoint selection. They're expected to diverge
from each other (different chance floors, different sensitivity to
representation collapse) -- that divergence is a finding, not a bug. What
would be a bug is if any one metric were computed differently across
different pruning levels, which is what the table above rules out.
