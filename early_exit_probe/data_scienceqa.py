"""ScienceQA train split (image-bearing rows) as a probe training source, with the same
answer-fit / confidence-fit / calibration partition as A-OKVQA (70/15/15).

Unlike the A-OKVQA split (row-level), this one is IMAGE-DISJOINT as the deck requires:
rows are grouped by the md5 of their image bytes and whole groups are assigned to a split,
so a picture never appears in two splits. ScienceQA repeats templated questions/images, so
this matters more here than for A-OKVQA.

The MMEB-eval ScienceQA rows are drawn from ScienceQA's *test* split (see data_mmeb_eval),
so this train source cannot leak eval rows by construction (templated near-duplicates across
ScienceQA splits are inherent to the dataset and not removed).
"""
from __future__ import annotations

import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset

from .data_aokvqa import Example, SPLIT_FRACTIONS, SplitName

TRAIN_FILE = "data/train-00000-of-00001-1028f23e353fbe3e.parquet"
MAX_SIDE = 768


def _flat(s: str) -> str:
    return " ".join(s.split())  # newlines would confuse the "\n<label>." choice-boundary search


@lru_cache(maxsize=1)
def _load_train(seed: int = 26091622) -> dict:
    import pyarrow.parquet as pq

    path = hf_hub_download("derek-thomas/ScienceQA", TRAIN_FILE, repo_type="dataset")
    table = pq.read_table(path, columns=["question", "choices", "answer", "image"]).to_pydict()
    keep = [i for i, im in enumerate(table["image"]) if im is not None]
    questions = [_flat(table["question"][i]) for i in keep]
    choices = [[_flat(c) for c in table["choices"][i]] for i in keep]
    answers = [table["answer"][i] for i in keep]
    images = [table["image"][i]["bytes"] for i in keep]
    hashes = [hashlib.md5(b).hexdigest() for b in images]

    unique = sorted(set(hashes))
    order = np.random.default_rng(seed).permutation(len(unique))
    n = len(unique)
    n_answer = int(round(n * SPLIT_FRACTIONS["answer_fit"]))
    n_conf = int(round(n * SPLIT_FRACTIONS["confidence_fit"]))
    split_of_hash = {}
    for rank, u in enumerate(order):
        split_of_hash[unique[u]] = ("answer_fit" if rank < n_answer
                                    else "confidence_fit" if rank < n_answer + n_conf else "calibration")
    row_split = [split_of_hash[h] for h in hashes]
    return {"questions": questions, "choices": choices, "answers": answers, "images": images,
            "row_split": row_split, "n_unique_images": n, "seed": seed}


def save_split_index(path: Path) -> dict:
    data = _load_train()
    summary = {
        "seed": data["seed"], "n_rows": len(data["row_split"]), "n_unique_images": data["n_unique_images"],
        "rows_per_split": {s: data["row_split"].count(s) for s in SPLIT_FRACTIONS},
        "row_split": data["row_split"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f)
    return summary


class ScienceQATrainDataset(Dataset):
    def __init__(self, split_filter: SplitName, limit: int | None = None):
        self.data = _load_train()
        self.rows = [i for i, s in enumerate(self.data["row_split"]) if s == split_filter]
        if limit is not None:
            self.rows = self.rows[:limit]
        self.choice_counts = [len(self.data["choices"][i]) for i in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx) -> Example:
        i = self.rows[idx]
        image = Image.open(io.BytesIO(self.data["images"][i])).convert("RGB")
        if max(image.size) > MAX_SIDE:
            image.thumbnail((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
        return Example(
            question_id=f"scienceqa-train:{i}",
            question=self.data["questions"][i],
            choices=self.data["choices"][i],
            correct_choice_idx=self.data["answers"][i],
            image=image,
        )
