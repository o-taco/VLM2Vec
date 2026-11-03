"""Image-classification sources from the deck's training pool (page 4: CUB-200, ImageNet,
iNaturalist; "not MMEB") as multiple-choice examples, with the same seeded 70/15/15
answer-fit / confidence-fit / calibration partition as the other sources. Every row is its
own image, so a row-level split is image-disjoint.

The deck gives only answer-fit counts for these sources, not their candidate sets or
questions, so those are design choices here: CUB-200 uses its native 200-way label set, and
candidates are shuffled per row exactly as in data_mmeb_eval.
"""
from __future__ import annotations

import io
import json
from functools import lru_cache

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset

from .data_aokvqa import Example, SPLIT_FRACTIONS, SplitName

SEED = 26091622
MAX_SIDE = 768

SOURCES = {
    "CUB-200": {
        "repo": "Donghyun99/CUB-200-2011",
        "files": ("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"),
        "question": "Which bird species is shown in this image?",
    },
}


@lru_cache(maxsize=None)
def _load(name: str, seed: int = SEED) -> dict:
    import pyarrow.parquet as pq

    spec = SOURCES[name]
    images, labels, names = [], [], None
    for f in spec["files"]:
        path = hf_hub_download(spec["repo"], f, repo_type="dataset")
        pf = pq.ParquetFile(path)
        names = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
        table = pf.read(columns=["image", "label"]).to_pydict()
        images += [im["bytes"] for im in table["image"]]
        labels += table["label"]
    n = len(labels)
    order = np.random.default_rng(seed).permutation(n)
    n_answer = int(round(n * SPLIT_FRACTIONS["answer_fit"]))
    n_conf = int(round(n * SPLIT_FRACTIONS["confidence_fit"]))
    row_split = [None] * n
    for rank, i in enumerate(order):
        row_split[i] = ("answer_fit" if rank < n_answer
                        else "confidence_fit" if rank < n_answer + n_conf else "calibration")
    return {"images": images, "labels": labels, "row_split": row_split,
            "classes": [" ".join(c.replace("_", " ").split()) for c in names]}


class ClassificationTrainDataset(Dataset):
    def __init__(self, name: str, split_filter: SplitName, limit: int | None = None, seed: int = SEED):
        self.name, self.seed = name, seed
        self.data = _load(name)
        self.rows = [i for i, s in enumerate(self.data["row_split"]) if s == split_filter]
        if limit is not None:
            self.rows = self.rows[:limit]
        self.choice_counts = [len(self.data["classes"])] * len(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx) -> Example:
        i = self.rows[idx]
        classes = self.data["classes"]
        perm = np.random.default_rng([self.seed, i]).permutation(len(classes))
        image = Image.open(io.BytesIO(self.data["images"][i])).convert("RGB")
        if max(image.size) > MAX_SIDE:
            image.thumbnail((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
        return Example(
            question_id=f"{self.name}-train:{i}",
            question=SOURCES[self.name]["question"],
            choices=[classes[p] for p in perm],
            correct_choice_idx=int(np.where(perm == self.data["labels"][i])[0][0]),
            image=image,
        )
