"""Image-classification sources from the deck's training pool (page 4: CUB-200, ImageNet,
iNaturalist; "not MMEB") as multiple-choice examples, with the same seeded 70/15/15
answer-fit / confidence-fit / calibration partition as the other sources. Every row is its
own image, so a row-level split is image-disjoint.

The deck gives only answer-fit counts for these sources, not their candidate sets or
questions, so those are design choices here: CUB-200 uses its native 200-way label set, and
candidates are shuffled per row exactly as in data_mmeb_eval. ImageNet (1,000 classes) and
iNaturalist 2021 (10,000 species) would make prompts 5-50x longer with full lists, so each row
gets its true class plus 199 seeded random distractors (K=200, as in ImageNet-A/R). Only one
parquet shard of each is used: Table 3 needs a few hundred rows per role, and disk is small.
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
    "ImageNet": {
        "repo": "ILSVRC/imagenet-1k",  # gated: needs HF_TOKEN with accepted terms
        "files": ("data/train-00000-of-00294.parquet",),
        "question": "Which object is shown in this image?",
        "n_choices": 200,
    },
    "iNaturalist": {
        "repo": "MVRL/iNat-2021-val",  # iNat 2021 val images; labels from category_id / file_name
        "files": ("data/train-00000-of-00017.parquet",),
        "question": "Which species is shown in this image?",
        "label_col": "category_id",
        "n_choices": 200,
    },
}


def _class_names(name, pf, table):
    """(class display names, per-row label index into them)."""
    if name == "iNaturalist":
        # val/03938_Animalia_..._Ptilotula_penicillata/x.jpg -> category 3938, "Ptilotula penicillata"
        species = {}
        for cid, fn in zip(table["category_id"], table["file_name"]):
            species[cid] = " ".join(fn.split("/")[1].split("_")[-2:])
        ids = sorted(species)
        index = {c: k for k, c in enumerate(ids)}
        return [species[c] for c in ids], [index[c] for c in table["category_id"]]
    names = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
    if name == "ImageNet":
        names = [n.split(",")[0] for n in names]  # first WordNet synonym
    return names, table["label"]


@lru_cache(maxsize=None)
def _load(name: str, seed: int = SEED) -> dict:
    import pyarrow.parquet as pq

    spec = SOURCES[name]
    images, labels, names = [], [], None
    cols = ["image", "category_id", "file_name"] if name == "iNaturalist" else ["image", "label"]
    for f in spec["files"]:  # multi-file sources share one ClassLabel list (CUB)
        path = hf_hub_download(spec["repo"], f, repo_type="dataset")
        pf = pq.ParquetFile(path)
        table = pf.read(columns=cols).to_pydict()
        names, row_labels = _class_names(name, pf, table)
        images += [im["bytes"] for im in table["image"]]
        labels += row_labels
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
        self.k = SOURCES[name].get("n_choices") or len(self.data["classes"])
        self.choice_counts = [self.k] * len(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx) -> Example:
        i = self.rows[idx]
        classes, label = self.data["classes"], self.data["labels"][i]
        rng = np.random.default_rng([self.seed, i])
        if self.k < len(classes):  # true class + (k-1) distractors, then shuffled
            others = np.delete(np.arange(len(classes)), label)
            perm = rng.permutation(np.append(rng.choice(others, self.k - 1, replace=False), label))
        else:
            perm = rng.permutation(len(classes))
        image = Image.open(io.BytesIO(self.data["images"][i])).convert("RGB")
        if max(image.size) > MAX_SIDE:
            image.thumbnail((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
        return Example(
            question_id=f"{self.name}-train:{i}",
            question=SOURCES[self.name]["question"],
            choices=[classes[p] for p in perm],
            correct_choice_idx=int(np.where(perm == label)[0][0]),
            image=image,
        )
