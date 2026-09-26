"""MMEB-eval classification sets as native-candidate multiple-choice examples for the
early-exit probe (deck pages 7 and 17).

MMEB-eval stores every row as retrieval-style `(qry_text, qry_img_path, tgt_text[])` with
the correct answer ALWAYS at `tgt_text[0]` (page 17: "Earlier numbers used rows whose
correct answer was always first; those numbers are superseded"). So the candidate list is
shuffled per row with seed 26091622 and the correct index tracked; an unshuffled list would
let the probe read "index 0 is right" off position.

Questions are natural-language ("Natural questions replace MMEB instructions", page 17)
because the MMEB instruction strings ("Represent the given image for classification") are
embedding-model prompts, not questions.

The 7 classification sets use the row's own `tgt_text` pool as the native candidate set
(identical across rows). Visual7W and ScienceQA instead carry a pooled 1000/736-entry
distractor list in MMEB, so their native choices (4-way / 2-5-way) are recovered from the
source datasets and matched to MMEB rows:
  * Visual7W: `slxhere/visual7w-telling` annotations; key = (image_id from `v7w_<id>.jpg`,
    question); choices = 3 `multiple_choices` + `answer`, shuffled with the seeded RNG
    (999/1000 rows match uniquely; the one duplicate takes the first entry).
  * ScienceQA: `derek-thomas/ScienceQA` test split; MMEB `image_<i>` is the i-th image-bearing
    test row (verified 1000/1000 on question + answer). Source choice order is kept as-is:
    the answer's position is already varied there, so no shuffle is needed.
"""
from __future__ import annotations

import collections
import json
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset

from .data_aokvqa import Example

SHUFFLE_SEED = 26091622
MMEB_IMAGE_ROOT = Path(__file__).resolve().parents[1] / "data_eval" / "mmeb_images"
MAX_SIDE = 768  # cap so multi-megapixel images (ObjectNet) don't become 10K+ vision tokens

QUESTIONS = {
    "HatefulMemes": "Is this meme hateful?",
    "VOC2007": "Which object is shown in this image?",
    "ImageNet-A": "What is the main object in this image?",
    "ImageNet-R": "What is the main object in this image?",
    "ObjectNet": "Which object is shown in this image?",
    "Country211": "Which country is this photo taken in?",
}
N24_QUESTION = "News caption: {caption}\nWhich news section does this article belong to?"
N24_CAPTION_MARKER = "for domain classification: "

DATASETS = ("N24News", "HatefulMemes", "VOC2007", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211")
NATIVE_CHOICE_DATASETS = ("Visual7W", "ScienceQA")
ALL_DATASETS = DATASETS + NATIVE_CHOICE_DATASETS
QRY_MARKER = "Represent the given image with the following question: "


def _mmeb_question(row) -> str:
    return _flat(row["qry_text"].split(QRY_MARKER, 1)[1])


def visual7w_native_choices(rows) -> list[tuple[list[str], str]]:
    """Per MMEB row: (3 distractors + answer as an unshuffled list, answer)."""
    zpath = hf_hub_download("slxhere/visual7w-telling", "dataset_v7w_telling.zip", repo_type="dataset")
    data = json.load(zipfile.ZipFile(zpath).open("dataset_v7w_telling.json"))
    index = collections.defaultdict(list)
    for im in data["images"]:
        for qa in im["qa_pairs"]:
            index[(qa["image_id"], _flat(qa["question"]))].append(qa)
    out = []
    for r in rows:
        image_id = int(r["qry_img_path"].split("v7w_")[1].split(".")[0])
        cands = [qa for qa in index[(image_id, _mmeb_question(r))] if qa["answer"] == r["tgt_text"][0]]
        assert cands, f"no Visual7W source row for {r['qry_img_path']}: {_mmeb_question(r)}"
        qa = cands[0]
        out.append((qa["multiple_choices"] + [qa["answer"]], qa["answer"]))
    return out


def scienceqa_native_choices(rows) -> list[tuple[list[str], str]]:
    """Per MMEB row: (source choices in source order, answer). MMEB image_<i> == i-th
    image-bearing row of the ScienceQA test split."""
    import pyarrow.parquet as pq

    path = hf_hub_download("derek-thomas/ScienceQA", "data/test-00000-of-00001-f0e719df791966ff.parquet",
                           repo_type="dataset")
    src = [s for s in pq.read_table(path, columns=["question", "choices", "answer", "image"]).to_pylist()
           if s["image"] is not None]
    out = []
    for r in rows:
        i = int(r["qry_img_path"].split("image_")[1].split(".")[0])
        s = src[i]
        answer = s["choices"][s["answer"]]
        assert _flat(s["question"]) == _mmeb_question(r) and answer == r["tgt_text"][0], \
            f"ScienceQA row {i} does not line up with its source row"
        out.append((list(s["choices"]), answer))
    return out


def _flat(s: str) -> str:
    return " ".join(s.split())  # newlines would confuse the "\n<label>." choice-boundary search


def _display(name: str, dataset: str) -> str:
    if dataset == "ImageNet-A":
        return _flat(name.split(",")[0])  # deck: "ImageNet-A synonym lists use the first name"
    return _flat(name.replace("_", " "))


def imagenet_a_200(pool: list[str], answers: set[str], seed: int = SHUFFLE_SEED) -> list[str]:
    """The deck labels ImageNet-A 200-way but MMEB's row pool has all 1000 ImageNet classes.
    Recover the 200 official classes by matching CLIP-benchmark's ImageNet-A class list to
    any lemma of a pool entry. This matches ~180/200 (CLIP's curated names differ from
    WordNet lemmas for the rest), so the remainder is filled with the answer classes that
    matched nothing plus a seeded random sample of the pool. APPROXIMATE -- not the deck's
    exact 200-way set."""
    try:
        path = hf_hub_download("clip-benchmark/wds_imagenet-a", "classnames.txt", repo_type="dataset")
        names = [l.strip() for l in open(path) if l.strip()]
    except Exception:
        names = []
    lemma_to_entries: dict[str, set[str]] = {}
    for entry in pool:
        for lemma in entry.split(","):
            lemma_to_entries.setdefault(lemma.strip().lower(), set()).add(entry)
    chosen: list[str] = []
    seen: set[str] = set()

    def add(entry):
        if entry not in seen:
            seen.add(entry)
            chosen.append(entry)

    for n in names:
        for entry in sorted(lemma_to_entries.get(n.lower(), ())):
            add(entry)
    for a in sorted(answers):
        add(a)
    rng = np.random.default_rng(seed)
    for idx in rng.permutation(len(pool)):
        if len(chosen) >= 200:
            break
        add(pool[idx])
    return chosen[:200]


class MMEBClassificationDataset(Dataset):
    def __init__(self, name: str, seed: int = SHUFFLE_SEED, limit: int | None = None):
        import pyarrow.parquet as pq

        assert name in ALL_DATASETS, f"{name} not a supported set: {ALL_DATASETS}"
        self.name = name
        self.seed = seed
        path = hf_hub_download("TIGER-Lab/MMEB-eval", f"{name}/test-00000-of-00001.parquet", repo_type="dataset")
        rows = pq.read_table(path).to_pylist()
        if limit is not None:
            rows = rows[:limit]
        self.rows = rows
        self.candidate_subset = None
        self.native = None  # per-row (choices, answer) for datasets whose choices come from the source
        if name == "Visual7W":
            self.native = visual7w_native_choices(rows)
        elif name == "ScienceQA":
            self.native = scienceqa_native_choices(rows)
        if name == "ImageNet-A":
            all_rows = pq.read_table(path).to_pylist()
            answers = {r["tgt_text"][0] for r in all_rows}
            self.candidate_subset = set(imagenet_a_200(rows[0]["tgt_text"], answers))
            missing = {r["tgt_text"][0] for r in rows} - self.candidate_subset
            assert not missing, f"answers outside the 200-way set: {missing}"
        self.choice_counts = [len(self._pool(i)) for i in range(len(rows))]
        self.n_choices = max(self.choice_counts)

    def _pool(self, idx):
        if self.native is not None:
            return self.native[idx][0]
        pool = self.rows[idx]["tgt_text"]
        if self.candidate_subset is not None:
            pool = [c for c in pool if c in self.candidate_subset]
        return pool

    def __len__(self):
        return len(self.rows)

    def _question(self, row) -> str:
        if self.name in NATIVE_CHOICE_DATASETS:
            return _mmeb_question(row)
        if self.name == "N24News":
            caption = row["qry_text"].split(N24_CAPTION_MARKER, 1)[1].strip()
            return N24_QUESTION.format(caption=_flat(caption))
        return QUESTIONS[self.name]

    def __getitem__(self, idx) -> Example:
        row = self.rows[idx]
        pool = self._pool(idx)
        correct = self.native[idx][1] if self.native is not None else row["tgt_text"][0]
        if self.name == "ScienceQA":
            shuffled = list(pool)  # source order already varies the answer position
        else:
            perm = np.random.default_rng([self.seed, idx]).permutation(len(pool))
            shuffled = [pool[p] for p in perm]
        correct_idx = shuffled.index(correct)
        image = Image.open(MMEB_IMAGE_ROOT / row["qry_img_path"]).convert("RGB")
        if max(image.size) > MAX_SIDE:
            image.thumbnail((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
        return Example(
            question_id=f"{self.name}:{idx}",
            question=self._question(row),
            choices=[_display(c, self.name) for c in shuffled],
            correct_choice_idx=correct_idx,
            image=image,
        )
