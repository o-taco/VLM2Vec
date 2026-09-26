"""Multi-source training pool for the early-exit probe.

`collate` requires every example in a batch to share one option count (A-OKVQA is 4-way,
ScienceQA is 2-5-way), so the pool is organised as (source, n_choices) groups per split and
each training step draws its whole batch from ONE group. Source choice is uniform across
sources (a design choice -- the deck does not give its pool mixture); the option-count group
within a source is drawn proportionally to its answer-fit size.

Pools: "aokvqa" (Phase-1 behaviour) and "aokvqa+scienceqa".
"""
from __future__ import annotations

import random
from pathlib import Path

from .data_aokvqa import AOKVQADataset, load_or_build_split_index

SPLITS = ("answer_fit", "confidence_fit", "calibration")


class _Source:
    def __init__(self, name: str, datasets: dict, counts: dict):
        self.name = name
        self.datasets = datasets  # split -> dataset
        self.groups = {}  # split -> {n_choices: [indices]}
        for split in SPLITS:
            g = {}
            for i, c in enumerate(counts[split]):
                g.setdefault(c, []).append(i)
            self.groups[split] = g


def _aokvqa_source(split_index_path: Path, cache_dir, limit) -> _Source:
    split_index = load_or_build_split_index(split_index_path, cache_dir=cache_dir)
    datasets = {s: AOKVQADataset("train", s, split_index, cache_dir=cache_dir, limit=limit) for s in SPLITS}
    counts = {s: [len(c) for c in datasets[s].ds["choices"]] for s in SPLITS}
    return _Source("A-OKVQA", datasets, counts)


def _scienceqa_source(limit) -> _Source:
    from .data_scienceqa import ScienceQATrainDataset

    datasets = {s: ScienceQATrainDataset(s, limit=limit) for s in SPLITS}
    counts = {s: datasets[s].choice_counts for s in SPLITS}
    return _Source("ScienceQA", datasets, counts)


class TrainingPool:
    def __init__(self, pool: str, aokvqa_split_index_path: Path, cache_dir=None, limit=None):
        names = pool.split("+")
        assert set(names) <= {"aokvqa", "scienceqa"} and "aokvqa" in names, f"unknown pool {pool}"
        self.sources = [_aokvqa_source(aokvqa_split_index_path, cache_dir, limit)]
        if "scienceqa" in names:
            self.sources.append(_scienceqa_source(limit))

    def describe(self) -> str:
        parts = []
        for src in self.sources:
            per_split = {s: {n: len(v) for n, v in sorted(src.groups[s].items())} for s in SPLITS}
            parts.append(f"{src.name}: {per_split}")
        return "\n".join(parts)

    def sample_step(self, rng: random.Random, k: int):
        """k answer-fit + k confidence-fit examples sharing one (source, n_choices) group."""
        while True:
            src = rng.choice(self.sources)
            counts = sorted(src.groups["answer_fit"])
            weights = [len(src.groups["answer_fit"][n]) for n in counts]
            n = rng.choices(counts, weights=weights)[0]
            if n in src.groups["confidence_fit"]:
                break
        return (self._draw(src, "answer_fit", n, rng, k), self._draw(src, "confidence_fit", n, rng, k))

    @staticmethod
    def _draw(src, split, n, rng, k):
        idxs = src.groups[split][n]
        picked = rng.sample(idxs, k) if len(idxs) >= k else rng.choices(idxs, k=k)
        return [src.datasets[split][i] for i in picked]

    def calibration_batches(self, batch_size: int, limit=None):
        """Yields (source_name, [examples]) over the whole calibration split, one group per batch."""
        for src in self.sources:
            for n, idxs in sorted(src.groups["calibration"].items()):
                if limit is not None:
                    idxs = idxs[:limit]
                for s in range(0, len(idxs), batch_size):
                    yield src.name, [src.datasets["calibration"][i] for i in idxs[s:s + batch_size]]
