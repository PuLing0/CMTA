from __future__ import annotations

import math
import os
import pickle
from collections import defaultdict
from typing import Iterable, Iterator, Sequence

import torch

_PT_LENGTHS_CACHE_BASENAME = "pt_lengths_v1.pkl"
_PT_LENGTHS_CACHE: dict[str, int] | None = None
_PT_LENGTHS_CACHE_DIRTY = False


def next_square(n: int) -> int:
    if n <= 0:
        raise ValueError(f"n must be > 0 (got {n})")
    k = int(math.ceil(math.sqrt(n)))
    return k * k


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _cache_dir() -> str:
    return os.path.join(_project_root(), ".cache")


def _cache_path() -> str:
    return os.path.join(_cache_dir(), _PT_LENGTHS_CACHE_BASENAME)


def _load_pt_lengths_cache() -> dict[str, int]:
    global _PT_LENGTHS_CACHE
    if _PT_LENGTHS_CACHE is not None:
        return _PT_LENGTHS_CACHE

    path = _cache_path()
    if not os.path.isfile(path):
        _PT_LENGTHS_CACHE = {}
        return _PT_LENGTHS_CACHE

    try:
        with open(path, "rb") as fp:
            data = pickle.load(fp)
        if not isinstance(data, dict):
            raise TypeError("cache is not a dict")
        _PT_LENGTHS_CACHE = {str(k): int(v) for k, v in data.items()}
    except Exception:
        _PT_LENGTHS_CACHE = {}
    return _PT_LENGTHS_CACHE


def _save_pt_lengths_cache() -> None:
    global _PT_LENGTHS_CACHE_DIRTY
    if not _PT_LENGTHS_CACHE_DIRTY:
        return

    cache = _load_pt_lengths_cache()
    os.makedirs(_cache_dir(), exist_ok=True)
    path = _cache_path()
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as fp:
        pickle.dump(cache, fp, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    _PT_LENGTHS_CACHE_DIRTY = False


def _pt_len(path: str) -> int:
    global _PT_LENGTHS_CACHE_DIRTY
    cache = _load_pt_lengths_cache()
    path = os.path.abspath(path)
    cached = cache.get(path, None)
    if cached is not None:
        return cached

    tensor = torch.load(path, map_location="cpu")
    if not hasattr(tensor, "shape") or len(getattr(tensor, "shape", ())) < 1:
        raise TypeError(f"Unsupported .pt payload at {path!r} (expected Tensor-like)")
    n = int(tensor.shape[0])
    cache[path] = n
    _PT_LENGTHS_CACHE_DIRTY = True
    return n


def _slide_id_to_pt_stem(slide_id: object) -> str:
    # Keep identical behavior to the dataset's `slide_id.rstrip('.svs')`.
    return str(slide_id).rstrip(".svs")


def _resolve_data_dir(dataset: object, idx: int) -> str:
    data_dir = getattr(dataset, "data_dir", None)
    if isinstance(data_dir, dict):
        slide_data = getattr(dataset, "slide_data")
        source = slide_data["oncotree_code"][idx]
        return data_dir[source]
    return data_dir


def compute_bucket_lengths(dataset: object) -> list[int]:
    """
    Compute per-index square-padded bag lengths L = next_square(min(total_patches, OOM)).

    This is used by a length-bucketed BatchSampler so that each batch has a single
    common L (no padding/masks needed across samples).
    """
    oom = int(getattr(dataset, "OOM", 0) or 0)
    slide_data = getattr(dataset, "slide_data")
    patient_dict = getattr(dataset, "patient_dict")

    bucket_lengths: list[int] = []
    for idx in range(len(dataset)):
        case_id = slide_data["case_id"][idx]
        slide_ids = patient_dict[case_id]
        if isinstance(slide_ids, str):
            slide_ids = [slide_ids]

        data_dir = _resolve_data_dir(dataset, idx)
        total_patches = 0
        for slide_id in slide_ids:
            wsi_path = os.path.join(data_dir, "pt_files", f"{_slide_id_to_pt_stem(slide_id)}.pt")
            if not os.path.isfile(wsi_path):
                continue
            total_patches += _pt_len(wsi_path)

        if total_patches <= 0:
            raise RuntimeError(f"Empty bag detected at dataset idx={idx} (case_id={case_id!r})")

        eff = min(total_patches, oom) if oom > 0 else total_patches
        bucket_lengths.append(next_square(int(eff)))

    _save_pt_lengths_cache()
    return bucket_lengths


class LengthBucketBatchSampler:
    """
    BatchSampler that groups indices by a precomputed bucket key (e.g. square bag length).

    Note: this intentionally does NOT implement __len__ since grouping depends on
    the sampling stream (especially for replacement samplers).
    """

    def __init__(
        self,
        sampler: Iterable[int],
        bucket_keys: Sequence[int],
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0 (got {batch_size})")
        self.sampler = sampler
        self.bucket_keys = bucket_keys
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

    def __iter__(self) -> Iterator[list[int]]:
        buckets: dict[int, list[int]] = defaultdict(list)
        for idx in self.sampler:
            key = int(self.bucket_keys[int(idx)])
            bucket = buckets[key]
            bucket.append(int(idx))
            if len(bucket) >= self.batch_size:
                batch = bucket[: self.batch_size]
                del bucket[: self.batch_size]
                yield batch

        if self.drop_last:
            return

        for bucket in buckets.values():
            while bucket:
                batch = bucket[: self.batch_size]
                del bucket[: self.batch_size]
                yield batch
