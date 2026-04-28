"""Dataset for the combined ASR features parquet.

Expects a single parquet produced by :mod:`src.data.preprocess`, containing:
    - ground_truth: str
    - {whisper, hubert, w2v2}_features: list[list[float]]  (variable-length [T, D])
    - {whisper, hubert, w2v2}_wer: float

Memory model:
    Two access strategies are supported, controlled by ``eager_load``.

    Eager mode (``eager_load=True``) — recommended on high-RAM hosts.
        At construction time the parquet is streamed into one flat
        ``(sum_T_k, D_k)`` ``np.float32`` buffer per expert plus a
        ``(N+1,)`` int64 offsets array, mirroring the variable-length
        layout used by Hugging Face / Arrow but kept on the heap as
        a single contiguous numpy allocation. ``__getitem__`` then
        does an O(1) slice into that buffer — no per-batch parquet
        decode and no Python object per frame, so training is
        bottlenecked on GPU instead of I/O. Multi-worker DataLoaders
        fork() and share the buffers via copy-on-write: the data
        pages are never refcounted (the numpy refcount lives in a
        tiny wrapper struct, not on the data pages), so workers
        read from the same physical pages without duplication.
        Estimated RAM = sum over experts of (N * mean_T * D * 4 B),
        e.g. ~38 GB for AMI with HuBERT-Large + Whisper-Base + W2V2.

    Lazy mode (``eager_load=False``) — the safe fallback for
        low-RAM environments such as Colab free tier. Opens the
        parquet as a :class:`pq.ParquetFile`, reads only the
        row-group containing the requested clip on demand, and
        keeps a small LRU cache so consecutive accesses within
        the same row group amortize the decode cost.

    The lazy strategy only works if the parquet's row groups are small
    (~few hundred clips). Many upstream pipelines (notably any path
    that goes through Pandas or :func:`pyarrow.parquet.write_table`
    without ``row_group_size``) write the whole table as a single huge
    row group, which makes "load one row group" mean "load the whole
    dataset" — and OOM-kills DataLoader workers on Colab. To make this
    robust regardless of how the input was written, we route the input
    parquet through :func:`src.data.parquet_cache.ensure_lazy_parquet`
    in ``__init__``: if the source has too-large row groups, a
    rechunked copy with small row groups is materialized to a local
    cache once, and all subsequent reads use that cached file. (This
    also bounds peak RAM during the eager-mode streaming decode.)
"""

from functools import lru_cache
import logging
from typing import Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.parquet_cache import DEFAULT_TARGET_ROW_GROUP_SIZE, ensure_lazy_parquet

logger = logging.getLogger(__name__)

MODEL_NAMES: List[str] = ["hubert", "whisper", "wav2vec2"]

FEATURE_COLUMNS: Dict[str, str] = {
    "hubert":   "hubert_features",
    "whisper":  "whisper_features",
    "wav2vec2": "w2v2_features",
}
WER_COLUMNS: Dict[str, str] = {
    "hubert":   "hubert_wer",
    "whisper":  "whisper_wer",
    "wav2vec2": "w2v2_wer",
}

DEFAULT_ROW_GROUP_CACHE_SIZE = 4


def _resolve_cache_size(num_row_groups: int, requested: Optional[int]) -> int:
    """Clamp the LRU cache size to a sensible range.

    Two row groups is rarely enough when the DataLoader uses random
    shuffled access (each batch's indices typically span many row
    groups), so default to 4 — a few hundred MB per worker. Cap at
    ``num_row_groups`` so we never allocate more cache slots than
    there are row groups.
    """
    if requested is None:
        requested = DEFAULT_ROW_GROUP_CACHE_SIZE
    return max(1, min(requested, max(1, num_row_groups)))


class ASRFeatureDataset(Dataset):
    """Variable-length frame-level embeddings + per-model WER.

    See the module docstring for the eager / lazy memory model.
    """

    def __init__(
        self,
        parquet_path: str,
        max_seq_len: int = 2000,
        cache_size: Optional[int] = None,
        auto_rechunk: bool = True,
        target_row_group_size: int = DEFAULT_TARGET_ROW_GROUP_SIZE,
        cache_dir: Optional[str] = None,
        eager_load: bool = False,
    ):
        """Args:
            parquet_path: Path to the unified combined parquet.
            max_seq_len: Frame sequences longer than this are truncated.
            cache_size: Number of row groups to keep decoded in memory
                per worker process (lazy mode only). ``None`` picks
                a sensible default (4) that handles random shuffled
                access without blowing up RAM.
            auto_rechunk: If True (default), stream-rewrite the input
                parquet with small row groups to a local cache when
                the source has too-large row groups. Fixes OOM kills
                in lazy mode and bounds peak RAM during eager-mode
                streaming decode.
            target_row_group_size: Rows per row group when rechunking.
            cache_dir: Where to write the rechunked file. Defaults to
                a local-disk cache (see :func:`ensure_lazy_parquet`).
            eager_load: If True, decode the entire feature parquet
                into per-expert flat ``np.float32`` buffers + offsets
                and serve all subsequent ``__getitem__`` calls from
                RAM. Recommended on high-RAM hosts (e.g. cloud GPU
                instances with 100+ GB system RAM). Eliminates per-
                batch parquet decode, so training becomes GPU-bound.
                Buffers are shared between DataLoader workers via
                fork+COW. Estimated RAM ≈ sum_k(N * mean_T * D_k * 4 B);
                a memory estimate is logged after loading.
        """
        self.source_parquet_path = str(parquet_path)
        self.parquet_path = ensure_lazy_parquet(
            parquet_path,
            target_row_group_size=target_row_group_size,
            cache_dir=cache_dir,
            auto_rechunk=auto_rechunk,
        )
        self.max_seq_len = max_seq_len

        self._pq_file: Optional[pq.ParquetFile] = None
        meta = pq.read_metadata(self.parquet_path)
        self.num_rows: int = meta.num_rows
        self.num_row_groups: int = meta.num_row_groups
        self.row_group_offsets: List[int] = []
        offset = 0
        for rg in range(self.num_row_groups):
            self.row_group_offsets.append(offset)
            offset += meta.row_group(rg).num_rows
        self.row_group_offsets.append(offset)

        self._cache_size = _resolve_cache_size(self.num_row_groups, cache_size)

        wer_cols = [WER_COLUMNS[n] for n in MODEL_NAMES]
        wer_table = pq.read_table(self.parquet_path, columns=wer_cols)
        self.wer_matrix = np.stack(
            [wer_table[c].to_numpy().astype(np.float32) for c in wer_cols],
            axis=-1,
        )

        self.eager_load = bool(eager_load)
        self._eager_buffers: Dict[str, np.ndarray] = {}
        self._eager_offsets: Dict[str, np.ndarray] = {}

        if self.eager_load:
            self._materialize_eager()

        self._read_row_group = lru_cache(maxsize=self._cache_size)(
            self._read_row_group_uncached
        )

        max_rg = (max(meta.row_group(i).num_rows for i in range(self.num_row_groups))
                  if self.num_row_groups else 0)
        mode = "EAGER (RAM-resident)" if self.eager_load else "LAZY (parquet)"
        logger.info(
            "Opened parquet %s (rows=%d, row_groups=%d, max_rg_rows=%d) — "
            "mode=%s, row-group cache size=%d.",
            self.parquet_path, self.num_rows, self.num_row_groups, max_rg,
            mode, self._cache_size,
        )
        if self.parquet_path != self.source_parquet_path:
            logger.info(
                "Note: served from rechunked cache. Original was %s.",
                self.source_parquet_path,
            )

    def _materialize_eager(self) -> None:
        """Stream-decode the parquet into flat numpy buffers, one per expert.

        We deliberately produce *one* contiguous ``np.float32`` array per
        expert rather than a list of per-clip arrays. Three reasons:

        1. **Fork+COW friendliness.** A single big numpy allocation
           has its refcount header in a separate page from the data.
           DataLoader workers fork() and read from the data pages
           without ever dirtying them, so all workers share the same
           physical RAM.
        2. **No Python object overhead.** Each clip is just an
           ``[offsets[i]:offsets[i+1]]`` view, not a wrapped Python
           object, so we pay 4 bytes/frame instead of 4 + ~30 bytes
           of CPython float overhead.
        3. **Cache-friendly slicing.** Slicing into a contiguous buffer
           is a single ``memcpy`` rather than a scatter from many
           small allocations.

        Decoding is done one row-group at a time so the *peak* RAM
        during construction is bounded (one row group's Arrow buffers
        + the partial output buffer), not the full table size.
        """
        feat_cols = [FEATURE_COLUMNS[n] for n in MODEL_NAMES]
        pf = pq.ParquetFile(self.parquet_path)

        # First pass: per-clip lengths, so we can pre-allocate the flat buffer.
        lengths: Dict[str, np.ndarray] = {n: np.empty(self.num_rows, dtype=np.int64)
                                          for n in MODEL_NAMES}
        dims: Dict[str, int] = {}
        clip_idx = 0
        for batch in pf.iter_batches(columns=feat_cols, batch_size=256):
            n_in_batch = batch.num_rows
            for name in MODEL_NAMES:
                col = batch.column(FEATURE_COLUMNS[name])
                for i in range(n_in_batch):
                    arr = col[i].as_py()
                    if not arr:
                        T = 0
                        D = dims.get(name, 0)
                    else:
                        T = min(len(arr), self.max_seq_len)
                        D = len(arr[0])
                    if name not in dims:
                        dims[name] = D
                    elif D != 0 and D != dims[name]:
                        raise ValueError(
                            f"Inconsistent feature dim for '{name}': "
                            f"row {clip_idx + i} has D={D}, expected D={dims[name]}."
                        )
                    lengths[name][clip_idx + i] = T
            clip_idx += n_in_batch

        offsets: Dict[str, np.ndarray] = {}
        buffers: Dict[str, np.ndarray] = {}
        total_bytes = 0
        for name in MODEL_NAMES:
            offs = np.empty(self.num_rows + 1, dtype=np.int64)
            offs[0] = 0
            np.cumsum(lengths[name], out=offs[1:])
            offsets[name] = offs
            total_T = int(offs[-1])
            D = dims[name]
            buffers[name] = np.empty((total_T, D), dtype=np.float32)
            total_bytes += buffers[name].nbytes

        logger.info(
            "Eager-load: allocating %.2f GB across %d experts (max_seq_len=%d).",
            total_bytes / 1e9, len(MODEL_NAMES), self.max_seq_len,
        )

        # Second pass: fill the flat buffers.
        clip_idx = 0
        for batch in pf.iter_batches(columns=feat_cols, batch_size=256):
            n_in_batch = batch.num_rows
            for name in MODEL_NAMES:
                col = batch.column(FEATURE_COLUMNS[name])
                buf = buffers[name]
                offs = offsets[name]
                for i in range(n_in_batch):
                    raw = col[i].as_py()
                    T = int(offs[clip_idx + i + 1] - offs[clip_idx + i])
                    if T == 0:
                        continue
                    arr = np.asarray(raw[:T], dtype=np.float32)
                    if arr.ndim != 2:
                        raise ValueError(
                            f"Expected 2D (T, D) embedding for '{name}', "
                            f"got shape {arr.shape}"
                        )
                    buf[offs[clip_idx + i]:offs[clip_idx + i + 1]] = arr
            clip_idx += n_in_batch

        self._eager_buffers = buffers
        self._eager_offsets = offsets

        # Sanity logging
        total_frames = {n: int(offsets[n][-1]) for n in MODEL_NAMES}
        logger.info(
            "Eager-load complete: %s frames; resident size %.2f GB.",
            ", ".join(f"{n}={total_frames[n]}" for n in MODEL_NAMES),
            total_bytes / 1e9,
        )

    @property
    def pq_file(self) -> pq.ParquetFile:
        """Lazy ParquetFile handle; opened in the worker process."""
        if self._pq_file is None:
            self._pq_file = pq.ParquetFile(self.parquet_path)
        return self._pq_file

    def _read_row_group_uncached(self, rg_idx: int) -> Dict[str, "pa.ChunkedArray"]:
        """Read the three feature columns of a single row group as Arrow arrays.

        We deliberately avoid ``.to_pylist()`` here — that would decode
        every clip in the row group into Python list objects up-front,
        which inflates memory by ~7-9× compared to the parquet on-disk
        size (Python float overhead). Instead we keep the Arrow buffers
        and decode one clip at a time inside :meth:`__getitem__` via
        ``Scalar.as_py()``. Combined with a small ``row_group_size``
        in the cached parquet, this caps peak RAM to roughly one row
        group's Arrow footprint per cache slot.
        """
        cols = [FEATURE_COLUMNS[n] for n in MODEL_NAMES]
        table = self.pq_file.read_row_group(rg_idx, columns=cols)
        return {n: table[FEATURE_COLUMNS[n]] for n in MODEL_NAMES}

    def _locate(self, idx: int) -> tuple[int, int]:
        """Find ``(row_group_index, row_in_group)`` for global ``idx``."""
        if idx < 0 or idx >= self.num_rows:
            raise IndexError(idx)
        lo, hi = 0, self.num_row_groups
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.row_group_offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid
        return lo, idx - self.row_group_offsets[lo]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pq_file"] = None
        state.pop("_read_row_group", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        cache_size = self.__dict__.get("_cache_size", DEFAULT_ROW_GROUP_CACHE_SIZE)
        self._read_row_group = lru_cache(maxsize=cache_size)(
            self._read_row_group_uncached
        )

    def __len__(self) -> int:
        return self.num_rows

    def _get_eager(self, idx: int) -> dict:
        """O(1) slice into pre-loaded flat buffers."""
        hidden_states: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, int] = {}
        for name in MODEL_NAMES:
            offs = self._eager_offsets[name]
            start = int(offs[idx])
            end = int(offs[idx + 1])
            # ``torch.from_numpy`` shares memory with the numpy buffer.
            # The downstream ``pad_sequence`` in ``collate_fn`` will copy
            # this view into a fresh padded tensor anyway, so we don't
            # need a defensive copy here.
            emb = self._eager_buffers[name][start:end]
            hidden_states[name] = torch.from_numpy(emb)
            seq_lens[name] = end - start
        return {
            "hidden_states": hidden_states,
            "seq_lens": seq_lens,
            "wer_scores": torch.from_numpy(self.wer_matrix[idx]),
            "sample_id": idx,
        }

    def __getitem__(self, idx: int) -> dict:
        if self.eager_load:
            return self._get_eager(idx)

        rg_idx, row = self._locate(idx)
        rg = self._read_row_group(rg_idx)

        hidden_states: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, int] = {}
        for name in MODEL_NAMES:
            # rg[name] is an Arrow ChunkedArray; indexing returns a
            # ListScalar, and as_py() decodes only that single clip's
            # nested list — not the whole row group.
            emb = np.asarray(rg[name][row].as_py(), dtype=np.float32)
            if emb.ndim != 2:
                raise ValueError(
                    f"Expected 2D (T, D) embedding for '{name}', got shape {emb.shape}"
                )
            if emb.shape[0] > self.max_seq_len:
                emb = emb[: self.max_seq_len]
            hidden_states[name] = torch.from_numpy(emb)
            seq_lens[name] = emb.shape[0]

        wer_scores = torch.from_numpy(self.wer_matrix[idx])

        return {
            "hidden_states": hidden_states,
            "seq_lens": seq_lens,
            "wer_scores": wer_scores,
            "sample_id": idx,
        }


def collate_fn(batch: List[dict]) -> dict:
    """Pad each model's frame sequence independently and build attention masks.

    Args:
        batch: List of dicts from :meth:`ASRFeatureDataset.__getitem__`.

    Returns:
        Dict with:
            hidden_states: Dict[model_name, (B, T_max_k, D_k)].
            attention_masks: Dict[model_name, (B, T_max_k) bool], True where valid.
            wer_scores: (B, n_models) float tensor in :data:`MODEL_NAMES` order.
            sample_ids: List of sample identifiers.
    """
    batch_size = len(batch)
    padded_hidden_states: Dict[str, torch.Tensor] = {}
    attention_masks: Dict[str, torch.Tensor] = {}

    for name in MODEL_NAMES:
        sequences = [b["hidden_states"][name] for b in batch]
        lengths = torch.tensor([b["seq_lens"][name] for b in batch])

        padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        padded_hidden_states[name] = padded

        max_len = padded.shape[1]
        mask = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1) < lengths.unsqueeze(1)
        attention_masks[name] = mask

    wer_scores = torch.stack([b["wer_scores"] for b in batch])
    sample_ids = [b["sample_id"] for b in batch]

    return {
        "hidden_states": padded_hidden_states,
        "attention_masks": attention_masks,
        "wer_scores": wer_scores,
        "sample_ids": sample_ids,
    }
