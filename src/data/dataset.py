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
        self.eager_load = bool(eager_load)
        
        if self.eager_load:
            self.parquet_path = self.source_parquet_path
        else:
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

        if self.eager_load:
            logger.info("Eager loading full parquet into flat float16 numpy buffers...")
            table = pq.read_table(self.parquet_path, columns=[FEATURE_COLUMNS[n] for n in MODEL_NAMES])
            
            self._flat_buffers: Dict[str, np.ndarray] = {}
            self._frame_offsets: Dict[str, np.ndarray] = {}
            self._model_dims: Dict[str, int] = {}
            
            for name in MODEL_NAMES:
                col_name = FEATURE_COLUMNS[name]
                chunked_arr = table[col_name].combine_chunks()
                
                # Extract PyArrow buffers
                outer_offsets = chunked_arr.offsets.to_numpy()
                inner_arr = chunked_arr.values
                inner_offsets = inner_arr.offsets.to_numpy()
                
                # Assume all frames have the same D
                D = inner_offsets[1] - inner_offsets[0]
                
                # Get raw floats and cast to float16 to save memory
                flat_floats = inner_arr.values.to_numpy(zero_copy_only=False).astype(np.float16)
                
                self._flat_buffers[name] = flat_floats
                self._frame_offsets[name] = outer_offsets
                self._model_dims[name] = D
                
            logger.info("Eager load complete. Flat buffer size per model: ~%.1f GB", 
                        len(flat_floats) * 2 / (1024**3))
            self._pq_file = None
        else:
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
        if not self.eager_load:
            state["_pq_file"] = None
            state.pop("_read_row_group", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not self.eager_load:
            cache_size = self.__dict__.get("_cache_size", DEFAULT_ROW_GROUP_CACHE_SIZE)
            self._read_row_group = lru_cache(maxsize=cache_size)(
                self._read_row_group_uncached
            )

    def __len__(self) -> int:
        return self.num_rows

    def __getitem__(self, idx: int) -> dict:
        hidden_states: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, int] = {}
        
        if self.eager_load:
            for name in MODEL_NAMES:
                start_frame = self._frame_offsets[name][idx]
                end_frame = self._frame_offsets[name][idx+1]
                D = self._model_dims[name]
                
                emb = self._flat_buffers[name][start_frame * D : end_frame * D].reshape(-1, D)
                # Cast back to float32 for PyTorch training stability
                emb = emb.astype(np.float32)
                
                if emb.shape[0] > self.max_seq_len:
                    emb = emb[: self.max_seq_len]
                
                hidden_states[name] = torch.from_numpy(emb)
                seq_lens[name] = emb.shape[0]
        else:
            rg_idx, row = self._locate(idx)
            rg = self._read_row_group(rg_idx)

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
