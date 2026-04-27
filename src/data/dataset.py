"""Dataset for the combined ASR features parquet.

Expects a single parquet produced by :mod:`src.data.preprocess`, containing:
    - ground_truth: str
    - {whisper, hubert, w2v2}_features: list[list[float]]  (variable-length [T, D])
    - {whisper, hubert, w2v2}_wer: float

Memory model:
    The combined parquet contains three frame-level feature columns whose
    total in-memory size for AMI/VoxPopuli is tens of gigabytes. Loading
    the whole table eagerly (e.g. via ``pq.read_table`` or
    ``datasets.load_dataset``) plus a multi-worker DataLoader fork
    triggers a copy-on-write blowup that exceeds Colab's RAM. To avoid
    that, this dataset opens the parquet lazily as a
    :class:`pq.ParquetFile`, reads only the row-group containing the
    requested clip on demand, and keeps a one-row-group LRU cache so
    that consecutive accesses within the same row group amortize the
    decode cost. Scalar columns (``ground_truth`` and the three
    ``*_wer`` columns) are eagerly loaded into NumPy arrays since they
    are tiny.
"""

import logging
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

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


class ASRFeatureDataset(Dataset):
    """Variable-length frame-level embeddings + per-model WER, lazily loaded."""

    def __init__(self, parquet_path: str, max_seq_len: int = 2000, cache_size: int = 2):
        """Args:
            parquet_path: Path to the unified combined parquet.
            max_seq_len: Frame sequences longer than this are truncated.
            cache_size: Number of row groups to keep decoded in memory.
                One is enough for sequential access; two helps when the
                DataLoader prefetches across a row-group boundary.
        """
        self.parquet_path = parquet_path
        self.max_seq_len = max_seq_len

        self._pq_file: pq.ParquetFile | None = None
        meta = pq.read_metadata(parquet_path)
        self.num_rows: int = meta.num_rows
        self.num_row_groups: int = meta.num_row_groups
        self.row_group_offsets: List[int] = []
        offset = 0
        for rg in range(self.num_row_groups):
            self.row_group_offsets.append(offset)
            offset += meta.row_group(rg).num_rows
        self.row_group_offsets.append(offset)

        wer_cols = [WER_COLUMNS[n] for n in MODEL_NAMES]
        wer_table = pq.read_table(parquet_path, columns=wer_cols)
        self.wer_matrix = np.stack(
            [wer_table[c].to_numpy().astype(np.float32) for c in wer_cols],
            axis=-1,
        )

        self._read_row_group = lru_cache(maxsize=cache_size)(self._read_row_group_uncached)

        logger.info(
            "Opened parquet %s (rows=%d, row_groups=%d) — features stay on disk, "
            "row-group cache size=%d.",
            parquet_path, self.num_rows, self.num_row_groups, cache_size,
        )

    @property
    def pq_file(self) -> pq.ParquetFile:
        """Lazy ParquetFile handle; opened in the worker process."""
        if self._pq_file is None:
            self._pq_file = pq.ParquetFile(self.parquet_path)
        return self._pq_file

    def _read_row_group_uncached(self, rg_idx: int) -> Dict[str, list]:
        """Read the three feature columns of a single row group as Python lists.

        Returns a dict keyed by model name, where each value is a list
        of frame-level embedding lists (one entry per row in the group).
        """
        cols = [FEATURE_COLUMNS[n] for n in MODEL_NAMES]
        table = self.pq_file.read_row_group(rg_idx, columns=cols)
        return {
            n: table[FEATURE_COLUMNS[n]].to_pylist()
            for n in MODEL_NAMES
        }

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
        self._read_row_group = lru_cache(maxsize=2)(self._read_row_group_uncached)

    def __len__(self) -> int:
        return self.num_rows

    def __getitem__(self, idx: int) -> dict:
        rg_idx, row = self._locate(idx)
        rg = self._read_row_group(rg_idx)

        hidden_states: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, int] = {}
        for name in MODEL_NAMES:
            emb = np.asarray(rg[name][row], dtype=np.float32)
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
