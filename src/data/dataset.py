"""Dataset for the combined ASR features parquet.

Expects a single parquet produced by `src.data.preprocess`, containing:
    - ground_truth: str
    - {whisper,hubert,w2v2}_features: list[list[float]]  (variable-length [T, D])
    - {whisper,hubert,w2v2}_wer: float
"""

import logging
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
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
    """Variable-length frame-level embeddings from three ASR encoders + per-model WER."""

    def __init__(self, parquet_path: str, max_seq_len: int = 2000):
        """Args:
            parquet_path: Path to the unified `combined_features.parquet`.
            max_seq_len: Frame sequences longer than this are truncated.
        """
        self.max_seq_len = max_seq_len
        self.ds = load_dataset("parquet", data_files=parquet_path, split="train")
        logger.info("Loaded %d samples from %s", len(self.ds), parquet_path)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        """Returns a dict with per-model hidden states, seq lengths, and WER scores."""
        row = self.ds[idx]

        hidden_states: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, int] = {}
        for name in MODEL_NAMES:
            emb = np.asarray(row[FEATURE_COLUMNS[name]], dtype=np.float32)
            if emb.ndim != 2:
                raise ValueError(
                    f"Expected 2D (T, D) embedding for '{name}', got shape {emb.shape}"
                )
            if emb.shape[0] > self.max_seq_len:
                emb = emb[: self.max_seq_len]
            hidden_states[name] = torch.from_numpy(emb)
            seq_lens[name] = emb.shape[0]

        wer_scores = torch.tensor(
            [float(row[WER_COLUMNS[n]]) for n in MODEL_NAMES],
            dtype=torch.float32,
        )

        return {
            "hidden_states": hidden_states,
            "seq_lens": seq_lens,
            "wer_scores": wer_scores,
            "sample_id": idx,
        }


def collate_fn(batch: List[dict]) -> dict:
    """Pad each model's frame sequence independently and build attention masks.

    Args:
        batch: List of dicts from `ASRFeatureDataset.__getitem__`.

    Returns:
        Dict with:
            hidden_states: Dict[model_name, (B, T_max_k, D_k)].
            attention_masks: Dict[model_name, (B, T_max_k) bool], True where valid.
            wer_scores: (B, n_models) float tensor, in MODEL_NAMES order.
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
