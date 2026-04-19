"""PyTorch Dataset for the ASR model selector.

Loads precomputed encoder hidden states and WER scores from disk.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


MODEL_NAMES = ["hubert", "whisper", "wav2vec2"]


class ASRSelectorDataset(Dataset):
    """Dataset of precomputed ASR encoder hidden states and WER scores.

    Each sample contains:
        - Encoder hidden states per model (variable-length tensors).
        - WER scores per model of shape (3,).
        - Metadata (sample_id).
    """

    def __init__(self, metadata_path: str, max_seq_len: int = 2000):
        """Args:
            metadata_path: Path to the metadata JSON file produced by preprocess.py.
            max_seq_len: Maximum sequence length; longer sequences are truncated.
        """
        with open(metadata_path, "r") as f:
            self.records = json.load(f)
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        """Args:
            idx: Sample index.

        Returns:
            Dict with keys:
                hidden_states: Dict of model_name → (T_k, D_k) tensor.
                seq_lens: Dict of model_name → int.
                wer_scores: (3,) float tensor.
                sample_id: Sample identifier.
        """
        record = self.records[idx]

        hidden_states = {}
        seq_lens = {}
        for model_name in MODEL_NAMES:
            emb = np.load(record[f"{model_name}_emb_path"])  # (T, D)
            if emb.shape[0] > self.max_seq_len:
                emb = emb[:self.max_seq_len]
            hidden_states[model_name] = torch.from_numpy(emb)
            seq_lens[model_name] = emb.shape[0]

        wer_scores = torch.tensor(
            [record[f"{model_name}_wer"] for model_name in MODEL_NAMES],
            dtype=torch.float32,
        )

        return {
            "hidden_states": hidden_states,
            "seq_lens": seq_lens,
            "wer_scores": wer_scores,
            "sample_id": record["sample_id"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate a list of samples into a batched dict.

    Pads each model's hidden states independently, creates boolean attention
    masks, and stacks WER scores.

    Args:
        batch: List of dicts returned by ASRSelectorDataset.__getitem__.

    Returns:
        Dict with keys:
            hidden_states: Dict of model_name → (B, T_max_k, D_k) tensor.
            attention_masks: Dict of model_name → (B, T_max_k) bool tensor,
                True where valid.
            wer_scores: (B, 3) float tensor.
            sample_ids: List of sample identifiers.
    """
    batch_size = len(batch)

    padded_hidden_states = {}
    attention_masks = {}

    for model_name in MODEL_NAMES:
        sequences = [sample["hidden_states"][model_name] for sample in batch]
        lengths = [sample["seq_lens"][model_name] for sample in batch]

        padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        padded_hidden_states[model_name] = padded

        max_len = padded.shape[1]
        mask = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1)
        mask = mask < torch.tensor(lengths).unsqueeze(1)
        attention_masks[model_name] = mask

    wer_scores = torch.stack([sample["wer_scores"] for sample in batch])
    sample_ids = [sample["sample_id"] for sample in batch]

    return {
        "hidden_states": padded_hidden_states,
        "attention_masks": attention_masks,
        "wer_scores": wer_scores,
        "sample_ids": sample_ids,
    }
