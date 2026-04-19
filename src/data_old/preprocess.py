"""Preprocessing script for the ASR model selector.

Loads HuggingFace datasets with encoder hidden states and transcriptions,
computes per-model WER, and saves a unified dataset with metadata JSON and
per-sample .npy embeddings.

Usage:
    python -m src.data.preprocess \
        --data_root data/processed \
        --output_dir data/unified

Important:
    Edit MODEL_CONFIGS and GROUND_TRUTH_COL to match your column names.
"""

import argparse
import json
import logging
import os

import numpy as np
from datasets import load_from_disk
from jiwer import wer as compute_wer

logger = logging.getLogger(__name__)

# Base model configs: edit to match your saved dataset column names.
MODEL_CONFIGS = {
    "hubert": {
        "dir_suffix": "hubert-large-ls960-ft",
        "hidden_states_col": "encoder_state",
        "transcription_col": "transcription",
    },
    "whisper": {
        "dir_suffix": "whisper-base",
        "hidden_states_col": "encoder_state",
        "transcription_col": "transcription",
    },
    "wav2vec2": {
        "dir_suffix": "wav2vec2-base-960h",
        "hidden_states_col": "encoder_state",
        "transcription_col": "transcription",
    },
}

GROUND_TRUTH_COL = "original_text"


def compute_sample_wer(prediction: str, reference: str) -> float:
    """Compute WER for a single sample.

    Args:
        prediction: Model's predicted transcription.
        reference: Ground truth transcription.

    Returns:
        WER value in [0, 1]. Returns 1.0 if either string is empty or on error.
    """
    if not reference or not reference.strip():
        return 1.0
    if not prediction or not prediction.strip():
        return 1.0
    try:
        return compute_wer(reference.lower().strip(), prediction.lower().strip())
    except Exception:
        return 1.0


def load_and_align_datasets(data_root: str) -> tuple[dict, int]:
    """Load all three model datasets and verify they are aligned by index.

    Args:
        data_root: Root directory containing model-specific feature datasets.

    Returns:
        datasets: Dict mapping model name to HuggingFace Dataset.
        n_samples: Number of samples (same across all datasets).

    Raises:
        AssertionError: If dataset lengths do not match.
    """
    datasets = {}
    for model_name, config in MODEL_CONFIGS.items():
        path = os.path.join(
            data_root, f"voxpopuli_asr_features_poc_{config['dir_suffix']}"
        )
        logger.info("Loading %s from %s", model_name, path)
        ds = load_from_disk(path)
        datasets[model_name] = ds
        logger.info("  → %d samples, columns: %s", len(ds), ds.column_names)

    lengths = {k: len(v) for k, v in datasets.items()}
    assert len(set(lengths.values())) == 1, f"Dataset length mismatch: {lengths}"
    n_samples = list(lengths.values())[0]
    logger.info("All datasets have %d samples.", n_samples)

    return datasets, n_samples


def build_unified_dataset(
    datasets: dict, n_samples: int, output_dir: str
) -> list[dict]:
    """Build a unified dataset from the aligned per-model datasets.

    Saves encoder hidden states as .npy files (memory-efficient for large
    tensors) and writes a metadata.json index file.

    Args:
        datasets: Dict mapping model name to HuggingFace Dataset.
        n_samples: Total number of samples.
        output_dir: Directory to write embeddings/ and metadata.json.

    Returns:
        records: List of per-sample metadata dicts.
    """
    os.makedirs(output_dir, exist_ok=True)
    embeddings_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    model_names = list(MODEL_CONFIGS.keys())
    first_model_ds = datasets[model_names[0]]

    records = []
    wer_stats = {m: [] for m in model_names}

    for i in range(n_samples):
        record: dict = {
            "sample_id": i,
            "ground_truth": first_model_ds[i][GROUND_TRUTH_COL],
        }

        for model_name in model_names:
            config = MODEL_CONFIGS[model_name]
            sample = datasets[model_name][i]

            hidden_states = np.array(
                sample[config["hidden_states_col"]], dtype=np.float32
            )
            emb_path = os.path.join(embeddings_dir, f"{i}_{model_name}.npy")
            np.save(emb_path, hidden_states)
            record[f"{model_name}_emb_path"] = emb_path
            record[f"{model_name}_seq_len"] = hidden_states.shape[0]
            record[f"{model_name}_feat_dim"] = hidden_states.shape[1]

            prediction = sample[config["transcription_col"]]
            reference = record["ground_truth"]
            sample_wer = compute_sample_wer(prediction, reference)
            record[f"{model_name}_wer"] = sample_wer
            record[f"{model_name}_transcription"] = prediction
            wer_stats[model_name].append(sample_wer)

        records.append(record)

        if (i + 1) % 100 == 0:
            logger.info("Processed %d/%d samples", i + 1, n_samples)

    logger.info("=== Per-Model WER Statistics ===")
    for model_name in model_names:
        wers = wer_stats[model_name]
        best_count = sum(
            1 for j in range(n_samples)
            if wers[j] == min(wer_stats[m][j] for m in model_names)
        )
        logger.info(
            "%s: mean=%.4f, median=%.4f, std=%.4f, best_count=%d",
            model_name,
            np.mean(wers),
            np.median(wers),
            np.std(wers),
            best_count,
        )

    oracle_wers = [
        min(wer_stats[m][i] for m in model_names) for i in range(n_samples)
    ]
    logger.info("Oracle WER: %.4f", np.mean(oracle_wers))
    best_model = min((np.mean(wer_stats[m]), m) for m in model_names)
    logger.info("Best single model: %s (%.4f)", best_model[1], best_model[0])

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(records, f)

    logger.info("Saved %d records to %s", len(records), meta_path)
    return records


def main():
    """Entry point for the preprocessing script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Preprocess ASR features for selector model"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/processed",
        help="Root directory containing model-specific feature datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/unified",
        help="Output directory for unified dataset",
    )
    args = parser.parse_args()

    datasets, n_samples = load_and_align_datasets(args.data_root)
    build_unified_dataset(datasets, n_samples, args.output_dir)


if __name__ == "__main__":
    main()
