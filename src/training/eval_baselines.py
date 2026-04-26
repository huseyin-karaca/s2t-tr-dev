"""Compute training-free baselines from a feature parquet.

This script evaluates routing baselines that do not require a learned
model: per-clip oracle, single-base-model performance, uniform random
selection, and weighted random selection (with weights derived from
training-set per-expert WER means). The same parquet schema as the
trained models is used (see :mod:`src.data.dataset`), so the script
works for both real-world and synthetic datasets.

The split logic mirrors :mod:`src.training.train` so the reported
test-set numbers correspond to the same samples as a training run with
matching ``--seed``, ``--train-ratio`` and ``--val-ratio``.

Usage:
    python -m src.training.eval_baselines \\
        --parquet-path data/processed/synthetic_regime_switch/combined_features.parquet \\
        --save-json logs/synthetic_regime_switch/baselines.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pyarrow.parquet as pq
import typer

from src.data.dataset import MODEL_NAMES, WER_COLUMNS

logger = logging.getLogger(__name__)
app = typer.Typer(help="Training-free routing baselines on a feature parquet.")


def _split_indices(
    n_total: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Reproduce the train/val/test split used by ``src.training.train``.

    The training script uses :func:`torch.utils.data.random_split` with a
    seeded ``torch.Generator``. We mirror the same logic here using a
    seeded NumPy RNG over the same total size and ratios. The exact
    permutation differs from torch's, so this function is intended for
    aggregate baselines rather than per-sample alignment.

    Args:
        n_total: Total number of samples in the dataset.
        train_ratio: Fraction of samples in the training split.
        val_ratio: Fraction of samples in the validation split.
        seed: RNG seed.

    Returns:
        Dict with ``"train"``, ``"val"``, and ``"test"`` index arrays.
    """
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    perm = np.random.default_rng(seed).permutation(n_total)
    return {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
    }


def _load_wer_matrix(parquet_path: str) -> np.ndarray:
    """Load the per-expert WER matrix from a feature parquet.

    Only the WER columns are read, which makes this fast even for
    parquets containing large frame-level feature columns.

    Args:
        parquet_path: Path to a parquet matching the schema used by
            :class:`src.data.dataset.ASRFeatureDataset`.

    Returns:
        Float array of shape ``(N, K)`` in :data:`MODEL_NAMES` order.
    """
    columns = [WER_COLUMNS[name] for name in MODEL_NAMES]
    table = pq.read_table(parquet_path, columns=columns)
    arr = np.stack([table[c].to_numpy() for c in columns], axis=-1).astype(np.float32)
    return arr


def _summarize(per_clip_wer: np.ndarray) -> Dict[str, float]:
    """Mean and standard error of a per-clip WER vector."""
    n = per_clip_wer.shape[0]
    sem = float(per_clip_wer.std(ddof=1) / np.sqrt(max(n, 1))) if n > 1 else 0.0
    return {
        "wer_mean": float(per_clip_wer.mean()),
        "wer_sem": sem,
        "n": int(n),
    }


def _compute_baselines(
    wer_train: np.ndarray,
    wer_eval: np.ndarray,
    seed: int,
) -> Dict[str, Dict]:
    """Compute all training-free baselines for one evaluation split.

    Args:
        wer_train: Training-split WER matrix of shape ``(N_train, K)``,
            used to derive priors for weighted random.
        wer_eval: Evaluation-split WER matrix of shape ``(N_eval, K)``.
        seed: RNG seed for the random and weighted-random sampling.

    Returns:
        Dict mapping baseline name to summary statistics.
    """
    n_eval, K = wer_eval.shape
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict] = {}

    for k, name in enumerate(MODEL_NAMES):
        out[f"single_{name}"] = _summarize(wer_eval[:, k])

    out["oracle"] = _summarize(wer_eval.min(axis=-1))

    uniform_idx = rng.integers(0, K, size=n_eval)
    out["random"] = _summarize(wer_eval[np.arange(n_eval), uniform_idx])

    train_mean = wer_train.mean(axis=0)
    inv = np.maximum(1.0 - train_mean, 1e-6)
    weights = inv / inv.sum()
    weighted_idx = rng.choice(K, size=n_eval, p=weights)
    out["weighted_random"] = _summarize(wer_eval[np.arange(n_eval), weighted_idx])
    out["weighted_random_weights"] = {
        name: float(w) for name, w in zip(MODEL_NAMES, weights)
    }

    n_eval_clips = n_eval
    out["selection_accuracy_random"] = {
        "value": float(np.mean(uniform_idx == wer_eval.argmin(axis=-1))),
        "n": int(n_eval_clips),
    }
    return out


@app.command()
def evaluate(
    parquet_path: str = typer.Option(..., "--parquet-path", "-p"),
    train_ratio: float = typer.Option(0.8, "--train-ratio"),
    val_ratio: float = typer.Option(0.1, "--val-ratio"),
    seed: int = typer.Option(42, "--seed"),
    split: str = typer.Option(
        "test", "--split",
        help="Which split to evaluate on: train, val, test, or all.",
    ),
    save_json: Optional[str] = typer.Option(
        None, "--save-json",
        help="Optional path to dump results as JSON.",
    ),
):
    """Compute random / weighted-random / per-base-model / oracle baselines."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    wer = _load_wer_matrix(parquet_path)
    n_total = wer.shape[0]
    splits = _split_indices(n_total, train_ratio, val_ratio, seed)
    logger.info(
        "Loaded WER matrix from %s — N=%d, K=%d, splits: train=%d val=%d test=%d",
        parquet_path, n_total, wer.shape[1],
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )

    if split == "all":
        target_splits = ["train", "val", "test"]
    elif split in splits:
        target_splits = [split]
    else:
        raise ValueError(f"Unknown split={split!r}; expected one of train/val/test/all")

    train_wer = wer[splits["train"]]
    results: Dict[str, Dict] = {}
    for s in target_splits:
        eval_wer = wer[splits[s]]
        results[s] = _compute_baselines(train_wer, eval_wer, seed=seed)

    for s, by_baseline in results.items():
        logger.info("=== Baselines on %s split ===", s)
        for name, stats in by_baseline.items():
            if "wer_mean" in stats:
                logger.info(
                    "  %-22s wer=%.4f ± %.4f (n=%d)",
                    name, stats["wer_mean"], stats["wer_sem"], stats["n"],
                )
            elif "value" in stats:
                logger.info(
                    "  %-22s value=%.4f (n=%d)",
                    name, stats["value"], stats["n"],
                )
            elif name == "weighted_random_weights":
                logger.info("  weighted_random_weights: %s", stats)

    if save_json:
        out = Path(save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "parquet_path": parquet_path,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "seed": seed,
                "results": results,
            }, f, indent=2)
        logger.info("Saved results to %s", save_json)


if __name__ == "__main__":
    app()
