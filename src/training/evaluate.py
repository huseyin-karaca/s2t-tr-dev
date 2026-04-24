"""Evaluate a trained ASR Model Selector checkpoint on the test split.

Usage:
    python -m src.training.evaluate \
        --checkpoint logs/ami_full/checkpoints/best-epoch=07.ckpt \
        --parquet-path data/processed/edinburghcstr_ami/combined_features.parquet

    # Evaluate on an arbitrary split ratio (must match training's seed/ratios to
    # reproduce the exact test set):
    python -m src.training.evaluate \
        --checkpoint <path> --split test --seed 42 \
        --train-ratio 0.8 --val-ratio 0.1
"""

import json
import logging
from typing import Optional

import pytorch_lightning as pl
import torch
import typer
from torch.utils.data import DataLoader, random_split

# We only load checkpoints we produced ourselves, so the weights_only security
# hardening from PyTorch 2.6 is unnecessary here. Force the old behavior.
_orig_torch_load = torch.load
def _torch_load_full(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_full

from src.data.dataset import ASRFeatureDataset, collate_fn
from src.training.train import ASRSelectorModule

logger = logging.getLogger(__name__)
app = typer.Typer(help="Evaluate a trained ASR Model Selector checkpoint.")


def _select_split(full_ds, split: str, train_ratio: float, val_ratio: float, seed: int):
    """Return the requested split using the same logic as `train.py`."""
    n_total = len(full_ds)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
    return {"train": train_ds, "val": val_ds, "test": test_ds}[split]


@app.command()
def evaluate(
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Path to .ckpt file."),
    parquet_path: str = typer.Option(
        "data/processed/edinburghcstr_ami/combined_features.parquet",
        "--parquet-path", "-p",
    ),
    split: str = typer.Option("test", "--split", help="Which split: train/val/test."),
    train_ratio: float = typer.Option(0.8, "--train-ratio"),
    val_ratio: float = typer.Option(0.1, "--val-ratio"),
    max_seq_len: int = typer.Option(2000, "--max-seq-len"),
    batch_size: int = typer.Option(8, "--batch-size", "-b"),
    num_workers: int = typer.Option(4, "--num-workers"),
    seed: int = typer.Option(42, "--seed"),
    save_json: Optional[str] = typer.Option(
        None, "--save-json",
        help="Optional path to dump results as JSON.",
    ),
):
    """Load a checkpoint and compute metrics on the chosen split."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    pl.seed_everything(seed)

    logger.info("Loading checkpoint: %s", checkpoint)
    model = ASRSelectorModule.load_from_checkpoint(checkpoint)
    model.eval()

    full_ds = ASRFeatureDataset(parquet_path=parquet_path, max_seq_len=max_seq_len)
    split_ds = _select_split(full_ds, split, train_ratio, val_ratio, seed)
    logger.info("Evaluating on split '%s' (%d samples)", split, len(split_ds))

    loader = DataLoader(
        split_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )

    trainer = pl.Trainer(
        accelerator="auto", devices=1,
        precision="16-mixed", logger=False,
        enable_progress_bar=True,
    )
    results = trainer.test(model, loader)[0]

    clean = {k.replace("test/", ""): float(v) for k, v in results.items()}
    logger.info("=== Results (%s) ===", split)
    for k, v in clean.items():
        logger.info("  %-22s: %.4f", k, v)

    if save_json:
        with open(save_json, "w") as f:
            json.dump({"checkpoint": checkpoint, "split": split, **clean}, f, indent=2)
        logger.info("Saved results to %s", save_json)


if __name__ == "__main__":
    app()
