"""PyTorch Lightning training script for the ASR Model Selector.

Loss:
    Primary: Weighted WER = sum(p_k * wer_k) per sample, averaged over batch.
        Differentiable through softmax; pushes probability toward the model
        with the lowest WER for each clip.
    Auxiliary: Cross-entropy on the hard best-model label (with label smoothing).
        Aids early convergence by providing a stronger gradient signal.

Full training run:
    python -m src.training.train \
        --parquet-path data/processed/edinburghcstr_ami/combined_features.parquet \
        --batch-size 8 \
        --max-epochs 50 \
        --experiment-name ami_full

Sample run (quick smoke test to verify the pipeline end-to-end):
    python -m src.training.train \
        --parquet-path data/processed/edinburghcstr_ami/combined_features.parquet \
        --batch-size 2 \
        --max-epochs 1 \
        --limit-batches 4 \
        --num-workers 0 \
        --experiment-name smoke_test
"""

import json
import logging
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import typer

# We only load checkpoints we produced ourselves, so the weights_only security
# hardening from PyTorch 2.6 is unnecessary here. Force the old behavior.
_orig_torch_load = torch.load
def _torch_load_full(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_full
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from src.data.dataset import MODEL_NAMES, ASRFeatureDataset, collate_fn
from src.models.selector import ASRModelSelector

logger = logging.getLogger(__name__)

app = typer.Typer(help="Train the ASR Model Selector on unified parquet features.")

MODEL_DIMS = {
    "hubert":   1024,  # facebook/hubert-large-ls960-ft
    "whisper":  512,   # openai/whisper-base
    "wav2vec2": 1024,   # facebook/wav2vec2-base-960h
}


class ASRSelectorModule(pl.LightningModule):
    """Lightning wrapper for the ASR Model Selector."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        stage1_layers: int = 2,
        stage2_layers: int = 1,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        use_cross_attention_bridge: bool = True,
        share_stage1_weights: bool = True,
        max_seq_len: int = 2000,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 200,
        total_steps: int = 5000,
        primary_weight: float = 1.0,
        aux_ce_weight: float = 0.3,
        soft_ce_weight: float = 0.0,
        soft_ce_temperature: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        """Args:
            d_model: Shared transformer hidden dimension.
            n_heads: Number of attention heads.
            stage1_layers: Transformer layers in Stage 1.
            stage2_layers: Transformer layers in Stage 2.
            ffn_dim: Feed-forward dimension.
            dropout: Dropout probability.
            use_cross_attention_bridge: Whether to use cross-attention bridge.
            share_stage1_weights: Whether Stage 1 weights are shared across models.
            max_seq_len: Maximum sequence length.
            learning_rate: Peak learning rate for AdamW.
            weight_decay: Weight decay for AdamW.
            warmup_steps: Linear warmup steps.
            total_steps: Total training steps for cosine annealing.
            aux_ce_weight: Weight of the auxiliary cross-entropy loss.
            label_smoothing: Label smoothing for auxiliary CE loss.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = ASRModelSelector(
            model_dims=MODEL_DIMS,
            model_names=MODEL_NAMES,
            d_model=d_model,
            n_heads=n_heads,
            stage1_layers=stage1_layers,
            stage2_layers=stage2_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_cross_attention_bridge=use_cross_attention_bridge,
            share_stage1_weights=share_stage1_weights,
            max_seq_len=max_seq_len,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.primary_weight = primary_weight
        self.aux_ce_weight = aux_ce_weight
        self.soft_ce_weight = soft_ce_weight
        self.soft_ce_temperature = soft_ce_temperature
        self.label_smoothing = label_smoothing

    def forward(
        self,
        hidden_states: dict[str, torch.Tensor],
        attention_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.model(hidden_states, attention_masks)

    def _compute_loss(
        self,
        probs: torch.Tensor,
        wer_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute primary + auxiliary loss and evaluation metrics."""
        weighted_wer = (probs * wer_scores).sum(dim=-1)
        primary_loss = weighted_wer.mean()

        best_model_idx = wer_scores.argmin(dim=-1)
        hard_ce = F.cross_entropy(
            torch.log(probs + 1e-8),
            best_model_idx,
            label_smoothing=self.label_smoothing,
        )

        # Soft CE: target distribution is softmax(-wer / T) — carries WER magnitudes.
        soft_target = F.softmax(-wer_scores / self.soft_ce_temperature, dim=-1)
        soft_ce = -(soft_target * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        total_loss = (
            self.primary_weight * primary_loss
            + self.aux_ce_weight * hard_ce
            + self.soft_ce_weight * soft_ce
        )

        selected_model = probs.argmax(dim=-1)
        oracle_wer = wer_scores.min(dim=-1).values
        selected_wer = wer_scores.gather(1, selected_model.unsqueeze(1)).squeeze(1)
        selection_accuracy = (selected_model == best_model_idx).float().mean()

        selection_freq = {
            MODEL_NAMES[i]: (selected_model == i).float().mean()
            for i in range(len(MODEL_NAMES))
        }

        metrics = {
            "primary_loss": primary_loss,
            "hard_ce": hard_ce,
            "soft_ce": soft_ce,
            "total_loss": total_loss,
            "selected_wer": selected_wer.mean(),
            "oracle_wer": oracle_wer.mean(),
            "wer_gap": (selected_wer - oracle_wer).mean(),
            "selection_accuracy": selection_accuracy,
            **{f"freq_{k}": v for k, v in selection_freq.items()},
        }
        return total_loss, metrics

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        probs = self(batch["hidden_states"], batch["attention_masks"])
        loss, metrics = self._compute_loss(probs, batch["wer_scores"])

        prog_bar_keys = {"total_loss", "selected_wer", "selection_accuracy"}
        for k, v in metrics.items():
            self.log(
                f"train/{k}", v,
                on_step=True, on_epoch=True,
                prog_bar=(k in prog_bar_keys),
            )
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        probs = self(batch["hidden_states"], batch["attention_masks"])
        loss, metrics = self._compute_loss(probs, batch["wer_scores"])

        prog_bar_keys = {"total_loss", "selected_wer", "selection_accuracy"}
        for k, v in metrics.items():
            self.log(
                f"val/{k}", v,
                on_epoch=True,
                prog_bar=(k in prog_bar_keys),
            )
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        probs = self(batch["hidden_states"], batch["attention_masks"])
        loss, metrics = self._compute_loss(probs, batch["wer_scores"])
        for k, v in metrics.items():
            self.log(f"test/{k}", v, on_epoch=True)
        return loss

    def configure_optimizers(self) -> dict:
        """AdamW with linear warmup + cosine annealing."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


@app.command()
def train(
    # Data
    parquet_path: str = typer.Option(
        "data/processed/edinburghcstr_ami/combined_features.parquet",
        "--parquet-path", "-p",
        help="Path to the unified combined_features.parquet.",
    ),
    train_ratio: float = typer.Option(0.8, "--train-ratio"),
    val_ratio: float = typer.Option(0.1, "--val-ratio"),
    max_seq_len: int = typer.Option(2000, "--max-seq-len"),
    batch_size: int = typer.Option(4, "--batch-size", "-b"),
    num_workers: int = typer.Option(4, "--num-workers"),
    # Model
    d_model: int = typer.Option(256, "--d-model"),
    n_heads: int = typer.Option(4, "--n-heads"),
    stage1_layers: int = typer.Option(2, "--stage1-layers"),
    stage2_layers: int = typer.Option(1, "--stage2-layers"),
    ffn_dim: int = typer.Option(512, "--ffn-dim"),
    dropout: float = typer.Option(0.15, "--dropout"),
    no_cross_attention: bool = typer.Option(False, "--no-cross-attention"),
    separate_stage1: bool = typer.Option(
        False, "--separate-stage1",
        help="Use separate Stage 1 weights per model (default: shared).",
    ),
    # Training
    learning_rate: float = typer.Option(1e-4, "--learning-rate", "--lr"),
    weight_decay: float = typer.Option(1e-2, "--weight-decay"),
    warmup_steps: int = typer.Option(200, "--warmup-steps"),
    max_epochs: int = typer.Option(50, "--max-epochs"),
    primary_weight: float = typer.Option(
        1.0, "--primary-weight",
        help="Weight of weighted-WER primary loss (set 0 for classification-only).",
    ),
    aux_ce_weight: float = typer.Option(
        0.3, "--aux-ce-weight",
        help="Weight of hard-label CE (argmin WER) auxiliary loss.",
    ),
    soft_ce_weight: float = typer.Option(
        0.0, "--soft-ce-weight",
        help="Weight of soft CE against softmax(-wer/T) target distribution.",
    ),
    soft_ce_temperature: float = typer.Option(
        0.1, "--soft-ce-temperature",
        help="Temperature for soft CE target; lower = sharper (closer to hard CE).",
    ),
    label_smoothing: float = typer.Option(0.1, "--label-smoothing"),
    gradient_clip_val: float = typer.Option(1.0, "--gradient-clip-val"),
    seed: int = typer.Option(42, "--seed"),
    limit_batches: Optional[int] = typer.Option(
        None, "--limit-batches",
        help="Limit train/val/test batches per epoch for smoke tests.",
    ),
    # Logging
    experiment_name: str = typer.Option("asr_selector", "--experiment-name"),
    log_dir: str = typer.Option("logs", "--log-dir"),
    progress_bar_refresh: int = typer.Option(
        20, "--progress-bar-refresh",
        help="tqdm refresh interval in steps (higher = less notebook spam).",
    ),
):
    """Train the ASR Model Selector."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pl.seed_everything(seed)

    full_dataset = ASRFeatureDataset(
        parquet_path=parquet_path,
        max_seq_len=max_seq_len,
    )

    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    logger.info("Dataset splits: train=%d, val=%d, test=%d", n_train, n_val, n_test)

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * max_epochs

    lightning_model = ASRSelectorModule(
        d_model=d_model,
        n_heads=n_heads,
        stage1_layers=stage1_layers,
        stage2_layers=stage2_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
        use_cross_attention_bridge=not no_cross_attention,
        share_stage1_weights=not separate_stage1,
        max_seq_len=max_seq_len,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        primary_weight=primary_weight,
        aux_ce_weight=aux_ce_weight,
        soft_ce_weight=soft_ce_weight,
        soft_ce_temperature=soft_ce_temperature,
        label_smoothing=label_smoothing,
    )

    param_counts = lightning_model.model.count_parameters()
    logger.info("=== Model Parameter Counts ===")
    for k, v in param_counts.items():
        logger.info("  %25s: %10d", k, v)

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(log_dir, experiment_name, "checkpoints"),
            filename="best-{epoch:02d}",
            monitor="val/selected_wer",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/selected_wer",
            patience=10,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=progress_bar_refresh),
    ]

    tb_logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name)

    trainer_kwargs = dict(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=tb_logger,
        gradient_clip_val=gradient_clip_val,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=10,
        deterministic=True,
    )
    if limit_batches is not None:
        trainer_kwargs.update(
            limit_train_batches=limit_batches,
            limit_val_batches=limit_batches,
            limit_test_batches=limit_batches,
        )

    trainer = pl.Trainer(**trainer_kwargs)

    logger.info("Starting training...")
    trainer.fit(lightning_model, train_loader, val_loader)

    logger.info("Running test evaluation...")
    trainer.test(lightning_model, test_loader, ckpt_path="best")

    results_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    config_snapshot = {
        "parquet_path": parquet_path,
        "train_ratio": train_ratio, "val_ratio": val_ratio,
        "max_seq_len": max_seq_len, "batch_size": batch_size,
        "num_workers": num_workers,
        "d_model": d_model, "n_heads": n_heads,
        "stage1_layers": stage1_layers, "stage2_layers": stage2_layers,
        "ffn_dim": ffn_dim, "dropout": dropout,
        "use_cross_attention_bridge": not no_cross_attention,
        "share_stage1_weights": not separate_stage1,
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "warmup_steps": warmup_steps, "max_epochs": max_epochs,
        "primary_weight": primary_weight,
        "aux_ce_weight": aux_ce_weight,
        "soft_ce_weight": soft_ce_weight,
        "soft_ce_temperature": soft_ce_temperature,
        "label_smoothing": label_smoothing,
        "gradient_clip_val": gradient_clip_val, "seed": seed,
        "limit_batches": limit_batches,
        "experiment_name": experiment_name, "log_dir": log_dir,
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)
    logger.info("Training complete. Logs saved to %s", results_dir)


if __name__ == "__main__":
    app()
