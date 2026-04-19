"""PyTorch Lightning training script for the ASR Model Selector.

Loss:
    Primary: Weighted WER = sum(p_k * wer_k) per sample, averaged over batch.
        Differentiable through softmax; pushes probability toward the model
        with lowest WER for each clip.
    Auxiliary: Cross-entropy on hard best-model label (with label smoothing).
        Aids early convergence by providing a stronger gradient signal.

Usage:
    python -m src.experiments.train \
        --metadata_path data/unified/metadata.json \
        --batch_size 4 \
        --max_epochs 50
"""

import argparse
import json
import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from src.data_old.dataset import ASRSelectorDataset, collate_fn, MODEL_NAMES
from src.models.selector import ASRModelSelector

logger = logging.getLogger(__name__)

MODEL_DIMS = {
    "hubert": 1024,
    "whisper": 512,
    "wav2vec2": 768,
}


class ASRSelectorModule(pl.LightningModule):
    """Lightning wrapper for the ASR Model Selector.

    Losses:
        - Primary: Weighted WER = sum(p_k * wer_k).
        - Optional auxiliary: Cross-entropy on hard best-model labels.
    """

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
        aux_ce_weight: float = 0.3,
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
        self.aux_ce_weight = aux_ce_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        hidden_states: dict[str, torch.Tensor],
        attention_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Args:
            hidden_states: Dict of model_name → (B, T_k, D_k).
            attention_masks: Dict of model_name → (B, T_k) bool.

        Returns:
            probs: (B, n_models) model selection probabilities.
        """
        return self.model(hidden_states, attention_masks)

    def _compute_loss(
        self,
        probs: torch.Tensor,
        wer_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute primary + auxiliary loss and all evaluation metrics.

        Args:
            probs: Predicted model probabilities of shape (B, n_models).
            wer_scores: Precomputed WER per model of shape (B, n_models).

        Returns:
            total_loss: Scalar loss tensor.
            metrics: Dict of named metric tensors.
        """
        weighted_wer = (probs * wer_scores).sum(dim=-1)
        primary_loss = weighted_wer.mean()

        best_model_idx = wer_scores.argmin(dim=-1)
        aux_loss = F.cross_entropy(
            torch.log(probs + 1e-8),
            best_model_idx,
            label_smoothing=self.label_smoothing,
        )

        total_loss = primary_loss + self.aux_ce_weight * aux_loss

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
            "aux_loss": aux_loss,
            "total_loss": total_loss,
            "selected_wer": selected_wer.mean(),
            "oracle_wer": oracle_wer.mean(),
            "wer_gap": (selected_wer - oracle_wer).mean(),
            "selection_accuracy": selection_accuracy,
            **{f"freq_{k}": v for k, v in selection_freq.items()},
        }

        return total_loss, metrics

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Args:
            batch: Collated batch dict.
            batch_idx: Batch index (unused).

        Returns:
            Scalar training loss.
        """
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
        """Args:
            batch: Collated batch dict.
            batch_idx: Batch index (unused).

        Returns:
            Scalar validation loss.
        """
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
        """Args:
            batch: Collated batch dict.
            batch_idx: Batch index (unused).

        Returns:
            Scalar test loss.
        """
        probs = self(batch["hidden_states"], batch["attention_masks"])
        loss, metrics = self._compute_loss(probs, batch["wer_scores"])

        for k, v in metrics.items():
            self.log(f"test/{k}", v, on_epoch=True)

        return loss

    def configure_optimizers(self) -> dict:
        """Configure AdamW optimizer with linear warmup + cosine annealing.

        Returns:
            Dict with optimizer and lr_scheduler config for Lightning.
        """
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


def main():
    """Entry point for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train ASR Model Selector")

    # Data
    parser.add_argument("--metadata_path", type=str, default="data/unified/metadata.json")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--stage1_layers", type=int, default=2)
    parser.add_argument("--stage2_layers", type=int, default=1)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--no_cross_attention", action="store_true")
    parser.add_argument(
        "--separate_stage1",
        action="store_true",
        help="Use separate Stage 1 weights per model (default: shared)",
    )

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--aux_ce_weight", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--experiment_name", type=str, default="asr_selector")
    parser.add_argument("--log_dir", type=str, default="logs")

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    full_dataset = ASRSelectorDataset(
        metadata_path=args.metadata_path,
        max_seq_len=args.max_seq_len,
    )

    n_total = len(full_dataset)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val

    logger.info("Dataset splits: train=%d, val=%d, test=%d", n_train, n_val, n_test)

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=True,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.max_epochs

    lightning_model = ASRSelectorModule(
        d_model=args.d_model,
        n_heads=args.n_heads,
        stage1_layers=args.stage1_layers,
        stage2_layers=args.stage2_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        use_cross_attention_bridge=not args.no_cross_attention,
        share_stage1_weights=not args.separate_stage1,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        aux_ce_weight=args.aux_ce_weight,
        label_smoothing=args.label_smoothing,
    )

    param_counts = lightning_model.model.count_parameters()
    logger.info("=== Model Parameter Counts ===")
    for k, v in param_counts.items():
        logger.info("  %25s: %10d", k, v)

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.log_dir, args.experiment_name, "checkpoints"),
            filename="best-{epoch:02d}-{val/selected_wer:.4f}",
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
    ]

    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=tb_logger,
        gradient_clip_val=args.gradient_clip_val,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=10,
        deterministic=True,
    )

    logger.info("Starting training...")
    trainer.fit(lightning_model, train_loader, val_loader)

    logger.info("Running test evaluation...")
    trainer.test(lightning_model, test_loader, ckpt_path="best")

    results_dir = os.path.join(args.log_dir, args.experiment_name)
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info("Training complete. Logs saved to %s", results_dir)


if __name__ == "__main__":
    main()
