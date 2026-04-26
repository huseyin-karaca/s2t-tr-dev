"""Post-hoc visualizations of trained ASR Model Selector checkpoints.

Two subcommands are provided:

* ``predictions`` runs one or more trained checkpoints over a chosen
  split of a feature parquet, then writes confusion-matrix, per-clip
  WER scatter, and selection-frequency figures alongside a JSON
  summary.

* ``curves`` reads the TensorBoard event files from one or more
  training runs and writes a multi-panel learning-curves figure that
  overlays the runs.

Both subcommands are agnostic to the dataset and to the architecture
(``hierarchical_transformer`` or ``mlp_pool``); they therefore apply
to the synthetic experiment, to AMI/VoxPopuli, or to any future
parquet/checkpoint that follows the same conventions.

Usage:
    python -m src.training.visualize predictions \\
        --parquet-path data/processed/synthetic_regime_switch/combined_features.parquet \\
        --ckpt proposed=logs/synthetic_regime_switch_hier/checkpoints/last.ckpt \\
        --ckpt mlp_pool=logs/synthetic_regime_switch_mlp_pool/checkpoints/last.ckpt \\
        --output-dir reports/figures/synthetic_regime_switch

    python -m src.training.visualize curves \\
        --logdir proposed=logs/synthetic_regime_switch_hier \\
        --logdir mlp_pool=logs/synthetic_regime_switch_mlp_pool \\
        --output-path reports/figures/synthetic_regime_switch/curves.pdf
"""

import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from torch.utils.data import DataLoader, random_split

# Match the security-hardening override in src.training.train so checkpoints
# we produced ourselves load without weights_only complaints.
_orig_torch_load = torch.load
def _torch_load_full(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_full

from src.data.dataset import MODEL_NAMES, ASRFeatureDataset, collate_fn
from src.training.train import ASRSelectorModule

logger = logging.getLogger(__name__)
app = typer.Typer(help="Post-hoc visualizations of trained ASR Model Selector checkpoints.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_named_options(values: List[str], flag_name: str) -> Dict[str, str]:
    """Parse a list of ``"name=path"`` strings into a dict.

    Args:
        values: Raw CLI inputs of the form ``"name=path"``.
        flag_name: The CLI flag name, used in error messages.

    Returns:
        Ordered dict mapping ``name`` to ``path``.
    """
    parsed: Dict[str, str] = {}
    for v in values:
        if "=" not in v:
            raise ValueError(
                f"--{flag_name} expects 'name=path' entries, got {v!r}."
            )
        name, _, path = v.partition("=")
        if not name or not path:
            raise ValueError(
                f"--{flag_name} entry {v!r} is missing a name or a path."
            )
        if name in parsed:
            raise ValueError(f"--{flag_name} name {name!r} given twice.")
        parsed[name] = path
    if not parsed:
        raise ValueError(f"At least one --{flag_name} entry is required.")
    return parsed


def _select_split(full_ds, split: str, train_ratio: float, val_ratio: float, seed: int):
    """Return one of train/val/test splits using the same logic as ``train.py``."""
    n_total = len(full_ds)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
    return {"train": train_ds, "val": val_ds, "test": test_ds}[split]


# ---------------------------------------------------------------------------
# predictions subcommand
# ---------------------------------------------------------------------------


@torch.no_grad()
def _collect_predictions(
    module: ASRSelectorModule,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader and collect routing probs and WER scores.

    Args:
        module: A loaded :class:`ASRSelectorModule`.
        loader: DataLoader over the evaluation split.
        device: Torch device to run on.

    Returns:
        ``(probs, wer)`` where ``probs`` has shape ``(N, K)`` and
        ``wer`` has shape ``(N, K)``, both in :data:`MODEL_NAMES` order.
    """
    module.eval().to(device)
    probs_chunks, wer_chunks = [], []
    for batch in loader:
        hs = {k: v.to(device) for k, v in batch["hidden_states"].items()}
        masks = {k: v.to(device) for k, v in batch["attention_masks"].items()}
        probs = module(hs, masks).detach().float().cpu().numpy()
        probs_chunks.append(probs)
        wer_chunks.append(batch["wer_scores"].numpy())
    return np.concatenate(probs_chunks, axis=0), np.concatenate(wer_chunks, axis=0)


def _per_checkpoint_metrics(probs: np.ndarray, wer: np.ndarray) -> Dict:
    """Compute the metrics that get logged alongside the figures."""
    selected = probs.argmax(axis=-1)
    oracle = wer.argmin(axis=-1)
    selected_wer = wer[np.arange(len(wer)), selected].mean()
    oracle_wer = wer.min(axis=-1).mean()
    sel_acc = float((selected == oracle).mean())
    freq = {
        MODEL_NAMES[k]: float((selected == k).mean()) for k in range(wer.shape[1])
    }
    return {
        "selected_wer": float(selected_wer),
        "oracle_wer":   float(oracle_wer),
        "wer_gap":      float(selected_wer - oracle_wer),
        "selection_accuracy": sel_acc,
        "selection_frequency": freq,
        "n":            int(len(wer)),
    }


def _confusion_matrix(probs: np.ndarray, wer: np.ndarray) -> np.ndarray:
    """Predicted-vs-oracle expert confusion matrix, normalised by oracle row."""
    K = wer.shape[1]
    selected = probs.argmax(axis=-1)
    oracle = wer.argmin(axis=-1)
    cm = np.zeros((K, K), dtype=np.float64)
    for o, s in zip(oracle, selected):
        cm[o, s] += 1.0
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm / np.maximum(row_sum, 1), 0.0)
    return cm_norm


def _plot_confusion(ax: plt.Axes, cm: np.ndarray, title: str) -> None:
    K = cm.shape[0]
    im = ax.imshow(cm, vmin=0.0, vmax=1.0, cmap="Blues", aspect="equal")
    for i in range(K):
        for j in range(K):
            ax.text(
                j, i, f"{cm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm[i, j] > 0.5 else "black",
                fontsize=10,
            )
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(MODEL_NAMES, rotation=20)
    ax.set_yticklabels(MODEL_NAMES)
    ax.set_xlabel("Predicted expert")
    ax.set_ylabel("Oracle expert")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised count")


def _plot_scatter(ax: plt.Axes, probs: np.ndarray, wer: np.ndarray, title: str) -> None:
    selected_wer = wer[np.arange(len(wer)), probs.argmax(axis=-1)]
    oracle_wer = wer.min(axis=-1)
    lo = min(float(oracle_wer.min()), float(selected_wer.min())) - 0.02
    hi = max(float(oracle_wer.max()), float(selected_wer.max())) + 0.02
    ax.scatter(oracle_wer, selected_wer, s=8, alpha=0.35, color="tab:blue", edgecolor="none")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="$y=x$ (oracle)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Oracle WER per clip")
    ax.set_ylabel("Router-selected WER per clip")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)


def _plot_selection_frequency(
    ax: plt.Axes,
    metrics_by_name: Dict[str, Dict],
) -> None:
    """Grouped bar chart of selection frequency per checkpoint."""
    K = len(MODEL_NAMES)
    names = list(metrics_by_name.keys())
    width = 0.8 / max(len(names), 1)
    x = np.arange(K)
    palette = plt.cm.tab10(np.arange(len(names)) % 10)
    for i, name in enumerate(names):
        freq = [metrics_by_name[name]["selection_frequency"][m] for m in MODEL_NAMES]
        ax.bar(x + i * width - 0.4 + width / 2, freq, width=width, color=palette[i], label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES)
    ax.set_ylabel("Selection frequency")
    ax.set_title("Per-base-model selection frequency on the test split")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


@app.command()
def predictions(
    parquet_path: str = typer.Option(..., "--parquet-path", "-p"),
    ckpt: List[str] = typer.Option(
        ..., "--ckpt", "-c",
        help="Checkpoint to evaluate, as 'name=path'. May be repeated.",
    ),
    output_dir: str = typer.Option(..., "--output-dir", "-o"),
    split: str = typer.Option("test", "--split"),
    train_ratio: float = typer.Option(0.8, "--train-ratio"),
    val_ratio: float = typer.Option(0.1, "--val-ratio"),
    max_seq_len: int = typer.Option(2000, "--max-seq-len"),
    batch_size: int = typer.Option(32, "--batch-size"),
    num_workers: int = typer.Option(2, "--num-workers"),
    seed: int = typer.Option(42, "--seed"),
):
    """Compute predictions and per-checkpoint figures for one or more checkpoints."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ckpts = _parse_named_options(ckpt, "ckpt")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    full_ds = ASRFeatureDataset(parquet_path=parquet_path, max_seq_len=max_seq_len)
    split_ds = _select_split(full_ds, split, train_ratio, val_ratio, seed)
    loader = DataLoader(
        split_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    logger.info("Device: %s; split=%s (n=%d)", device, split, len(split_ds))

    metrics_by_name: Dict[str, Dict] = {}
    probs_by_name: Dict[str, np.ndarray] = {}
    wer_array: Optional[np.ndarray] = None
    for name, path in ckpts.items():
        logger.info("=== checkpoint %s ← %s ===", name, path)
        module = ASRSelectorModule.load_from_checkpoint(path, map_location=device)
        probs, wer = _collect_predictions(module, loader, device)
        if wer_array is None:
            wer_array = wer
        metrics = _per_checkpoint_metrics(probs, wer)
        metrics_by_name[name] = metrics
        probs_by_name[name] = probs
        logger.info("  metrics: %s", json.dumps(metrics, indent=2))

        cm = _confusion_matrix(probs, wer)
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        _plot_confusion(ax, cm, f"Confusion ({name})")
        fig.tight_layout()
        fig.savefig(out / f"confusion_{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"confusion_{name}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        _plot_scatter(ax, probs, wer, f"Per-clip WER scatter ({name})")
        fig.tight_layout()
        fig.savefig(out / f"scatter_{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"scatter_{name}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _plot_selection_frequency(ax, metrics_by_name)
    fig.tight_layout()
    fig.savefig(out / "selection_frequency.pdf", bbox_inches="tight")
    fig.savefig(out / "selection_frequency.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    summary = {
        "parquet_path": parquet_path,
        "split":        split,
        "train_ratio":  train_ratio,
        "val_ratio":    val_ratio,
        "seed":         seed,
        "n":            int(len(split_ds)),
        "metrics":      metrics_by_name,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote figures and summary.json to %s", out)


# ---------------------------------------------------------------------------
# curves subcommand
# ---------------------------------------------------------------------------


def _read_scalar_series(logdir: str, tags: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Read epoch-aggregated scalar series from a TensorBoard log dir.

    Args:
        logdir: A directory containing one or more ``version_*/`` subdirs
            with TensorBoard event files.
        tags: Scalar tag names to extract.

    Returns:
        Dict mapping tag → ``(steps, values)`` numpy arrays. Missing
        tags map to ``(empty, empty)``.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = sorted(glob.glob(f"{logdir}/version_*/events.out.tfevents.*"))
    series: Dict[str, Tuple[List[int], List[float]]] = {tag: ([], []) for tag in tags}
    for f in files:
        try:
            ea = EventAccumulator(f, size_guidance={"scalars": 100000})
            ea.Reload()
            available = set(ea.Tags().get("scalars", []))
            for tag in tags:
                if tag not in available:
                    continue
                for s in ea.Scalars(tag):
                    series[tag][0].append(int(s.step))
                    series[tag][1].append(float(s.value))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Could not read %s: %s", f, e)
    return {tag: (np.asarray(s[0]), np.asarray(s[1])) for tag, s in series.items()}


@app.command()
def curves(
    logdir: List[str] = typer.Option(
        ..., "--logdir", "-l",
        help="Training log directory, as 'name=path'. May be repeated.",
    ),
    output_path: str = typer.Option(
        "reports/figures/training_curves.pdf", "--output-path", "-o",
    ),
    tags: List[str] = typer.Option(
        ["val/selected_wer", "val/oracle_wer", "val/selection_accuracy"],
        "--tag",
        help="TensorBoard scalar tag to plot. May be repeated.",
    ),
    smoothing_window: int = typer.Option(
        1, "--smoothing-window",
        help="Centred moving-average window size; 1 disables smoothing.",
    ),
):
    """Overlay training curves from multiple TensorBoard log directories."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    runs = _parse_named_options(logdir, "logdir")

    series_by_run: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for name, path in runs.items():
        series_by_run[name] = _read_scalar_series(path, tags)
        for tag, (steps, _) in series_by_run[name].items():
            logger.info("  %s :: %s — %d points", name, tag, len(steps))

    n_panels = len(tags)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.6), squeeze=False)
    palette = plt.cm.tab10(np.arange(len(runs)) % 10)
    for j, tag in enumerate(tags):
        ax = axes[0, j]
        for i, (name, series) in enumerate(series_by_run.items()):
            steps, values = series[tag]
            if len(steps) == 0:
                continue
            order = np.argsort(steps)
            steps = steps[order]
            values = values[order]
            if smoothing_window > 1 and len(values) >= smoothing_window:
                kernel = np.ones(smoothing_window) / smoothing_window
                values = np.convolve(values, kernel, mode="same")
            ax.plot(steps, values, color=palette[i], label=name, linewidth=1.6)
        ax.set_xlabel("Training step")
        ax.set_ylabel(tag.split("/")[-1])
        ax.set_title(tag)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if out.suffix.lower() == ".pdf":
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Wrote curves figure to %s", out)


if __name__ == "__main__":
    app()
