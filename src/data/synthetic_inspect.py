"""Visualize the structure of a synthetic regime-switch parquet.

The figure produced by this script is intended for sanity-checking the
generated dataset and as a notebook demo. It reproduces the regime
geometry (regime centers in latent space), the per-expert WER table
(values and best-expert argmax), and a few sample latent trajectories
projected to 2-D. Together these three views make it visible that
(i) the regimes are separable, (ii) the WER table is asymmetric in
the ordered pair ``(r1, r2)``, and (iii) clips show a clean
regime-switch in their per-frame embedding trajectory.

The script reads the regime centers and the WER table from the parquet
schema metadata written by :mod:`src.data.synthetic`, and the per-clip
embeddings from the regular feature columns. No retraining or model
forward pass is involved.

Usage:
    python -m src.data.synthetic_inspect \\
        --parquet-path data/processed/synthetic_regime_switch/combined_features.parquet \\
        --output-path  reports/figures/synthetic_inspect.pdf \\
        --num-trajectories 4
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import typer
from sklearn.decomposition import PCA

from src.data.synthetic import (
    EXPERT_ORDER,
    METADATA_KEY_GEN_PARAMS,
    METADATA_KEY_REGIME_CENTERS,
    METADATA_KEY_WER_TABLE,
)

logger = logging.getLogger(__name__)
app = typer.Typer(help="Visualize the structure of a synthetic regime-switch parquet.")


def _read_synthetic_metadata(parquet_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load regime centers, WER table, and generation params from the parquet.

    Args:
        parquet_path: Path to a parquet produced by :mod:`src.data.synthetic`.

    Returns:
        Tuple of ``(regime_centers, wer_table, gen_params)`` where
        ``regime_centers`` has shape ``(R, regime_dim)``, ``wer_table``
        has shape ``(R, R, K)``, and ``gen_params`` is the dict written
        at generation time.
    """
    schema = pq.read_table(parquet_path, columns=["ground_truth"]).schema
    md = schema.metadata or {}
    if METADATA_KEY_REGIME_CENTERS not in md or METADATA_KEY_WER_TABLE not in md:
        raise ValueError(
            f"{parquet_path} does not contain the synthetic metadata keys; "
            "regenerate with the current src.data.synthetic."
        )
    regime_centers = np.asarray(json.loads(md[METADATA_KEY_REGIME_CENTERS]), dtype=np.float32)
    wer_table = np.asarray(json.loads(md[METADATA_KEY_WER_TABLE]), dtype=np.float32)
    gen_params = json.loads(md.get(METADATA_KEY_GEN_PARAMS, b"{}"))
    return regime_centers, wer_table, gen_params


def _read_sample_trajectories(
    parquet_path: str,
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the hubert frame-level features for a subset of clips.

    Args:
        parquet_path: Synthetic parquet path.
        indices: 1-D array of row indices to load.

    Returns:
        Tuple ``(features, regime_first, switch_position)`` where
        ``features`` has shape ``(len(indices), T, D_hubert)``, and the
        other two are 1-D arrays of length ``len(indices)``.
    """
    table = pq.read_table(
        parquet_path,
        columns=["hubert_features", "regime_first", "regime_second", "switch_position"],
    )
    features = []
    for i in indices:
        row = np.asarray(table["hubert_features"][int(i)].as_py(), dtype=np.float32)
        features.append(row)
    return (
        np.stack(features, axis=0),
        table["regime_first"].to_numpy()[indices],
        table["switch_position"].to_numpy()[indices],
    )


def _plot_regime_centers(ax: plt.Axes, regime_centers: np.ndarray) -> None:
    """Scatter the regime mean vectors after projecting them to 2-D."""
    R = regime_centers.shape[0]
    coords = PCA(n_components=2).fit_transform(regime_centers)
    colors = plt.cm.tab10(np.arange(R) % 10)
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=180, edgecolor="black", linewidth=0.7, zorder=3)
    for r, (x, y) in enumerate(coords):
        ax.annotate(f"r={r}", (x, y), textcoords="offset points", xytext=(8, 4), fontsize=11)
    ax.set_title("Regime centers (PCA → 2-D)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(alpha=0.3)


def _plot_wer_table(axes: list, wer_table: np.ndarray) -> None:
    """Three side-by-side R×R heatmaps, one per expert.

    Each heatmap shows ``WER[r1, r2, k]`` with rows indexed by ``r1``
    and columns by ``r2``. Asymmetry between cells ``(r1, r2)`` and
    ``(r2, r1)`` is visible to the eye.
    """
    R = wer_table.shape[0]
    vmax = float(wer_table.max())
    vmin = float(wer_table.min())
    for k, (ax, name) in enumerate(zip(axes, EXPERT_ORDER)):
        im = ax.imshow(wer_table[..., k], vmin=vmin, vmax=vmax, cmap="viridis", aspect="equal")
        for r1 in range(R):
            for r2 in range(R):
                ax.text(
                    r2, r1, f"{wer_table[r1, r2, k]:.2f}",
                    ha="center", va="center",
                    color="white" if wer_table[r1, r2, k] < 0.5 * (vmin + vmax) else "black",
                    fontsize=9,
                )
        ax.set_xticks(range(R))
        ax.set_yticks(range(R))
        ax.set_xlabel(r"$r_2$ (second half)")
        if k == 0:
            ax.set_ylabel(r"$r_1$ (first half)")
        ax.set_title(f"WER[$r_1, r_2$, k={name}]")
    plt.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="WER")


def _plot_best_expert_grid(ax: plt.Axes, wer_table: np.ndarray) -> None:
    """Per-(r1, r2) argmin-expert grid; visualises the asymmetry directly."""
    R, _, K = wer_table.shape
    best = wer_table.argmin(axis=-1)
    cmap = plt.get_cmap("tab10", max(K, 3))
    ax.imshow(best, cmap=cmap, vmin=-0.5, vmax=K - 0.5, aspect="equal")
    for r1 in range(R):
        for r2 in range(R):
            ax.text(r2, r1, EXPERT_ORDER[best[r1, r2]],
                    ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(range(R))
    ax.set_yticks(range(R))
    ax.set_xlabel(r"$r_2$ (second half)")
    ax.set_ylabel(r"$r_1$ (first half)")
    ax.set_title("Best expert per ordered pair $(r_1, r_2)$")


def _plot_trajectories(
    ax: plt.Axes,
    features: np.ndarray,
    regime_first: np.ndarray,
    switch_position: np.ndarray,
) -> None:
    """Project per-clip frame embeddings to 2-D and plot their trajectories.

    Frames before the regime switch are drawn in one colour, frames
    after the switch in another. A small marker indicates the switch
    position itself.
    """
    n_clips, T, D = features.shape
    flat = features.reshape(n_clips * T, D)
    coords = PCA(n_components=2).fit_transform(flat).reshape(n_clips, T, 2)
    palette = plt.cm.tab10(np.arange(n_clips) % 10)
    for i in range(n_clips):
        rho = int(switch_position[i])
        first  = coords[i, :rho]
        second = coords[i, rho:]
        ax.plot(first[:, 0],  first[:, 1],  "-",  color=palette[i], alpha=0.85, label=f"clip {i} (1st half)")
        ax.plot(second[:, 0], second[:, 1], "--", color=palette[i], alpha=0.85, label=f"clip {i} (2nd half)")
        ax.scatter([coords[i, rho, 0]], [coords[i, rho, 1]],
                   marker="X", color=palette[i], s=80, edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_title("Per-clip frame trajectories (PCA → 2-D)\nsolid = first half, dashed = second half, X = switch frame")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)


@app.command()
def inspect(
    parquet_path: str = typer.Option(..., "--parquet-path", "-p"),
    output_path: str = typer.Option(
        "reports/figures/synthetic_inspect.pdf", "--output-path", "-o",
        help="Output figure path; format inferred from extension (.pdf/.png).",
    ),
    num_trajectories: int = typer.Option(
        4, "--num-trajectories",
        help="Number of sample clips to show in the trajectory panel.",
    ),
    seed: int = typer.Option(0, "--seed",
        help="Seed for the random selection of trajectory clips."),
):
    """Render the four-panel inspection figure for a synthetic parquet."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    regime_centers, wer_table, gen_params = _read_synthetic_metadata(parquet_path)
    logger.info(
        "Loaded synthetic metadata from %s — R=%d, regime_dim=%d, K=%d",
        parquet_path, *regime_centers.shape, wer_table.shape[-1],
    )

    n_total = pq.read_metadata(parquet_path).num_rows
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=min(num_trajectories, n_total), replace=False)
    indices.sort()
    features, regime_first, switch_position = _read_sample_trajectories(parquet_path, indices)

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.1], hspace=0.45, wspace=0.35)

    ax_centers = fig.add_subplot(gs[0, 0])
    _plot_regime_centers(ax_centers, regime_centers)

    ax_best = fig.add_subplot(gs[0, 1:])
    _plot_best_expert_grid(ax_best, wer_table)

    ax_wer = [fig.add_subplot(gs[1, k]) for k in range(3)]
    _plot_wer_table(ax_wer, wer_table)

    ax_traj = fig.add_subplot(gs[2, :])
    _plot_trajectories(ax_traj, features, regime_first, switch_position)

    title_bits = [f"R={regime_centers.shape[0]}", f"K={wer_table.shape[-1]}"]
    if gen_params:
        title_bits += [
            f"regime_dim={gen_params.get('regime_dim','?')}",
            f"noise_std={gen_params.get('noise_std','?')}",
            f"seed={gen_params.get('seed','?')}",
        ]
    fig.suptitle("Synthetic regime-switch dataset — " + ", ".join(title_bits), fontsize=13)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Wrote inspection figure to %s", out)


if __name__ == "__main__":
    app()
