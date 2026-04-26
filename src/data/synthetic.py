"""Generate a synthetic regime-switch dataset for the model selector.

The output parquet matches the schema consumed by
:class:`src.data.dataset.ASRFeatureDataset` (whisper_features,
hubert_features, w2v2_features, plus the corresponding _wer columns and
``ground_truth``). The training script can therefore consume the
synthetic data without any modification.

Each clip contains ``frame_length`` frames split into two halves by a
regime-switch position ``rho``. The first half is drawn from regime
``r1`` and the second half from regime ``r2`` with ``r2 != r1``,
sampled uniformly from a finite set of ``num_regimes`` regimes.

The per-expert WER scores depend on the ordered pair ``(r1, r2)``,
which makes the best expert non-recoverable from any pooled summary
that is symmetric in the two halves. Frame-level routers can read the
ordering from the temporal sequence and are expected to outperform
pooled baselines.

Usage:
    python -m src.data.synthetic \\
        --output-path data/processed/synthetic_regime_switch/combined_features.parquet \\
        --num-samples 10000 --frame-length 128 --num-regimes 4 --seed 42
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer

logger = logging.getLogger(__name__)
app = typer.Typer(help="Generate a synthetic regime-switch parquet for the ASR Model Selector.")


# Per-expert encoder dimensions; matches the real-world configuration in
# src.training.train.MODEL_DIMS so the resulting parquet is a drop-in
# replacement.
EXPERT_DIMS: Dict[str, int] = {
    "hubert":   1024,
    "whisper":  512,
    "wav2vec2": 1024,
}
# Order matches src.data.dataset.MODEL_NAMES.
EXPERT_ORDER = ["hubert", "whisper", "wav2vec2"]


def _build_wer_table(
    num_regimes: int,
    num_experts: int,
    best_wer: float,
    worst_wer: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct an asymmetric WER lookup table over ordered regime pairs.

    The best expert for the ordered pair ``(r1, r2)`` is determined by
    a deterministic permutation that depends on both ``r1`` and ``r2``,
    so the optimal selection cannot be recovered from a representation
    that is symmetric in the two halves of the clip. A small per-cell
    perturbation breaks ties and avoids exact equality across cells.

    Args:
        num_regimes: Number of regimes ``R``.
        num_experts: Number of experts ``K``.
        best_wer: Target WER for the best expert in each cell.
        worst_wer: Target WER for the non-best experts in each cell.
        rng: Random number generator.

    Returns:
        Array of shape ``(R, R, K)`` with non-negative WER values; the
        diagonal ``r1 == r2`` is unused at sampling time but is filled
        for completeness.
    """
    table = np.full((num_regimes, num_regimes, num_experts), worst_wer, dtype=np.float32)
    for r1 in range(num_regimes):
        for r2 in range(num_regimes):
            best_k = (r1 + 2 * r2) % num_experts
            table[r1, r2, best_k] = best_wer
    table = table + rng.normal(0.0, 0.02, size=table.shape).astype(np.float32)
    return np.clip(table, 0.01, 1.0)


def _generate_clip(
    rng: np.random.Generator,
    regime_centers: np.ndarray,
    expert_projections: Dict[str, np.ndarray],
    frame_length: int,
    num_regimes: int,
    noise_std: float,
    wer_table: np.ndarray,
    wer_noise_std: float,
) -> dict:
    """Generate a single clip and its per-expert features and WER scores.

    Args:
        rng: Random number generator.
        regime_centers: Regime mean vectors of shape ``(R, regime_dim)``.
        expert_projections: Per-expert projection matrices that map the
            regime latent space to the expert hidden dimension.
        frame_length: Number of frames per clip.
        num_regimes: Number of regimes ``R``.
        noise_std: Standard deviation of the per-frame Gaussian noise
            added in the regime latent space.
        wer_table: Per-(r1, r2, k) WER lookup table.
        wer_noise_std: Standard deviation of the additive per-clip
            Gaussian WER noise.

    Returns:
        Dictionary with the per-expert frame-level features (as float32
        nested lists), the per-expert WER scores, and metadata fields.
    """
    r1 = int(rng.integers(num_regimes))
    r2 = int(rng.integers(num_regimes))
    while r2 == r1:
        r2 = int(rng.integers(num_regimes))
    rho = int(rng.integers(frame_length // 4, 3 * frame_length // 4))

    eta = rng.normal(0.0, noise_std, size=(frame_length, regime_centers.shape[1])).astype(np.float32)
    is_first_half = (np.arange(frame_length) < rho)[:, None]
    latent = np.where(
        is_first_half,
        regime_centers[r1] + eta,
        regime_centers[r2] + eta,
    ).astype(np.float32)

    features = {name: latent @ proj for name, proj in expert_projections.items()}

    wer_clean = wer_table[r1, r2]
    wer_noisy = np.clip(
        wer_clean + rng.normal(0.0, wer_noise_std, size=wer_clean.shape).astype(np.float32),
        0.0,
        1.0,
    )

    return {
        "ground_truth": "",
        "hubert_features":  features["hubert"].astype(np.float32).tolist(),
        "whisper_features": features["whisper"].astype(np.float32).tolist(),
        "w2v2_features":    features["wav2vec2"].astype(np.float32).tolist(),
        "hubert_wer":   float(wer_noisy[EXPERT_ORDER.index("hubert")]),
        "whisper_wer":  float(wer_noisy[EXPERT_ORDER.index("whisper")]),
        "w2v2_wer":     float(wer_noisy[EXPERT_ORDER.index("wav2vec2")]),
        "regime_first":     r1,
        "regime_second":    r2,
        "switch_position":  rho,
    }


@app.command()
def generate(
    output_path: str = typer.Option(
        "data/processed/synthetic_regime_switch/combined_features.parquet",
        "--output-path", "-o",
        help="Output parquet path; created if it does not exist.",
    ),
    num_samples: int = typer.Option(10000, "--num-samples", "-n"),
    frame_length: int = typer.Option(128, "--frame-length", "-T"),
    num_regimes: int = typer.Option(4, "--num-regimes", "-R"),
    regime_dim: int = typer.Option(32, "--regime-dim",
        help="Latent dimensionality of the regime mean vectors before "
             "the per-expert projection."),
    noise_std: float = typer.Option(0.5, "--noise-std",
        help="Standard deviation of the per-frame Gaussian noise added "
             "in the regime latent space."),
    best_wer: float = typer.Option(0.15, "--best-wer",
        help="Target WER for the best expert per (r1, r2) cell."),
    worst_wer: float = typer.Option(0.50, "--worst-wer",
        help="Target WER for the non-best experts per (r1, r2) cell."),
    wer_noise_std: float = typer.Option(0.02, "--wer-noise-std",
        help="Per-sample additive Gaussian noise on the WER scores."),
    seed: int = typer.Option(42, "--seed"),
    batch_log: int = typer.Option(2000, "--batch-log",
        help="Log progress every N samples."),
):
    """Generate a synthetic regime-switch parquet matching the real schema."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    rng = np.random.default_rng(seed)

    regime_centers = rng.standard_normal((num_regimes, regime_dim)).astype(np.float32)
    expert_projections = {
        name: (rng.standard_normal((regime_dim, dim)) / np.sqrt(regime_dim)).astype(np.float32)
        for name, dim in EXPERT_DIMS.items()
    }
    wer_table = _build_wer_table(
        num_regimes=num_regimes,
        num_experts=len(EXPERT_ORDER),
        best_wer=best_wer,
        worst_wer=worst_wer,
        rng=rng,
    )

    logger.info(
        "Generating %d clips: T=%d, R=%d, regime_dim=%d, dims=%s",
        num_samples, frame_length, num_regimes, regime_dim, EXPERT_DIMS,
    )
    rows = []
    for n in range(num_samples):
        rows.append(_generate_clip(
            rng=rng,
            regime_centers=regime_centers,
            expert_projections=expert_projections,
            frame_length=frame_length,
            num_regimes=num_regimes,
            noise_std=noise_std,
            wer_table=wer_table,
            wer_noise_std=wer_noise_std,
        ))
        if (n + 1) % batch_log == 0:
            logger.info("  generated %d / %d", n + 1, num_samples)

    table = pa.Table.from_pylist(rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out.as_posix())
    logger.info("Wrote %d rows to %s", len(rows), out)

    wer_array = np.stack([
        np.asarray([r["hubert_wer"] for r in rows]),
        np.asarray([r["whisper_wer"] for r in rows]),
        np.asarray([r["w2v2_wer"] for r in rows]),
    ], axis=-1)
    oracle = wer_array.min(axis=-1).mean()
    random_avg = wer_array.mean()
    logger.info("Sanity stats — random WER: %.4f, oracle WER: %.4f", random_avg, oracle)


if __name__ == "__main__":
    app()
