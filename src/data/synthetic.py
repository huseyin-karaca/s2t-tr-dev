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
that is symmetric in the two halves of the clip. Frame-level routers
can read the ordering from the temporal sequence and are expected to
outperform pooled baselines.

Memory and IO. The clips are produced and written to parquet in
streamed batches via :class:`pyarrow.parquet.ParquetWriter`, with the
nested-list embedding columns built directly from numpy arrays via
offset-based :class:`pyarrow.ListArray.from_arrays`. This avoids the
~30x inflation that ``ndarray.tolist()`` causes when the floats are
materialized as Python objects, and caps peak RAM at roughly one
batch worth of embeddings.

Usage:
    python -m src.data.synthetic \\
        --output-path data/processed/synthetic_regime_switch/combined_features.parquet \\
        --num-samples 10000 --frame-length 128 --num-regimes 4 --seed 42
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

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
FEATURE_COL = {"hubert": "hubert_features", "whisper": "whisper_features", "wav2vec2": "w2v2_features"}
WER_COL     = {"hubert": "hubert_wer",      "whisper": "whisper_wer",      "wav2vec2": "w2v2_wer"}

DTYPE_MAP = {"float16": np.float16, "float32": np.float32}


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
    that is symmetric in the two halves of the clip.

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


def _generate_batch(
    rng: np.random.Generator,
    regime_centers: np.ndarray,
    expert_projections: Dict[str, np.ndarray],
    wer_table: np.ndarray,
    batch_size: int,
    frame_length: int,
    num_regimes: int,
    noise_std: float,
    wer_noise_std: float,
    feature_dtype: np.dtype,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
    """Generate a batch of clips fully vectorised.

    Args:
        rng: Random number generator.
        regime_centers: Regime mean vectors of shape ``(R, regime_dim)``.
        expert_projections: Per-expert projection matrices that map the
            regime latent space to the expert hidden dimension.
        wer_table: Per-(r1, r2, k) WER lookup table.
        batch_size: Number of clips to generate in this batch.
        frame_length: Number of frames per clip.
        num_regimes: Number of regimes ``R``.
        noise_std: Standard deviation of the per-frame Gaussian noise
            added in the regime latent space.
        wer_noise_std: Standard deviation of the additive per-clip
            Gaussian WER noise.
        feature_dtype: Output dtype for the per-expert features.

    Returns:
        Tuple of ``(features, wer_scores, metadata)`` where ``features``
        maps expert name to a ``(B, T, D_k)`` array, ``wer_scores`` is
        of shape ``(B, K)`` in :data:`EXPERT_ORDER`, and ``metadata``
        contains the per-clip ``regime_first``, ``regime_second``, and
        ``switch_position`` integer arrays.
    """
    B, T = batch_size, frame_length
    K = len(EXPERT_ORDER)

    r1 = rng.integers(0, num_regimes, size=B)
    r2 = rng.integers(0, num_regimes - 1, size=B)
    r2 = np.where(r2 >= r1, r2 + 1, r2)
    rho = rng.integers(T // 4, 3 * T // 4, size=B)

    is_first = (np.arange(T)[None, :] < rho[:, None])
    centers_first  = regime_centers[r1]
    centers_second = regime_centers[r2]
    centers_per_frame = np.where(
        is_first[..., None],
        centers_first[:, None, :],
        centers_second[:, None, :],
    )
    eta = rng.normal(0.0, noise_std, size=centers_per_frame.shape).astype(np.float32)
    latent = (centers_per_frame + eta).astype(np.float32)

    features: Dict[str, np.ndarray] = {}
    for name, proj in expert_projections.items():
        feat = np.einsum("btd,de->bte", latent, proj)
        features[name] = feat.astype(feature_dtype, copy=False)

    wer_clean = wer_table[r1, r2]
    wer_noise = rng.normal(0.0, wer_noise_std, size=wer_clean.shape).astype(np.float32)
    wer_scores = np.clip(wer_clean + wer_noise, 0.0, 1.0)

    metadata = {
        "regime_first":    r1.astype(np.int32),
        "regime_second":   r2.astype(np.int32),
        "switch_position": rho.astype(np.int32),
    }
    return features, wer_scores, metadata


def _nested_list_column(arr: np.ndarray) -> pa.Array:
    """Build a ``list<list<floatX>>`` PyArrow array from a (N, T, D) tensor.

    The outer list is over time frames, the inner list is over the
    encoder hidden dimension. Construction is offset-based and avoids
    materializing Python lists of floats, which is what would otherwise
    cause RAM to balloon for large ``N``.

    Args:
        arr: 3-D numpy array of shape ``(N, T, D)``.

    Returns:
        PyArrow ``ListArray`` whose logical length is ``N``.
    """
    N, T, D = arr.shape
    flat_values = pa.array(np.ascontiguousarray(arr).reshape(-1), type=pa.from_numpy_dtype(arr.dtype))
    inner_offsets = pa.array(np.arange(0, N * T * D + 1, D, dtype=np.int32))
    inner_list = pa.ListArray.from_arrays(inner_offsets, flat_values)
    outer_offsets = pa.array(np.arange(0, N * T + 1, T, dtype=np.int32))
    return pa.ListArray.from_arrays(outer_offsets, inner_list)


def _record_batch(
    features: Dict[str, np.ndarray],
    wer_scores: np.ndarray,
    metadata: Dict[str, np.ndarray],
) -> pa.RecordBatch:
    """Pack a generated batch into a :class:`pyarrow.RecordBatch`.

    Args:
        features: Mapping of expert name to a ``(B, T, D_k)`` array.
        wer_scores: ``(B, K)`` per-clip WER scores in
            :data:`EXPERT_ORDER`.
        metadata: Per-clip integer metadata fields.

    Returns:
        PyArrow record batch with the unified parquet schema.
    """
    B = wer_scores.shape[0]
    columns = {
        "ground_truth": pa.array([""] * B, type=pa.string()),
    }
    for name in EXPERT_ORDER:
        columns[FEATURE_COL[name]] = _nested_list_column(features[name])
    for k, name in enumerate(EXPERT_ORDER):
        columns[WER_COL[name]] = pa.array(wer_scores[:, k].astype(np.float32), type=pa.float32())
    for k, v in metadata.items():
        columns[k] = pa.array(v, type=pa.int32())
    return pa.RecordBatch.from_pydict(columns)


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
    feature_dtype: str = typer.Option(
        "float16", "--feature-dtype",
        help="dtype of the embedding columns; one of float16, float32. "
             "float16 matches the real-world parquet and halves memory.",
    ),
    write_batch_size: int = typer.Option(
        500, "--write-batch-size",
        help="Number of clips generated and flushed per parquet row "
             "group. Larger = fewer row groups, more peak RAM.",
    ),
    seed: int = typer.Option(42, "--seed"),
):
    """Generate a synthetic regime-switch parquet matching the real schema."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if feature_dtype not in DTYPE_MAP:
        raise ValueError(f"--feature-dtype must be one of {list(DTYPE_MAP)}, got {feature_dtype!r}")
    np_feature_dtype = DTYPE_MAP[feature_dtype]

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

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    bytes_per_clip_per_expert = frame_length * sum(EXPERT_DIMS.values()) * np.dtype(np_feature_dtype).itemsize
    total_bytes = num_samples * bytes_per_clip_per_expert
    peak_batch_bytes = write_batch_size * bytes_per_clip_per_expert
    logger.info(
        "Generating %d clips → %s (T=%d, R=%d, regime_dim=%d, dims=%s, dtype=%s)",
        num_samples, out, frame_length, num_regimes, regime_dim, EXPERT_DIMS, feature_dtype,
    )
    logger.info(
        "Embedding payload: %.2f GB total, peak per write batch: %.2f GB (write_batch_size=%d)",
        total_bytes / 1e9, peak_batch_bytes / 1e9, write_batch_size,
    )

    writer: pq.ParquetWriter = None
    wer_running = []
    n_written = 0
    try:
        while n_written < num_samples:
            this_batch = min(write_batch_size, num_samples - n_written)
            features, wer_scores, metadata = _generate_batch(
                rng=rng,
                regime_centers=regime_centers,
                expert_projections=expert_projections,
                wer_table=wer_table,
                batch_size=this_batch,
                frame_length=frame_length,
                num_regimes=num_regimes,
                noise_std=noise_std,
                wer_noise_std=wer_noise_std,
                feature_dtype=np_feature_dtype,
            )
            batch = _record_batch(features, wer_scores, metadata)
            if writer is None:
                writer = pq.ParquetWriter(out.as_posix(), batch.schema, compression="snappy")
            writer.write_batch(batch)
            wer_running.append(wer_scores)

            del features, wer_scores, metadata, batch
            n_written += this_batch
            if n_written % (max(write_batch_size, 1) * 4) == 0 or n_written == num_samples:
                logger.info("  wrote %d / %d clips", n_written, num_samples)
    finally:
        if writer is not None:
            writer.close()

    wer_array = np.concatenate(wer_running, axis=0)
    oracle = wer_array.min(axis=-1).mean()
    random_avg = wer_array.mean()
    logger.info(
        "Sanity stats — random WER: %.4f, oracle WER: %.4f, "
        "per-expert mean WER: %s",
        random_avg, oracle,
        {name: float(wer_array[:, k].mean()) for k, name in enumerate(EXPERT_ORDER)},
    )
    logger.info("Done.")


if __name__ == "__main__":
    app()
