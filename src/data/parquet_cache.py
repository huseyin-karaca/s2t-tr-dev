"""Make a Parquet file safe for lazy, low-memory random access.

This module exists because :class:`src.data.dataset.ASRFeatureDataset`'s
"lazy" loading strategy implicitly assumes that each row group of the
combined-features parquet is small enough to keep in RAM at most a
few at a time (one per worker × LRU cache size). When the upstream
parquet was written with a single huge row group (the default for
``pyarrow.parquet.write_table`` and many Pandas pipelines), reading
"the row group containing index *i*" actually loads the entire dataset
into memory — and once the DataLoader forks ``num_workers`` copies of
that, system RAM blows up and the workers get OOM-killed by the
kernel:

    RuntimeError: DataLoader worker (pid X) is killed by signal: Killed.

To make training robust to whatever row-group layout the input was
written with, :func:`ensure_lazy_parquet` checks the row-group sizes
of the source file and, if needed, streams a rechunked copy with
small row groups (default 256 rows) to a cache directory. The
rechunked file is reused on subsequent runs as long as it is newer
than the source. Subsequent training runs read from the cached file,
which guarantees that each ``read_row_group`` call decodes at most
``target_row_group_size`` clips' worth of features.

Typical usage from the dataset layer::

    cached_path = ensure_lazy_parquet(parquet_path, target_row_group_size=256)
    pq_file = pq.ParquetFile(cached_path)
    ...
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DEFAULT_TARGET_ROW_GROUP_SIZE = 256
DEFAULT_LARGE_ROW_GROUP_RATIO = 4


def _default_cache_root() -> Path:
    """Return a writable cache directory with reasonable defaults.

    Order of preference:
        1. ``$S2T_PARQUET_CACHE`` if set (explicit user control).
        2. ``$TMPDIR/s2t-tr-dev-cache`` (fast local disk on Colab).
        3. ``/tmp/s2t-tr-dev-cache`` as a last resort.

    We deliberately avoid the source directory because the source is
    often a slow / network-mounted location (e.g. Google Drive on Colab).
    """
    env = os.environ.get("S2T_PARQUET_CACHE")
    if env:
        return Path(env)
    tmp = os.environ.get("TMPDIR") or tempfile.gettempdir() or "/tmp"
    return Path(tmp) / "s2t-tr-dev-cache"


def _cache_path_for(
    src: Path,
    cache_dir: Path,
    target_row_group_size: int,
) -> Path:
    """Stable, deterministic cache path for a given (source, row-group) pair.

    We embed an MD5 of the source's absolute parent directory as a
    ``.<hash>`` suffix so that two parquets with the same basename in
    different directories don't collide in the shared cache. We use
    MD5 explicitly (not Python's built-in :func:`hash`) because the
    latter is randomized per interpreter run via ``PYTHONHASHSEED``,
    which would defeat caching entirely.
    """
    parent = str(src.parent.resolve()).encode("utf-8")
    digest = hashlib.md5(parent).hexdigest()[:8]
    name = f"{src.stem}.rg{target_row_group_size}.{digest}.parquet"
    return cache_dir / name


def _is_fresh(cache_path: Path, src: Path) -> bool:
    """Return True if ``cache_path`` exists and is at least as new as ``src``."""
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        return False
    try:
        return cache_path.stat().st_mtime >= src.stat().st_mtime
    except OSError:
        return False


def _rechunk_streaming(
    src_path: Path,
    dst_path: Path,
    target_row_group_size: int,
) -> None:
    """Stream-rewrite ``src_path`` to ``dst_path`` with small row groups.

    Reads one batch at a time via :meth:`pq.ParquetFile.iter_batches`
    so peak memory is bounded by a single batch's Arrow footprint —
    not the full table. Writes to a ``.tmp`` sibling and renames on
    success so an interrupted rechunk never leaves a half-written
    file in the cache.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    src = pq.ParquetFile(str(src_path))
    total_rows = src.metadata.num_rows

    logger.info(
        "Rechunking parquet %s -> %s (rows=%d, src_row_groups=%d, "
        "target_row_group_size=%d).",
        src_path, dst_path, total_rows, src.metadata.num_row_groups,
        target_row_group_size,
    )

    writer: Optional[pq.ParquetWriter] = None
    written = 0
    try:
        for batch in src.iter_batches(batch_size=target_row_group_size):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(
                    str(tmp_path),
                    table.schema,
                    compression="snappy",
                )
            writer.write_table(table, row_group_size=target_row_group_size)
            written += table.num_rows
            if total_rows and written % (20 * target_row_group_size) == 0:
                logger.info("  rechunk progress: %d / %d rows", written, total_rows)
    finally:
        if writer is not None:
            writer.close()

    os.replace(tmp_path, dst_path)
    logger.info("Rechunk complete: wrote %d rows to %s.", written, dst_path)


def ensure_lazy_parquet(
    parquet_path: str | os.PathLike,
    target_row_group_size: int = DEFAULT_TARGET_ROW_GROUP_SIZE,
    cache_dir: Optional[str | os.PathLike] = None,
    large_row_group_ratio: int = DEFAULT_LARGE_ROW_GROUP_RATIO,
    auto_rechunk: bool = True,
) -> str:
    """Return a path to a Parquet that supports cheap lazy row-group reads.

    If the input parquet already has reasonably small row groups
    (max rows per group ≤ ``large_row_group_ratio * target_row_group_size``),
    it is returned unchanged. Otherwise a rechunked copy is materialized
    in ``cache_dir`` (default :func:`_default_cache_root`) and that path
    is returned. The rechunked file is cached by mtime, so repeated runs
    skip the rewrite.

    Args:
        parquet_path: Source parquet path.
        target_row_group_size: Rows per row group in the rechunked file.
            256 keeps each row group's Arrow footprint to a few hundred
            MB even for HuBERT-Large-style 1024-dim frame features.
        cache_dir: Where to write the rechunked file. Defaults to a
            local-disk path (see :func:`_default_cache_root`).
        large_row_group_ratio: A row group is considered "too large" if
            it contains more than ``large_row_group_ratio *
            target_row_group_size`` rows. The 4× cushion avoids
            re-rewriting parquets that are only mildly chunkier than
            the target (e.g. 512-row groups when the target is 256).
        auto_rechunk: If False, never rewrite — return the original
            path even if it has huge row groups (useful for debugging
            or when disk space is too tight to cache).

    Returns:
        Path (string) to a parquet that is safe for the lazy-loading
        strategy in :class:`ASRFeatureDataset`.
    """
    src = Path(parquet_path)
    if not src.exists():
        return str(src)

    meta = pq.read_metadata(str(src))
    n_rows = meta.num_rows
    n_groups = meta.num_row_groups
    if n_groups == 0:
        return str(src)

    max_rg_rows = max(meta.row_group(i).num_rows for i in range(n_groups))
    threshold = max(target_row_group_size * large_row_group_ratio, target_row_group_size)
    if max_rg_rows <= threshold:
        logger.info(
            "Parquet %s already has small row groups (max=%d rows ≤ threshold=%d); "
            "using as-is.",
            src, max_rg_rows, threshold,
        )
        return str(src)

    if not auto_rechunk:
        logger.warning(
            "Parquet %s has huge row groups (max=%d rows; %d total rows in %d groups), "
            "but auto_rechunk=False. Lazy access may exhaust system RAM.",
            src, max_rg_rows, n_rows, n_groups,
        )
        return str(src)

    cache_root = Path(cache_dir) if cache_dir is not None else _default_cache_root()
    cached = _cache_path_for(src, cache_root, target_row_group_size)

    if _is_fresh(cached, src):
        logger.info(
            "Reusing cached rechunked parquet %s (source max_rg_rows=%d > threshold=%d).",
            cached, max_rg_rows, threshold,
        )
        return str(cached)

    free_bytes = shutil.disk_usage(cache_root.parent if cache_root.parent.exists()
                                   else Path("/")).free
    src_bytes = src.stat().st_size
    if free_bytes < src_bytes * 1.2:
        logger.warning(
            "Not enough disk space at %s to cache a rechunked copy "
            "(need ~%.1f GB, have %.1f GB free). Falling back to original parquet — "
            "training may OOM if row groups are too large.",
            cache_root, src_bytes / 1e9 * 1.2, free_bytes / 1e9,
        )
        return str(src)

    logger.warning(
        "Parquet %s has huge row groups (max=%d rows; %d total rows in %d groups). "
        "Rechunking to %s with row_group_size=%d for lazy random access.",
        src, max_rg_rows, n_rows, n_groups, cached, target_row_group_size,
    )
    _rechunk_streaming(src, cached, target_row_group_size)
    return str(cached)
