"""Rewrite a Parquet file with smaller row groups, streaming.

The combined feature parquets produced by older versions of
:mod:`src.data.preprocess` (or by Pandas-based tooling) tend to use a
single huge row group, which forces
:class:`src.data.dataset.ASRFeatureDataset` to materialize the entire
table when it caches a row group. This script reads the source parquet
in row-group-batched batches and writes them out with a much smaller
``row_group_size``, without ever holding the whole table in memory.

Usage:
    python -m src.data.rechunk_parquet \\
        --input  data/processed/edinburghcstr_ami/combined_features_with_transcripts.parquet \\
        --output data/processed/edinburghcstr_ami/combined_features_rechunked.parquet \\
        --row-group-size 256
"""

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import typer

logger = logging.getLogger(__name__)
app = typer.Typer(help="Rewrite a Parquet file with smaller row groups (streaming).")


@app.command()
def rechunk(
    input: str = typer.Option(..., "--input", "-i"),
    output: str = typer.Option(..., "--output", "-o"),
    row_group_size: int = typer.Option(256, "--row-group-size"),
    read_batch_size: int = typer.Option(
        256, "--read-batch-size",
        help="Rows pulled per iter_batches() call. Should match or exceed "
             "row_group_size; the writer re-chunks downstream.",
    ),
):
    """Stream ``input`` to ``output`` with a smaller ``row_group_size``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    src = pq.ParquetFile(input)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Rechunking %s -> %s (rows=%d, src_row_groups=%d, target_row_group_size=%d)",
        input, output, src.metadata.num_rows, src.metadata.num_row_groups,
        row_group_size,
    )

    writer: pq.ParquetWriter | None = None
    written = 0
    for batch in src.iter_batches(batch_size=read_batch_size):
        table = pa.Table.from_batches([batch])
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table, row_group_size=row_group_size)
        written += table.num_rows
        if written % (10 * row_group_size) == 0:
            logger.info("  wrote %d / %d rows", written, src.metadata.num_rows)

    if writer is not None:
        writer.close()
    logger.info("Done — wrote %d rows to %s", written, output)


if __name__ == "__main__":
    app()
