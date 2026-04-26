"""Combine per-expert interim parquets into a single processed parquet.

Reads the three interim parquets written by :mod:`src.data.generate_features`
and emits a combined table with one row per clip. Each interim file carries
its own ``{safe_model_name}_embedding``, ``_transcription`` and ``_wer``
columns; this script renames them to expert-agnostic columns
(``hubert_features``, ``hubert_transcription``, ``hubert_wer`` ...) so that
downstream training and ROVER baselines can address them uniformly.

The output filename defaults to ``combined_features_with_transcripts.parquet``
to avoid clobbering existing ``combined_features.parquet`` files produced
by older versions of this script (which dropped the transcription columns).
"""

import argparse
import logging
import os

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

WHISPER_MODEL = "openai_whisper-base"
HUBERT_MODEL = "facebook_hubert-large-ls960-ft"
W2V2_MODEL = "facebook_wav2vec2-base-960h"


def process_dataset(
    dataset_name: str,
    interim_dir: str = "data/interim",
    processed_dir: str = "data/processed",
    output_name: str = "combined_features_with_transcripts.parquet",
) -> str:
    """Combine the three interim per-expert parquets for ``dataset_name``.

    Args:
        dataset_name: Hugging Face dataset id (e.g. ``edinburghcstr/ami``).
        interim_dir: Root for the per-expert interim parquets.
        processed_dir: Root for the combined output parquet.
        output_name: Filename written under ``processed_dir/{safe_name}/``.

    Returns:
        The absolute path of the written parquet.
    """
    safe_name = dataset_name.replace("/", "_")
    interim_path = os.path.join(interim_dir, safe_name)
    processed_path = os.path.join(processed_dir, safe_name)

    logger.info("Loading interim datasets for %s ...", dataset_name)
    w_table = pq.read_table(f"{interim_path}/{WHISPER_MODEL}.parquet")
    h_table = pq.read_table(f"{interim_path}/{HUBERT_MODEL}.parquet")
    v_table = pq.read_table(f"{interim_path}/{W2V2_MODEL}.parquet")

    assert w_table.num_rows == h_table.num_rows == v_table.num_rows, (
        "Interim parquets have mismatched row counts."
    )

    logger.info("Combining %d rows ...", w_table.num_rows)
    arrays = [
        w_table["ground_truth"],
        w_table[f"{WHISPER_MODEL}_embedding"],
        h_table[f"{HUBERT_MODEL}_embedding"],
        v_table[f"{W2V2_MODEL}_embedding"],
        w_table[f"{WHISPER_MODEL}_transcription"],
        h_table[f"{HUBERT_MODEL}_transcription"],
        v_table[f"{W2V2_MODEL}_transcription"],
        w_table[f"{WHISPER_MODEL}_wer"],
        h_table[f"{HUBERT_MODEL}_wer"],
        v_table[f"{W2V2_MODEL}_wer"],
    ]
    names = [
        "ground_truth",
        "whisper_features",
        "hubert_features",
        "w2v2_features",
        "whisper_transcription",
        "hubert_transcription",
        "w2v2_transcription",
        "whisper_wer",
        "hubert_wer",
        "w2v2_wer",
    ]
    combined_table = pa.Table.from_arrays(arrays, names=names)

    os.makedirs(processed_path, exist_ok=True)
    out_path = os.path.join(processed_path, output_name)
    logger.info("Writing combined parquet to %s", out_path)
    pq.write_table(combined_table, out_path)
    logger.info("Done.")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Preprocess and combine ASR features.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of the dataset (e.g., edinburghcstr/ami)")
    parser.add_argument("--interim_dir", type=str, default="data/interim")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_name", type=str,
                        default="combined_features_with_transcripts.parquet",
                        help="Output filename under processed_dir/{safe_name}/")
    args = parser.parse_args()

    process_dataset(
        args.dataset, args.interim_dir, args.processed_dir, args.output_name,
    )
