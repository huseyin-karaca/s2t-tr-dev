import os
import re
import pandas as pd
import torch
import evaluate
from datasets import Audio
from typing import List
import typer

from src.data.config import ALL_DATASETS
from src.base_models.config import ALL_BASE_MODELS
from src.config import PROJ_ROOT

app = typer.Typer(help="Generates outputs (embedding/wer) for multiple datasets and models.")


def process_combination(ds_cfg, model_cfg, device: str, batch_size: int):
    """Processes a single dataset and model combination."""
    print(f"\n🚀 Starting process: Dataset [ {ds_cfg.name} ] ⬇️ Model [ {model_cfg.model_id} ]")

    # 1. Load Raw Data
    ds = ds_cfg.load()
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Filter out very short or corrupted audio files (< 0.1 sec)
    min_sample_length = 1600
    original_len = len(ds)
    ds = ds.filter(
        lambda x: len(x["audio"]["array"]) > min_sample_length,
        num_proc=os.cpu_count() or 4,
        desc="Filtering (Removing too short/corrupted audio)",
    )
    if len(ds) < original_len:
        typer.secho(
            f"🧹 Cleanup: {original_len - len(ds)} short/corrupted audio files filtered out.",
            fg=typer.colors.YELLOW,
        )

    # 2. Prepare Model and Metrics
    loaded_model, processor = model_cfg.load(device=device)
    wer_metric = evaluate.load("wer")

    def clean_text(text):
        if text is None:
            return ""
        return re.sub(r"[^\w\s]", "", str(text).lower()).strip()

    safe_model_name = model_cfg.model_id.replace("/", "_")

    # 3. Batch Processing — uses predict_batch
    def process_batch(batch):
        audio_arrays = [a["array"] for a in batch["audio"]]
        srs = [a["sampling_rate"] for a in batch["audio"]]

        text_col = "text" if "text" in batch else "transcription"
        targets = [clean_text(t) for t in batch[text_col]]

        # Note: Ensure that predict_batch returns frame-level (2D) embeddings 
        # (e.g., shape: [num_frames, hidden_dim]) rather than pooled 1D embeddings.
        results = model_cfg.predict_batch(
            loaded_model, processor, audio_arrays, srs, device
        )

        preds = [clean_text(r["transcription"]) for r in results]
        wers = [
            wer_metric.compute(predictions=[p], references=[t]) if t else 0.0
            for p, t in zip(preds, targets)
        ]

        # Ensure embeddings are nested Python lists. 
        # If the model returns a 2D array (num_frames, 768), tolist() converts it 
        # to a list of lists, which Arrow handles perfectly as variable-length arrays.
        embeddings = []
        for r in results:
            emb = r["embedding"]
            if hasattr(emb, "tolist"):
                emb = emb.tolist()
            embeddings.append(emb)

        return {
            "ground_truth":                     targets, # 1) Ground truth added here
            f"{safe_model_name}_embedding":     embeddings,
            f"{safe_model_name}_transcription": preds,
            f"{safe_model_name}_wer":           wers,
        }

    # 4. Inference
    feature_ds = ds.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=ds.column_names,
        desc=f"Running {model_cfg.model_id} inference",
    )

    # 5. Save as a single Parquet file
    safe_ds_name = ds_cfg.name.replace("/", "_")
    output_dir = os.path.join(ds_cfg.interim_path, safe_ds_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{safe_model_name}.parquet")
    
    # Save directly to parquet
    feature_ds.to_parquet(output_file)
    
    typer.secho(
        f"✅ Combination completed → {output_file}", fg=typer.colors.GREEN
    )


@app.command()
def generate_features(
    datasets: List[str] = typer.Option(..., "--dataset", "-d", help="Dataset alias"),
    models:   List[str] = typer.Option(..., "--model",   "-m", help="Model alias"),
    batch_size: int     = typer.Option(8,   "--batch-size", "-b", help="Batch size (adjust based on GPU memory)"),
    cpu: bool           = typer.Option(False, "--cpu", help="Use CPU instead of GPU"),
):
    """Generates features for all given dataset × model combinations."""
    device = "cpu" if cpu or not torch.cuda.is_available() else "cuda"
    print(f"⚙️  Device: {device.upper()}  |  Batch size: {batch_size}")

    valid_datasets = []
    for d_alias in datasets:
        if d_alias in ALL_DATASETS:
            valid_datasets.append(ALL_DATASETS[d_alias])
        else:
            typer.secho(f"⚠️  Dataset '{d_alias}' not found, skipping.", fg=typer.colors.YELLOW)

    valid_models = []
    for m_alias in models:
        if m_alias in ALL_BASE_MODELS:
            valid_models.append(ALL_BASE_MODELS[m_alias])
        else:
            typer.secho(f"⚠️  Model '{m_alias}' not found, skipping.", fg=typer.colors.YELLOW)

    if not valid_datasets or not valid_models:
        typer.secho("❌ No valid dataset/model combinations left.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    total = len(valid_datasets) * len(valid_models)
    typer.secho(f"\n🔥 Total {total} combinations will be computed...", fg=typer.colors.CYAN)

    for ds_cfg in valid_datasets:
        for model_cfg in valid_models:
            process_combination(ds_cfg, model_cfg, device, batch_size)

    typer.secho("\n🎉 All tasks completed successfully!", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
