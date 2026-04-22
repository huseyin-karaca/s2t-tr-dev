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

app = typer.Typer(help="Çoklu Dataset ve Model için çıktı (embedding/wer) üretir.")


def process_combination(ds_cfg, model_cfg, device: str, batch_size: int):
    """Tek bir dataset + model kombinasyonunu işler."""
    print(f"\n🚀 İşlem Başlıyor: Dataset [ {ds_cfg.name} ] ⬇️ Model [ {model_cfg.model_id} ]")

    # 1. Ham Veriyi Yükle
    ds = ds_cfg.load()
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Çok kısa / bozuk ses dosyalarını filtrele (< 0.1 sn)
    min_sample_length = 1600
    original_len = len(ds)
    ds = ds.filter(
        lambda x: len(x["audio"]["array"]) > min_sample_length,
        num_proc=os.cpu_count() or 4,
        desc="Filtreleme (Çok kısa/bozuk sesler temizleniyor)",
    )
    if len(ds) < original_len:
        typer.secho(
            f"🧹 Temizlik: {original_len - len(ds)} adet bozuk/kısa ses filtrelendi.",
            fg=typer.colors.YELLOW,
        )

    # 2. Model ve Metrik Hazırlığı
    loaded_model, processor = model_cfg.load(device=device)
    wer_metric = evaluate.load("wer")

    def clean_text(text):
        if text is None:
            return ""
        return re.sub(r"[^\w\s]", "", str(text).lower()).strip()

    safe_model_name = model_cfg.model_id.replace("/", "_")

    # 3. Batch İşleme — predict_batch kullanır
    def process_batch(batch):
        audio_arrays = [a["array"] for a in batch["audio"]]
        srs = [a["sampling_rate"] for a in batch["audio"]]

        text_col = "text" if "text" in batch else "transcription"
        targets = [clean_text(t) for t in batch[text_col]]

        results = model_cfg.predict_batch(
            loaded_model, processor, audio_arrays, srs, device
        )

        preds = [clean_text(r["transcription"]) for r in results]
        wers = [
            wer_metric.compute(predictions=[p], references=[t]) if t else 0.0
            for p, t in zip(preds, targets)
        ]

        # MODIFIED: Ensure embeddings are nested Python lists, not Tensors/Numpy arrays.
        # This prevents PyArrow from trying to vertically stack variable-length arrays.
        embeddings = []
        for r in results:
            emb = r["embedding"]
            # Convert PyTorch Tensor or Numpy array to nested list
            if hasattr(emb, "tolist"):
                emb = emb.tolist()
            embeddings.append(emb)

        return {
            f"{safe_model_name}_embedding":     embeddings,
            f"{safe_model_name}_transcription": preds,
            f"{safe_model_name}_wer":           wers,
        }

    # 4. Çıkarım
    feature_ds = ds.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=ds.column_names,
        desc=f"Running {model_cfg.model_id} inference",
    )

    # 5. Kayıt (as directory, hf native)
    # safe_ds_name = ds_cfg.name.replace("/", "_")
    # output_dir = os.path.join(ds_cfg.interim_path, safe_ds_name, safe_model_name)
    # os.makedirs(output_dir, exist_ok=True)
    # feature_ds.save_to_disk(output_dir)
    # typer.secho(
    #    f"✅ Kombinasyon tamamlandı → {output_dir}", fg=typer.colors.GREEN
    # )

    # 5. Kayıt (MODIFIED: Save as a single Parquet file instead of a dataset directory)
    safe_ds_name = ds_cfg.name.replace("/", "_")
    output_dir = os.path.join(ds_cfg.interim_path, safe_ds_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{safe_model_name}.parquet")
    
    # Save directly to parquet
    feature_ds.to_parquet(output_file)
    
    typer.secho(
        f"✅ Kombinasyon tamamlandı → {output_file}", fg=typer.colors.GREEN
    )


@app.command()
def generate_features(
    datasets: List[str] = typer.Option(..., "--dataset", "-d", help="Dataset alias"),
    models:   List[str] = typer.Option(..., "--model",   "-m", help="Model alias"),
    batch_size: int     = typer.Option(8,   "--batch-size", "-b", help="Batch size (GPU belleğine göre ayarla)"),
    cpu: bool           = typer.Option(False, "--cpu", help="GPU yerine CPU kullan"),
):
    """Verilen tüm dataset × model kombinasyonları için özellikleri üretir."""
    device = "cpu" if cpu or not torch.cuda.is_available() else "cuda"
    print(f"⚙️  Cihaz: {device.upper()}  |  Batch size: {batch_size}")

    valid_datasets = []
    for d_alias in datasets:
        if d_alias in ALL_DATASETS:
            valid_datasets.append(ALL_DATASETS[d_alias])
        else:
            typer.secho(f"⚠️  Dataset '{d_alias}' bulunamadı, atlanıyor.", fg=typer.colors.YELLOW)

    valid_models = []
    for m_alias in models:
        if m_alias in ALL_BASE_MODELS:
            valid_models.append(ALL_BASE_MODELS[m_alias])
        else:
            typer.secho(f"⚠️  Model '{m_alias}' bulunamadı, atlanıyor.", fg=typer.colors.YELLOW)

    if not valid_datasets or not valid_models:
        typer.secho("❌ Geçerli dataset/model kombinasyonu kalmadı.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    total = len(valid_datasets) * len(valid_models)
    typer.secho(f"\n🔥 Toplam {total} kombinasyon hesaplanacak...", fg=typer.colors.CYAN)

    for ds_cfg in valid_datasets:
        for model_cfg in valid_models:
            process_combination(ds_cfg, model_cfg, device, batch_size)

    typer.secho("\n🎉 Tüm görevler başarıyla tamamlandı!", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
