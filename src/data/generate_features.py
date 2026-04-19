import os
import re
import torch
import evaluate
from datasets import Audio
from typing import List
import typer 

# Projenin kendi modüllerini import ediyoruz
# (Sözlük oldukları ve key'lerin alias, value'ların config class olduğu varsayımıyla)
from src.data.config import ALL_DATASETS
from src.base_models.config import ALL_BASE_MODELS

app = typer.Typer(help="Çoklu Dataset ve Model için çıktı (embedding/wer) üretir.")

def process_combination(ds_cfg, model_cfg, device: str):
    """Tek bir dataset + model kombinasyonunu işler."""
    print(f"\n🚀 İşlem Başlıyor: Dataset [ {ds_cfg.name} ] ⬇️ Model [ {model_cfg.name} ]")
    
    # 1. Ham Veriyi Yükle
    ds = ds_cfg.load()
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    
    # 2. Model ve Metrik Hazırlığı
    loaded_model, processor = model_cfg.load(device=device)
    wer_metric = evaluate.load("wer")
    
    def clean_text(text):
        if text is None: return ""
        return re.sub(r'[^\w\s]', '', str(text).lower()).strip()

    # Model isimlerinde "/" veya geçersiz karakter varsa dosya ve kolon isimleri için temizle
    safe_model_name = model_cfg.name.replace("/", "_")

    # 3. Batch İşleme Fonksiyonu
    def process_batch(batch):
        results_list = []
        for i in range(len(batch["audio"])):
            audio_data = batch["audio"][i]
            target = clean_text(batch.get("text", batch.get("transcription", [""])[i]))
            
            # Model çıkarımı
            res = model_cfg.predict(
                loaded_model, processor, 
                audio_data["array"], 
                audio_data["sampling_rate"], 
                device
            )
            
            # WER Hesapla
            pred = clean_text(res["transcription"])
            wer = wer_metric.compute(predictions=[pred], references=[target]) if target else 0.0
            
            # Sütun isimlerinde de alias yerine safe_model_name kullanıyoruz ki çakışma olmasın
            results_list.append({
                f"{safe_model_name}_embedding": res["embedding"],
                f"{safe_model_name}_transcription": pred,
                f"{safe_model_name}_wer": wer
            })
            
        return {k: [dic[k] for dic in results_list] for k in results_list[0]}

    # 4. Çıkarım (Inference)
    feature_ds = ds.map(
        process_batch, 
        batched=True, 
        batch_size=1, 
        remove_columns=ds.column_names,
        desc=f"Running {model_cfg.name} inference"
    )

    # 5. Kayıt İşlemi (Alias YERİNE Name kullanılıyor)
    # ds_cfg.name ve model_cfg.name (temizlenmiş haliyle) path'te kullanıldı
    safe_ds_name = ds_cfg.name.replace("/", "_")
    output_dir = os.path.join(ds_cfg.interim_path, safe_ds_name, safe_model_name)
    
    os.makedirs(output_dir, exist_ok=True)
    feature_ds.save_to_disk(output_dir)
    typer.secho(f"✅ Kombinasyon tamamlandı, buraya kaydedildi: {output_dir}", fg=typer.colors.GREEN)


@app.command()
def generate_features(
    datasets: List[str] = typer.Option(..., "--dataset", "-d", help="Dataset alias (Örn: -d ds1 -d ds2)"),
    models: List[str] = typer.Option(..., "--model", "-m", help="Model alias (Örn: -m w2v2 -m whisper)"),
    cpu: bool = typer.Option(False, "--cpu", help="GPU yerine CPU kullan")
):
    """
    Verilen tüm dataset ve model kombinasyonları için sırayla özellikleri üretir.
    """
    device = "cpu" if cpu or not torch.cuda.is_available() else "cuda"
    print(f"⚙️  Cihaz: {device.upper()}")

    # 1. Gelen alias'ları sözlüklerden (dictionary) doğrula ve config objelerini topla
    valid_datasets = []
    for d_alias in datasets:
        if d_alias in ALL_DATASETS:
            valid_datasets.append(ALL_DATASETS[d_alias])
        else:
            typer.secho(f"⚠️ Uyarı: Dataset '{d_alias}' sözlükte bulunamadı, atlanıyor.", fg=typer.colors.YELLOW)

    valid_models = []
    for m_alias in models:
        if m_alias in ALL_BASE_MODELS:
            valid_models.append(ALL_BASE_MODELS[m_alias])
        else:
            typer.secho(f"⚠️ Uyarı: Model '{m_alias}' sözlükte bulunamadı, atlanıyor.", fg=typer.colors.YELLOW)

    # İşlem yapılacak geçerli eleman kalmadıysa çık
    if not valid_datasets or not valid_models:
        typer.secho("❌ Hata: Çalıştırılacak geçerli dataset/model kombinasyonu kalmadı.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Toplam iş miktarını göster
    total_combinations = len(valid_datasets) * len(valid_models)
    typer.secho(f"\n🔥 Toplam {total_combinations} farklı kombinasyon hesaplanacak...", fg=typer.colors.CYAN)

    # 2. Cartesian Product (Tüm eşleşmeler)
    for ds_cfg in valid_datasets:
        for model_cfg in valid_models:
            process_combination(ds_cfg, model_cfg, device)

    typer.secho("\n🎉 Tüm görevler başarıyla tamamlandı!", fg=typer.colors.GREEN, bold=True)

if __name__ == "__main__":
    app()