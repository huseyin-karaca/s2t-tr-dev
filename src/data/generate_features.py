import os
import argparse
import torch
import evaluate
from datasets import load_from_disk, Audio
from typing import Optional

# Projenin kendi modüllerini import ediyoruz
from src.data.config import ALL_DATASETS, DatasetConfig
from src.base_models.config import ACTIVE_MODELS, BaseModelConfig

def generate_features(ds_cfg: DatasetConfig, model_cfg: BaseModelConfig, device: str = "cuda"):
    """
    Belirtilen dataset ve model için encoding/wer hesaplar ve interim'e kaydeder.
    """
    print(f"\n🚀 İşlem Başlıyor: {ds_cfg.name} ({ds_cfg.subset}) ⬇️ Model: {model_cfg.alias}")
    
    # 1. Ham Veriyi Yükle
    # if not os.path.exists(ds_cfg.raw_path):
    #     raise FileNotFoundError(f"Ham veri bulunamadı: {ds_cfg.raw_path}. Önce get_raw.py çalıştırılmalı.")
    
    ds = ds_cfg.load()
    
    # Tüm modeller 16kHz beklediği için garantiye alıyoruz
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    
    # 2. Model ve Metrik Hazırlığı
    model, processor = model_cfg.load(device=device)
    wer_metric = evaluate.load("wer")
    
    # Temizlik fonksiyonu (WER için)
    def clean_text(text):
        import re
        if text is None: return ""
        return re.sub(r'[^\w\s]', '', str(text).lower()).strip()

    # 3. Batch İşleme Fonksiyonu
    def process_batch(batch):
        # NOT: datasets.map(batched=True) içinde batch bir sözlüktür (listeler içerir)
        # Ancak bizim model_cfg.predict tekil örnek bekliyor (şimdilik basitleştirilmiş hali)
        
        results_list = []
        for i in range(len(batch["audio"])):
            audio_data = batch["audio"][i]
            # Dataset'teki hedef metin kolonun ismine göre burayı güncelle (örn: 'text' veya 'sentence')
            target = clean_text(batch.get("text", batch.get("transcription", [""])[i]))
            
            # Model çıkarımı
            res = model_cfg.predict(
                model, processor, 
                audio_data["array"], 
                audio_data["sampling_rate"], 
                device
            )
            
            # WER Hesapla
            pred = clean_text(res["transcription"])
            wer = wer_metric.compute(predictions=[pred], references=[target]) if target else 0.0
            
            results_list.append({
                f"{model_cfg.alias}_embedding": res["embedding"],
                f"{model_cfg.alias}_transcription": pred,
                f"{model_cfg.alias}_wer": wer
            })
            
        # Listeyi sözlük formatına geri döndür (HF Map standardı)
        return {k: [dic[k] for dic in results_list] for k in results_list[0]}

    # 4. Çıkarım (Inference) - Sadece yeni kolonları oluştur
    # remove_columns ile ham veriyi siliyoruz çünkü raw_path'te zaten varlar. 
    # Sadece yeni feature'ları kaydederek diskten %90 tasarruf ediyoruz.
    feature_ds = ds.map(
        process_batch, 
        batched=True, 
        batch_size=1, # Bellek yönetimi için 1 (Gerekirse artırılabilir)
        remove_columns=ds.column_names,
        desc=f"Running {model_cfg.alias} inference"
    )

    # 5. Verimli Kayıt
    output_dir = os.path.join(ds_cfg.interim_path, model_cfg.alias)
    feature_ds.save_to_disk(output_dir)
    print(f"✅ Özellikler kaydedildi: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Base model çıktılarını (embedding/wer) üretir.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset alias (örn: AMI_IHM_TEST)")
    parser.add_argument("--model", type=str, required=True, help="Model alias (örn: w2v2)")
    parser.add_argument("--cpu", action="store_true", help="GPU yerine CPU kullan")
    
    args = parser.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    # Config'leri bul
    ds_map = {d.name.replace("/", "_") + "_" + d.subset: d for d in ALL_DATASETS}
    # Basitleştirmek için config dosyasındaki değişken isimlerini eşleştirebilirsin:
    # (Bu kısım senin config.py'daki instance isimlerine göre düzenlenmeli)
    
    # Örnek seçici:
    selected_ds = next((d for d in ALL_DATASETS if args.dataset in d.name or args.dataset in d.subset), None)
    selected_model = next((m for m in ACTIVE_MODELS if m.alias == args.model), None)

    if not selected_ds or not selected_model:
        print(f"❌ Hata: Dataset ({args.dataset}) veya Model ({args.model}) bulunamadı.")
        return

    generate_features(selected_ds, selected_model, device)

if __name__ == "__main__":
    main()