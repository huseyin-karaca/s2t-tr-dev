import os
import torch
import librosa
import numpy as np
import pandas as pd
import evaluate
from pathlib import Path
from datasets import load_from_disk, Audio, concatenate_datasets
from transformers import (
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor, # HuBERT processor genelde Wav2Vec2Processor ile aynı çalışır
    HubertForCTC,
    WhisperProcessor, WhisperForConditionalGeneration
)

# 1. KLASÖR YAPILARININ OLUŞTURULMASI
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Cihaz seçimi (GPU varsa çok hızlandırır)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kullanılan cihaz: {device}")

# 2. MODELLERİN VE İŞLEMCİLERİN YÜKLENMESİ
print("Modeller yükleniyor...")
# Wav2Vec2
w2v_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", cache_dir = MODELS_DIR)
w2v_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", cache_dir = MODELS_DIR).to(device)

#bu hata verdi
## HuBERT (Processor olarak Wav2Vec2 processor kullanılır)
#hubert_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960", cache_dir = MODELS_DIR)
#hubert_model = HubertForCTC.from_pretrained("facebook/hubert-base-ls960", cache_dir = MODELS_DIR).to(device)

# HuBERT (Fine-tune edilmiş resmi Large modeli)
hubert_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft",cache_dir=MODELS_DIR)
hubert_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft", cache_dir = MODELS_DIR).to(device)


# Whisper
# whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base", cache_dir = MODELS_DIR)
# whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base", cache_dir = MODELS_DIR).to(device)

# data2vec
data2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft",cache_dir=MODELS_DIR)
data2vec_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft", cache_dir = MODELS_DIR).to(device)


# WER Metriği
wer_metric = evaluate.load("wer")

# Modelleri evaluation moduna alalım
w2v_model.eval()
hubert_model.eval()
whisper_model.eval()

# 3. VERİ SETİNİ YÜKLEME VE HAZIRLAMA
# Not: HF datasetini data/raw içine kaydettiğini varsayıyoruz. 
# Eğer doğrudan indireceksen: load_dataset("PolyAI/minds14", "en-US", cache_dir=RAW_DIR)
print("Veri seti yükleniyor...")

locales = ["en-US", "en-AU", "en-GB"]

print("Veri setleri indiriliyor ve yükleniyor...")

# 1. Her bir aksanı ayrı ayrı yükleyip bir listeye atıyoruz.
# Not: minds14 sadece 'train' split'ine sahip olduğu için doğrudan split="train" diyerek
# DatasetDict yerine Dataset objesi alıyoruz, bu sayede birleştirmesi kolaylaşıyor.
dataset_list = [
    load_dataset("PolyAI/minds14", name=loc, split="train", cache_dir=RAW_DIR) 
    for loc in locales
]

# 2. Listeyi tek bir birleşik Dataset haline getiriyoruz
combined_dataset = concatenate_datasets(dataset_list)
# Tüm modeller 16kHz beklediği için veriyi 16kHz'e resample ediyoruz.
dataset = combined_dataset.cast_column("audio", Audio(sampling_rate=16000))

# Metinleri temizleme fonksiyonu (WER hesabı için önemli)
def clean_text(text):
    import re
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# 4. ANA İŞLEME FONKSİYONU
def process_audio_batch(batch):
    audio_array = batch["audio"]["array"]
    sr = batch["audio"]["sampling_rate"]
    ground_truth = clean_text(batch["transcription"]) # dataset'teki hedefin 'transcription' olduğunu varsayıyoruz
    
    # --- A. MFCC ÇIKARIMI ---
    # 13 MFCC katsayısı çıkarıp zaman ekseninde ortalamasını (pooling) alıyoruz
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1).tolist() 

    with torch.no_grad():
        # --- B. WAV2VEC2 İŞLEMLERİ ---
        w2v_inputs = w2v_processor(audio_array, sampling_rate=sr, return_tensors="pt").to(device)
        w2v_outputs = w2v_model(**w2v_inputs, output_hidden_states=True)
        # Pooling: (Batch, Time, Feature) -> (Feature)
        w2v_emb = w2v_outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy().tolist()
        w2v_pred_ids = torch.argmax(w2v_outputs.logits, dim=-1)
        w2v_transcription = clean_text(w2v_processor.batch_decode(w2v_pred_ids)[0])

        # --- C. HUBERT İŞLEMLERİ ---
        hubert_inputs = hubert_processor(audio_array, sampling_rate=sr, return_tensors="pt").to(device)
        hubert_outputs = hubert_model(**hubert_inputs, output_hidden_states=True)
        # Pooling
        hubert_emb = hubert_outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy().tolist()
        hubert_pred_ids = torch.argmax(hubert_outputs.logits, dim=-1)
        hubert_transcription = clean_text(hubert_processor.batch_decode(hubert_pred_ids)[0])

        # --- D. WHISPER İŞLEMLERİ ---
        whisper_inputs = whisper_processor(audio_array, sampling_rate=sr, return_tensors="pt").to(device)
        input_features = whisper_inputs.input_features
        # Encoder özelliklerini alma
        whisper_encoder_outputs = whisper_model.get_encoder()(input_features)
        # Pooling
        whisper_emb = whisper_encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
        # Transcription için generate
        whisper_pred_ids = whisper_model.generate(input_features)
        whisper_transcription = clean_text(whisper_processor.batch_decode(whisper_pred_ids, skip_special_tokens=True)[0])

    # --- E. WER HESAPLAMA ---
    # Eğer ground_truth boşsa hata vermemesi için küçük bir kontrol
    if len(ground_truth.strip()) > 0:
        w2v_wer = wer_metric.compute(predictions=[w2v_transcription], references=[ground_truth])
        hubert_wer = wer_metric.compute(predictions=[hubert_transcription], references=[ground_truth])
        # whisper_wer = wer_metric.compute(predictions=[whisper_transcription], references=[ground_truth])
        data2vec_wer = wer_metric.compute(predictions=[data2vec_transcription], references=[ground_truth])
    else:
        w2v_wer, hubert_wer, data2vec_wer = 0.0, 0.0, 0.0

    # Sonuçları batch'e ekle
    batch["w2v_embedding"] = w2v_emb
    batch["w2v_transcription"] = w2v_transcription
    batch["w2v_wer"] = w2v_wer

    batch["hubert_embedding"] = hubert_emb
    batch["hubert_transcription"] = hubert_transcription
    batch["hubert_wer"] = hubert_wer

    batch["data2vec_embedding"] = data2vec_emb
    batch["data2vec_transcription"] = data2vec_transcription
    batch["data2vec_wer"] = data2vec_wer

    batch["whisper_embedding"] = whisper_emb
    batch["whisper_transcription"] = whisper_transcription
    batch["whisper_wer"] = whisper_wer

    batch["mfcc_features"] = mfccs_mean

    return batch

# 5. DATASET'E UYGULAMA VE KAYDETME
print("Özellikler çıkarılıyor... (Bu işlem veri boyutuna ve GPU'ya göre zaman alabilir)")
# Tüm veri seti üzerinde fonksiyonu çalıştır (Audio datasını da bellekte tuttuğu için işlem uzun sürebilir)
processed_dataset = dataset.map(process_audio_batch, remove_columns=[])

print(f"Tam işlenmiş veri seti (Interim) kaydediliyor: {INTERIM_DIR}")
# Data/interim'e ses dosyaları, modellerin çıktıları dahil her şeyi HF formatında kaydet
processed_dataset.save_to_disk(INTERIM_DIR)

print("Data interim'e kaydedildi!")