import os
import re
import argparse
import torch
import evaluate
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForCTC, AutoModelForSpeechSeq2Seq
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- ADIM 1: PYTORCH DATASET SINIFI ---
class VoxpopuliPyTorchDataset(Dataset):
    """Hugging Face veri setini saf PyTorch Dataset'ine çeviren sarmalayıcı (wrapper)"""
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        # Encoder matrisini PyTorch Tensörüne çeviriyoruz
        encoder_tensor = torch.tensor(item["encoder_state"], dtype=torch.float32)
        
        return {
            "encoder_state": encoder_tensor,
            "original_text": item["original_text"],
            "transcription": item["transcription"],
            "wer": torch.tensor(item["wer"], dtype=torch.float32),
            "cer": torch.tensor(item["cer"], dtype=torch.float32)
        }

# --- ADIM 2: DATALOADER İÇİN COLLATOR ---
def asr_collate_fn(batch):
    encoder_states = [item["encoder_state"] for item in batch]
    padded_encoder_states = pad_sequence(encoder_states, batch_first=True, padding_value=0.0)
    
    return {
        "encoder_state": padded_encoder_states,
        "original_text": [item["original_text"] for item in batch],
        "transcription": [item["transcription"] for item in batch],
        "wer": torch.stack([item["wer"] for item in batch]),
        "cer": torch.stack([item["cer"] for item in batch])
    }


def main(model_id="facebook/wav2vec2-base-960h"):
    print(f"Başlıyoruz! Seçilen Model: {model_id}")

    model_cache_dir = "models"
    os.makedirs(model_cache_dir, exist_ok=True)

    print("Veri seti yükleniyor...")
    dataset = load_dataset("facebook/voxpopuli", "en_accented", split="test", cache_dir="data/raw")
    dataset = dataset.select(range(100))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=model_cache_dir)
    
    is_whisper = "whisper" in model_id.lower()
    # if is_whisper:
    #     model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, cache_dir=model_cache_dir).to(device)
    if is_whisper:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, cache_dir=model_cache_dir, torch_dtype=torch.float32
        ).to(device)
    else:
        model = AutoModelForCTC.from_pretrained(model_id, cache_dir=model_cache_dir).to(device)

    def extract_features(batch):
        audio_arrays = []
        valid_indices = []
        original_texts = []
        
        for i, audio_data in enumerate(batch["audio"]):
            try:
                arr = audio_data["array"]
                if arr is not None and len(arr) > 0:
                    audio_arrays.append(arr)
                    valid_indices.append(i)
                    original_texts.append(batch["normalized_text"][i])
            except Exception:
                continue

        if not audio_arrays:
            return {"original_text": [], "transcription": [], "encoder_state": [], "wer": [], "cer": []}

        # --- FIX: Whisper needs padding="max_length" to always produce 3000 mel frames ---
        if is_whisper:
            inputs = processor(
                audio_arrays,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",      # Pad to 30s (3000 frames)
                return_attention_mask=True, # So we know which frames are real
            ).to(device)
        else:
            inputs = processor(
                audio_arrays,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).to(device)
        
        with torch.no_grad():
            if is_whisper:
                input_features = inputs.input_features
                # Fix: specify language='en' to avoid the language detection warning/issue
                predicted_ids = model.generate(input_features, language="en", task="transcribe")
                transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
                encoder_outputs = model.get_encoder()(input_features, output_hidden_states=True)
                last_hidden_state = encoder_outputs.hidden_states[-1].cpu().numpy()
            else:
                outputs = model(**inputs, output_hidden_states=True)
                predicted_ids = torch.argmax(outputs.logits, dim=-1)
                transcriptions = processor.batch_decode(predicted_ids)
                last_hidden_state = outputs.hidden_states[-1].cpu().numpy()
        
        final_original, final_transcriptions, final_encoder_states, final_wer, final_cer = ([] for _ in range(5))
        
        for idx, valid_idx in enumerate(valid_indices):
            ref_text = re.sub(r'[^\w\s]', '', original_texts[idx]).lower().strip()
            pred_text = re.sub(r'[^\w\s]', '', transcriptions[idx]).lower().strip()
            
            final_original.append(original_texts[idx])
            final_transcriptions.append(transcriptions[idx])
            
            if len(ref_text) > 0 and len(pred_text) > 0:
                final_wer.append(wer_metric.compute(predictions=[pred_text], references=[ref_text]))
                final_cer.append(cer_metric.compute(predictions=[pred_text], references=[ref_text]))
            else:
                hata = 0.0 if ref_text == pred_text else 1.0
                final_wer.append(hata)
                final_cer.append(hata)
            
            # Trim Whisper's encoder output to actual audio duration
            if is_whisper:
                ses_saniyesi = len(audio_arrays[idx]) / 16000.0
                actual_seq_len = min(int(ses_saniyesi * 50), last_hidden_state.shape[1])
            else:
                actual_seq_len = model._get_feat_extract_output_lengths(len(audio_arrays[idx]))
                
            unpadded_state = last_hidden_state[idx, :actual_seq_len, :]
            final_encoder_states.append(unpadded_state.tolist())

        return {
            "original_text": final_original,
            "transcription": final_transcriptions,
            "encoder_state": final_encoder_states,
            "wer": final_wer,
            "cer": final_cer
        }

    print("Veri işleniyor... (Batch Size: 2)")
    processed_dataset = dataset.map(extract_features, batched=True, batch_size=2, remove_columns=dataset.column_names)

    clean_model_name = model_id.split('/')[-1]
    output_dir = f"data/processed/voxpopuli_asr_features_poc_{clean_model_name}"
    
    print(f"İşlem tamamlandı! Yeni veri seti diske kaydediliyor: {output_dir}")
    processed_dataset.save_to_disk(output_dir)

    # --- ADIM 3: PYTORCH DATASET'İNİ OLUŞTUR VE TEST ET ---
    pytorch_dataset = VoxpopuliPyTorchDataset(processed_dataset)
    
    print("\n--- PYTORCH DATASET HAZIR! ---")
    print(f"Toplam Örnek: {len(pytorch_dataset)}")
    
    dataloader = DataLoader(
        pytorch_dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=asr_collate_fn
    )
    
    ilk_batch = next(iter(dataloader))
    print("\n--- DATALOADER BATCH TESTİ ---")
    print(f"Encoder Tensör Boyutu (Batch Size, Sequence Length, Hidden Dim): {ilk_batch['encoder_state'].shape}")
    print(f"WER Değerleri Tensörü: {ilk_batch['wer']}")
    print(f"CER Değerleri Tensörü: {ilk_batch['cer']}")
    
    return pytorch_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxPopuli ASR feature extractor")
    parser.add_argument(
        "--model-id",
        default="facebook/hubert-large-ls960-ft",
        choices=["facebook/hubert-large-ls960-ft", "openai/whisper-large-v3", "facebook/wav2vec2-base-960h", "microsoft/wavlm-base-plus", "microsoft/wavlm-large", "openai/whisper-base"],
    )
    args = parser.parse_args()

    my_pytorch_dataset = main(model_id=args.model_id)