from dataclasses import dataclass
from typing import Any, Dict, List, Type
import torch
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor,
    Wav2Vec2ForCTC, Wav2Vec2Processor,
    HubertForCTC, Data2VecAudioForCTC,
)


@dataclass
class BaseModelConfig:
    alias: str
    model_id: str
    model_class: Type
    processor_class: Type
    model_type: str = "ctc"  # "ctc" or "whisper"

    def load(self, device: str, cache_dir: str = "models"):
        processor = self.processor_class.from_pretrained(self.model_id, cache_dir=cache_dir)
        model = self.model_class.from_pretrained(self.model_id, cache_dir=cache_dir).to(device)
        model.eval()
        return model, processor

    def predict(self, model, processor, audio_array, sr, device):
        """Tek örnek — batch metoduna yönlendirir."""
        return self.predict_batch(model, processor, [audio_array], [sr], device)[0]

    def predict_batch(self, model, processor, audio_arrays: List, srs: List, device: str):
        """
        Batch inference. Padding yapılır ama masked mean ile embedding
        kalitesi tek-örnek çıktısıyla birebir aynı tutulur.
        """
        with torch.no_grad():
            if self.model_type == "whisper":
                inputs = processor(
                    audio_arrays,
                    sampling_rate=srs[0],       # Whisper sabit 16 kHz bekler
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                input_features = inputs.input_features  # (B, 80, T)

                # Embedding: encoder temporal ortalaması
                encoder_out = model.get_encoder()(input_features)
                embeddings = encoder_out.last_hidden_state.mean(dim=1)  # (B, D)

                # Transcription
                predicted_ids = model.generate(input_features)
                transcriptions = processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )

            else:
                # CTC: Wav2Vec2, HuBERT, Data2Vec
                inputs = processor(
                    audio_arrays,
                    sampling_rate=srs[0],
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=True,
                ).to(device)

                outputs = model(**inputs, output_hidden_states=True)

                # attention_mask raw sample uzunluğunda → frame uzunluğuna indir
                input_lengths = inputs.attention_mask.sum(dim=-1)           # (B,) — her örneğin gerçek sample sayısı
                output_lengths = model._get_feat_extract_output_lengths(input_lengths)  # (B,) — frame sayısı

                hidden = outputs.hidden_states[-1]                          # (B, T_frames, D)
                B, T, D = hidden.shape

                # Frame maskesi oluştur: her örnek için ilk output_lengths[i] frame geçerli
                frame_mask = torch.arange(T, device=hidden.device).unsqueeze(0) < output_lengths.unsqueeze(1)
                frame_mask = frame_mask.unsqueeze(-1).float()               # (B, T, 1)

                embeddings = (hidden * frame_mask).sum(dim=1) / frame_mask.sum(dim=1)  # (B, D)

                predicted_ids = torch.argmax(outputs.logits, dim=-1)
                transcriptions = processor.batch_decode(predicted_ids)

        return [
            {
                "embedding": embeddings[i].cpu().numpy().tolist(),
                "transcription": transcriptions[i],
            }
            for i in range(len(audio_arrays))
        ]


# --- MODEL KATALOĞU ---
WAV2VEC2_BASE = BaseModelConfig(
    alias="w2v2",
    model_id="facebook/wav2vec2-base-960h",
    model_class=Wav2Vec2ForCTC,
    processor_class=Wav2Vec2Processor,
)

HUBERT_LARGE = BaseModelConfig(
    alias="hubert",
    model_id="facebook/hubert-large-ls960-ft",
    model_class=HubertForCTC,
    processor_class=Wav2Vec2Processor,
)

DATA2VEC_BASE = BaseModelConfig(
    alias="d2v",
    model_id="facebook/data2vec-audio-base-960h",
    model_class=Data2VecAudioForCTC,
    processor_class=Wav2Vec2Processor,
)

WHISPER_BASE = BaseModelConfig(
    alias="whisper",
    model_id="openai/whisper-base",
    model_class=WhisperForConditionalGeneration,
    processor_class=WhisperProcessor,
    model_type="whisper",
)

ALL_BASE_MODELS = {
    "w2v2":    WAV2VEC2_BASE,
    "hubert":  HUBERT_LARGE,
    "d2v":     DATA2VEC_BASE,
    "whisper": WHISPER_BASE,
}
