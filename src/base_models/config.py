from dataclasses import dataclass
from typing import Any, Dict, Type
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, HubertForCTC, Data2VecAudioForCTC

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
        """Different inference according to the model type."""
        with torch.no_grad():
            if self.model_type == "whisper":
                # --- WHISPER ÖZEL ---
                inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt").to(device)
                input_features = inputs.input_features
                
                # Encoder Output (Embedding)
                encoder_outputs = model.get_encoder()(input_features)
                embedding = encoder_outputs.last_hidden_state.mean(dim=1).squeeze()
                
                # Decoder Generation (Transcription)
                predicted_ids = model.generate(input_features)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
            else:
                # --- CTC MODELS (Wav2Vec2, HuBERT, Data2Vec) ---
                inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt").to(device)
                outputs = model(**inputs, output_hidden_states=True)
                
                # Last Hidden State (Embedding)
                embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
                
                # Argmax Decoding (Transcription)
                predicted_ids = torch.argmax(outputs.logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]

        return {
            "embedding": embedding.cpu().numpy().tolist(),
            "transcription": transcription
        }
    



# --- MODEL KATALOĞUNUZ ---
WAV2VEC2_BASE = BaseModelConfig(
    alias="w2v2",
    model_id="facebook/wav2vec2-base-960h",
    model_class=Wav2Vec2ForCTC,
    processor_class=Wav2Vec2Processor
)

HUBERT_LARGE = BaseModelConfig(
    alias="hubert",
    model_id="facebook/hubert-large-ls960-ft",
    model_class=HubertForCTC,
    processor_class=Wav2Vec2Processor
)

DATA2VEC_BASE = BaseModelConfig(
    alias="d2v",
    model_id="facebook/data2vec-audio-base-960h",
    model_class=Data2VecAudioForCTC,
    processor_class=Wav2Vec2Processor
)

WHISPER_BASE = BaseModelConfig(
    alias="whisper",
    model_id="openai/whisper-base",
    model_class=WhisperForConditionalGeneration,
    processor_class=WhisperProcessor,
    model_type="whisper" # <--- Kritik fark
)

ACTIVE_MODELS = [WAV2VEC2_BASE, HUBERT_LARGE, DATA2VEC_BASE, WHISPER_BASE]