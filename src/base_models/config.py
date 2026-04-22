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
        """Single example — routes to the batch method."""
        return self.predict_batch(model, processor, [audio_array], [sr], device)[0]

    def predict_batch(self, model, processor, audio_arrays: List, srs: List, device: str):
        """
        Batch inference. Padding is applied for batch processing, but valid 
        frame-level embeddings are extracted directly without padding contamination.
        """
        with torch.no_grad():
            if self.model_type == "whisper":
                inputs = processor(
                    audio_arrays,
                    sampling_rate=srs[0],       # Whisper expects exactly 16 kHz
                    return_tensors="pt",
                    max_length=480000,
                    padding="max_length",
                ).to(device)

                input_features = inputs.input_features  # strictly  (B, 80, 3000)

                # Get frame-level sequence from the encoder
                encoder_out = model.get_encoder()(input_features)
                hidden = encoder_out.last_hidden_state  # (B, T_frames, D) -> normally (B, 1500, D)

                # Whisper processes at 100 frames/sec for mel spec, and the encoder 
                # has a stride of 2 (50 frames/sec). So 1 frame = 20ms = 320 samples at 16kHz.
                # We calculate the true valid frame length to remove Whisper's 30-second padding.
                audio_lengths = [len(a) for a in audio_arrays]
                valid_lengths = [min(hidden.shape[1], l // 320) for l in audio_lengths]

                # Slice to get variable-length frame arrays (float 16 yaptım)
                embeddings_list = [
                    hidden[i, :valid_lengths[i], :].half().cpu().numpy().tolist()
                    for i in range(len(audio_arrays))
                ]

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

                # Calculate valid lengths to slice off padding
                input_lengths = inputs.attention_mask.sum(dim=-1)                       # (B,) — true sample count
                output_lengths = model._get_feat_extract_output_lengths(input_lengths)  # (B,) — true frame count

                hidden = outputs.hidden_states[-1]                                      # (B, max_T_frames, D)

                # Slice padding for each batch item individually
                embeddings_list = [
                    hidden[i, :output_lengths[i], :].half().cpu().numpy().tolist()
                    for i in range(len(audio_arrays))
                ]

                # Transcription
                predicted_ids = torch.argmax(outputs.logits, dim=-1)
                transcriptions = processor.batch_decode(predicted_ids)

        return [
            {
                "embedding": embeddings_list[i],
                "transcription": transcriptions[i],
            }
            for i in range(len(audio_arrays))
        ]


# --- MODEL CATALOG ---
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
