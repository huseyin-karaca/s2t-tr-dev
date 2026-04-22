import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

class ASRFeatureDataset(Dataset):
    """Dataset for loading parallel ASR model embeddings from parquet files."""
    def __init__(self, dataset_name, data_dir="data/interim", split="train"):
        # Format dataset name to match folder structure (e.g., edinburghcstr/ami -> edinburghcstr_ami)
        safe_name = dataset_name.replace("/", "_")
        base_path = os.path.join(data_dir, safe_name)
        
        # Load datasets (HF load_dataset returns a DatasetDict, we take the default split 'train')
        self.whisper_ds = load_dataset("parquet", data_files=f"{base_path}/openai_whisper-base.parquet", split="train")
        self.hubert_ds = load_dataset("parquet", data_files=f"{base_path}/facebook_hubert-large-ls960-ft.parquet", split="train")
        self.w2v2_ds = load_dataset("parquet", data_files=f"{base_path}/facebook_wav2vec2-base-960h.parquet", split="train")
        
        assert len(self.whisper_ds) == len(self.hubert_ds) == len(self.w2v2_ds), "Dataset lengths do not match!"
        
    def __len__(self):
        return len(self.whisper_ds)
        
    def __getitem__(self, idx):
        # Assuming the features are stored in a column named 'features' or similar.
        # Adjust the key based on your actual parquet schema.
        w_feat = torch.tensor(self.whisper_ds[idx]["features"])
        h_feat = torch.tensor(self.hubert_ds[idx]["features"])
        v_feat = torch.tensor(self.w2v2_ds[idx]["features"])
        
        # TODO: Add logic to fetch the target labels (like WER) if they exist in the parquet
        # labels = torch.tensor(self.whisper_ds[idx]["labels"])
        
        return {
            "whisper": w_feat,
            "hubert": h_feat,
            "wav2vec2": v_feat
            # "labels": labels
        }

def collate_fn(batch):
    """Pads sequences in a batch to the maximum length."""
    whisper_batch = [item["whisper"] for item in batch]
    hubert_batch = [item["hubert"] for item in batch]
    w2v2_batch = [item["wav2vec2"] for item in batch]
    
    whisper_padded = pad_sequence(whisper_batch, batch_first=True)
    hubert_padded = pad_sequence(hubert_batch, batch_first=True)
    w2v2_padded = pad_sequence(w2v2_batch, batch_first=True)
    
    batch_dict = {
        "whisper": whisper_padded,
        "hubert": hubert_padded,
        "wav2vec2": w2v2_padded
    }
    
    # if "labels" in batch[0]:
    #     batch_dict["labels"] = torch.stack([item["labels"] for item in batch])
        
    return batch_dict
