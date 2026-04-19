from dataclasses import dataclass, field
from datasets import load_dataset
from typing import Dict

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

@dataclass
class DatasetConfig:
    name: str
    subset: str
    split_name: str  # 'split' yerine 'split_name' diyelim, karışmasın
    revision: str = "refs/convert/parquet"
    
    # Path logic'i burada kapsüllüyoruz
    @property
    def data_files(self) -> Dict[str, str]:
        return {self.split_name: f"{self.subset}/{self.split_name}/*.parquet"}

    def load(self, cache_dir=RAW_DATA_DIR):
        return load_dataset(
            self.name,
            revision=self.revision,
            data_files=self.data_files,
            cache_dir=cache_dir,
            split=self.split_name # Doğrudan Dataset objesi döner (DatasetDict değil)
        )

# Dataset tanımlamaları
ALL_DATASETS = {
    "ami": DatasetConfig(
        name="edinburghcstr/ami", 
        subset="ihm", 
        split_name="test"
        ),
    "libri": DatasetConfig(
        name="openslr/librispeech_asr", 
        subset="other", 
        split_name="test"
        ),
    "voxpopuli": DatasetConfig(
        name="facebook/voxpopuli",
        subset="en_accented",
        split_name="test"
    ),
}

# # Tek satırda yükleme (Dictionary Comprehension)
# # Anahtar olarak dataset ismini (veya subset'i) kullanabilirsin
# loaded_datasets = {cfg.name: cfg.load() for cfg in ALL_DATASETS}

# # Kullanım:
# # ami_data = loaded_datasets["edinburghcstr/ami"]