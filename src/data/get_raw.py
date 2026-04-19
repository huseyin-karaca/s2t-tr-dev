from datasets import load_dataset
from src.data.config import ALL_DATASETS # Merkezi yerden alıyoruz

# just runs a dummy load to trigger Hugging Face's caching mechanism. cache_dir (download path is data/raw)
def fetch_data():
    for cfg in ALL_DATASETS:
        print("Yükleniyor:", cfg.name)
        cfg.load()

if __name__ == "__main__":
    fetch_data()