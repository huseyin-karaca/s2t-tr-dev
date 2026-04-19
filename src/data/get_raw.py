from datasets import load_dataset
from src.data.config import ALL_DATASETS # Merkezi yerden alıyoruz

import typer
from loguru import logger
from typing import Annotated

app = typer.Typer()


# just runs a dummy load to trigger Hugging Face's caching mechanism. cache_dir (download path is data/raw)

@app.command()
def fetch_data(
  dataset: list[str] = None
):
  logger.info(dataset)
  DATASETS = [ALL_DATASETS[name] for name in dataset if name in ALL_DATASETS]
  logger.info(f"{DATASETS}")

  for cfg in DATASETS:
      print("Yükleniyor:", cfg.name)
      cfg.load()

if __name__ == "__main__":
    app()