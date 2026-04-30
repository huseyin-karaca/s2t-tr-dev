import os
import gdown
import typer
from loguru import logger

app = typer.Typer(help="Download processed datasets from Google Drive")

# Registry of known processed datasets and their Google Drive IDs
DATASETS = {
    "ami": {
        "file_id": "1FgN4FQ422HPDCwG-Wl4jrXQGl0tgX6ZP",
        "output_path": "data/processed/edinburghcstr_ami/combined_features_with_transcripts.parquet"
    },
    "voxpopuli": {
        "file_id": "1yf-G-DWhhZLlqeGXZ77GbuhTBmhqyyXA",
        "output_path": "data/processed/facebook_voxpopuli/combined_features_with_transcripts.parquet"
    }
    # Future datasets can be added here
    # "libri": { ... }
}

@app.command()
def fetch(
    dataset: str = typer.Option(..., "-d", "--dataset", help="Name of the dataset to download (e.g., ami)"),
    output_path: str = typer.Option(None, "-o", "--output", help="Override the default output path"),
    file_id: str = typer.Option(None, "-f", "--file-id", help="Override the default Google Drive file ID")
):
    """
    Downloads the processed dataset features and transcripts from Google Drive.
    """
    dataset = dataset.lower()
    
    if dataset not in DATASETS and not (output_path and file_id):
        logger.error(f"Dataset '{dataset}' is not registered. Available datasets: {list(DATASETS.keys())}")
        logger.error("If you want to download an unregistered dataset, you must provide both -o/--output and -f/--file-id.")
        raise typer.Exit(code=1)
        
    # Get config, fallback to empty dict if not registered (in which case overrides must be present)
    cfg = DATASETS.get(dataset, {})
    
    target_file_id = file_id or cfg.get("file_id")
    target_output_path = output_path or cfg.get("output_path")
    
    if not target_file_id or not target_output_path:
        logger.error("Missing file ID or output path.")
        raise typer.Exit(code=1)
    
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(target_output_path), exist_ok=True)
    
    logger.info(f"Downloading processed {dataset.upper()} dataset to {target_output_path}...")
    
    # Download using gdown
    gdown.download(id=target_file_id, output=target_output_path, quiet=False)
    
    if os.path.exists(target_output_path):
        logger.info(f"Download completed successfully: {target_output_path}")
    else:
        logger.error("Download failed or file was not saved correctly.")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
