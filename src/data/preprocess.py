import os
import argparse
import pyarrow.parquet as pq
import pyarrow as pa

def process_dataset(dataset_name, interim_dir="data/interim", processed_dir="data/processed"):
    safe_name = dataset_name.replace("/", "_")
    interim_path = os.path.join(interim_dir, safe_name)
    processed_path = os.path.join(processed_dir, safe_name)
    
    print(f"Loading interim datasets for {dataset_name}...")
    # Use pyarrow to safely bypass HF/Pandas schema parsing issues with nested variable-length arrays
    w_table = pq.read_table(f"{interim_path}/openai_whisper-base.parquet")
    h_table = pq.read_table(f"{interim_path}/facebook_hubert-large-ls960-ft.parquet")
    v_table = pq.read_table(f"{interim_path}/facebook_wav2vec2-base-960h.parquet")
    
    assert w_table.num_rows == h_table.num_rows == v_table.num_rows, "Dataset lengths do not match!"
    
    print("Combining and aligning features...")
    
    # Build new PyArrow table directly from the loaded columns
    arrays = [
        w_table["ground_truth"],
        w_table["openai_whisper-base_embedding"],
        h_table["facebook_hubert-large-ls960-ft_embedding"],
        v_table["facebook_wav2vec2-base-960h_embedding"],
        w_table["openai_whisper-base_wer"],
        h_table["facebook_hubert-large-ls960-ft_wer"],
        v_table["facebook_wav2vec2-base-960h_wer"]
    ]
    
    names = [
        "ground_truth",
        "whisper_features",
        "hubert_features",
        "w2v2_features",
        "whisper_wer",
        "hubert_wer",
        "w2v2_wer"
    ]
    
    combined_table = pa.Table.from_arrays(arrays, names=names)
    
    os.makedirs(processed_path, exist_ok=True)
    out_path = f"{processed_path}/combined_features.parquet"
    print(f"Saving processed dataset to {out_path}...")
    pq.write_table(combined_table, out_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and combine ASR features.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., edinburghcstr/ami)")
    parser.add_argument("--interim_dir", type=str, default="data/interim")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    args = parser.parse_args()
    
    process_dataset(args.dataset, args.interim_dir, args.processed_dir)
