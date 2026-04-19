import os
import argparse
from dataclasses import dataclass
from typing import Optional, Dict
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download

# @dataclass
# class DatasetConfig:
#     output_filename: str
#     repo_id: str
#     config_name: str
#     split: str
#     # data_files is optional. Used for bypassing HF's default scripts (like our LibriSpeech fix)
#     data_files: Optional[Dict[str, str]] = None 






if __name__ == "__main__":

    # repo = "edinburghcstr/ami"
    # rev = "refs%2Fconvert%2Fparquet"
    # cfg = "ihm"

    # total = 4
    # urls = [
    #     f"https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/{cfg}/test/{i:05d}.parquet"
    #     for i in range(total)
    # ]

    # print(urls)

    

    snapshot_download(
        repo_id="openslr/librispeech_asr",
        repo_type="dataset",
        revision="refs/convert/parquet",
        allow_patterns=["all/test.other/*.parquet"],
        local_dir="./data/raw/librispeech_test_other",
    )

    # ds_stream = load_dataset(
    #     "edinburghcstr/ami", "ihm",
    #     split="test",
    #     streaming=True,
    # )

    # ds = Dataset.from_list(list(ds_stream))
    # ds.save_to_disk("data/raw/ami_ihm_test")

    # parser = argparse.ArgumentParser(description="Download raw datasets and save them as consolidated Parquet files.")
    
    # # Allow the user to specify one dataset, or use 'all'
    # parser.add_argument(
    #     "--dataset", 
    #     type=str, 
    #     required=True,
    #     choices=list(DATASET_REGISTRY.keys()) + ["all"],
    #     help="The registry key of the dataset to download, or 'all' to download everything."
    # )
    
    # parser.add_argument(
    #     "--output_dir", 
    #     type=str, 
    #     default="data/raw",
    #     help="Target directory for the downloaded parquet files."
    # )
    
    # args = parser.parse_args()

    # print(args)
    
    # if args.dataset == "all":
    #     for key in DATASET_REGISTRY.keys():
    #         download_raw_data(key, args.output_dir)
    # else:
    #     download_raw_data(args.dataset, args.output_dir)


# "test-00000-of-00004.parquet"
# "test-00001-of-00004.parquet"
# "test-00002-of-00004.parquet"
# "test-00003-of-00004.parquet"

