import pyarrow.parquet as pq

path = "data/processed/edinburghcstr_ami/combined_features_with_transcripts.parquet"
table = pq.read_table(path, columns=["hubert_features"])
chunked_arr = table["hubert_features"].combine_chunks()

print("Chunked Array Type:", chunked_arr.type)
print("Outer Offsets length:", len(chunked_arr.offsets))

inner_arr = chunked_arr.values
print("Inner Array Type:", inner_arr.type)
print("Inner Offsets length:", len(inner_arr.offsets))

flat_floats = inner_arr.values
print("Flat Floats Type:", flat_floats.type)
print("Flat Floats length:", len(flat_floats))

import numpy as np
outer_offsets = chunked_arr.offsets.to_numpy()
inner_offsets = inner_arr.offsets.to_numpy()
flat_np = flat_floats.to_numpy()

print(f"First clip has frames: {outer_offsets[0]} to {outer_offsets[1]}")
print(f"Dimension of first frame: {inner_offsets[1] - inner_offsets[0]}")
