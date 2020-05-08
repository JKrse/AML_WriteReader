
# ============================================================================================================
import pandas as pd 
from pathlib import Path
from config import *
import numpy as np

# ============================================================================================================

config = Config()

file_name = "ATEL evaluation data - Evaluation_v1_with_images.tsv"
path_data = config.local_path_temp / "data"

file_path = path_data / file_name
# file_path = config.local_path / "data" / file_name

df = pd.read_csv(file_path, sep="\t", header=1)
df = df[~df["BadProposal"].isnull()]

data = df.to_dict("list")

# Convert to pickle: 
np.savez(path_data/"test_data.npz", **data)

print(f"tsv has been converted to npz \n Samples: \n {df.count()}")

# df.to_pickle(path_data/"test_data.npz")
# np.load(f"{str(path_data)}/test_data.npz",allow_pickle=True)
