import pandas as pd
import numpy as np
import re
import argparse
from config import *
from discriminator import *
import os
import glob
import copy

# ============================================================================================================

parser = argparse.ArgumentParser(description='Get test results')
parser.add_argument(
        '--name', type=str, default="proposals",
        help='Name of the method used')
parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='threshold for when model is confused')
# parser.add_argument(
#         '--model_architecture', type=str, default="mlp_1_img_1_512_0",
#         help='Name of model architecture used')

args = parser.parse_args()

# ============================================================================================================

local_path = Config.local_path
name = args.name
data_path = f"{local_path}/data/{name}"

# ============================================================================================================
exp_name = f"{name}_scoring"
txt_file = f"{local_path}/{exp_name}/test/*.txt"

output_path = f"{local_path}/test_examples"

if not os.path.exists(output_path):
    os.mkdir(output_path)

models = glob.glob(f"{txt_file}")

threshold = args.threshold

for model in models:
    test_results = pd.read_csv(model, sep="\t", engine='python')

    idx = []
    scores = []
    cat = []
    for i in range(len(test_results)):
        all_idx = [int(x) for x in re.findall("\d{1,4}", test_results["idx_batch"][i])]
        idx += all_idx

        all_scores = [float(x) for x in re.findall(".\.\d{1,6}", test_results["score"][i])]
        scores += all_scores

        cat_long = [test_results["cat"][i]] * len(all_idx)
        cat += cat_long

    data = pd.DataFrame([idx, scores, cat]).T
    data.columns = ["idx", "score", "cat"]

    mc = data[data["cat"] == 0]
    mc_hard = mc[mc["score"] > threshold]["idx"]


    [data_train, data_val, data_test, word_embedding] = data_loader(
            data_path, use_mc_samples=False)

    word_to_idx = data_train[f'word_to_idx']
    idx_to_word = dict((v, k) for k, v in word_to_idx.items())

    filename = data_val['file_names']

    examples = []

    for j in range(len(mc_hard)):
        example = copy.deepcopy(data_val['captions']['dis'][filename[int(mc_hard.iloc[j])]][name])
        sent = " ".join(pd.Series(example).map(idx_to_word))
        examples.append(sent)

    model_name = os.path.splitext(os.path.split(model)[1])[0]

    with open(f'{output_path}/{model_name}.txt', 'w', encoding='utf-8') as f:
        for item in examples:
            f.write(f"{item} \n")
    f.close()