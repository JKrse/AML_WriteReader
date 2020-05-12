import pandas as pd
import numpy as np
import re
import argparse
from config import *
from discriminator import *
import os
import glob
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy import stats

# ==============================================================================

local_path = Config.local_path_temp
name = "proposals"
data_path = f"{local_path}/data/{name}"
threshold = 0.5

# ==============================================================================

parser = argparse.ArgumentParser(description='Get test results')
parser.add_argument(
        '--name', type=str, default="proposals",
        help='Name of the method used')
parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='threshold for when model is confused')

args = parser.parse_args()

# ==============================================================================

local_path = Config.local_path
name = args.name
data_path = f"{local_path}/data/{name}"
threshold = args.threshold

save_fig = True

# ==============================================================================
exp_name = f"{name}_scoring"
txt_file = f"{local_path}/{exp_name}/test/*.txt"

output_path = f"{local_path}/test_examples"

if not os.path.exists(output_path):
    os.mkdir(output_path)

models = glob.glob(f"{txt_file}")

[data_train, data_val, _, _] = data_loader(
    data_path, use_mc_samples=False)

train_file = os.path.join(data_path, f"data_train_full_{Config.vocab_size}.npy")
val_file = os.path.join(data_path, f"data_val_full_{Config.vocab_size}.npy")

data_train = np.load(train_file, allow_pickle=True).item()
data_val = np.load(val_file, allow_pickle=True).item()

word_to_idx = data_train[f'word_to_idx']
idx_to_word = dict((v, k) for k, v in word_to_idx.items())

filename = data_val['file_names']

def get_examples(df, kind, category):
    examples = []
    f = open(f'{output_path}/{model_name}_{kind}.txt', 'w', encoding='utf-8')

    for j in range(len(df)):
        example = copy.deepcopy(data_val['captions']['dis'][filename[int(df.iloc[j]["idx"])]][category])
        sent = " ".join(pd.Series(example).map(idx_to_word))
        examples.append(sent)

        f.write(f"{sent}, \t")
        f.write(f"{df.iloc[j]['score']} \n")

    f.close()


# ==============================================================================

fontsize = 20
minus_font = 4
bins = 40
quality = 1200

model_name = os.path.splitext(os.path.split(model)[1])[0]

for model in models:
    test_results = pd.read_csv(model, sep="\t", engine='python')

    idx = []
    scores = []
    cat = []
    for i in range(len(test_results)):
        all_idx = [int(x) for x in re.findall("\d{1,4}", test_results["idx_batch"][i])]
        idx += all_idx

        all_scores = [float(x) for x in re.findall(".\.\d{1,16}e-\d{1,2}|.\.\d{1,16}", test_results["score"][i])]
        scores += all_scores

        cat_long = [test_results["cat"][i]] * len(all_idx)
        cat += cat_long

    bp = data_val["BadProposal"] + data_val["BadProposal"]

    data = pd.DataFrame([idx, scores, cat, bp]).T
    data.columns = ["idx", "score", "cat", "bp"]

    mc = data[data["cat"] == 0]
    human = data[data["cat"] == 1]

    # plt.figure()
    # sns.distplot(human["score"], hist=True, norm_hist=True)
    # sns.distplot(mc["score"], hist=True, norm_hist=True)
    # plt.legend(["Human", "Proposals"])
    # plt.xlim(0,1)
    # plt.xlabel("Score", fontsize = 10)
    # plt.title("Distribution of scores for the two classes", fontsize = 12)
    # if save_fig: 
    #     plt.savefig(f"{output_path}/{model_name}_dist.png", dpi = 600)

    plt.figure()
    plt.hist(mc["score"], bins = bins, alpha = 0.7, color="orange")
    plt.hist(human["score"], bins = bins, alpha = 0.7, color="blue")
    plt.xlim(0,)
    plt.legend(["Machine", "Human"], fontsize=fontsize-minus_font)
    plt.xlabel("Score", fontsize = fontsize-minus_font)
    plt.ylabel("Count", fontsize = fontsize-minus_font)
    plt.title("Distribution of scores for the two classes", fontsize = fontsize)
            
    plt.xticks(fontsize=(fontsize-minus_font))
    plt.yticks(fontsize=(fontsize-minus_font))
    plt.tight_layout()
    if save_fig: 
        plt.savefig(f"{output_path}/{model_name}_dist.png", dpi = quality)

    # ==============================================================================
    proposal_1 = mc[mc["bp"] == 1]
    proposal_2 = mc[mc["bp"] == 2]
    proposal_3 = mc[mc["bp"] == 3]

    # sns.distplot(proposal_1["score"], hist = True, bins = 40, norm_hist=True)
    # sns.distplot(proposal_2["score"], hist=True, bins= 40, norm_hist=True)
    # sns.distplot(proposal_3["score"], hist=True, bins = 40, norm_hist=True)
    
    plt.figure()
    plt.hist(proposal_1["score"], bins = bins, color="lightgreen", lw=3, fc=(0.5, 0.5, 1, 0.5))
    plt.hist(proposal_2["score"], bins = bins, color="r", lw=3, fc=(0, 0.5, 0, 0.5))
    plt.hist(proposal_3["score"], bins = bins, color="k", lw=3, fc=(0, 0, 0, 0.5))
    
    plt.legend([f"Cat. 1, \u03BC: {np.round(proposal_1['score'].mean(),2)}, \u03C3: {np.round(proposal_1['score'].std(), 2)}",
                f"Cat. 2, \u03BC: {np.round(proposal_2['score'].mean(),2)}, \u03C3: {np.round(proposal_2['score'].std(), 2)}",
                f"Cat. 3, \u03BC: {np.round(proposal_3['score'].mean(),2)}, \u03C3: {np.round(proposal_3['score'].std(), 2)}"],
                fontsize=fontsize-minus_font)
    
    plt.xlim(0,)

    plt.xlabel("Score", fontsize = fontsize-minus_font)
    plt.ylabel("Count", fontsize = fontsize-minus_font)
    plt.title("Distribution of scores for the \n three sentence categories", fontsize = fontsize)
            
    plt.xticks(fontsize=(fontsize-minus_font))
    plt.yticks(fontsize=(fontsize-minus_font))
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"{output_path}/{model_name}.png", dpi=quality)

    mc_hard = mc[mc["score"] > threshold][["idx", "score"]]
    mc_easy = mc[mc["score"] < (1-threshold)][["idx", "score"]]

    human_easy = human[human["score"] > threshold][["idx", "score"]]
    human_hard = human[human["score"] < (1-threshold)][["idx", "score"]]

    get_examples(mc_hard, "FP", name)
    get_examples(mc_easy, "TN", name)
    get_examples(human_hard, "FN", "human")
    get_examples(human_easy, "TP", "human")





   #stat, p = ttest_ind(proposal_1["score"], proposal_3["score"])

   #plt.figure()
   #plt.hist(proposal_1["score"], bins="auto")
   #plt.hist(proposal_2["score"], bins="auto")
   #plt.hist(proposal_3["score"], bins="auto")
   #plt.legend(["BadProposal = 1", "BadProposal = 2", "BadProposal = 3"])
   #plt.show()

   #plt.figure()
   #n, bins, rectangles = plt.hist(proposal_1["score"], 50, density=True)
   ##proposal_1["score"].plot.hist(normed=True)
   ##plt.hist(proposal_2["score"], bins="auto")
   ##plt.hist(proposal_3["score"], bins="auto")
   ##plt.legend(["BadProposal = 1", "BadProposal = 2", "BadProposal = 3"])
   #plt.show()