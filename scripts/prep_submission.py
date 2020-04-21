# ============================================================================================================

from config import *
from pathlib import Path
import os 
import numpy as np
import progressbar
import argparse
import pandas as pd
import string

from sklearn.model_selection import train_test_split
import random

# ============================================================================================================
output_path = f"{Config.local_path_temp}/data"
word_to_idx = f"{Config.local_path_temp}/data/word_to_idx.npy"
data_path = f"{Config.local_path_temp}/data/proposals2.npz"
image_feat = f"{Config.local_path_temp}/data/image_features.npz"
name_human = "adult_texts"
name_mc = "proposals"
num_steps = 15
# ============================================================================================================

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
        '--data', type=str, default=f"{Config.local_path}/data/proposals2.npz",
        help='Path to the embeddings.')
parser.add_argument(
        '--image_feat', type=str, default=f"{Config.local_path}/data/image_features.npz",
        help='Path to the image features.')
parser.add_argument(
        '--word-to-idx', type=str, default=f"{Config.local_path}/data/word_to_idx.npy",
        help='Path to the npy file that contains mapping from word to index.')
parser.add_argument(
        '--output_path', type=str, default=f"{Config.local_path}/data",
        help='Path to the JSON file that contains how to split the dataset')
parser.add_argument(
        '--noicy', type=bool, default=False,
        help='If the data suppose to be noice (machine generated).')
parser.add_argument(
        '--num-steps', type=int, default=15,
        help='Length of all captions (default 15).')
parser.add_argument(
        '--name_human', type=str, default='adult_texts',
        help='Name of the method.')
parser.add_argument(
        '--name_mc', type=str, default='proposals',
        help='Name of the method.')

args = parser.parse_args()

assert args.name_human != 'human' # Prevent naming conflits

output_path = args.output_path
word_to_idx = args.word_to_idx 
data_path = args.data
image_feat = args.image_feat
name_human = args.name_human
name_mc = args.name_mc
num_steps = args.num_steps

# If data has been augmentated 
if args.noicy == True: 
    name_placeholder = args.name
    args.name = "random_word_gen"

# ============================================================================================================

def clean_string(txt, lower_all=True, lower_beg_punc=False, punctuation=True): 
    """
    Pre-process string by cleaning the input text. Output the input string and the tokenized form. 
    
    Parameters
    lower_all
        Lower all words
    lower_beg_punc - ** NOT WORKING AS INTENDED! ** 
        Lower the first word in a sentence and words after "."
    punctuation
        Whether or not to remove punctuation 
    """
    lower_beg_punc = False # ** Disabled: ask if it's good to have else remove. 

    if lower_all:
        txt = txt.lower()

    if punctuation:
        remove_stuff = string.punctuation + string.digits + ""
        translation = str.maketrans(remove_stuff, " " * len(remove_stuff))
        txt = txt.translate(translation)
        
        sentence2token = txt.split(" ")
        
        for _ in range(sentence2token.count("")):
            sentence2token.remove("")

    tokens = sentence2token
    # sentence = txt
    sentence = " ".join(tokens)
    return sentence, tokens


def image_name(url):
    name = str(url.split('/')[-1])
    return name


def prep_caption_feat(filenames, data, name_idx):
    feat = {}
    for i, filename in enumerate(filenames): 
        feat[filename] = {}

        gt = data[name_idx[0]][i]
        nt = data[name_idx[1]][i]
        # feat[filename][name_human] = np.zeros((len(gt), num_steps), dtype=np.int32)
        feat[filename][name_human] = np.zeros((num_steps), dtype=np.int32)
        feat[filename][name_mc] = np.zeros((num_steps), dtype=np.int32)    

        for k in range(min(len(gt), num_steps)):
            ann = gt[k]
            if ann in word2idx:
                feat[filename][name_human][k] = word2idx[ann]
            else: 
                feat[filename][name_human][k] = word2idx['<unk>']
        if len(ann) < num_steps:
            feat[filename][name_human][len(ann)] = word2idx['<eos>']

        for k in range(min(len(nt), num_steps)):
            ann = nt[k]
            if ann in word2idx:
                feat[filename][name_mc][k] = word2idx[ann]
            else: 
                feat[filename][name_mc][k] = word2idx['<unk>']
        if len(ann) < num_steps:
            feat[filename][name_mc][len(ann)] = word2idx['<eos>']
    return feat


def gen_new_data(filenames, data, word2idx, features):
    image_idxs = []
    for i in range(len(filenames)):
        # image_idxs.extend([i] * len(prep_data[filenames[i]][name_human]))
        image_idxs.extend([i])
    
    captions = np.zeros((len(image_idxs), num_steps), dtype=np.int32) 

    for c, i in enumerate(range(len(filenames))):
        caps = data[filenames[i]][name_human]
        for j in range(caps.shape[0]):
            captions[c, :] = caps


    ret = {}
    ret['file_names'] = filenames
    ret['image_idxs'] = image_idxs
    ret['word_to_idx'] = word2idx
    ret['captions'] = {}
    ret['captions']['gen'] = captions
    ret['captions']['dis'] = data
    ret['features'] = {}
    ret['features']['dis'] = features

    return ret


# ============================================================================================================

output_path =  Path(os.path.join(output_path, name_mc))

if not os.path.exists(output_path):
    os.makedirs(output_path)

# ============================================================================================================


#####################
# Load embedded data:  
data = dict(np.load(data_path, allow_pickle=True))
# For merging the image features: 
data["Images"] = [image_name(url) for url in data["url"]]

# Load image features: 
img_feat = dict(np.load(image_feat, allow_pickle=True))
# Reduce dimension (1,1,2048) -> (2048,)
img_feat["Image_features"] = [np.reshape(feat, [-1]) for feat in img_feat["Image_features"]]

# Convert to dataframes for innger merge (exclude all images without images features)
df_feat = pd.DataFrame(img_feat)
df_data = pd.DataFrame(data)
df = pd.merge(df_feat, df_data, how="inner", on="Images")

# Convert back to dict: 
data = df.to_dict("list")

data["human"] = data.pop(name_human)
name_human = "human"

# word2idx mapping: 
word2idx = np.load(word_to_idx, allow_pickle=True).item()

#####################

# ============================================================================================================

# Generate tokens and clean sentence: 
data["adult_sent"] = [clean_string(i)[0] for i in data[name_human]]
data["adult_token"] = [clean_string(i)[1] for i in data[name_human]]

data["proposals_sent"] = [clean_string(i)[0] for i in data[name_mc]]
data["proposals_token"] = [clean_string(i)[1] for i in data[name_mc]]

filenames = data["Images"]
name_idx = ["adult_token", "proposals_token"]


img_feat = []
for i, filename in enumerate(filenames): 
    img_feat.append([filename, data["Image_features"][i]])
features = dict(img_feat)

# Prep caption: 
prep_data = prep_caption_feat(filenames, data, name_idx)
data_ready = gen_new_data(filenames, prep_data, word2idx, features)


# ============================================================================================================
print("[INFO] Saving ...")
np.save(os.path.join(output_path, 'data_train_full.npy'), data_ready)
np.save(os.path.join(output_path, 'data_val_full.npy'),   data_ready)
np.save(os.path.join(output_path, 'data_test_full.npy'),  data_ready)
print("[INFO] Done")
# DO IT AT THE DATAFRAME: 
# Train, test, val: 
# idx_train = random.sample(range(len(data_ready["file_names"])), int(len(data_ready["file_names"])*0.8))
# idx_test = range(len(data_ready["file_names"]))- idx_train

# samples_no = range(len(data_ready["file_names"]))

