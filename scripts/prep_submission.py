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
# output_path = f"{Config.local_path_temp}/data"
# word_to_idx = f"{Config.local_path_temp}/data/word_to_idx_{Config.vocab_size}.npy"
# data_path = "test_data.npz"
# data_path = f"{Config.local_path_temp}/data/{data_path}"
# # data_path = f"{Config.local_path_temp}/data/proposals2.npz"
# image_feat = "image_features_test.npz"
# image_feat = f"{Config.local_path_temp}/data/{image_feat}"
# url_name = "ImageUrl"
# # url_name = "url"
# name_human = "AdultText"
# name_mc = "Text Proposal"
# num_steps = 15
# train_test_val_split = False
# single_name = ""
# scoring_proposal = "BadProposal"

# ============================================================================================================

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
        '--data', type=str, default=f"proposals2.npz",
        help='Name of the data with embeddings.')
parser.add_argument(
        '--image_feat', type=str, default=f"image_features.npz",
        help='Path to the image features.')
parser.add_argument(
        '--word-to-idx', type=str, default=f"{Config.local_path}/data/word_to_idx_{Config.vocab_size}.npy",
        help='Path to the npy file that contains mapping from word to index.')
parser.add_argument(
        '--output_path', type=str, default=f"{Config.local_path}/data",
        help='Path to the JSON file that contains how to split the dataset')
parser.add_argument(
        '--num-steps', type=int, default=15,
        help='Length of all captions (default 15).')
parser.add_argument(
        '--train_test_val_split', type=str, default="True")
parser.add_argument(
        '--single_name', type=str, default="",
        help='Specifying name for a single file.')
parser.add_argument(
        '--scoring_proposal', type=str, default="",
        help='The scoring added')
parser.add_argument(
        '--name_human', type=str, default='adult_texts',
        help='Name of the method.')
parser.add_argument(
        '--name_mc', type=str, default='proposals',
        help='Name of the method.')
parser.add_argument(
        '--url_name', type=str, default="url", 
        help='Name of the title for the URL')

args = parser.parse_args()

assert args.name_human != 'human' # Prevent naming conflits

output_path = args.output_path
word_to_idx = args.word_to_idx 
data_path = args.data
data_path = f"{Config.local_path}/data/{data_path}"
image_feat = args.image_feat
image_feat = f"{Config.local_path}/data/{image_feat}"
name_human = args.name_human
name_mc = args.name_mc
num_steps = args.num_steps
train_test_val_split = args.train_test_val_split
single_name = args.single_name
url_name = args.url_name
scoring_proposal = args.scoring_proposal 

if train_test_val_split == "False":
    train_test_val_split = False
    print('Not splitting data in train, test, and validation')
else:
    train_test_val_split = True
    print("Splitting data in train, test, and validation")

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
    try:
        name = str(url.split('/')[-1])
    except:
        name = ""
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


def add_sent_token(data):
    """
    Data is a dictionary
    """
    data["adult_sent"] = [clean_string(i)[0] for i in data[name_human]]
    data["adult_token"] = [clean_string(i)[1] for i in data[name_human]]

    data["proposals_sent"] = [clean_string(i)[0] for i in data[name_mc]]
    data["proposals_token"] = [clean_string(i)[1] for i in data[name_mc]]

    return data


def add_features(filenames, data): 
    img_feat = []
    for i, filename in enumerate(filenames): 
        img_feat.append([filename, data[i]])
    features = dict(img_feat)

    return features


# ============================================================================================================

output_path =  Path(os.path.join(output_path, name_mc))

if not os.path.exists(output_path):
    os.makedirs(output_path)

# ============================================================================================================


#####################
# Load embedded data:  
data = dict(np.load(data_path, allow_pickle=True))
# For merging the image features: 
data["Images"] = [image_name(url) for url in data[url_name]]

# Load image features: 
img_feat = dict(np.load(image_feat, allow_pickle=True))
# Reduce dimension (1,1,2048) -> (2048,)
img_feat["Image_features"] = [np.reshape(feat, [-1]) for feat in img_feat["Image_features"]]

# Convert to dataframes for innger merge (exclude all images without images features)
df_feat = pd.DataFrame(img_feat)
df_data = pd.DataFrame(data)
df = pd.merge(df_feat, df_data, how="inner", on="Images")

df = df.rename(columns = {name_human: "human"})
df = df.rename(columns = {name_mc: "proposals"})

name_human = "human"
name_mc = 'proposals'

# Split data training, testing and validation sets
if train_test_val_split: 
    df_train, temp  = train_test_split(df, train_size=0.8, shuffle=True, random_state=42)
    df_test, df_val = train_test_split(temp, train_size=0.95, shuffle=True, random_state=42)

    # Convert back to dict: 
    data_train = df_train.to_dict("list")
    data_test = df_test.to_dict("list")
    data_val = df_val.to_dict("list")

    # word2idx mapping: 
    word2idx = np.load(word_to_idx, allow_pickle=True).item()

    #####################

    # ============================================================================================================

    # Generate tokens and clean sentence: 
    data_train = add_sent_token(data_train)
    data_test = add_sent_token(data_test)
    data_val = add_sent_token(data_val)

    name_idx = ["adult_token", "proposals_token"]

    #####################
    # Train: 
    filenames_train = data_train["Images"]
    features_train = add_features(filenames_train, data_train["Image_features"]) 

    prep_data_train = prep_caption_feat(filenames_train, data_train, name_idx)
    data_ready_train = gen_new_data(filenames_train, prep_data_train, word2idx, features_train)

    #####################
    # Test
    filenames_test = data_test["Images"]
    features_test = add_features(filenames_test, data_test["Image_features"]) 

    prep_data_test = prep_caption_feat(filenames_test, data_test, name_idx)
    data_ready_test = gen_new_data(filenames_test, prep_data_test, word2idx, features_test)

    #####################
    # Validation
    filenames_val = data_val["Images"]
    features_val = add_features(filenames_val, data_val["Image_features"]) 

    prep_data_val = prep_caption_feat(filenames_val, data_val, name_idx)
    data_ready_val = gen_new_data(filenames_val, prep_data_val, word2idx, features_val)


# ============================================================================================================
    print("[INFO] Saving ...")
    
    np.save(os.path.join(output_path, f"data_train_full_{Config.vocab_size}.npy"), data_ready_train)
    np.save(os.path.join(output_path, f"data_test_full_{Config.vocab_size}.npy"),  data_ready_test)
    np.save(os.path.join(output_path, f"data_val_full_{Config.vocab_size}.npy"),   data_ready_val)
     
    print("[INFO] Done")
else: 
    # Convert back to dict: 
    data = df.to_dict("list")
    
    # word2idx mapping: 
    word2idx = np.load(word_to_idx, allow_pickle=True).item()

    # ============================================================================================================

    # Generate tokens and clean sentence: 
    data = add_sent_token(data)
    
    name_idx = ["adult_token", "proposals_token"]

    #####################
    # Train: 
    filenames = data["Images"]
    features = add_features(filenames, data["Image_features"]) 

    prep_data = prep_caption_feat(filenames, data, name_idx)
    data_ready = gen_new_data(filenames, prep_data, word2idx, features)
    
    if len(scoring_proposal) > 0:
        data_ready[scoring_proposal] = data[scoring_proposal] 

    # ============================================================================================================
    print("[INFO] Saving ...")
    np.save(os.path.join(output_path, f"data_single_{single_name}_full_{Config.vocab_size}.npy"), data_ready)
    print("[INFO] Done")


