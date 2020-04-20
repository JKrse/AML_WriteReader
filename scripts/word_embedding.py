# ============================================================================================================
import numpy as np
from config import *
import string
import fasttext
import re
import argparse

# ============================================================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--fname', default="data/proposals2.npz", help='File with image links')
parser.add_argument('--model', type=str, default="wiki.da/wiki.da.bin", help='The model to be loaded')
args = vars(parser.parse_args())

f_name = args['fname']
model = args['model']
local_files = Config.local_path

# # Debug
# local_files = Config.local_path_temp 
# f_name = "data/proposals2.npz"
# model = "wiki.da/wiki.da.bin"

# Arg: datafile, load_model

# ============================================================================================================
# Generate all folders needed: 

# Load data:
dir_data = local_files / f_name
data = np.load(dir_data, encoding="latin1")
# list(data.keys())
# list(data.items())

child = data["child_texts"]
adult = data["adult_texts"]
proposals = data["proposals"]
# vocab = data["vocab"] # vocab is in proposals

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
    # else: 
    #     sentence2token = re.findall(r"[\w']+|[.,!?;]", txt)

    # if lower_beg_punc: 
    #     sentence2token[0] = sentence2token[0].lower()

        # for i in range(len(sentence2token)-2): 
        #     if sentence2token[i] == ".":
        #         sentence2token[i+1] = sentence2token[i+1].lower()
    
    tokens = sentence2token
    # sentence = txt
    sentence = " ".join(tokens)
    return sentence, tokens


# ============================================================================================================

# =================== 
# Load model .bin file: 
print("[INFO] Loading model")
dir_model = f"{str(local_files)}/{model}"
model = fasttext.load_model(dir_model)
print("[INFO] Done loading model")

idx_sent, sentences, tokens, embeddings = [], [], [], []

print("[INFO] Generating embeddings")
for idx, sent in enumerate(adult): 
    # Clean sentence and tokens: 
    sent_temp, tokens_temp = clean_string(sent, lower_all=True, punctuation=True)
    # Generate embedding vector: 
    embedding_temp = model.get_sentence_vector(sent_temp)
    
    idx_sent.append(idx)
    sentences.append(sent_temp)
    tokens.append(tokens_temp)
    
    embeddings.append(embedding_temp)


# Make a dictionary: 
embeding_dict = {}
embeding_dict["word_idx"] = idx_sent
embeding_dict["sentence"] = sentences
embeding_dict["token"] = tokens
embeding_dict["embedding"] = embeddings

print("[INFO] Saving...")
np.savez(f"{local_files}/data/prep_data.npz",**embeding_dict)
print("[INFO] Done...")


