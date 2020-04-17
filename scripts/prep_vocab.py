# ============================================================================================================
import json
import argparse
import time
import os
import numpy as np
from tqdm import tqdm
from config import *
import string
import operator
import fasttext
# ============================================================================================================

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
        '--f_name', type=str, default=f"{Config.local_path}/data/proposals2.npz",
        help='Path to the file that contains the dataset')
parser.add_argument(
        '--f_vocab', type=str, default=f"{Config.local_path}/data/proposals.npz",
        help='Path to the file that vocabulary')
parser.add_argument(
        '--output_path', type=str, default=f"{Config.local_path}/data",
        help='Path to the to the output files')
parser.add_argument(
        '--model', type=str, default="wiki.da/wiki.da.bin", 
        help='The model to be loaded')

args = parser.parse_args()

f_name = args.f_name
f_vocab = args.f_vocab
output_path = args.output_path
model = args.model

local_files = Config.local_path

# ============================================================================================================


def clean_string(txt, lower_all=True, punctuation=True): 
    """
    Pre-process string by cleaning the input text. Output the input string and the tokenized form. 
    
    lower_all
        Lower all words
    punctuation
        Whether or not to remove punctuation 
    """
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
    sentence = " ".join(tokens) # Easier to find mistakes if occured
    return sentence, tokens


# Generate vocabulary and a frequency count for each word based on textural data: 
def vocabulary_gen(txt_data):
    """
    txt_data
        Arrays of every sentence
    """
    write_dict = {}
    token_len = []
    for i in tqdm(range(len(txt_data))):
        anns = txt_data[i]
        _, tokens = clean_string(anns)
        token_len.append(len(tokens))

        for token in tokens:
            if token in write_dict:
                write_dict[token] += 1
            else:
                write_dict[token] = 1

    # Sort the dictionary High -> Low occurrence: 
    sorted_dict = sorted(write_dict.items(), key=operator.itemgetter(1))
    sorted_dict = sorted_dict[::-1] # **Only 6904 words**
    
    return sorted_dict


def word_embedding(words, model, dim=300):
    """
    idx2word:
        dictionary with format {index: word}
    model:
        model to be used for embedding
    """

    W = np.zeros((len(words), dim), dtype=float)
    W_dict = {}
    for i, word in enumerate(words):
        # Fasttext embedding 
        embedding_vec_temp = model.get_word_vector(word)
        # Save embedding vector and map to dictionary 
        W[i, :] = embedding_vec_temp
        W_dict[word] = embedding_vec_temp    

    return W, W_dict


# ============================================================================================================
# ============================================================================================================

# Debug: 
# f_name = f"{Config.local_path_temp}/data/proposals2.npz"
# f_vocab = f"{Config.local_path_temp}/data/proposals.npz"
# output_path =f"{Config.local_path_temp}/data"
# model = "wiki.da/wiki.da.bin"
# local_files = Config.local_path_temp
# # Make own function here: 
# vocab = np.load(f_vocab)["vocab"]

# ============================================================================================================
# ============================================================================================================

# Load data: 
data = np.load(f_name)
# txt_data = data["adult_texts"]

################################################
#### Prepare the embedding for the vocabulary: 

# sorted_dict = vocabulary_gen(txt_data)
# With frequency: 
sorted_dict = np.load(f_vocab)["vocab"]

# Without frequency: 
write_word_10k = []
write_word_10k.append('<pad>') # Padding 
write_word_10k.append('<unk>') # Unknown
write_word_10k.append('<sos>') # Start of sentence
write_word_10k.append('<eos>') # End of sentence

for i in range(Config.vocab_size):
    write_word_10k.append(sorted_dict[i][0])
    if (i+1) == len(sorted_dict):
        break

print(f"Vocabulary length: {len(write_word_10k)}")

# Making a mapping for a word to a index 
word_to_idx = {}
for i, word in enumerate(write_word_10k):
    word_to_idx[word] = i
# Making a reverse mapping, from index to a word
idx_to_word = {i: w for w, i in word_to_idx.items()}

# ============================================================================================================

################################################
#### Load fasttext model and embed vocabulary

# Load model .bin file: 
print("[INFO] Loading model")
dir_model = f"{str(local_files)}/{model}"
model = fasttext.load_model(dir_model)
print("[INFO] Done loading model")

# Embed the vocabulary: 
# word_embedding_300, word_embedding_300_dict = word_embedding(write_word_10k[4:], model)
word_embedding_300, word_embedding_300_dict = word_embedding(write_word_10k, model)

# ============================================================================================================

# def load_glove(filename):
#     ret = {}
#     with open(filename) as f:
#         for l in f:
#             lst = l.split()
#             key = lst[0]
#             value = np.array(lst[1:]).astype(np.float)
#             ret[key] = value
#     return ret

# D300  = load_glove('../local_files/glove.6B.300d.txt')

################################################ *** Not sure this is needed ***
#### Embed every word in the adult text: 
# print("[INFO] Extract all words in the text")
# words_freq = vocabulary_gen(txt_data)
# words = [i[0] for i in words_freq]

# list(vocab_embeddings_dict) 
# vocab_dict = vocab_embeddings_dict

# embeddings = D_to_W(words, vocab_dict)

# # Dictionary to word
# def D_to_W(words, vocab_dict, dim=300):
#     W = np.zeros((len(words), dim), dtype=float)

#     for i, word in enumerate(words):
#         if word in vocab_dict:
#             W[i, :] = vocab_dict[word]
#         else: 
#             W[i, :] = np.zeros(dim) # ATM no better way
#     return W


# ============================================================================================================
# ============================================================================================================
print("[INFO] Saving ...")
###### Save data ######

# Saving the word to index mapping 
np.save(os.path.join(output_path, 'word_to_idx.npy'),  word_to_idx)

# Saving the embedding matrix: 
np.save(os.path.join(output_path, 'word_embedding_300.npy'), word_embedding_300)

# Vocabulary and embeddings: 
f = open(f"{output_path}/fasttext.txt","w")
f.write( str(word_embedding_300_dict) )
f.close()

print("[INFO] Done")