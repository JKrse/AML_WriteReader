# ============================================================================================================
import numpy as np
from config import *
import string
from progressbar import ProgressBar 
import fasttext
import re

# ============================================================================================================
# Generate all folders needed: 
local_files = Config.local_path

# Load data:
data = np.load(local_files / "data" / "proposals2.npz", encoding="latin1")
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
    else: 
        sentence2token = re.findall(r"[\w']+|[.,!?;]", txt)

    if lower_beg_punc: 
        sentence2token[0] = sentence2token[0].lower()

        for i in range(len(sentence2token)-2): 
            if sentence2token[i] == ".":
                sentence2token[i+1] = sentence2token[i+1].lower()
    
    tokens = sentence2token
    sentence = " ".join(tokens)
    return sentence, tokens


# ============================================================================================================

# =================== 
# Load model .bin file: 
model = fasttext.load_model(f"{str(local_files)}/wiki.da/wiki.da.bin")

idx_sent, sentences, tokens, embeddings = [], [], [], []

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


# Problem with Tokens ()
# z = 0
# adult[z]
# txt = adult[z]
# adult[0:10]

# ft.tokenize(adult[0]) # Problem: "Noah." and not "Noah", "."
# sentence, _ = clean_string(adult[z], lower_all=True, punctuation=True)


# model.get_nearest_neighbors("abe", k=5)

# What is the difference: 
# from gensim.models.wrappers import FastText # import fasttext
# from gensim.models.fasttext import FastText # -> same as:
# from gensim.models import FastText

# model.get_sentence_vector("hej med dig").shape

# Load model .vec file: 
# vec_file = open(f"{str(local_files)}/wiki.da/wiki.da.vec", 'r', encoding='utf-8').readlines()
# vec_file[2]

# Load trained FastText model: 
# model2 = FastText.load_fasttext_format(f"{str(local_files)}/wiki.da/wiki.da.bin")

# model.get_nearest_neighbors("abe", k=10)
# model2.most_similar("abe")
