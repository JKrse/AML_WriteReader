# AML_WriteReader

# Get started! 
To generate the folder structure for this project: 
```
python config.py
```

# Preparation: 
We use the python package called ‘fasttext’ which introduces high level interface to use the vector files along with some other fastText functionalities:

```
pip install fasttext
```

## Download pre-trained model: 
To download the pre-trained fastText "da" model download the binary file: https://fasttext.cc/docs/en/pretrained-vectors.html (bin+text)

## Generate dataset with word embeddings:
To generate the data with word embeddings: 
```
python scripts/word_embedding.py -fname "data/proposals2.npz" -model "wiki.da/wiki.da.bin"
```

## Download images for the experiment: 
To download all images needed for feature extraction: 

```
pip install grequests
```

```
python scripts/image_scraper.py -fname "data/proposals2.npz" -url_name "url"
```


# Dependencies: 
- gensim: 3.8.1
- request: 2.22.0
- urllib.request: 3.7