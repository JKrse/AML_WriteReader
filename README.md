# AML_WriteReader

# Get started! 
To generate the folder structure for this project: 
```
python config.py
```

# Preparation: 
## Download images for the experiment: 
First we download all images needed for feature extraction: 

```
python scripts/image_scraper.py -fname "data/proposals2.npz" -url_name "url"
```

This process can easily take an hour. For the 5k sample size we experinced an error for 10 of the urls. Since this is a significantly small amount of images, these samples will simple be removed at a later state when the images features are merged in "" (script). 

## Download pre-trained model: 
To download the pre-trained fastText "da" model download the binary file: https://fasttext.cc/docs/en/pretrained-vectors.html (bin+text)

## Generate dataset with word embeddings:
We use the python package called ‘fasttext’ which introduces high level interface to use the vector files along with some other fastText functionalities:

```
pip install fasttext
```

To generate the data with word embeddings: 
```
python scripts/word_embedding.py -fname "data/proposals2.npz" -model "wiki.da/wiki.da.bin"
```





# Dependencies: 
- request: 2.22.0
- urllib.request: 3.7