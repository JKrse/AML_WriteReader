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

## Download images for the experiment: 
To download all images needed for feature extraction: 
```
python scripts/image_scraper.py -fname "data/proposals2.npz" -url_name "url"
```


# Dependencies: 
- gensim: 3.8.1