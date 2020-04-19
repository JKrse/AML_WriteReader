# AML_WriteReader

# Get started! 
To generate the folder structure for this project: 
```
python config.py
```

# Preparation: 
Downloading and performing feature extraction for all images takes quite a lot of time 2-3 hours. Hence, to ease this process we recommend downloading preprocessed dataset (LINK)

To manully do this please follow the insctructions: 
## Download images for the experiment: 
First we download all images needed for feature extraction: 

```
python scripts/image_scraper.py --fname "data/proposals2.npz" --url_name "url"
```

This process can easily take an hour. For the 5k sample size we experinced an error for 10 of the urls. Since this is a significantly small amount of images, these samples will simple be removed at a later state when the images features are merged in "" (script). 

## Feature extraxtion 
Having all the images downloaded the next step is to perfom feature extraction for each image. 

```
pip install opencv-python
```

To get the features from each image: 
```
python scripts/image_feature_extraction.py --fname "./local_files/images"
```
This process takes 1-2 hours, hence we recommend downloading the preprossed data: (LINK).

## Download pre-trained model: 
To download the pre-trained fastText "da" model download the binary file: https://fasttext.cc/docs/en/pretrained-vectors.html (bin+text)

## Generate dataset with word embeddings:
We use the python package called ‘fasttext’ which introduces high level interface to use the vector files along with some other fastText functionalities:

```
pip install fasttext
```

To generate the data with word embeddings: 
```
python scripts/word_embedding.py --fname "data/proposals2.npz" --model "wiki.da/wiki.da.bin"
```

## Prepare the final preprossed dataset: 




# Dependencies: 
- request: 2.22.0
- urllib.request: 3.7
- keras-2.3.1