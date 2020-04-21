# AML_WriteReader


# Learning to Evaluate Image Captioning
TensorFlow implementation for the paper:

[*Learning to Evaluate Image Captioning*](https://vision.cornell.edu/se3/wp-content/uploads/2018/03/1501.pdf)\
[Yin Cui](http://www.cs.cornell.edu/~ycui/), [Guandao Yang](http://www.guandaoyang.com/), [Andreas Veit](https://www.cs.cornell.edu/~andreas/), [Xun Huang](http://www.cs.cornell.edu/~xhuang/), [Serge Belongie](http://blogs.cornell.edu/techfaculty/serge-belongie/)\
CVPR 2018

This repository contains a discriminator that could be trained to evaluate image captioning systems. The discriminator is trained to distinguish between machine generated captions and human written ones. During testing, the trained discriminator take the cadidate caption, the reference caption, and optionally the image to be captioned as input. Its output probability of how likely the candidate caption is human written can be used to evaluate the candidate caption. Please refer to our paper [[link]](https://vision.cornell.edu/se3/wp-content/uploads/2018/03/1501.pdf) for more detail.

<p align="center">
  <img src="figs/TrainingDiagram.png" width="100%">
</p>

## Dependencies: 
+ Python (3.7)
+ Tensorflow (1.15.0)
+ request: 2.22.0
+ urllib.request: 3.7
+ keras-2.3.1

## Preparation
The project uses different data source provided by WriteReaders. The objective is to implement the TensorFlow model and use it to between machine generated captions and human written ones. 

1. To generate the folder structure for this project: 
```
python config.py
```

Downloading and performing feature extraction for all images takes quite a lot of time 2-3 hours. Hence, to ease this process we recommend downloading preprocessed dataset (LINK)

2. Manully download the images run: 
```
python scripts/image_scraper.py --fname "data/proposals2.npz" --url_name "url"
```

This process can easily take an hour. For the 5k sample size we experinced an error for 10 of the urls. Since this is a significantly small amount of images, these samples will simple be removed at a later state (```prep_submission.py```). 

3. Having all the images downloaded the next step is to perfom feature extraction for each image. 
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


## Evaluation




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


# RUNNING

```
python scripts/score.py --name proposals
```


