# AML_WriteReader

## Get started! 
To generate the folder structure for this project: 
```
python config.py
```

# Preparation: 
There’s a python package called ‘fasttext’ which introduces high level interface to use the vector files along with some other fastText functionalities. But working with that package in my case, I didn’t find any convenience. Instead we used  ‘gensim’,  
Instal fasttext: 

```
pip install -U gensim
```

To download the pre-trained fastText "da" model download the binary file: https://fasttext.cc/docs/en/pretrained-vectors.html (bin+text)


# Dependencies: 
- gensim: 3.8.1