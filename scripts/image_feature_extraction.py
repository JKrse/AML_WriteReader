# ============================================================================================================
import keras
from config import *
import os
from cv2 import cv2
import numpy as np
import argparse
# from keras.datasets import cifar10 # (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# ============================================================================================================
local_path = Config.local_path

parser = argparse.ArgumentParser()
parser.add_argument('--fname', default=f"./{local_path}/images", help='Path to folder(s) with images')
args = vars(parser.parse_args())

f_name = args['fname']

# ============================================================================================================

# Load Keras ResNet model: (Add to config - models section)
model = keras.applications.resnet.ResNet152(include_top=False, weights='imagenet', input_tensor=None, 
                                            input_shape=None, pooling='avg', classes=1000)


# ============================================================================================================

# rootdir = f"{Config.local_path_temp}/images"
rootdir = f_name

img_name = []
img_feat = []

for dirpath, dirnames, filenames in os.walk(rootdir):
    for i, file in enumerate(filenames):
        try:
            # Load images:
            img_path = os.path.join(dirpath, file)   
            img = cv2.imread(img_path)   

            # Images shape: [batch_size, image_width, image_height, number_of_channels]
            shape = img.shape

            # Feature extraction:
            features = model.predict(img.reshape(1, shape[0], shape[1], shape[2]))

            # Append features and file name: 
            img_name.append(file)
            img_feat.append(features)

            if i % 50 == 0: 
                print(f"[INFO] {i} of {len(filenames)} images have been feature extracted.")    
        except:
            print(f"Error: {file}")
    print(f"[INFO] {i+1} of {len(filenames)} images have been feature extracted.")

img_feat_dict = {"Images": img_name, "Image_features": img_feat}

# Saving Data: 
print(f"[INFO] Saving: {local_path}/data/image_features.npz")
np.savez(f"{local_path}/data/image_features.npz",**img_feat_dict)