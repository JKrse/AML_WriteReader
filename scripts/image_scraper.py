# ============================================================================================================
# import packages
import requests
import urllib.request
import os
import argparse
from config import *
import numpy as np

# ============================================================================================================
config = Config()

# # For debug: " 
f_name = "data/test_data.npz"
url_name = "ImageUrl"
local_files = config.local_path_temp
test = True

#Define the argument parser to read in the URL
parser = argparse.ArgumentParser()
parser.add_argument('--fname', default="data/proposals2.npz", 
                    help='File with image links')
parser.add_argument('--url_name', type=str, default="url", 
                    help='Name of the title for the URL')
parser.add_argument('--test', type=bool, default=False)
args = parser.parse_args()

f_name = args.fname
url_name = args.url_name
local_files = config.local_path
test = args.test


if test == "True":
    test = True
elif test == "False":
    test = False
else:
    print("test has to be 'True' or 'False'")


# ============================================================================================================


# Function to take an image url and save the image in the given directory
def download_image(url, save_path):
    # Find png file name: 
    name = str(url.split('/')[-1])
    # Save image:
    urllib.request.urlretrieve(url,os.path.join(save_path, name))


def image_name(url):
    name = str(url.split('/')[-1])
    return name


# ============================================================================================================

# Create the directory name where the images will be saved
if test:
    dir_img = local_files / "images_test"
else:
    dir_img = local_files / "images"
# Create the directory if already not there
if not os.path.exists(dir_img):
    os.mkdir(dir_img)

# Load data: 
dir_data = local_files / f_name
data = np.load(dir_data, encoding="latin1", allow_pickle=True)

# ============================================================================================================
links = data[url_name]
error_img = []

for i, img in enumerate(links): 
    try:
        download_image(img, dir_img)
    except:
        error_img.append([i, img])
        print(f"Error for image: {i}")    
    if i % 50 == 0: 
        print(f"[INFO] Downloaded {i} of {len(links)} images.")

print(f"[INFO] Downloaded {i+1-len(error_img)} of {len(links)} images.")