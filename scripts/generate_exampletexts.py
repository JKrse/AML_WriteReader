# ============================================================================================================

import tensorflow as tf
from config import *
import argparse
import os

config = Config()

# ============================================================================================================
model_architecture = "mlp_1_img_1_512_0"
local_files = config.local_path_temp
name = "proposals"
# ============================================================================================================

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument(
        "--model_architecture", type=str, default="mlp_1_img_1_512_0",
        help='Path to the file that contains the dataset')
parser.add_argument(
        "--name", type=str, default="proposals",
        help='Name of the method used')

args = parser.parse_args()

model_architecture = args.model_architecture
local_files = config.local_path
name = args.proposals


# ============================================================================================================

path = local_files / f"{name}_scoring"

# The network graph defined in .meta 
if not os.path.exists(f"{path}/{model_architecture}-5.meta"):
    print(".meta naming wrong")

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(f"{path}/{model_architecture}-5.meta")
    saver.restore(sess, tf.train.latest_checkpoint('./'))

