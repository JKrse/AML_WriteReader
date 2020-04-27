# ============================================================================================================
import matplotlib.pyplot as plt
from pathlib import Path
from config import *
import argparse
import pandas as pd

import numpy as np
import os
# ============================================================================================================

local_path = Config.local_path_temp
name = "proposals"
model_architecture = "mlp_1_img_1_512_0"

# ============================================================================================================

parser = argparse.ArgumentParser(description='Generate visualizations')
parser.add_argument(
        '--name', type=str, default="proposals",
        help='Name of the method used')
parser.add_argument(
        '--model_architecture', type=str, default="mlp_1_img_1_512_0",
        help='Name of model architecture used')
parser.add_argument(
        '--output_path', type=str, default=f"{Config.local_path}/data",
        help='Path to the to the output files')


args = parser.parse_args()

# ============================================================================================================


def plot(x, y, x_label=None, y_label=None, img_title=None, save_plt=False, img_name=None, fontsize=16):
    plt.figure()
    fig = plt.plot(x, y)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.yticks(fontsize=(fontsize-2))
    plt.xticks(x, fontsize=(fontsize-2))
    plt.title(img_title, fontsize=(fontsize+2))
    plt.tight_layout()
    if save_plt: 
        plt.savefig(f"{path_image}/{img_name}", dpi=1200)
    plt.show(fig)


# ============================================================================================================

local_path = Config.local_path
name = args.name
model_architecture = args.model_architecture

# ============================================================================================================

exp_name = f"{name}_scoring"
txt_file = f"{local_path}/{exp_name}/{model_architecture}.txt"

data = pd.read_csv(txt_file, sep="\\t")

# ============================================================================================================

colors = ['red', 'black', 'blue', 'brown', 'green']
fontsize = 14

# Subplots: 
for i, col in enumerate(data.columns):
    x,y = range(len(data[col])), data[col]
    plt.subplot(2,2,(i+1))
    plt.plot(x,y, c=colors[i])

    plt.title(col, fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.ylabel("Accurcy", fontsize=fontsize)
    plt.xticks(x, fontsize=(fontsize-2))
    plt.yticks(fontsize=(fontsize-2))
    plt.tight_layout() 

# Single plots: 
for i, col in enumerate(data.columns):
    plot(x=range(len(data[col])), y=data[col], y_label="Accurcy", x_label="Epoch", img_title=f"{col}")

