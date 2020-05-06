# ============================================================================================================
import matplotlib.pyplot as plt
from pathlib import Path
from config import *
import argparse
import pandas as pd
import os
import glob
# ============================================================================================================

# local_path = Config.local_path_temp
# name = "proposals"
# save_plt = True
# show_plt = False
# model_architecture = "mlp_1_img_1_512_0"

# ============================================================================================================

parser = argparse.ArgumentParser(description='Generate visualizations')
parser.add_argument(
        '--name', type=str, default="proposals",
        help='Name of the method used')
# parser.add_argument(
#         '--model_architecture', type=str, default="mlp_1_img_1_512_0",
#         help='Name of model architecture used')
parser.add_argument(
        '--show_plt', type=str, default=False,
        help='Whether to show the figure')
parser.add_argument(
        '--save_plt', type=str, default=False,
        help='Whether to save the figures')

args = parser.parse_args()

# ============================================================================================================


def plot(x, y, x_label=None, y_label=None, img_title=None, save_plt=False, img_name=None, fontsize=16, fig_no=None):
    plt.figure(fig_no)
    fig = plt.plot(x, y)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.yticks(fontsize=(fontsize-2))
    plt.xticks(x, fontsize=(fontsize-2))
    plt.title(img_title, fontsize=(fontsize+2))
    plt.tight_layout()
    if save_plt: 
        plt.savefig(f"{output_path}/{img_name}", dpi=1200)
    plt.show(fig)


# ============================================================================================================

local_path = Config.local_path
name = args.name
save_plt = args.save_plt
show_plt = args.show_plt
# model_architecture = args.model_architecture

# ============================================================================================================

exp_name = f"{name}_scoring"
# txt_file = f"{local_path}/{exp_name}/{model_architecture}.txt"
txt_file = f"{local_path}/{exp_name}/*.txt"
txt_file_train = f"{local_path}/{exp_name}/train/*.txt"
output_path = f"{local_path}/figures"

if not os.path.exists(output_path): 
    os.mkdir(output_path)

models = glob.glob(f"{txt_file}")
trains = glob.glob(f"{txt_file_train}")

colors = ['red', 'black', 'blue', 'brown', 'green']
fontsize = 14

for model in models: 
    data = pd.read_csv(model, sep="\\t", engine='python')
    cols = data.columns
    data = data[[cols[0], cols[2], cols[1], cols[3]]]

    # Subplots: 
    for i in range(3):
        grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.4)
        if i < 2:
            x,y = range(len(data.iloc[:,i])), data.iloc[:,i]
            plt.subplot(grid[0, i])
            plt.plot(x,y, c=colors[i])

            plt.title(data.columns[i], fontsize=fontsize)
            plt.xlabel("Epoch", fontsize=fontsize)
            plt.ylabel("Accuracy", fontsize=fontsize)
            plt.xticks(x, fontsize=(fontsize-2))
            plt.yticks(fontsize=(fontsize-2))
            plt.tight_layout()
        if i == 2:
            x, y1 = range(len(data.iloc[:, i])), data.iloc[:, i]
            y2 = data.iloc[:, i+1]

            plt.subplot(grid[1,:])
            plt.plot(x, y1, c=colors[i])
            plt.plot(x, y2, c=colors[i+1])

            plt.legend(["Human", "Pro"], loc="upper left")

            plt.title("Error Rate", fontsize=fontsize)
            plt.xlabel("Epoch", fontsize=fontsize)
            plt.ylabel("Accuracy", fontsize=fontsize)
            plt.xticks(x, fontsize=(fontsize - 2))
            plt.yticks(fontsize=(fontsize - 2))
            plt.tight_layout()

    if save_plt:
        model_name = os.path.splitext(os.path.split(model)[1])[0]
        plt.savefig(f"{output_path}/{model_name}.png", dpi=600)
    if show_plt:
        plt.show()
    else:
        plt.close()

# Single plots: 
# for i, col in enumerate(data.columns):
#     plot(x=range(len(data[col])), y=data[col], y_label="Accurcy", x_label="Epoch", 
#         img_title=f"{col}", fig_no=(i+1))

