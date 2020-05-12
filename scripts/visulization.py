# ============================================================================================================
import matplotlib.pyplot as plt
from pathlib import Path
from config import *
import argparse
import pandas as pd
import os
import glob
import numpy as np
import re
from sklearn import metrics
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

    model_name = os.path.splitext(os.path.split(model)[1])[0]

    r = re.compile(f".*{model_name}")
    train = list(filter(r.match, trains))

    data_train = pd.read_csv(train[0], sep="\\t", engine='python')

    samples = data_train.groupby("Epoch")[" Loss"].count()[0]

    new_x = []
    for e in set(data_train["Epoch"]):
        range_x = np.arange(e, e+1, 1/samples)
        new_x.append(range_x)

    new_x_flat = [y for x in new_x for y in x]
    new_x_flat = pd.DataFrame(new_x_flat, columns=["new_x"])

    data_train = pd.concat([data_train, new_x_flat], axis = 1)

    # Subplots: 
    for i in range(3):
        grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.4)
        if i < 2:
            x = [x+1 for x in range(len(data.iloc[:,i]))]
            y = data.iloc[:,i]
            plt.subplot(grid[0, i])
            plt.plot(x,y, c=colors[i])

            plt.title(data.columns[i], fontsize=fontsize)
            plt.xlabel("Epoch", fontsize=fontsize)
            plt.ylabel("Score", fontsize=fontsize)
            plt.xticks(x, fontsize=(fontsize-8))
            plt.yticks(fontsize=(fontsize-8))
            plt.tight_layout()
        if i == 2:
            x = [x + 1 for x in range(len(data.iloc[:, i]))]
            y1 = data.iloc[:, i]
            y2 = data.iloc[:, i+1]
            x_new, y3 = data_train["new_x"], data_train[" Accuracy"]

            plt.subplot(grid[1,:])
            plt.plot(x, y1, c=colors[i])
            plt.plot(x, y2, c=colors[i+1])
            plt.plot(x_new, y3, c=colors[i+2])

            plt.legend(["Prop", "Human", "Train"], loc="upper left")

            #plt.title("Accuracy", fontsize=fontsize)
            plt.xlabel("Epoch", fontsize=fontsize)
            plt.ylabel("Sensitivity", fontsize=fontsize)
            plt.xticks(x, fontsize=(fontsize - 6))
            plt.yticks(fontsize=(fontsize - 6))
            plt.tight_layout()

    if save_plt:
        plt.savefig(f"{output_path}/{model_name}.png", dpi=600)
    if show_plt:
        plt.show()
    else:
        plt.close()

    # Generating ROC
    ytrue1 = np.ones(len(data))
    ytrue0 = np.zeros(len(data))
    ytrue = pd.DataFrame(np.concatenate([ytrue1, ytrue0], axis = 0))
    yprobs = pd.concat([data[cols[2]], data[cols[0]]], axis=0)

    fpr, tpr, _ = metrics.roc_curve(ytrue, yprobs)
    auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC with AUC: {auc:.2f}")

    if save_plt:
        plt.savefig(f"{output_path}/{model_name}_ROC.png", dpi=600)
    if show_plt:
        plt.show()
    else:
        plt.close()



# Single plots: 
# for i, col in enumerate(data.columns):
#     plot(x=range(len(data[col])), y=data[col], y_label="Accurcy", x_label="Epoch", 
#         img_title=f"{col}", fig_no=(i+1))

