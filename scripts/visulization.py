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

local_path = Config.local_path_temp
name = "neuraltalk"
save_plt = False
show_plt = True
model_architecture = "mlp_1_img_1_512_0"

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

colors = ['blue', 'orange', 'green',  'brown']
fontsize = 14
minus_font = 2

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
        grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.6)
        if i == 0:
            x1 = [x+1 for x in range(len(data.iloc[:,i]))]
            y1 = data.iloc[:,i]
            x = [x+1 for x in range(len(data.iloc[:,i+1]))]
            y = data.iloc[:,i+1]

            plt.subplot(grid[i, 0])
            plt.plot(x,y, c=colors[i])
            plt.plot(x1,y1, c=colors[i+1])

            plt.legend(["Human", "Machine"]) #, loc="upper left")

            plt.title("Score - Inference", fontsize=fontsize)
            # plt.xlabel("Epoch", fontsize=fontsize)
            plt.ylabel("Score", fontsize=fontsize)
            plt.xticks(fontsize=(fontsize-minus_font))
            plt.yticks(fontsize=(fontsize-minus_font))
            plt.tight_layout()
        
        if i == 1:
            x = [x + 1 for x in range(len(data.iloc[:, i]))]
            y1 = data.iloc[:, 1]
            y2 = data.iloc[:, i+1]

            plt.subplot(grid[0,1])
            plt.plot(x, y2, c=colors[0])
            plt.plot(x, y1, c=colors[1])

            # plt.legend(["Human", "Machine"], loc="upper left")

            plt.title("Sensitivity - Inference", fontsize=fontsize)
            #plt.xlabel("Epoch", fontsize=fontsize)
            plt.ylabel("Accuracy", fontsize=fontsize)
            plt.xticks(fontsize=(fontsize - minus_font))
            plt.yticks(fontsize=(fontsize - minus_font))
            plt.tight_layout()
        
        if i == 2: 
            
            y3 = data_train[" Accuracy"]
            y3_avg = [np.mean(y3[i:i+10]) for i in range(0, len(y3), int(len(y3)/x[-1])) ]
            
            y_acc = (data.iloc[:, i] + data.iloc[:, i+1])*0.5 #  (sent_mc*prev) + (sent_hum*(1-prev))

            plt.subplot(grid[1,:])
            plt.plot(x, y_acc, c=colors[3]) # Test
            plt.plot(x, y3_avg, c=colors[2]) # Training

            plt.legend(["Test", "Train"]) #, loc="upper left")

            plt.title("Model accuracy when training and testing", fontsize=fontsize)
            plt.xlabel("Epoch", fontsize=fontsize)
            plt.ylabel("Accuracy", fontsize=fontsize)
            plt.xticks(fontsize=(fontsize - minus_font))
            plt.yticks(fontsize=(fontsize - minus_font))
            plt.tight_layout()
        
    if save_plt:
        plt.savefig(f"{output_path}/{model_name}.png", dpi=600)
    if show_plt:
        plt.show()
    else:
        plt.close()

    
    # ROC NOT DONE PRETTY YET

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

