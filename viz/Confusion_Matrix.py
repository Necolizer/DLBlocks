import pandas as pd
import numpy as np
import re
import torch.nn as nn
import torch
import seaborn as sns
import matplotlib.pyplot as plt

def draw_confusion_matrix(m, classes, save_path="./Confusion_Matrix.jpg", annot=False, log=False):
    if log:
        mdf = pd.DataFrame(np.log(m+1),index=classes,columns=classes)
        title = "Log Confusion Matrix"
    else:
        mdf = pd.DataFrame(m.astype(int),index=classes,columns=classes)
        title = "Confusion Matrix"
    fig = sns.heatmap(mdf, cmap="PuBu", fmt='d', annot=annot, annot_kws={'size':5,})
    plt.xlabel("Prediction")
    plt.ylabel("Label")
    plt.title(title)
    fig.get_figure().savefig(save_path, dpi = 400, bbox_inches='tight')