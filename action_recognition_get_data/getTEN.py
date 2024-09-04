import torch
import torch.nn as nn
import numpy as np
import os
import sys
from shutil import copyfile
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

def read_single(path):
    df = pd.read_csv(path, delimiter='\t', header=None, names=['frame', 'label'])
    print(df.head())
    res = df.groupby(['label']).agg({'label':'count'})
    print(res)



if __name__ == '__main__':
    path = r'D:\PersonalProject\Action2Text\Tennis\annotations\labels\V008.txt'
    read_single(path)