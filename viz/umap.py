import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import linalg as LA
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
import umap
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import seaborn as sns
import pandas as pd


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    data_path = r''
    checkpoint_path = r''

    f = Feeder(data_path=data_path, p_interval=[0.95], split='test', window_size=64)
    graph = 'graph.ntu_rgb_d.Graph'
    init_seed(1)

    data_loader = DataLoader(
        dataset=f,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        worker_init_fn=init_seed)

    data_list = []

    model = Model(num_class=26, num_point=25, num_person=2, graph='model.CTRGCN.graph.ntu_rgb_d.Graph', graph_args=dict(labeling_mode='spatial'))
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()
    model.eval()

    # cnt = 0
    for batch, (data, label, sample) in enumerate(tqdm(data_loader, desc="Checkpoint Evaluating", ncols=80)):
        # if cnt == 90:
        #     break
        with torch.no_grad():
            data = data.float().cuda()
            output = model(data)
            data_list.append(output.cpu().numpy())
        # cnt += 1

    data = np.concatenate(data_list, axis=0)
    N, C = data.shape # N, 26

    # 初始化UMAP模型
    umap_model = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine')

    # 拟合模型并进行降维
    umap_result = umap_model.fit_transform(data)

    # 创建DataFrame以便绘制可视化图表
    umap_df = pd.DataFrame(data=umap_result, columns=["UMAP 1", "UMAP 2"])
    umap_df["categories"] = f.label.astype(int)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制UMAP可视化图表
    sns.scatterplot(data=umap_df, x="UMAP 1", y="UMAP 2", hue="categories", palette=sns.color_palette("hls",26), legend=False)

    # ax.axes.xaxis.set_ticklabels([])
    # ax.axes.yaxis.set_ticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    # plt.legend(
    #     labels = [
    #         'punch/slap', 'kicking', 'pushing', 'pat on back',
    #         'point finger', 'hugging', 'giving object', 'touch pocket',
    #         'shaking hands', 'walking towards', 'walking apart', 'hit with object',
    #         'wield knife', 'knock over', 'grab stuff', 'shoot with gun',
    #         'step on foot', 'high-five', 'cheers and drink', 'carry object',
    #         'take a photo', 'follow', 'whisper', 'exchange things',
    #         'support somebody', 'rock-paper-scissors',
    #         ],
    #     frameon=False, 
    #     loc='upper left', 
    #     bbox_to_anchor=(1, 1), 
    #     ncol=2
    # )
    plt.gca().xaxis.set_major_locator(MultipleLocator(5))
    plt.gca().yaxis.set_major_locator(MultipleLocator(5))

    plt.savefig('umap2.svg', format='svg', transparent=True, bbox_inches = 'tight', pad_inches=0.0)
    # plt.savefig('umap.png', format='png', transparent=False, bbox_inches = 'tight', pad_inches=0.1)
