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

def copy_skeleton():
    root_dir = r'E:\SBU-Kinect-Interaction\Clean'
    save_dir = r'E:\SBU-Kinect-Interaction-Skeleton\Clean'

    label_name = ['01', '02', '03', '04', '05', '06', '07', '08']

    pair_name = os.listdir(root_dir)
    pair_name = [i for i in pair_name if i[-4:] != '.zip']

    with tqdm(total=len(pair_name)*len(label_name), desc='Copying') as pbar:
        for pair in pair_name:
            for label in label_name:
                seq_list_dir = os.path.join(root_dir, pair, label)

                if os.path.exists(seq_list_dir):
                    seq_name = os.listdir(seq_list_dir)
                    if '.DS_Store' in seq_name:
                        seq_name.remove('.DS_Store')
                    for seq in seq_name:
                        ske_path = os.path.join(seq_list_dir, seq, 'skeleton_pos.txt')
                        target_dir = os.path.join(save_dir, pair, label, seq)
                        if not os.path.exists(target_dir):
                            os.makedirs(target_dir)
                        copyfile(ske_path, os.path.join(target_dir, 'skeleton_pos.txt'))
                pbar.update(1)

def get_frame_num(path):
    df = pd.read_csv(path, header=None)
    return len(df)

def get_SBU_max_and_min_frame_num():
    max_frame_num = 0
    max_frame_num_name = ''
    min_frame_num = 9999
    min_frame_num_name = ''
    root_dir = r'E:\SBU-Kinect-Interaction-Skeleton\Clean'
    label_name = ['01', '02', '03', '04', '05', '06', '07', '08']
    pair_name = os.listdir(root_dir)

    with tqdm(total=len(pair_name)*len(label_name), desc='Processing') as pbar:
        for pair in pair_name:
            for label in label_name:
                seq_list_dir = os.path.join(root_dir, pair, label)
                if os.path.exists(seq_list_dir):
                    seq_name = os.listdir(seq_list_dir)
                    for seq in seq_name:
                        ske_path = os.path.join(seq_list_dir, seq, 'skeleton_pos.txt')
                        f_num = get_frame_num(ske_path)
                        if f_num > max_frame_num:
                            max_frame_num = f_num
                            max_frame_num_name = ske_path
                        if f_num < min_frame_num:
                            min_frame_num = f_num
                            min_frame_num_name = ske_path
                pbar.update(1)

    print(max_frame_num) # 46 #noisy:102
    print(max_frame_num_name) # Clean\s02s07\05\001\skeleton_pos.txt
    print(min_frame_num) # 10 #noisy:18
    print(min_frame_num_name) # Clean\s01s02\08\002\skeleton_pos.txt


def get_SBU_frame_num_plot():
    frame_num_list = []
    root_dir = r'E:\SBU-Kinect-Interaction-Skeleton\Noisy'
    label_name = ['01', '02', '03', '04', '05', '06', '07', '08']
    pair_name = os.listdir(root_dir)

    with tqdm(total=len(pair_name)*len(label_name), desc='Processing') as pbar:
        for pair in pair_name:
            for label in label_name:
                seq_list_dir = os.path.join(root_dir, pair, label)
                if os.path.exists(seq_list_dir):
                    seq_name = os.listdir(seq_list_dir)
                    for seq in seq_name:
                        ske_path = os.path.join(seq_list_dir, seq, 'skeleton_pos.txt')
                        f_num = get_frame_num(ske_path)
                        frame_num_list.append(f_num)
                pbar.update(1)

    frame_num_np = np.array(frame_num_list, dtype=np.int)
    sns.distplot(frame_num_np, hist=True, kde=False, norm_hist=False,
            rug=False, vertical=False, label='Frequency',
            axlabel='frame number', fit=norm)
    plt.axvline(frame_num_np.mean(), label='Mean',linestyle='-.', color='r')
    plt.axvline(np.median(frame_num_np), label='Median',linestyle='-.', color='g')
    plt.legend()
    plt.savefig ('Noisy.png', bbox_inches='tight')


def read_tensor(path):
    df = pd.read_csv(path, header=None)
    df.drop(0, axis=1, inplace=True)
    # print(df)
    assert df.shape[1] == 90

    frame_list = []
    for t in range(len(df)):
        person_list = []
        tuple_xyz_list = []
        for i in range(0, 44, 3):
            tuple_xyz_list.append(torch.from_numpy(np.array(df.iloc[t][i:i+3])))
        person_list.append(torch.stack(tuple_xyz_list, dim=0))
        tuple_xyz_list = []
        for i in range(45, 89, 3):
            tuple_xyz_list.append(torch.from_numpy(np.array(df.iloc[t][i:i+3])))
        person_list.append(torch.stack(tuple_xyz_list, dim=0))
        frame_list.append(torch.stack(person_list, dim=0))

    sample_tensor = torch.stack(frame_list, dim=0)
    # print(sample_tensor.shape) # torch.Size([45, 2, 15, 3])
    # T, M, V, C
    return sample_tensor

def pad_tensor(sample_tensor, max_frame_num=46):
    if sample_tensor.size(0) < max_frame_num:
        zero_tensor = torch.zeros((max_frame_num-sample_tensor.size(0), sample_tensor.size(1), sample_tensor.size(2), sample_tensor.size(3)))
        sample_tensor = torch.cat([sample_tensor, zero_tensor], dim=0)

    if sample_tensor.size(0) > max_frame_num:
        sample_tensor = sample_tensor[:max_frame_num,:,:,:]
    
    return sample_tensor


def get_SBU(root_dir, split='all', fold=0):
    assert (fold >= 0) and (fold <= 4)
    SBU_tensor_list = []
    label_list = []
    label_name = ['01', '02', '03', '04', '05', '06', '07', '08']

    fold_pair_name = [['s01s02', 's03s04', 's05s02', 's06s04'],
            ['s02s03', 's02s07', 's03s05', 's05s03'],
            ['s01s03', 's01s07', 's07s01', 's07s03'],
            ['s02s01', 's02s06', 's03s02', 's03s06'],
            ['s04s02', 's04s03', 's04s06', 's06s02', 's06s03']]

    if split == 'all':
        pair_name = []
        for i in range(len(fold_pair_name)):
            pair_name += fold_pair_name[i]
        #os.listdir(root_dir)
        tqdm_desc = 'Get All SBU'
    elif split == 'train':
        pair_name = []
        for i in range(len(fold_pair_name)):
            if i == fold:
                continue
            pair_name += fold_pair_name[i]
        tqdm_desc = 'Get SBU Train Fold'+str(fold)
    elif split == 'test':
        pair_name = fold_pair_name[fold]
        tqdm_desc = 'Get SBU Test Fold'+str(fold)
    else:
        raise NotImplementedError('data split only supports train/test/all')

    with tqdm(total=len(pair_name)*len(label_name), desc=tqdm_desc) as pbar:
        for pair in pair_name:
            for label in label_name:
                seq_list_dir = os.path.join(root_dir, pair, label)
                if os.path.exists(seq_list_dir):
                    seq_name = os.listdir(seq_list_dir)
                    for seq in seq_name:
                        ske_path = os.path.join(seq_list_dir, seq, 'skeleton_pos.txt')
                        SBU_tensor_list.append(pad_tensor(read_tensor(ske_path)))
                        label_list.append(int(label)-1)
                pbar.update(1)
    
    data = torch.stack(SBU_tensor_list, dim=0)
    ground_truth = torch.tensor(label_list)

    assert data.size(0) == ground_truth.size(0)

    # B, T, M, V, C # B
    return data, ground_truth

if __name__ == '__main__':
    # root_dir = r'E:\SBU-Kinect-Interaction-Skeleton\Clean'
    # data, ground_truth = get_SBU(root_dir, split='train', fold=0)
    # print(data.shape)
    # print(ground_truth.shape)
    # # data, ground_truth = get_SBU(root_dir, split='test', fold=0)
    # # print(data.shape)
    # # print(ground_truth.shape)
    # x = 1280 - (data[:,:,:,:,0] * 2560)
    # y = 960 - (data[:,:,:,:,1] * 1920)
    # z = data[:,:,:,:,2] * 10000 / 7.8125
    # res = torch.stack([x,y,z],dim=-1)
    # print(res.shape)
    # print(data[0,0,0,0,0])
    # print(data[0,0,0,0,1])
    # print(data[0,0,0,0,2])
    # print(res[0,0,0,0,0])
    # print(res[0,0,0,0,1])
    # print(res[0,0,0,0,2])

    # copy_skeleton()

    # get_SBU_max_and_min_frame_num()
    get_SBU_frame_num_plot()
