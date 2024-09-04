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

Two_hands = [7, 8, 9, 10, 11, 23, 36, 47, 52, 53]

train_ids = [3, 4, 5, 6, 8, 10, 15, 16, 17, 20, 21, 22, 23, 25, 26, 27, 30, 32, 36, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50]
val_ids = [1, 7, 12, 13, 24, 29, 33, 34, 35, 37]
test_ids = [2, 9, 11, 14, 18, 19, 28, 31, 41, 47]

def read_single(path):
    try:
        df = pd.read_csv(path, names=['label', 'onset', 'offset'], header=None).dropna().astype(int)
    except:
        print(path)
    df = df[df['label'].isin(Two_hands)].reset_index(drop=True)

    if not df.empty:
        dir = os.path.dirname(path)
        subject = os.path.basename(os.path.dirname(dir)).capitalize()
        scene = os.path.basename(dir)
        group = os.path.splitext(os.path.basename(path))[0].replace('Group', 'rgb', 1).replace('group', 'rgb', 1)

        s = pd.Series([subject] * len(df), name='subject')
        c = pd.Series([scene] * len(df), name='scene')
        g = pd.Series([group] * len(df), name='group')

        df = pd.concat([s, c, g, df], axis=1)
        return df
    else:
        return None

def filter_two_hand_gestures(base_path, save_csv_path):
    df = pd.DataFrame(columns=['subject', 'scene', 'group', 'label', 'onset', 'offset'])

    all_subjects = os.listdir(base_path)
    if '.DS_Store' in all_subjects:
        all_subjects.remove('.DS_Store')
    
    with tqdm(total=len(all_subjects), ncols=100, desc='Filtering') as pbar:
        for subject in all_subjects:
            subject_path = os.path.join(base_path, subject)
            all_scenes = os.listdir(subject_path)
            if '.DS_Store' in all_scenes:
                all_scenes.remove('.DS_Store')
            
            for scene in all_scenes:
                scene_path = os.path.join(base_path, subject, scene)
                all_groups = os.listdir(scene_path)
                if '.DS_Store' in all_groups:
                    all_groups.remove('.DS_Store')

                for group in all_groups:
                    group_csv = os.path.join(base_path, subject, scene, group)
                    temp = read_single(group_csv)
                    if temp is not None:
                        df = pd.concat([df, temp])
            pbar.update(1)

    df.to_csv(save_csv_path, index=False)

def two_hand_gestures_stat(save_csv_path):
    df = pd.read_csv(save_csv_path)
    
    df['framelen'] = df['offset'] - df['onset'] + 1

    df = df.drop(columns=['onset', 'offset'])

    print(df['framelen'].describe())

    def df_groupby_agg(df):
        all_df = df.drop(columns=['subject', 'scene', 'group'])
        res = all_df.groupby(['label']).agg({'label':'count', 'framelen': 'sum'})
        print(res)

    df_groupby_agg(df)

    train_subs = ['Subject' + (2-len(str(i)))*'0' + str(i) for i in train_ids]
    val_subs = ['Subject' + (2-len(str(i)))*'0' + str(i) for i in val_ids]
    test_subs = ['Subject' + (2-len(str(i)))*'0' + str(i) for i in test_ids]

    train_df = df[df['subject'].isin(train_subs)]
    val_df = df[df['subject'].isin(val_subs)]
    test_df = df[df['subject'].isin(test_subs)]

    df_groupby_agg(train_df)
    df_groupby_agg(val_df)
    df_groupby_agg(test_df)

def visualize_stat(save_csv_path):
    df = pd.read_csv(save_csv_path)
    
    df['framelen'] = df['offset'] - df['onset'] + 1

    df = df.drop(columns=['onset', 'offset'])

    def df_groupby_agg(df, split):
        all_df = df.drop(columns=['subject', 'scene', 'group'])
        res = all_df.groupby(['label']).agg({'label':'count', 'framelen': 'sum'})
        res1 = pd.DataFrame({'count' : all_df.groupby(['label']).size()}).reset_index()
        res1.columns = ['label', split+' count']

        res2 = pd.DataFrame({'sum' : all_df.groupby(['label'])['framelen'].sum()}).reset_index()
        res2.columns = ['label', split+' frame num']
        return res1, res2

    train_subs = ['Subject' + (2-len(str(i)))*'0' + str(i) for i in train_ids]
    val_subs = ['Subject' + (2-len(str(i)))*'0' + str(i) for i in val_ids]
    test_subs = ['Subject' + (2-len(str(i)))*'0' + str(i) for i in test_ids]

    train_df = df[df['subject'].isin(train_subs)]
    val_df = df[df['subject'].isin(val_subs)]
    test_df = df[df['subject'].isin(test_subs)]

    t1, t2 = df_groupby_agg(train_df, 'train')
    v1, v2 = df_groupby_agg(val_df, 'val')
    e1, e2 = df_groupby_agg(test_df, 'test')
    p1 = pd.merge(t1, v1, on='label')
    p1 = pd.merge(p1, e1, on='label')
    p2 = pd.merge(t2, v2, on='label')
    p2 = pd.merge(p2, e2, on='label')

    fig1 = p1.plot(kind="bar",stacked=True, x='label')
    plt.grid(linestyle="-.", axis='y', alpha=0.4)
    plt.ylabel('Number of Samples')
    height_list = np.zeros(len(p1))
    half_height_list = np.zeros(len(p1))
    for i, rect in enumerate(fig1.patches):
        height = rect.get_height()
        half_height_list[i % len(p1)] = height_list[i % len(p1)] + height / 2
        height_list[i % len(p1)] += height 
        plt.text(rect.get_x() + rect.get_width() / 2, half_height_list[i % len(p1)], str(height), size=8, ha='center', va='bottom')

    for i, rect in enumerate(fig1.patches[-len(p1):]):
        plt.text(rect.get_x() + rect.get_width() / 2, height_list[i], str(int(height_list[i])), size=8, ha='center', va='bottom', color='red')
    fig1.legend(loc=10, bbox_to_anchor=(0.5,-0.15), frameon=False, ncol=3)
    plt.tight_layout()
    fig1.figure.savefig('label_sample_num_distribution.png')

    fig2 = p2.plot(kind="bar",stacked=True, x='label')
    plt.grid(linestyle="-.", axis='y', alpha=0.4)
    plt.ylabel('Total Number of Frames')
    height_list = np.zeros(len(p1))
    half_height_list = np.zeros(len(p1))
    for i, rect in enumerate(fig2.patches):
        height = rect.get_height()
        half_height_list[i % len(p1)] = height_list[i % len(p1)] + height / 3
        height_list[i % len(p1)] += height 
        plt.text(rect.get_x() + rect.get_width() / 2, half_height_list[i % len(p1)], str(height), size=8, ha='center', va='bottom', rotation='vertical')

    for i, rect in enumerate(fig2.patches[-len(p1):]):
        plt.text(rect.get_x() + rect.get_width() / 2, height_list[i], str(int(height_list[i])), size=8, ha='center', va='bottom', color='red')
    fig2.legend(loc=10, bbox_to_anchor=(0.5,-0.15), frameon=False, ncol=3)
    plt.tight_layout()
    fig2.figure.savefig('label_frame_num_distribution.png')

def split_dataset_csv(save_csv_path):
    df = pd.read_csv(save_csv_path)

    train_subs = ['Subject' + (2-len(str(i)))*'0' + str(i) for i in train_ids]
    val_subs = ['Subject' + (2-len(str(i)))*'0' + str(i) for i in val_ids]
    test_subs = ['Subject' + (2-len(str(i)))*'0' + str(i) for i in test_ids]
    train_df = df[df['subject'].isin(train_subs)]
    val_df = df[df['subject'].isin(val_subs)]
    test_df = df[df['subject'].isin(test_subs)]

    csv_dir_name = os.path.dirname(save_csv_path)
    csv_base_name = os.path.basename(save_csv_path)
    train_df.reset_index(drop=True).to_csv(os.path.join(csv_dir_name, 'train_' + csv_base_name), index=True)
    val_df.reset_index(drop=True).to_csv(os.path.join(csv_dir_name, 'val_' + csv_base_name), index=True)
    test_df.reset_index(drop=True).to_csv(os.path.join(csv_dir_name, 'test_' + csv_base_name), index=True)


def make_two_hand_version_dataset(img_path, save_path, save_csv_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    csv_dir_name = os.path.dirname(save_csv_path)
    csv_base_name = os.path.basename(save_csv_path)
    train_csv_path = os.path.join(csv_dir_name, 'train_' + csv_base_name)
    val_csv_path = os.path.join(csv_dir_name, 'val_' + csv_base_name)
    test_csv_path = os.path.join(csv_dir_name, 'test_' + csv_base_name)

    def make_split(img_path, save_path, csv_path, split_name):
        save_split_path = os.path.join(save_path, split_name)
        df = pd.read_csv(csv_path, index_col=0)
        with tqdm(total=len(df), ncols=100, desc=split_name) as pbar:
            for i in range(len(df)):
                inter = os.path.join(df.at[i, 'subject'], df.at[i, 'scene'], 'Color', df.at[i, 'group'])
                for f in range(df.at[i, 'onset'], df.at[i, 'offset']+1):
                    img_name = (6-len(str(f)))*'0' + str(f) + '.jpg'
                    target_path = os.path.join(img_path, inter, img_name)
                    target_save_dir = os.path.join(save_split_path, str(i))
                    if not os.path.exists(target_save_dir):
                        os.makedirs(target_save_dir)
                    copyfile(target_path, os.path.join(target_save_dir, img_name))
                pbar.update(1)
    
    make_split(img_path, save_path, train_csv_path, 'train')
    make_split(img_path, save_path, val_csv_path, 'val')
    make_split(img_path, save_path, test_csv_path, 'test')

if __name__ == '__main__':
    base_path = r'E:\EgoGestureDataset\labels-final-revised1'
    img_path = r'E:\EgoGestureDataset\images_unziped'
    save_csv_path = r'E:\EgoGestureDataset\two_hands.csv'
    save_path = r'E:\EgoGestureDataset\EgoGestureDataset_2Hands'

    # filter_two_hand_gestures(base_path, save_csv_path)
    # two_hand_gestures_stat(save_csv_path)
    # split_dataset_csv(save_csv_path)
    # make_two_hand_version_dataset(img_path, save_path, save_csv_path)

    visualize_stat(save_csv_path)