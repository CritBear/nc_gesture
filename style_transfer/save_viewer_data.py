import random

import torch

import numpy as np

from torch.utils.data import DataLoader

from datasets.nc_mocap import NCMocapDataset
from datasets.ActionStyleDataset import ActionStyleDataset
from modules.networks import Generator

import pickle

import os
from config import Config


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from collections import Counter
import seaborn as sns
from modules.transformer_vae import *

from utils.utils import *
from utils.tensors import *

import matplotlib.pyplot as plt

def process_file_name(file_name):
    d = file_name.split('_')
    action =''.join(d[:4])
    return action



def load_result(result_name):
    with open(result_name, 'rb') as f:
        latent_data = pickle.load(f)

    show_tSNE(latent_data)


def show_tSNE_by_action(latent_data):
    tsne = TSNE(n_components=2, perplexity=30).fit_transform(np.array(latent_data["latent_list"]))
    # print(tsne)

    di_idx_list = []
    de_idx_list = []
    mi_idx_list = []
    me_idx_list = []


    for idx, style in enumerate(latent_data['style_list']):
        latent_data['style_list'][idx] = process_file_name(style)

    cnt = dict(Counter(latent_data['style_list']))
    for k in cnt.keys():
        cnt[k] = []
    for idx, style in enumerate(latent_data['style_list']):
        cnt[style].append(idx)

    di_tsne = tsne[di_idx_list]
    de_tsne = tsne[de_idx_list]
    mi_tsne = tsne[mi_idx_list]
    me_tsne = tsne[me_idx_list]

    colors = sns.color_palette("husl", 63)
    print(colors)

    print(len(cnt.keys()))


    for idx,k in enumerate(cnt.keys()):
        cur_tsne = tsne[cnt[k]]
        plt.scatter(cur_tsne[:,0],cur_tsne[:,1],c=colors[idx],label= k)


    plt.xlabel('tsne_0')
    plt.ylabel('tsne_1')
    plt.legend()
    plt.show()

def show_tSNE(latent_data):
    tsne = TSNE(n_components=2, perplexity=40).fit_transform(np.array(latent_data["latent_list"]))
    # print(tsne)

    di_idx_list = []
    de_idx_list = []
    mi_idx_list = []
    me_idx_list = []

    for idx, style in enumerate(latent_data['style_list']):
        if style == 'di':
            di_idx_list.append(idx)
        elif style == 'de':
            de_idx_list.append(idx)
        elif style == 'mi':
            mi_idx_list.append(idx)
        elif style == 'me':
            me_idx_list.append(idx)

    di_tsne = tsne[di_idx_list]
    de_tsne = tsne[de_idx_list]
    mi_tsne = tsne[mi_idx_list]
    me_tsne = tsne[me_idx_list]

    # print(di_tsne)
    # print(de_tsne)
    # print(mi_tsne)
    # print(me_tsne)

    plt.scatter(di_tsne[:, 0], di_tsne[:, 1], color='pink', label='di')
    plt.scatter(de_tsne[:, 0], de_tsne[:, 1], color='purple', label='de')
    plt.scatter(mi_tsne[:, 0], mi_tsne[:, 1], color='green', label='mi')
    plt.scatter(me_tsne[:, 0], me_tsne[:, 1], color='blue', label='me')

    plt.xlabel('tsne_0')
    plt.ylabel('tsne_1')
    plt.legend()
    plt.show()
def show_tSNE():
    with open("datasets/data/classifier_900_high.pkl", 'rb') as f:
        motion_data = pickle.load(f)

    motion = []
    style = []
    for m in motion_data:
        motion.append(m['joint_rotation_matrix'].reshape(-1))
        style.append(m['persona'])
    tsne = TSNE(n_components=2, perplexity=50).fit_transform(np.array(motion))
    # print(tsne)

    di_idx_list = []
    de_idx_list = []
    mi_idx_list = []
    me_idx_list = []

    for idx, style in enumerate(style):
        if style == 'di':
            di_idx_list.append(idx)
        elif style == 'de':
            de_idx_list.append(idx)
        elif style == 'mi':
            mi_idx_list.append(idx)
        elif style == 'me':
            me_idx_list.append(idx)

    di_tsne = tsne[di_idx_list]
    de_tsne = tsne[de_idx_list]
    mi_tsne = tsne[mi_idx_list]
    me_tsne = tsne[me_idx_list]

    plt.scatter(di_tsne[:, 0], di_tsne[:, 1], color='pink', label='di')
    plt.scatter(de_tsne[:, 0], de_tsne[:, 1], color='purple', label='de')
    plt.scatter(mi_tsne[:, 0], mi_tsne[:, 1], color='green', label='mi')
    plt.scatter(me_tsne[:, 0], me_tsne[:, 1], color='blue', label='me')

    plt.xlabel('tsne_0')
    plt.ylabel('tsne_1')
    plt.legend()
    plt.show()
def result_visualize(model_name):
    options = Config()
    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO | Device : {options.device}")

    data_path = os.path.join(options.data_dir, "motion_body_fixed_nohand_all.pkl")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    dataset = NCMocapDataset(data)

    test_content_dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False, drop_last=True)
    test_style_dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True, drop_last=True)
    different_style_dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True, drop_last=True)

    options.use_cuda = (True if options.device == 'cuda' else False)

    model = Generator(options).to(options.device)
    model.load_state_dict(torch.load('Result/'+model_name+'.pt'))

    result = {"model_name":model_name,
              "data_name" : options.data_file_name,
              "motion_list":[],
              "style_list" :[],
              "latent_list":[]}

    style2_motion_list = []

    with torch.no_grad():
        for idx, (content_motion_label, content_motion) in enumerate(test_content_dataloader):
            content_motion = content_motion.to(options.device)

            style_motion_label, style_motion = next(iter(test_style_dataloader))
            style_motion = style_motion.to(options.device)
            output_motion = model(content_motion, style_motion)
            motion_style_code = model.get_style_code(style_motion)

            style_motion_label2, style_motion2 = next(iter(different_style_dataloader))
            style_motion2 = style_motion2.to(options.device)
            output_motion2 = model(content_motion, style_motion2)
            motion_style_code2 = model.get_style_code(style_motion2)

            style2_zip = zip(motion_style_code2, output_motion2)

            for i, (s, m) in enumerate(zip(motion_style_code, output_motion)):
                print(data[idx * model.config.batch_size + i]['file_name'])
                origin_data = data[idx * model.config.batch_size + i]
                d = origin_data.copy()
                d["joint_rotation_matrix"] = m.reshape(origin_data["n_frames"], 26, 3, 2).cpu().numpy()
                d['target_style'] = style_motion_label[i]
                result["motion_list"].append(d)
                result["latent_list"].append(s.cpu().numpy().squeeze())
                result["style_list"].append(style_motion_label[i])

                s2, m2 = next(style2_zip)
                d2 = origin_data.copy()
                d2["joint_rotation_matrix"] = m2.reshape(origin_data["n_frames"], 26, 3, 2).cpu().numpy()
                d2['target_style'] = style_motion_label2[i]
                style2_motion_list.append(d2)

            print(f'processing... [{idx + 1}/{len(test_content_dataloader)}]')

        with open(f"datasets/data/train_result_{model_name}.pkl", 'wb') as f:
            pickle.dump(result, f)
        with open(f"datasets/data/decord_result_{model_name}.pkl", 'wb') as f:
            pickle.dump(result['motion_list'], f)
        with open(f"datasets/data/decord_result_{model_name}_2.pkl", 'wb') as f:
            pickle.dump(style2_motion_list, f)

    show_tSNE(result)

def result_tvae(model_name):
    options = Config()
    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO | Device : {options.device}")

    data_path = os.path.join(options.data_dir, options.data_file_name)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    dataset = ActionStyleDataset(data)
    origin_dataset = data
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False, collate_fn=collate)

    options.use_cuda = (True if options.device == 'cuda' else False)

    model = TVAE(options).to(options.device)
    model.load_state_dict(torch.load('Result/' + model_name + '.pt'))
    result = []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = {key: val.to(options.device) for key, val in batch.items()}
            batch = model(batch)
            for i,m in enumerate(batch['output']):
                origin_data = origin_dataset[idx * model.config.batch_size + i]
                d = origin_data.copy()
                mask = batch['mask'][i]
                outmasked = m[mask]
                d['n_frames'] = len(outmasked)
                d["output"] = outmasked.reshape(-1, 26, 3, 2).cpu().numpy()
                #d['target_motion'] = batch['x'][i].reshape(len(batch['x'][i]),26,3,2).cpu().numpy()
                d['target_style'] = origin_data['target_style']
                result.append(d)

    with open(f"datasets/data/decord_result_{model_name}.pkl", 'wb') as f:
        pickle.dump(result, f)
def create_random_tensor_excluding(n, exclude_value):
    random_tensor = torch.randint(low=0, high=4, size=(n,))
    for i in range(n):
        if random_tensor[i] == exclude_value[i]:
            random_tensor[i] = (exclude_value[i]+1)%4

    return random_tensor
def result_tvae_decode(model_name):
    options = Config()
    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO | Device : {options.device}")

    data_path = os.path.join(options.data_dir, options.data_file_name)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    dataset = ActionStyleDataset(data)
    origin_dataset = data
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False, collate_fn=collate)

    options.use_cuda = (True if options.device == 'cuda' else False)

    model = TVAE(options).to(options.device)
    model.load_state_dict(torch.load('Result/' + model_name + '.pt'))
    result = []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = {key: val.to(options.device) for key, val in batch.items()}
            batch['z'] = torch.randn(batch['x'].shape[0],options.latent_dim, device=options.device)
            batch = model.decode(batch)
            batch2 = batch.copy()
            batch2['style'] = create_random_tensor_excluding(len(batch['x']),batch['style'].cpu())
            batch2 = model.decode(batch2)
            for i,m in enumerate(batch['output']):
                origin_data = origin_dataset[idx * model.config.batch_size + i]
                d = origin_data.copy()
                d["output"] = m.reshape(-1, 26, 3, 2).cpu().numpy()
                d["output2"] = batch2['output'][i].reshape(-1, 26,3, 2).cpu().numpy()
                d['target_style2'] = batch2['style'][i]
                #d['target_motion'] = batch['x'][i].reshape(len(batch['x'][i]),26,3,2).cpu().numpy()
                d['target_style'] = batch['style'][i]

                result.append(d)

    with open(f"datasets/data/decord_result_{model_name}.pkl", 'wb') as f:
        pickle.dump(result, f)

def result_tvae(model_name):
    options = Config()
    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO | Device : {options.device}")

    data_path = os.path.join(options.data_dir, options.data_file_name)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    dataset = ActionStyleDataset(data)
    origin_dataset = data
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False, collate_fn=collate)

    options.use_cuda = (True if options.device == 'cuda' else False)


    model = TVAE(options).to(options.device)
    model.load_state_dict(torch.load('Result/' + model_name + '.pt'))
    result = []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = {key: val.to(options.device) for key, val in batch.items()}
            batch = model(batch)
            for i,m in enumerate(batch['output']):
                origin_data = origin_dataset[idx * model.config.batch_size + i]
                d = origin_data.copy()
                mask = batch['mask'][i]
                outmasked = m[mask]
                first_false_indices = (mask == False).nonzero()
                if first_false_indices.numel() > 0:
                    first_false_index = first_false_indices.min().item()
                    d['n_frames'] = first_false_index
                d["output"] = outmasked.reshape(-1, options.num_joints, 3, 2).cpu().numpy()
                #d['target_motion'] = batch['x'][i].reshape(len(batch['x'][i]),26,3,2).cpu().numpy()
                d['target_style'] = origin_data['target_style']
                result.append(d)

    with open(f"datasets/data/decord_result_{model_name}.pkl", 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    #show_tSNE()
    result_tvae("tVAE_best")