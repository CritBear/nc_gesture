import torch

import numpy as np

from torch.utils.data import DataLoader

from datasets.nc_mocap import NCMocapDataset
from modules.networks import Generator

import pickle

import os
from config import Config


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def show_tSNE(latent_data):
    tsne = TSNE(n_components=2, perplexity=2).fit_transform(np.array(latent_data["latent_list"]))
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

def result_visualize(model_name):
    options = Config()
    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO | Device : {options.device}")

    data_path = os.path.join(options.data_dir, options.data_file_name)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    dataset = NCMocapDataset(data)

    test_content_dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False, drop_last=True)
    test_style_dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True, drop_last=True)
    different_style_dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True, drop_last=True)

    options.use_cuda = (True if options.device == 'cuda' else False)

    model = Generator(options).to(options.device)
    model.load_state_dict(torch.load('Results/'+model_name+'.pt'))

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


if __name__ == "__main__":
    result_visualize("BaseMST_motion_body_fixed_nohand_all_9000")