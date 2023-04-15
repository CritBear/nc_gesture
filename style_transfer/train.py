
import torch
import numpy as np

from torch.utils.data import DataLoader

from nc_gesture.simple_rvae.datasets.nc_mocap import NCMocapDataset
from modules.cvae import Generator

import pickle
from tqdm import tqdm

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import os
from config import Config

def train():

    options = Config()
    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO | Device : {options.device}")

    print(options.data_dir)
    data_path = os.path.join(options.data_dir, options.data_file_name)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(data[0]["joint_rotation_matrix"].shape)
    dataset = NCMocapDataset(data)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=False)

    options.use_cuda = (True if options.device == 'cuda' else False)

    model = Generator(options).to(options.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr_gen)

    criterion = torch.nn.MSELoss()

    for epoch in range(options.num_epochs):
        # train
        model.train()
        optimizer.zero_grad()
        train_iterator = tqdm(
            enumerate(train_dataloader), total=len(train_dataloader), desc="training"
        )

        for idx, (style, motion) in train_iterator:
            motion = motion.to(options.device)

            out = model(motion)
            loss = criterion(out,motion)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            train_iterator.set_postfix({"train_loss": float(loss.mean())})
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"Results\BaseMST_{options.data_file_name}_{epoch}.pt")



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

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_content_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    options.batch_size = 59
    test_content_dataloader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=False)
    test_style_dataloader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=True)

    options.use_cuda = (True if options.device == 'cuda' else False)

    model = Generator(options).to(options.device)
    model.load_state_dict(torch.load(model_name))

    result = {"model_name":model_name,
              "data_name" : options.data_file_name,
              "motion_list":[],
              "style_list" :[],
              "latent_list":[]}

    with torch.no_grad():
        for idx, (content_motion_label, content_motion) in enumerate(test_content_dataloader):
            content_motion = content_motion.to(options.device)
            style_motion_label, style_motion = next(iter(test_style_dataloader))
            style_motion = style_motion.to(options.device)
            output_motion = model(content_motion, style_motion)
            motion_style_code = model.get_style_code(content_motion)

            for i, (s, m) in enumerate(zip(motion_style_code, output_motion)):
                print(data[idx * model.config.batch_size + i]['file_name'])
                origin_data = data[idx * model.config.batch_size + i]
                d = origin_data.copy()
                d["joint_rotation_matrix"] = m.reshape(origin_data["n_frames"], 26, 3, 2).cpu().numpy()
                d['target_style'] = style_motion_label[i]
                result["motion_list"].append(d)
                result["latent_list"].append(s.cpu().numpy().squeeze())
                result["style_list"].append(style_motion_label[i])
            print(f'processing... [{idx + 1}/{len(test_content_dataloader)}]')

        with open(f"train_result_{model_name}.pkl", 'wb') as f:
            pickle.dump(result, f)
        with open(f"decord_result_{model_name}.pkl", 'wb') as f:
            pickle.dump(result['motion_list'], f)

    show_tSNE(result)



if __name__ == "__main__":
    result_visualize("BaseMST_motion_body_fixed_nohand_all.pkl_59340.pt")
    #train()