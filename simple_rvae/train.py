import os
import sys
import torch
import numpy as np

from torch.utils.data import DataLoader

from datasets.nc_mocap import NCMocapDataset
from modules.rvae import RVAE

import pickle
from tqdm import tqdm

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import bvh_parser


class TrainingOptions:

    def __init__(self):
        self.device = 'cpu'
        self.use_cuda = False

        self.input_size = 666

        self.encoder_rnn_size = 2048
        self.encoder_num_layers = 2

        self.latent_variable_size = 1024

        self.decoder_rnn_size = 2048
        self.decoder_num_layers = 2

        self.output_size = 666

        self.num_epochs = 10
        self.batch_size = 1
        self.learning_rate = 0.0001
        self.drop_prob = 0.3

def train():

    options = TrainingOptions()

    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO | Device : {options.device}")

    data_path = 'datasets/data/refined_motion_HJK.pkl'

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    dataset = NCMocapDataset(data)

    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True)

    options.use_cuda = (True if options.device == 'cuda' else False)

    rvae = RVAE(options).to(options.device)

    optimizer = torch.optim.Adam(rvae.parameters(), lr=options.learning_rate)

    recon_criterion = torch.nn.MSELoss()
    def criterion(input, recon, kld):
        recon_loss = recon_criterion(input, recon)
        return recon_loss + kld * 0.5

    # train
    for epoch in range(options.num_epochs):
        rvae.train()
        optimizer.zero_grad()
        train_iterator = tqdm(
            enumerate(dataloader), total=len(dataloader), desc="training"
        )

        for idx, (style, motion) in train_iterator:
            motion = motion.to(options.device)

            loss, recon_output, info = rvae(motion)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            train_iterator.set_postfix({"train_loss": float(loss.mean())})

            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
            #     epoch + 1, options.num_epochs, idx + 1, len(dataloader), loss.item() / 1))

    torch.save(rvae.state_dict(), f"rvae_{'HJK_0412'}.pt")


def extract_latent_from_data():
    with open('rvae_HJK_0412_refined_motion_HJK.pkl', 'rb') as f:
        latent_data = pickle.load(f)

    print('Complete.')

    tsne = TSNE(n_components=2).fit_transform(np.array(latent_data['latent_list']))
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


def extract_latent_space():
    options = TrainingOptions()

    data_path = 'datasets/data/refined_motion_HJK.pkl'

    print('Data loading...')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    dataset = NCMocapDataset(data)

    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True)

    rvae = RVAE(options)
    rvae.load_state_dict(torch.load(f"rvae_{'HJK_0412'}.pt"))
    rvae.eval()

    latent_data = {
        'data_name': 'refined_motion_test_HJK.pkl',
        'latent_size': options.latent_variable_size,
        'style_list': [],
        'latent_list': []
    }

    for idx, (style, motion) in enumerate(dataloader):
        latent_data['style_list'].append(style[0])
        latent_data['latent_list'].append(rvae.get_latent_space(motion).numpy().squeeze())
        print(f'processing... [{idx + 1}/{len(dataloader)}]')

    with open('rvae_HJK_0412_refined_motion_HJK.pkl', 'wb') as f:
        pickle.dump(latent_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Complete.')

    tsne = TSNE(n_components=2, perplexity=2).fit_transform(np.array(latent_data['latent_list']))
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


if __name__ == "__main__":
    # train()
    # extract_latent_space()
    extract_latent_from_data()