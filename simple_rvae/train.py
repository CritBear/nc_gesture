import torch
import numpy as np

from torch.utils.data import DataLoader

from datasets.nc_mocap import NCMocapDataset
from modules.rvae import RVAE

import os
import pickle
from tqdm import tqdm

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class TrainingOptions:

    def __init__(self):
        self.data_dir_path = 'datasets/data'
        self.model_dir_path = 'trained_model'

        self.data_file_name = 'motion_body_HJK.pkl'

        self.train_dataset_ratio = 0.8

        self.device = 'cpu'
        self.use_cuda = False

        self.input_size = 156

        self.encoder_rnn_size = 1024
        self.encoder_num_layers = 2

        self.latent_variable_size = 512

        self.decoder_rnn_size = 1024
        self.decoder_num_layers = 2

        self.output_size = 156

        self.num_epochs = 100
        self.batch_size = 1
        self.learning_rate = 0.00005
        self.drop_prob = 0.3
        self.kld_weight = 1

def train():

    options = TrainingOptions()

    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO | Device : {options.device}")

    data_path = os.path.join(options.data_dir_path, options.data_file_name)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    dataset = NCMocapDataset(data)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=False)

    options.use_cuda = (True if options.device == 'cuda' else False)

    rvae = RVAE(options).to(options.device)

    optimizer = torch.optim.Adam(rvae.parameters(), lr=options.learning_rate)

    recon_criterion = torch.nn.MSELoss()

    def criterion(input, recon, kld):
        recon_loss = recon_criterion(input, recon)
        return recon_loss + kld * 0.5

    for epoch in range(options.num_epochs):

        # train
        rvae.train()
        optimizer.zero_grad()
        train_iterator = tqdm(
            enumerate(train_dataloader), total=len(train_dataloader), desc="training"
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

        if (epoch + 1) % 5 == 0:
            # test
            rvae.eval()
            eval_loss = 0
            test_iterator = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="testing")

            with torch.no_grad():
                for idx, (style, motion) in test_iterator:
                    motion = motion.to(options.device)

                    loss, recon_output, info = rvae(motion)

                    eval_loss += loss.mean().item()

                    test_iterator.set_postfix({"eval_loss": float(loss.mean())})

            eval_loss = eval_loss / len(test_dataloader)
            print("Evaluation Score : [{}]".format(eval_loss))

        if (epoch + 1) % 10 == 0 or (epoch + 1) == options.num_epochs:
            model_name = f"rvae_{options.data_file_name}_lr0005_epc{epoch + 1}.pt"
            torch.save(rvae.state_dict(), os.path.join(options.model_dir_path, model_name))
            print(f'Trained model saved. EPOCH: {epoch + 1}')




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


def extract_latent_and_sample():

    options = TrainingOptions()

    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO | Device : {options.device}")
    options.use_cuda = (True if options.device == 'cuda' else False)

    data_path = os.path.join(options.data_dir_path, options.data_file_name)

    print('Data loading...')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    dataset = NCMocapDataset(data)
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True)

    rvae = RVAE(options).to(options.device)
    rvae.load_state_dict(torch.load('trained_model/rvae_motion_body_HJK.pkl_lr0005_epc100.pt'))
    rvae.eval()

    inference_data = {
        'data_name': options.data_file_name,
        'latent_size': options.latent_variable_size,
        'style_list': [],
        'latent_list': [],
        'recon_motion': []
    }

    for idx, (style, motion) in enumerate(dataloader):
        motion = motion.to(options.device)
        inference_data['style_list'].append(style[0])
        z, recon_motion = rvae.get_latent_and_sample(motion)
        z = z.cpu().numpy().squeeze()
        recon_motion = recon_motion.cpu().numpy().squeeze()

        inference_data['latent_list'].append(z)
        inference_data['recon_motion'].append(recon_motion)

        print(f'\rprocessing... [{idx + 1}/{len(dataloader)}]', end='')

    with open('inference_rvae_motion_body_HJK.pkl_lr0005_epc100', 'wb') as f:
        pickle.dump(inference_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Complete.')

    tsne = TSNE(n_components=2, perplexity=50).fit_transform(np.array(inference_data['latent_list']))

    di_idx_list = []
    de_idx_list = []
    mi_idx_list = []
    me_idx_list = []

    for idx, style in enumerate(inference_data['style_list']):
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


def extract_latent_space():
    options = TrainingOptions()

    data_path = 'datasets/data/motion_body_test_HJK.pkl'

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

    tsne = TSNE(n_components=2, perplexity=50).fit_transform(np.array(latent_data['latent_list']))
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
    extract_latent_and_sample()
    # extract_latent_space()
    # extract_latent_from_data()