import os
import sys
import torch

from torch.utils.data import DataLoader

from datasets.nc_mocap import NCMocapDataset
from modules.rvae import RVAE

import pickle
from tqdm import tqdm
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








if __name__ == "__main__":
    train()