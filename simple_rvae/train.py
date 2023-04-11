import os
import sys
import torch

from torch.utils.data import DataLoader

from datasets.nc_mocap import NCMocapDataset
from modules.rvae import RVAE

import pickle

import bvh_parser


class TrainingOptions:

    def __init__(self):
        self.use_cuda = False

        self.input_size = 666

        self.encoder_rnn_size = 800
        self.encoder_num_layers = 2

        self.latent_variable_size = 128

        self.decoder_rnn_size = 800
        self.decoder_num_layers = 2

        self.output_size = 666

        self.num_iterations = 1000
        self.batch_size = 1
        self.learning_rate = 0.00005
        self.dropout = 0.3

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO | Device : {device}")

    options = TrainingOptions()

    data_path = 'datasets/data/refined_motion_test_HJK.pkl'

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    dataset = NCMocapDataset(data)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    options.use_cuda = (True if device == 'cuda' else False)

    rvae = RVAE(options)

    if options.use_cuda:
        rvae.cuda()

    optimizer = torch.optim.Adam(rvae.parameters(), lr=options.learning_rate)

    # train
    for idx, (style, motion) in enumerate(dataloader):
        print(style)
        print(motion.shape)






if __name__ == "__main__":
    train()