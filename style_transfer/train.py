
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


    torch.save(model.state_dict(), f"rvae_{options.data_file_name}_last.pt")

if __name__ == "__main__":
    train()