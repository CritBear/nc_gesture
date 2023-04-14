
import torch
import numpy as np

from torch.utils.data import DataLoader

from datasets.nc_mocap import NCMocapDataset
from modules.networks import Generator

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

    data_path = os.path.join(options.data_dir, options.data_file_name)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    dataset = NCMocapDataset(data)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_content_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_content_dataloader = DataLoader(train_content_dataset, batch_size=options.batch_size, shuffle=True)
    train_style_dataloader = DataLoader(train_content_dataset, batch_size=options.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=False)

    options.use_cuda = (True if options.device == 'cuda' else False)

    model = Generator(options).to(options.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr_gen)

    MSELoss = torch.nn.MSELoss() # reconstruction loss

    loss_history = []

    for epoch in range(options.num_epochs):
        # train
        model.train()
        optimizer.zero_grad()
        train_content_iterator = tqdm(
            enumerate(train_content_dataloader), total=len(train_content_dataloader), desc="training"
        )
        train_style_iterator = iter(train_style_dataloader)

        for idx, (content_motion_label, content_motion) in train_content_iterator:

            content_motion = content_motion.to(options.device)
            style_motion_label, style_motion = next(train_style_iterator)
            style_motion = style_motion.to(options.device)

            output_motion = model(content_motion, style_motion)

            with torch.no_grad():
                output_motion_style_code = model.get_style_code(output_motion)
                style_motion_style_code = model.get_style_code(output_motion)

            style_loss = MSELoss(output_motion_style_code, style_motion_style_code)
            content_loss = MSELoss(output_motion, content_motion) # reconstruction loss

            loss = options.style_loss_weight * style_loss + options.content_loss_weight*content_loss

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            train_content_iterator.set_postfix({"train_loss": float(loss.mean())})

        print(f'Epoch {epoch + 1}/{options.num_epochs}')
        loss_history.append(loss.item())

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"BaseMST_{options.data_file_name}_{epoch}.pt")

    plt.plot(loss_history)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

if __name__ == "__main__":
    train()