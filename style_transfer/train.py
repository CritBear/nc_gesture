
import torch
import numpy as np

from torch.utils.data import DataLoader


from nc_gesture.simple_rvae.datasets.nc_mocap import NCMocapDataset
from modules.networks import Generator

import pickle
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import os
from config import Config
import random

from utils.utils import *
from utils.tensors import *
from datasets.ActionStyleDataset import *
#from modules.transformer_vae import *
from sklearn.model_selection import train_test_split
from modules.transformer_vae import *

def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))

def get_same_action(action,dataset,usedId = -1):
    sames = []
    for idx,d in enumerate(dataset):
        if idx != usedId and process_file_name(d[0]) == action:
            sames.append(dataset[idx])

    if len(sames) == 0:
        return random.choice(dataset)
    return random.choice(sames)

def train():
    options = Config()
    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO | Device : {options.device}")

    data_path = os.path.join(options.data_dir, options.data_file_name)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    origin = data['origin_data']
    style_data = data['style_data']
    same_style = data['same_style_data']
    diff_style = data['diff_style_data']

    tmp = [[x, y, z, k] for x, y, z, k in zip(origin, style_data,same_style,diff_style)]
    random.shuffle(tmp)

    origin = [n[0] for n in tmp]
    style_data = [n[1] for n in tmp]
    same_style = [n[2] for n in tmp]
    diff_style = [n[3] for n in tmp]

    dataset = NCMocapDataset(origin)
    style_dataset = NCMocapDataset(style_data)
    same_style_dataset = NCMocapDataset(same_style)
    diff_style_dataset = NCMocapDataset(diff_style)
    #
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_content_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_content_dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False)

    print(len(style_dataset),len(same_style_dataset),len(diff_style_dataset))
    train_style_dataloader = DataLoader(style_dataset,batch_size=options.batch_size,shuffle=False) #shuffle 하면 안됨.
    train_same_style_dataloader = DataLoader(same_style_dataset,batch_size=options.batch_size,shuffle=False)
    train_diff_style_dataloader = DataLoader(diff_style_dataset,batch_size=options.batch_size,shuffle=False)

    print(len(train_style_dataloader),len(train_same_style_dataloader),len(train_diff_style_dataloader))
    options.use_cuda = (True if options.device == 'cuda' else False)

    model = Generator(options).to(options.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr_gen)

    MSELoss = torch.nn.MSELoss() # reconstruction loss
    triplet_loss = torch.nn.TripletMarginLoss(margin=options.triplet_margin)
    loss_history = []

    writer = SummaryWriter('logs/')


    for epoch in range(options.num_epochs):
        # train
        model.train()
        optimizer.zero_grad()
        train_content_iterator = tqdm(
            enumerate(train_content_dataloader), total=len(train_content_dataloader), desc="training"
        )
        train_style_iterator = iter(train_style_dataloader)
        train_same_iterator = iter(train_same_style_dataloader)
        train_diff_iterator = iter(train_diff_style_dataloader)

        for idx, (content_motion_label, content_motion) in train_content_iterator:
            content_motion = content_motion.to(options.device)
            style_motion_label, style_motion = next(train_style_iterator)
            _, same_style_motion = next(train_same_iterator)
            _, diff_style_motion = next(train_diff_iterator)

            style_motion = style_motion.to(options.device)

            output_motion = model(content_motion, style_motion)
            recon_motion = model(content_motion,content_motion)

            with torch.no_grad():
                output_motion_style_code = model.get_style_code(output_motion)
                style_motion_style_code = model.get_style_code(style_motion)

            #-----------------------------loss calculation---------------------------------------
            #style_loss = MSELoss(output_motion_style_code, style_motion_style_code)
            content_loss = recon_criterion(recon_motion,content_motion) #MSELoss(output_motion, content_motion) # reconstruction loss
            loss = options.content_loss_weight * content_loss

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            writer.add_scalar("Loss/train",loss,epoch)
            writer.add_scalar("Loss/recon_loss", content_loss, epoch)
            #writer.add_scalar("Loss/style_loss", style_loss, epoch)
            train_content_iterator.set_postfix({"train_loss": float(loss.mean())})

        print(f'Epoch {epoch + 1}/{options.num_epochs}')
        loss_history.append(loss.item())

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f"Result/BaseMST_{options.data_file_name}_{epoch}.pt")

    torch.save(model.state_dict(), f"Result/BaseMST_{options.data_file_name}_last.pt")
    writer.close()

    plt.plot(loss_history)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def getLoss(batch):
    MSELoss = torch.nn.MSELoss()
    beta = 0.0005
    recon_loss,vel_loss,avg_vel_loss,kl_loss = compute_mse_loss(batch),compute_vel_mse_loss(batch),compute_avg_vel_mse_loss(batch),kl_divergence(batch["mu"], batch["logvar"])
    loss = (1 - beta) * (recon_loss + 10 * vel_loss) + beta * kl_loss#+ 5*compute_vel_mse_loss(batch)
    lossdict = {"recon Loss":recon_loss,"vel_loss":vel_loss,'avg_vel_loss':avg_vel_loss,"kl_loss":kl_loss} #,"kl_loss":kl_loss
    return loss,lossdict

def train_tvae():
    options = Config()
    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO | Device : {options.device}")
    options.use_cuda = (True if options.device == 'cuda' else False)

    data_path = os.path.join(options.data_dir, options.data_file_name)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    writer = SummaryWriter('logs/')

    model = TVAE(options).to(options.device)
    #model.load_state_dict(torch.load('Result/tVAE_best.pt'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.lr_gen)


    dataset = ActionStyleDataset(data)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    datasets = {"train": train_dataset,"val":test_dataset}
    iterators = {key: DataLoader(datasets[key], batch_size=options.batch_size,
                                 shuffle=True, collate_fn=collate)
                 for key in datasets.keys()}

    min_val_loss = 10
    for epoch in range(options.num_epochs):
        model.train()
        train_iter = tqdm(enumerate(iterators['train']), total=len(iterators['train']), desc="training")

        with torch.enable_grad():
            for i,batch in train_iter:
                batch = {key: val.to(options.device) for key, val in batch.items()}
                batch = model(batch)

                loss,lossdict = getLoss(batch)

                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

                train_iter.set_postfix({"train_loss": float(loss.mean())})

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, batch in enumerate(iterators['val']):
                batch = {key: val.to(options.device) for key, val in batch.items()}
                batch = model(batch)
                loss,lossdict  = getLoss(batch)
                val_loss += loss
            val_loss /= len(iterators['val'])
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), f"Result/tVAE_best.pt")

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        for key in lossdict.keys():
            writer.add_scalar(f'{key}/train', lossdict[key], epoch)

        print(f"Epoch {epoch + 1}/{options.num_epochs}, Loss: {loss.mean()}, Val Loss: {val_loss}")
        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"Result/tVAE_{epoch}.pt")


if __name__ == "__main__":
    #load_result("train_result_BaseMST_motion_body_fixed_nohand_all.pkl_59340.pt.pkl")
    #result_visualize("BaseMST_motion_body_fixed_nohand_all.pkl_2000.pt")
    train_tvae()


#tensorboard --logdir=C:\Users\user\Desktop\NC\git\nc_gesture\style_transfer\logs
