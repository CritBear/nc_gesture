
import torch
import numpy as np

from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter


from nc_gesture.simple_rvae.datasets.nc_mocap import NCMocapDataset
from modules.networks import Generator

import pickle
from tqdm import tqdm

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import os
from config import Config
import random

from utils.utils import *


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

        if epoch % 2:
            for idx, (content_motion_label, content_motion) in train_content_iterator:
                content_motion = content_motion.to(options.device)
                style_motion_label, style_motion = next(train_style_iterator)
                style_motion = style_motion.to(options.device)
                label1 =  torch.LongTensor(get_onehot_labels(style_motion_label))
                with torch.no_grad():
                    output_motion = model(content_motion, style_motion)
                    recon_motion = model(content_motion, content_motion)

                label2 = torch.LongTensor(get_onehot_labels(content_motion_label))
                adv_fake_loss1,_, _ = model.disc.calc_dis_fake_loss(output_motion.detach(),label1)
                adv_fake_loss2, _, _ = model.disc.calc_dis_fake_loss(recon_motion.detach(), label2)

                adv_real_loss, _, _ = model.disc.calc_dis_real_loss(content_motion, label2)

                dis_loss = (adv_fake_loss1+ adv_fake_loss2)/2 + adv_real_loss
                optimizer.zero_grad()
                dis_loss.backward()
                optimizer.step()
                writer.add_scalar("Loss/dis_loss", dis_loss, epoch)
                train_content_iterator.set_postfix({"train_loss": float(loss.mean())})
        else:
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

                same_style_code = model.get_style_code(same_style_motion.to(options.device)) # for triplet loss
                diff_style_code = model.get_style_code(diff_style_motion.to(options.device)) # for triplet loss

                label1 = torch.LongTensor(get_onehot_labels(style_motion_label))
                label2 = torch.LongTensor(get_onehot_labels(content_motion_label))


                adv_loss1,acc1,gen_style_feat = model.disc.calc_gen_loss(output_motion,label1.to(options.device))
                adv_loss2,acc2,gen_content_feat = model.disc.calc_gen_loss(recon_motion,label2.to(options.device))

                _,content_feat = model.disc(content_motion,label2)
                _,style_feat = model.disc(style_motion,label1)
                content_feat_loss = recon_criterion(gen_content_feat.mean(2),content_feat.mean(2))
                style_feat_loss = recon_criterion(gen_style_feat.mean(2),style_feat.mean(2))
                ft_loss = (content_feat_loss + style_feat_loss)/2
                #self.trans_p * trans + self.rec_p * rec
                acc = (acc1 + acc2)/2
                adv_loss = ((adv_loss1 + adv_loss2))/2

                #-----------------------------loss calculation---------------------------------------
                #style_loss = MSELoss(output_motion_style_code, style_motion_style_code)
                content_loss = recon_criterion(recon_motion,content_motion) #MSELoss(output_motion, content_motion) # reconstruction loss
                l_triplet = triplet_loss(style_motion_style_code, same_style_code ,diff_style_code)
                loss = options.content_loss_weight * content_loss \
                       + l_triplet * options.triplet_loss_weight + adv_loss * options.adv_loss_weight + ft_loss * options.ft_loss_weight

                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

                writer.add_scalar("Loss/train",loss,epoch)
                writer.add_scalar("Loss/triplet_loss", l_triplet, epoch)
                writer.add_scalar("Loss/recon_loss", content_loss, epoch)
                #writer.add_scalar("Loss/style_loss", style_loss, epoch)
                writer.add_scalar("Loss/adv_loss",adv_loss,epoch)
                writer.add_scalar("Lss/ft_loss",ft_loss)
                writer.add_scalar("Acc/dis_acc/train",acc,epoch)
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


if __name__ == "__main__":
    #load_result("train_result_BaseMST_motion_body_fixed_nohand_all.pkl_59340.pt.pkl")
    #result_visualize("BaseMST_motion_body_fixed_nohand_all.pkl_2000.pt")
    train()