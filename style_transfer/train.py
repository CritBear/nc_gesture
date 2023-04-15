
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

from collections import Counter
import seaborn as sns
import random

def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))

def get_style_data(cur_style, e_dataset, i_dataset):
    same = []
    diff = []
    for s in cur_style:
        if 'e' in get_style_from_name(s):
            same.append(random.choice(e_dataset)[1].numpy())
            diff.append(random.choice(i_dataset)[1].numpy())
        else:
            same.append(random.choice(i_dataset)[1].numpy())
            diff.append(random.choice(e_dataset)[1].numpy())
    #print(len(same),len(diff))
    same_batch = torch.Tensor(same)
    diff_batch = torch.Tensor(diff)
    #print(same_batch.shape,diff_batch.shape)
    return same_batch,diff_batch

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

    style_e_data = []
    style_i_data = []

    for d in data:
        if d['persona'] == "de" or d['persona'] == "me":
            style_e_data.append(d)
        else:
            style_i_data.append(d)

    e_dataset = NCMocapDataset(style_e_data)
    i_dataset = NCMocapDataset(style_i_data)

    train_content_dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True)
    style_dataset = []

    index_i = 0
    index_e = 0
    print(len(e_dataset),len(i_dataset))

    for idx,(filenames,motions) in enumerate(train_content_dataloader):
        for j,(f,m) in enumerate(zip(filenames,motions)):
            #print(idx*32 + j,index_e,index_i)
            if 'e' in get_style_from_name(f):
                style_dataset.append(e_dataset[index_e])
                index_e +=1
            else:
                style_dataset.append(i_dataset[index_i])
                index_i += 1

    train_style_dataloader = DataLoader(style_dataset,batch_size=options.batch_size,shuffle=False) #shuffle 하면 안됨.

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

        for idx, (content_motion_label, content_motion) in train_content_iterator:
            content_motion = content_motion.to(options.device)
            style_motion_label, style_motion = next(train_style_iterator)
            same_style_motion,diff_style_motion = get_style_data(style_motion_label,e_dataset,i_dataset)
            style_motion = style_motion.to(options.device)

            output_motion = model(content_motion, style_motion)

            with torch.no_grad():
                output_motion_style_code = model.get_style_code(output_motion)
                style_motion_style_code = model.get_style_code(style_motion)
            same_style_code = model.get_style_code(same_style_motion.to(options.device))
            diff_style_code = model.get_style_code(diff_style_motion.to(options.device))
            style_loss = MSELoss(output_motion_style_code, style_motion_style_code)
            content_loss = recon_criterion(output_motion,content_motion) #MSELoss(output_motion, content_motion) # reconstruction loss

            l_triplet = triplet_loss(style_motion_style_code, same_style_code ,diff_style_code)

            loss = options.style_loss_weight * style_loss + options.content_loss_weight * content_loss + l_triplet * options.triplet_loss_weight

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            writer.add_scalar("Loss/train",loss,epoch)
            train_content_iterator.set_postfix({"train_loss": float(loss.mean())})

        print(f'Epoch {epoch + 1}/{options.num_epochs}')
        loss_history.append(loss.item())


        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f"Result/BaseMST_{options.data_file_name}_{epoch}.pt")
            writer.flush()

    writer.close()

    plt.plot(loss_history)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def process_file_name(file_name):
    d = file_name.split('_')
    action =''.join(d[:4])
    return action

def get_style_from_name(file_name):
    d = file_name.split('_')
    return d[-2]

def load_result(result_name):
    with open(result_name, 'rb') as f:
        latent_data = pickle.load(f)

    show_tSNE(latent_data)


def show_tSNE(latent_data):
    tsne = TSNE(n_components=2, perplexity=30).fit_transform(np.array(latent_data["latent_list"]))
    # print(tsne)

    di_idx_list = []
    de_idx_list = []
    mi_idx_list = []
    me_idx_list = []


    for idx, style in enumerate(latent_data['style_list']):
        latent_data['style_list'][idx]= process_file_name(style)

    cnt = dict(Counter(latent_data['style_list']))
    for k in cnt.keys():
        cnt[k] = []
    for idx, style in enumerate(latent_data['style_list']):
        cnt[style].append(idx)

    di_tsne = tsne[di_idx_list]
    de_tsne = tsne[de_idx_list]
    mi_tsne = tsne[mi_idx_list]
    me_tsne = tsne[me_idx_list]

    colors = sns.color_palette("husl", 63)
    print(colors)

    print(len(cnt.keys()))


    for idx,k in enumerate(cnt.keys()):
        cur_tsne = tsne[cnt[k]]
        plt.scatter(cur_tsne[:,0],cur_tsne[:,1],c=colors[idx],label= k)


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

    options.batch_size = 32
    test_content_dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False,drop_last=True)
    test_style_dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True,drop_last=True)

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
                #print(data[idx * model.config.batch_size + i]['file_name'])
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
    #load_result("train_result_BaseMST_motion_body_fixed_nohand_all.pkl_59340.pt.pkl")
    #result_visualize("BaseMST_motion_body_fixed_nohand_all.pkl_2000.pt")
    train()