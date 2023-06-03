import torch
import numpy as np

from torch.utils.data import DataLoader

from datasets.nc_mocap import NCMocapDataset
from modules.rvae import RVAE

import pickle
from tqdm import tqdm

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import os

class TrainingOptions:

    def __init__(self):
        self.device = 'cpu'
        self.use_cuda = False

        self.input_size = 444

        self.condition_size = 4

        self.encoder_rnn_size = 2048
        self.encoder_num_layers = 2

        self.latent_variable_size = 512

        self.decoder_rnn_size = 2048
        self.decoder_num_layers = 2

        self.output_size = 444

        self.num_epochs = 128
        self.batch_size = 20
        self.learning_rate = 0.00005
        self.drop_prob = 0.3
        self.kld_weight = 1

def train():

    options = TrainingOptions()

    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO | Device : {options.device}")

    data_path = 'datasets/data/motion_body_fixed_HJKKTG.pkl'

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

    torch.save(rvae.state_dict(), f"rvae_{'_0414'}.pt")


def extract_latent_from_data():
    with open('rvae_HJK_0412_refined_motion_HJK.pkl', 'rb') as f:
        latent_data = pickle.load(f)

    print('Complete.')

    tsne = TSNE(n_components=2,perplexity=30).fit_transform(np.array(latent_data['latent_list']))
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


def extract_latent_space(dataset_name, model_name):
    options = TrainingOptions()

    data_path = 'datasets/data/' + dataset_name

    print('Data loading...')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    dataset = NCMocapDataset(data)

    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True)

    rvae = RVAE(options)
    rvae.load_state_dict(torch.load(model_name))
    rvae.eval()

    latent_data = {
        'data_name': dataset_name+'.pkl',
        'latent_size': options.latent_variable_size,
        'style_list': [],
        'latent_list': []
    }
    with torch.no_grad():
        for idx, (style, motion) in enumerate(dataloader):
            out = rvae.get_latent_space(motion).numpy().squeeze()
            for j,z in enumerate(out):
                latent_data['style_list'].append(style[0])
                latent_data['latent_list'].append(z)
            try:
                print(np.array(latent_data['latent_list']).shape)
            except:
                print(motion.shape)
                for data in latent_data['latent_list']:
                    print(np.array(data).shape)
            print(latent_data['latent_list'])
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


def make_result_data(dataset_name, model_name):
    options = TrainingOptions()

    data_path = 'datasets/data/' + dataset_name

    print('Data loading...')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    dataset = NCMocapDataset(data)

    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False)

    rvae = RVAE(options)
    rvae.load_state_dict(torch.load(model_name))
    rvae.eval()
    decorded_data = []
    with torch.no_grad():
        for idx, (style, motion) in enumerate(dataloader):
            out = rvae.forward(motion)[1]
            for i,(s,m) in enumerate(zip(style,out)):
                origin_data = data[idx*rvae.options.batch_size+i]
                d = origin_data.copy()
                d["joint_rotation_matrix"] = m.reshape(origin_data["n_frames"],74,3,2)
                decorded_data.append(d)
                #print(m)
            print(f'processing... [{idx + 1}/{len(dataloader)}]')
    with open("decored_data_0414.pkl", 'wb') as f:
         pickle.dump(decorded_data, f)


def split_data(data,origin_data, fixed_size):
    n = 0
    for d in origin_data:
        if d["n_frames"] < 600: # 너무 긴 모션을 일단 버리고자 함.
            n += 1
            for i in range(d["n_frames"] // fixed_size):
                dd = d.copy()
                dd['n_frames'] = fixed_size
                over_lap_num = (int(fixed_size * 1 / 4))
                s = i * (fixed_size - over_lap_num)
                e = s + fixed_size
                dd['joint_rotation_matrix'] = d['joint_rotation_matrix'][s:e]
                dd['root_position'] = d['root_position'][s: e]
                #print(d["n_frames"], i, "size:", len(dd['joint_rotation_matrix']))
                data.append(dd)
    print("600 보다 작은 데이터 수 ",n)



def make_short_data():
    data_path1 = 'datasets/data/motion_body_HJK.pkl'
    data_path2 = 'datasets/data/motion_body_KTG.pkl'

    with open(data_path1, 'rb') as f:
        data1 = pickle.load(f)

    with open(data_path2, 'rb') as f:
        data2 = pickle.load(f)

    #print((data1[0]['joint_rotation_matrix'][0]))
    data = []
    fixed_size = 100
    print("------------------data1 size:", len(data1))
    split_data(data, data1,fixed_size)
    print("------------------data size:",len(data))
    print("------------------data2 size:", len(data2))
    split_data(data, data2, fixed_size)
    print("------------------data size:",len(data))
    # for d in data2:
    #     if d["n_frames"] < 600:
    #         data.append(d)

    with open(os.path.join("datasets/data/", 'motion_body_fixed_nohand_all.pkl'), 'wb') as f:
         pickle.dump(data, f)

if __name__ == "__main__":
    with open("datasets/data/motion_body_slow_fast.pkl", 'rb') as f:
        data1 = pickle.load(f)
    print(len(data1))
    #make_short_data()
    #train()
    #make_short_data()
    #make_result_data("motion_body_fixed_HJKKTG.pkl","rvae__0414.pt")
    #extract_latent_space("motion_body_fixed_HJKKTG.pkl","rvae__0414.pt")
    #extract_latent_from_data()

