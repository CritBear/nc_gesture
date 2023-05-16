import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import nc_gesture.classifier.tensors
from modules.classifier import Encoder_TRANSFORMER
import numpy as np
from config import Config

import pickle
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

def swap_dict_keys_and_values(dictionary):
    return {value: key for key, value in dictionary.items()}

def process_file_name(file_name):
    d = file_name.split('_')
    action =''.join(d[:-2])
    return action

def action_to_index_dict(data):
    dic = dict()
    index = 0
    for d in data:
        p = process_file_name(d['file_name'])
        if p not in dic:
            dic[p] = index
            index += 1
    return dic

class NCMocapDataset(Dataset):
    data_name = 'nc_mocap'

    def __init__(self, data):
        self._data = data

        # with open(data_path, 'rb') as f:
        #     self._data = pickle.load(f)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        action_dic = action_to_index_dict(self._data)
        style = self._data[idx]['persona']
        file_index = action_dic[process_file_name(self._data[idx]['file_name'])]
        motion = self._data[idx]['joint_rotation_matrix'].reshape(
            self._data[idx]['joint_rotation_matrix'].shape[0],
            -1
        ).astype(np.float32)  # [frames x joints x 3 x 3] -> [frames x pose data]
        return style,file_index, torch.from_numpy(motion)
# 하이퍼파라미터 설정


# 데이터셋 및 데이터로더 생성
config =  Config()

def train():
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO | Device : {config.device}")
    config.use_cuda = (True if config.device == 'cuda' else False)

    data_path = os.path.join(config.data_dir, config.data_file_name)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    dataset = NCMocapDataset(data)
    # 모델 초기화
    model = Encoder_TRANSFORMER("TVAE",config.num_joints,config.num_feats,
                                               config.num_classes,latent_dim= config.hidden_dim,num_heads=config.num_heads).to(config.device)

    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    datasets = {"train": train_dataset,"val":test_dataset}
    iterators = {key: DataLoader(datasets[key], batch_size=config.batch_size,
                                 shuffle=True, collate_fn=nc_gesture.classifier.tensors.collate)
                 for key in datasets.keys()}

    min_val_loss = 10

    for epoch in range(config.num_epochs):
        model.train()

        train_iter = tqdm(enumerate(iterators['train']), total=len(iterators['train']), desc="training")
        with torch.enable_grad():
            for i,batch in train_iter:
                # Forward 계산
                batch = {key: val.to(config.device) for key, val in batch.items()}

                batch = model(batch)
                loss = criterion(batch['output'], batch['style'])

                # Backward 및 경사도 업데이트
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        with torch.no_grad():
            model.eval()
            val_loss = 0
            total =0
            correct = 0
            for i, batch in enumerate(iterators['val']):
                batch = {key: val.to(config.device) for key, val in batch.items()}
                batch = model(batch)
                loss = criterion(batch['output'], batch['style'])
                val_loss += loss
                _,predict = torch.max(batch['output'],1)
                total += batch['output'].size(0)
                correct += (predict == batch['style']).sum()

            val_loss /= len(iterators['val'])
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), f"Result/classifier_best.pt")
        # 현재 에폭의 손실 출력
        print(f"Epoch [{epoch + 1}/{config.num_epochs}], Loss: {loss.item()} Val_loss: {val_loss.item()},ACC: {correct/total}")

    model.load_state_dict(torch.load('Result/classifier_best.pt'))
    cur_index_dic = swap_dict_keys_and_values(action_to_index_dict(data))
    num_correct = np.array([0] * len(cur_index_dic))
    num_total = np.array([0] * len(cur_index_dic))
    dataloader = DataLoader(dataset, batch_size=config.batch_size,
               shuffle=True, collate_fn=nc_gesture.classifier.tensors.collate)
    for aogai in range(10):
        with torch.no_grad():
            model.eval()
            val_loss = 0
            total = 0
            correct = 0
            for i, batch in enumerate(dataloader):
                batch = {key: val.to(config.device) for key, val in batch.items()}
                batch = model(batch)
                _, predict = torch.max(batch['output'], 1)
                total += batch['output'].size(0)
                correct += (predict == batch['style']).sum()
                matching_indices = (predict == batch['style']).nonzero().squeeze()
                for idx in range(len(batch['file'])):
                    cur_idx = batch['file'][idx]
                    num_total[cur_idx] += 1
                    if idx in matching_indices:
                        num_correct[cur_idx] += 1
            print(f"ACC: {correct/total}")

    high = []
    for i, tt in enumerate(zip(num_total, num_correct)):
        acc = tt[1] / tt[0]
        if acc > 0.5:
            high.append(cur_index_dic[i])
            print(cur_index_dic[i],":",acc)

    with open(f'high_action.pkl', 'wb') as f:
        pickle.dump(high,f)

if __name__ == "__main__":
    #load_result("train_result_BaseMST_motion_body_fixed_nohand_all.pkl_59340.pt.pkl")
    #result_visualize("BaseMST_motion_body_fixed_nohand_all.pkl_2000.pt")
    train()


#tensorboard --logdir=C:\Users\user\Desktop\NC\git\nc_gesture\style_transfer\logs