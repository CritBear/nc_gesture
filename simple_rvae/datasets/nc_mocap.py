import os
import os.path
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import pickle
from torch.utils.data import Dataset


class NCMocapDataset(Dataset):
    data_name = 'nc_mocap'

    def __init__(self, data):
        self._data = data

        # with open(data_path, 'rb') as f:
        #     self._data = pickle.load(f)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        style = self._data[idx]['file_name']
        motion = self._data[idx]['joint_rotation_matrix'].reshape(
            self._data[idx]['joint_rotation_matrix'].shape[0],
            -1
        ).astype(np.float32)  # [frames x joints x 3 x 3] -> [frames x pose data]
        return style, torch.from_numpy(motion)


# data_path = 'data/refined_motion_test_HJK.pkl'
# a = NCMocapDataset(data_path)