import numpy as np
import os
import pickle
from torch.utils.data import Dataset


class NCMocapDataset(Dataset):
    data_name = 'nc_mocap'

    def __init__(self, data_dir_path=''):
        self.data_path = data_dir_path

        file_name = 'refiend_mo' \
                    'tion_test_HJK.pkl'

        self._data = None

        data_file_path = os.path.join(data_dir_path, file_name)
        with open(data_file_path, 'rb') as f:
            self._data = pickle.load(f)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        style = self._data[idx]['persona']
        motion = self._data[idx]['joint_rotation_matrix'].reshape(
            self._data[idx]['joint_rotation_matrix'].shape[0],
            -1
        )  # [frames x joints x 3 x 3] -> [frames x pose data]
        return style, motion

