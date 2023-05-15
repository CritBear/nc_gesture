
from nc_gesture.style_transfer.utils.utils import *
from nc_gesture.style_transfer.utils.tensors import *
from torch.utils.data import Dataset
import numpy as np

# self.dataset[index] = {
#     'file_name': '',
#     'n_joints': 0,
#     'n_frames': 0,
#     'frame_time': 0,
#     'joint_name': [],         (n_joints, string)
#     'parent_indices': [],     (n_joints, int
#     'children_indices': [],   (n_joints, int list)
#     'end_site_indices': [],   (n_end_site, int)
#     'base_pose': [],          (n_joints x 3, initial joint position)
#     'root_position': [],      (n_frames x 3)
#     'joint_rotation_matrix': None,  (n_frames x n_joints x 3 x 2)
#     'persona': None
#     'action' :
#     'actionIdx':
#        'target_style':
#     'target_motion': (n_frames x n_joints x 3 x 2)
# }
class ActionStyleDataset(Dataset):
    data_name = 'nc_mocap'

    def __init__(self, data):
        self._data = data

        # with open(data_path, 'rb') as f:
        #     self._data = pickle.load(f)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        target_style = to_index(self._data[idx]['target_style'])
        target_motion = self._data[idx]['target_motion'].reshape(
            self._data[idx]['target_motion'].shape[0],
             -1 ).astype(np.float32)

        target = {'target_motion':target_motion,'style':target_style}

        motion = self._data[idx]['joint_rotation_matrix'].reshape(
            self._data[idx]['joint_rotation_matrix'].shape[0],
            -1
        ).astype(np.float32)  # [frames x joints x 3 x 3] -> [frames x pose data]
        return torch.from_numpy(motion),target