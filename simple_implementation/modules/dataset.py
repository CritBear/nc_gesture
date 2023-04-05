import torch

import numpy as np
import random

from ..utils.misc import to_torch
from ..utils.tensors import collate
from ..utils import rotation_conversions as geometry


class Dataset(torch.utils.data.Dataset):

    def __init__(self, options):
        super().__init__()

        self.num_frames = options.dataset.num_frames
        self.sampling = options.dataset.sampling
        self.sampling_step = options.dataset.sampling_step
        self.split = options.dataset.split
        self.pose_rep = options.dataset.pose_rep
        self.translation = options.datsaet.translation
        self.glob = options.dataset.glob
        self.max_len = options.dataset.max_len
        self.min_len = options.dataset.min_len
        self.num_seq_max = options.dataset.num_seq_max

        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f'{self.split} is not a valid split')

        self._original_train = None
        self._original_test = None

        # TBD
        # self._personas
        # self._train/self._test
        # self._num_frames_in_data[data_index]
        # self._persona_to_label[action]
        # self._label_to_persona[label]
        # self._load_pose(data_idx, frame_idx)
        # self._personas[idx] # => carefull changed here
        # self._persona_classes[action]

    def persona_to_label(self, persona):
        return self._persona_to_label[persona]

    def label_to_persona(self, label):
        import numbers
        if isinstance(label, numbers.Integral):
            return self._label_to_persona[label]
        else:
            label = np.argmax(label)
            return self._label_to_persona[label]

    def get_pose_data(self, data_idx, frame_ix):
        pose = self._load(data_idx, frame_ix)
        label = self.get_label(data_idx)
        return pose, label

    def get_label(self, idx):
        persona = self.get_persona(idx)
        return self.persona_to_label(persona)

    # def parse_persona(self, path, return_int=True):
    #     info = parse_info_name(path)["A"]
    #     if return_int:
    #         return int(info)
    #     else:
    #         return info

    def get_persona(self, ind):
        return self._personas[ind]

    def persona_to_persona_name(self, persona):
        return self._persona_classes[persona]

    def label_to_persona_name(self, label):
        persona = self.label_to_persona(label)
        return self.persona_to_persona_name(persona)

    def __getitem__(self, idx):
        if self.split == 'train':
            data_idx = self._train[idx]
        else:
            data_idx = self._test[idx]

        inp, target = self._get_item_data_index(data_idx)
        return inp, target

    def _load(self, ind, frame_ix):
        pose_rep = self.pose_rep
        if pose_rep == 'xyz' or self.translation:
            if getattr(self, '_load_joints3D', None) is not None:
                joints3D = self._load_joints3D(ind, frame_ix)
                joints3D = joints3D - joints3D[0, 0, :]
                ret = to_torch(joints3D)
                if self.translation:
                    ret_tr = ret[:, 0, :]
            else:
                if pose_rep == 'xyz':
                    raise ValueError("This representation is not possible.")
                if getattr(self, "_load_translation") is None:
                    raise ValueError("Can't extract translations.")
                ret_tr = self._load_translation(ind, frame_ix)
                ret_tr = to_torch(ret_tr - ret_tr[0])

        if pose_rep != 'xyz':
            if getattr(self, '_load_rotvec', None) is None:
                raise ValueError("This representation is not possible.")
            else:
                pose = self._load_rotvec(ind, frame_ix)
                if not self.glob:
                    pose = pose[:, 1:, :]
                pose = to_torch(pose)
                if pose_rep == "rotvec":
                    ret = pose
                elif pose_rep == "rotmat":
                    ret = geometry.axis_angle_to_matrix(pose).view(*pose.shape[:2], 9)
                elif pose_rep == "rotquat":
                    ret = geometry.axis_angle_to_quaternion(pose)
                elif pose_rep == "rot6d":
                    ret = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose))
        if pose_rep != 'xyz' and self.translation:
            padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
            padded_tr[:, :3] = ret_tr
            ret = torch.cat((ret, padded_tr[:, None]), 1)

        ret = ret.permute(1, 2, 0).contiguous()
        return ret.float()

    def _get_item_data_index(self, data_idx):
        nframes = self._num_frames_in_data[data_idx]

        if self.num_frames == -1 and (self.max_len == -1 or nframes <= self.max_len):
            frame_ix = np.arange(nframes)
        else:
            if self.num_frames == -2:
                if self.min_len <= 0:
                    raise ValueError("You should put a min_len > 0 for num_frames == -2 mode")
                if self.max_len != -1:
                    max_frame = min(nframes, self.max_len)
                else:
                    max_frame = nframes

                num_frames = random.randint(self.min_len, max(max_frame, self.min_len))
            else:
                num_frames = self.num_frames if self.num_frames != -1 else self.max_len
            # sampling goal: input: ----------- 11 nframes
            #                       o--o--o--o- 4  ninputs
            #
            # step number is computed like that: [(11-1)/(4-1)] = 3
            #                   [---][---][---][-
            # So step = 3, and we take 0 to step*ninputs+1 with steps
            #                   [o--][o--][o--][o-]
            # then we can randomly shift the vector
            #                   -[o--][o--][o--]o
            # If there are too much frames required
            if num_frames > nframes:
                fair = False
                if fair:
                    choices = np.random.choice(range(nframes), num_frames, replace=True)
                    frame_ix = sorted(choices)
                else:
                    ntoadd = max(0, num_frames - nframes)
                    lastframe = nframes - 1
                    padding = lastframe * np.ones(ntoadd, dtype=int)
                    frame_ix = np.concatenate((np.arange(0, nframes), padding))

            elif self.sampling in ['conseq', 'random_conseq']:
                step_max = (nframes - 1) // (num_frames - 1)
                if self.sampling == 'conseq':
                    if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                        step = step_max
                    else:
                        step = self.sampling_step
                elif self.sampling == 'random_conseq':
                    step = random.randint(1, step_max)

                lastone = step * (num_frames - 1)
                shift_max = nframes - lastone - 1
                shift = random.randint(0, max(0, shift_max - 1))
                frame_ix = shift + np.arange(0, lastone + 1, step)

            elif self.sampling == 'random':
                choices = np.random.choice(range(nframes), num_frames, replace=False)
                frame_ix = sorted(choices)

            else:
                raise ValueError('Sampling not recognized.')

        inp, target = self.get_pose_data(data_idx, frame_ix)
        return inp, target

    def get_label_sample(self, label, n=1, return_labels=False, return_index=False):
        if self.split == 'train':
            index = self._train
        else:
            index = self._test

        action = self.label_to_action(label)
        choices = np.argwhere(np.array(self._actions)[index] == action).squeeze(1)

        if n == 1:
            data_index = index[np.random.choice(choices)]
            x, y = self._get_item_data_index(data_index)
            assert (label == y)
            y = label
        else:
            data_index = np.random.choice(choices, n)
            x = np.stack([self._get_item_data_index(index[di])[0] for di in data_index])
            y = label * np.ones(n, dtype=int)
        if return_labels:
            if return_index:
                return x, y, data_index
            return x, y
        else:
            if return_index:
                return x, data_index
            return x

    def get_label_sample_batch(self, labels):
        samples = [self.get_label_sample(label, n=1, return_labels=True, return_index=False) for label in labels]
        batch = collate(samples)
        x = batch["x"]
        mask = batch["mask"]
        lengths = mask.sum(1)
        return x, mask, lengths

    def get_mean_length_label(self, label):
        if self.num_frames != -1:
            return self.num_frames

        if self.split == 'train':
            index = self._train
        else:
            index = self._test

        action = self.label_to_action(label)
        choices = np.argwhere(self._actions[index] == action).squeeze(1)
        lengths = self._num_frames_in_video[np.array(index)[choices]]

        if self.max_len == -1:
            return np.mean(lengths)
        else:
            # make the lengths less than max_len
            lengths[lengths > self.max_len] = self.max_len
        return np.mean(lengths)

    def get_stats(self):
        if self.split == 'train':
            index = self._train
        else:
            index = self._test

        numframes = self._num_frames_in_video[index]
        allmeans = np.array([self.get_mean_length_label(x) for x in range(self.num_classes)])

        stats = {"name": self.dataname,
                 "number of classes": self.num_classes,
                 "number of sequences": len(self),
                 "duration: min": int(numframes.min()),
                 "duration: max": int(numframes.max()),
                 "duration: mean": int(numframes.mean()),
                 "duration mean/action: min": int(allmeans.min()),
                 "duration mean/action: max": int(allmeans.max()),
                 "duration mean/action: mean": int(allmeans.mean())}
        return stats

    def __len__(self):
        num_seq_max = getattr(self, "num_seq_max", -1)
        if num_seq_max == -1:
            from math import inf
            num_seq_max = inf

        if self.split == 'train':
            return min(len(self._train), num_seq_max)
        else:
            return min(len(self._test), num_seq_max)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"

    def update_parameters(self, parameters):
        self.njoints, self.nfeats, _ = self[0][0].shape
        parameters["num_classes"] = self.num_classes
        parameters["nfeats"] = self.nfeats
        parameters["njoints"] = self.njoints

    def shuffle(self):
        if self.split == 'train':
            random.shuffle(self._train)
        else:
            random.shuffle(self._test)

    def reset_shuffle(self):
        if self.split == 'train':
            if self._original_train is None:
                self._original_train = self._train
            else:
                self._train = self._original_train
        else:
            if self._original_test is None:
                self._original_test = self._test
            else:
                self._test = self._original_test