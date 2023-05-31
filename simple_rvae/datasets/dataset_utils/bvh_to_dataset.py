import os
import pickle
import copy
import numpy as np
import gc

from bvh_parser import Bvh


class DatasetGenerator:

    def __init__(self, except_hand=False):

        self.dataset = []
        # self.dataset[index] = {       # 데이터 형식
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
        # }
        self.except_hand = except_hand  # 손가락 관절 제외할지 선택

    def reset(self):
        del self.dataset
        gc.collect()  # python garbage collector 비워서 메모리 확보

        self.dataset = []

    def save(self, dir_path, file_name):
        with open(os.path.join(dir_path, f'{file_name}.pkl'), 'wb') as f:
            pickle.dump(self.dataset, f)

    def read(self, dir_path, file_name):
        data = {
            'file_name': '',
            'n_joints': 0,
            'n_frames': 0,
            'frame_time': 0,
            'joint_name': [],
            'parent_indices': [],
            'children_indices': [],
            'end_site_indices': [],
            'base_pose': [],
            'root_position': None,
            # 'joint_local_transform': None,
            # 'joint_position': None,
            # 'joint_axis_angle': None,
            'joint_rotation_matrix': None,
            'persona': None
        }

        with open(os.path.join(dir_path, file_name)) as f:
            bvh = Bvh(f.read())

        if file_name.find("_de_") != -1:
            data['persona'] = 'de'
        elif file_name.find("_di_") != -1:
            data['persona'] = 'di'
        elif file_name.find("_me_") != -1:
            data['persona'] = 'me'
        elif file_name.find("_mi_") != -1:
            data['persona'] = 'mi'
        else:
            data['persona'] = 'undefined'

        data['file_name'] = file_name
        data['n_frames'] = bvh.n_frames
        data['frame_time'] = bvh.frame_time

        # per_joint_local_transform = []
        # per_joint_position = []
        # per_joint_axis_angle = []
        per_joint_rotation_matrix = []

        for node in bvh.nodes:
            if node.parent_index == -1:
                data['root_position'] = np.array(node.position)
                break

        conf_joints_idx = []
        for node_idx, node in enumerate(bvh.nodes):
            if self.except_hand:
                if 'Finger' in node.joint_name:
                    continue

            conf_joints_idx.append(node_idx)

        data['n_joints'] = len(conf_joints_idx)

        for node_idx, node in enumerate(bvh.nodes):
            if node_idx not in conf_joints_idx:
                continue

            is_temp_end_site = False
            if self.except_hand:
                if 'Hand' in node.joint_name:
                    is_temp_end_site = True

            data['joint_name'].append(node.joint_name)
            if node.parent_index == -1:
                data['parent_indices'].append(node.parent_index)
            else:
                data['parent_indices'].append(conf_joints_idx.index(node.parent_index))

            children_idx = []
            for child in node.children:
                if child.index in conf_joints_idx:
                    children_idx.append(conf_joints_idx.index(child.index))

            data['children_indices'].append(children_idx)
            data['base_pose'].append(node.offset)

            if node.is_end_site or is_temp_end_site:
                data['end_site_indices'].append(conf_joints_idx.index(node_idx))

            per_joint_rotation_matrix.append(node.rotation)

            # per_frame_axis_angle = []
            # per_frame_local_transform = []

            # for rot_mat in node.rotation:
            #     per_frame_axis_angle.append(self.rotation_matrix_to_axis_angle(rot_mat).reshape(-1))
            #
            #     local_transform = np.identity(4)
            #     local_transform[:3, :3] = rot_mat
            #     local_transform[0, 3] = node.offset[0]
            #     local_transform[1, 3] = node.offset[1]
            #     local_transform[2, 3] = node.offset[2]
            #     per_frame_local_transform.append(copy.deepcopy(local_transform))

            # per_joint_axis_angle.append(per_frame_axis_angle)
            # per_joint_local_transform.append(per_frame_local_transform)

        # data['joint_local_transform'] = np.swapaxes(per_joint_local_transform, 0, 1)
        # data['joint_axis_angle'] = np.swapaxes(per_joint_axis_angle, 0, 1)
        data['joint_rotation_matrix'] = np.swapaxes(per_joint_rotation_matrix, 0, 1)[:, :, :, :2]

        # for i in data:
        #     print(i)
        #     if hasattr(data[i], 'shape'):
        #         print(data[i].shape)
        #     elif hasattr(data[i], '__len__'):
        #         print(len(data[i]))
        #     else:
        #         print(data[i])

        self.dataset.append(data)

    @staticmethod
    def rotation_matrix_to_axis_angle(R):
        """
        Convert a 3x3 rotation matrix R to an axis-angle representation.
        """
        # Compute the trace of the matrix
        trace = np.trace(R)

        # Compute the angle of rotation
        angle = np.arccos((trace - 1) / 2)

        # Compute the axis of rotation
        if np.abs(angle) < 1e-6:
            axis = np.array([0, 0, 1])
        elif np.abs(angle - np.pi) < 1e-6:
            # In this case, R = -I, so any vector is a valid rotation axis
            axis = np.random.rand(3)
            axis /= np.linalg.norm(axis)
        else:
            axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            axis /= np.linalg.norm(axis)

        # Return the axis-angle representation
        return angle * axis


def main():
    generator = DatasetGenerator(except_hand=False)

    raw_dataset_dir_path = '..\\bvh_data'  # ..\\bvh_data\sub_dir\*.bvh
    refined_dataset_dir_path = '..\\data'  # pkl 형식의 데이터셋을 저장할 디렉토리

    for dir in os.listdir(raw_dataset_dir_path):
        generator.reset()  # generator에 쌓여진 bvh 파일들 비우기

        sub_dir = os.path.join(raw_dataset_dir_path, dir)  # raw_dataset_dir_path 내부의 sub_dir
        if os.path.isdir(sub_dir):
            for file_name in [f for f in os.listdir(sub_dir) if f.endswith(".bvh")]:  # bvh 파일만 가져오기
                generator.read(sub_dir, file_name)
                print(f'complete. {file_name}')

        # sub_dir 에 있는 bvh 파일들을 하나의 dataset으로 묶어서 저장
        generator.save(refined_dataset_dir_path, f'motion_body_hand_{dir}')


if __name__ == '__main__':
    main()
