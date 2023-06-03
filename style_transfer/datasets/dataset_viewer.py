import random as rand

import matplotlib.pyplot as plt
import vpython.no_notebook
from vpython import *

import numpy as np
import math
import pickle
import time
from scipy.signal import savgol_filter
from nc_gesture.simple_rvae.datasets.dataset_utils.bvh_parser import Bvh
from nc_gesture.style_transfer.utils.utils import *

import pandas as pd



class BodyModel:
    def __init__(self, data, model_offset):
        self.data = data
        self.bones = []
        self.joints = []
        self.model_offset = model_offset

        self.init_model()

    def init_model(self):
        # Virtual function
        raise NotImplementedError

    def update_joint(self, frame_idx, joint_idx):
        # Virtual function
        raise NotImplementedError

    def update_pose(self, frame_idx):
        # update joints
        self.update_joint(frame_idx, 0)

        # update bones
        for bone in self.bones:
            bone.obj.pos = self.joints[bone.oji].obj.pos
            bone.obj.axis = self.joints[bone.dji].obj.pos - self.joints[bone.oji].obj.pos

    def update(self, frame_idx):
        self.update_pose(frame_idx)


class PklModel(BodyModel):

    def __init__(self, data, model_offset=vector(0, 0, 0)):
        super().__init__(data, model_offset)

    def init_model(self):
        for joint_idx in range(self.data['n_joints']):
            if self.data['parent_indices'][joint_idx] == -1:
                self.joints.append(Joint(
                    joint_idx,
                    self.data['parent_indices'][joint_idx],
                    self.data['base_pose'][joint_idx]
                ))
            else:
                self.joints.append(Joint(
                    joint_idx,
                    self.data['parent_indices'][joint_idx],
                    self.data['base_pose'][joint_idx]
                ))
                self.bones.append(Bone(
                    self.data['parent_indices'][joint_idx],
                    joint_idx
                ))

    def update_joint(self, frame_idx, joint_idx):
        data = self.data

        rot_u = data['joint_rotation_matrix'][frame_idx][joint_idx][:, 0]
        rot_v = data['joint_rotation_matrix'][frame_idx][joint_idx][:, 1]
        rot_w = np.cross(rot_u, rot_v)
        rotation = np.stack([rot_u, rot_v, rot_w], 1)

        if joint_idx == 0:
            self.joints[joint_idx].update_local_transform(
                rotation=rotation,
                position=data['root_position'][frame_idx]
            )
            self.joints[joint_idx].global_transform = self.joints[joint_idx].local_transform
        else:
            self.joints[joint_idx].update_local_transform(
                rotation=rotation
            )
            self.joints[joint_idx].global_transform = \
                self.joints[self.joints[joint_idx].parent_idx].global_transform @ self.joints[joint_idx].local_transform

        self.joints[joint_idx].update_obj(self.model_offset)

        for child_idx in data['children_indices'][joint_idx]:
            self.update_joint(frame_idx, child_idx)

    def reset_model(self):

        for bone in self.bones:
            bone.obj.visible = False

        for joint in self.joints:
            joint.obj.visible = False

        del self.data
        del self.bones
        del self.joints


class BvhModel(BodyModel):

    def __init__(self, data, model_offset=vector(0, 0, 0)):
        super().__init__(data, model_offset)

        self.joints_parent_idx = None
        self.joints_children_idx = None

    def init_model(self):
        for node in self.data.nodes:
            if node.parent == -1 or node.parent is None:  # Root
                self.joints.append(Joint(node.index, node.parent_index, node.offset))
            else:  # Joint, End site
                self.joints.append(Joint(node.index, node.parent_index, node.offset))
                self.bones.append(Bone(node.parent_index, node.index))

    def update_joint(self, frame_idx, joint_idx):
        bvh = self.data
        if joint_idx == 0:  # Root
            self.joints[joint_idx].update_local_transform(
                rotation=bvh.nodes[joint_idx].rotation[frame_idx],
                position=bvh.nodes[joint_idx].position[frame_idx]
            )
            self.joints[joint_idx].global_transform = self.joints[joint_idx].local_transform
        else:
            self.joints[joint_idx].update_local_transform(
                rotation=bvh.nodes[joint_idx].rotation[frame_idx]
            )
            self.joints[joint_idx].global_transform = \
                self.joints[self.joints[joint_idx].parent_idx].global_transform @ self.joints[joint_idx].local_transform

        self.joints[joint_idx].update_obj(self.model_offset)

        for child_node in bvh.nodes[joint_idx].children:
            self.update_joint(frame_idx, child_node.index)


class Joint:
    def __init__(self, idx, parent_idx, offset):
        if idx == 0:
            color = vpython.color.red
        elif idx == 1 or idx == 6:
            color = vpython.color.green
        elif idx == 11:
            color = vpython.color.blue
        else:
            color = vpython.color.cyan
        self.obj = sphere(radius=1, color=color)
        self.idx = idx
        self.parent_idx = parent_idx
        self.local_transform = np.array([
            [1, 0, 0, offset[0]],
            [0, 1, 0, offset[1]],
            [0, 0, 1, offset[2]],
            [0, 0, 0, 1]
        ])
        self.global_transform = np.zeros((4, 4))

    def update_obj(self, model_offset):
        self.obj.pos = vector(*self.global_transform[:3, 3]) + model_offset

    def update_local_transform(self, rotation, position=None):
        if position is not None:
            self.local_transform[:3, 3] = position

        self.local_transform[:3, :3] = rotation


class Bone:
    def __init__(self, origin_joint_idx, destination_joint_idx):
        self.obj = cylinder(radius=0.5, color=color.white)

        self.oji = origin_joint_idx
        self.dji = destination_joint_idx


class Viewer:
    def __init__(self):
        self.window = canvas(x=0, y=0, width=1200, height=1200, center=vector(0, 0, 0), background=vector(0, 0, 0))

        # self.axis_x = arrow(pos=vector(0, 0, 0), axis=vector(40, 0, 0), shaftwidth=1, color=vpython.color.red)
        # self.axis_y = arrow(pos=vector(0, 0, 0), axis=vector(0, 40, 0), shaftwidth=1, color=vpython.color.green)
        # self.axis_z = arrow(pos=vector(0, 0, 0), axis=vector(0, 0, 40), shaftwidth=1, color=vpython.color.blue)

        self.texts = []

        self.max_frame_length = 0
        self.models = []
        self.n_frames = []
        self.frame_times = []

    def init_model(self):
        n_frames = 0
        frame_time = 0
        n_joints = 0

        for model in self.models:
            model.reset_model()

        for t in self.texts:
            t.visible = False
            del t

        self.texts = []

        self.models = []
        self.n_frames = []
        self.frame_times = []

    def load_file(self, path, pkl_idx=0, model_offset=vector(0, 0, 0), inference_path=None, is_original_data = True, is_model = False,labelText= None,denoising = False,motion_data = None):

        if 'bvh' in path:
            with open(path) as f:
                bvh = Bvh(f.read())

            model = BvhModel(bvh, model_offset)

            n_frames = bvh.n_frames
            frame_time = bvh.frame_time
            n_joints = len(bvh.nodes)

            if labelText:
                self.texts.append(text(text=labelText,pos=vector(model_offset.x, model_offset.y - 10, model_offset.z), align="center", height=10))

        elif 'pkl' in path or motion_data:
            if motion_data:
                pkl_data = motion_data
            else:
                with open(path, 'rb') as f:
                    dataset = pickle.load(f)
                    pkl_data = dataset[pkl_idx]

            style = pkl_data['persona']
            if is_model:
                if labelText in pkl_data:
                    if denoising:
                        pkl_data['joint_rotation_matrix'] = motion_denosing(pkl_data[labelText],joint_num=len(pkl_data['joint_rotation_matrix'][0]))
                    else:
                        pkl_data['joint_rotation_matrix'] = pkl_data['output']
                    style = pkl_data['target_style']
                    if labelText == "output2":
                        style = pkl_data['target_style2']

            else:
                if not is_original_data:
                    if 'target_motion' in pkl_data:
                        pkl_data['joint_rotation_matrix'] = pkl_data['target_motion']
                        style = pkl_data['target_style']

            if labelText:

                self.texts.append(text(text=f'{labelText} ({style})',pos=vector(model_offset.x, model_offset.y - 30, model_offset.z), align="center", height=10))

            if inference_path is not None:
                with open(inference_path, 'rb') as f:
                    inference_data = pickle.load(f)
                    recon_motion = inference_data['recon_motion'][pkl_idx]

                    pkl_data['joint_rotation_matrix'] = recon_motion.reshape(
                        recon_motion.shape[0],
                        -1,
                        3,
                        2
                    )
                    print(pkl_data['joint_rotation_matrix'].shape)

            model = PklModel(pkl_data, model_offset)

            n_frames = pkl_data['n_frames']
            frame_time = pkl_data['frame_time']
            n_joints = pkl_data['n_joints']

        else:
            raise ValueError("file format has to be 'bvh' or 'pkl'.")

        # print(f'Frame length : {n_frames}')
        # print(f'Frame time : {frame_time}')
        # print(f'Num of joints : {n_joints}')

        if self.max_frame_length < n_frames:
            self.max_frame_length = n_frames

        self.models.append(model)
        self.n_frames.append(n_frames)
        self.frame_times.append(frame_time)


    def run_motion(self):
        for i in range(len(self.frame_times) - 1):
            if self.frame_times[i] != self.frame_times[i + 1]:
                print("Warning: Frame times are not same.")

        for frame_idx in range(self.max_frame_length-100):
            for model_idx in range(len(self.models)):
                if min(self.n_frames[model_idx],800)> frame_idx:
                    self.models[model_idx].update(frame_idx)

            time.sleep(self.frame_times[0])

def view4style(path):
    data_path = path
    with open(data_path, 'rb') as f:
        motion_data = pickle.load(f)
    viewer = Viewer()

    start = 0
    end = len(motion_data)
    for data_idx in range(start, end, 4):
        viewer.init_model()
        cur_labels = [motion_data[data_idx+i]['persona'] for i in range(0,4)]
        print(motion_data[data_idx]['file_name'])
        viewer.load_file("C:/Users/user/Desktop/NC/git/nc_gesture/simple_rvae/datasets/data/motion_body_hand_KTG.pkl",
                         pkl_idx=data_idx, model_offset=vector(-50, 20, 0),
                         is_original_data=True,labelText= cur_labels[0],motion_data=motion_data[data_idx])
        viewer.load_file("C:/Users/user/Desktop/NC/git/nc_gesture/simple_rvae/datasets/data/motion_body_hand_KTG.pkl",
                         pkl_idx=data_idx + 1, model_offset=vector(50, 20, 0),
                         is_original_data=True, labelText= cur_labels[1], motion_data=motion_data[data_idx + 1])
        viewer.load_file("C:/Users/user/Desktop/NC/git/nc_gesture/simple_rvae/datasets/data/motion_body_hand_KTG.pkl",
                         pkl_idx=data_idx + 2, model_offset=vector(-50, -200, 0),
                         is_original_data=True,labelText= cur_labels[2], motion_data=motion_data[data_idx + 2])
        viewer.load_file("C:/Users/user/Desktop/NC/git/nc_gesture/simple_rvae/datasets/data/motion_body_hand_KTG.pkl",
                         pkl_idx=data_idx + 3, model_offset=vector(50, -200, 0),
                         is_original_data=True,labelText= cur_labels[3], motion_data=motion_data[data_idx + 3])
        viewer.run_motion()
def view_result(model_result):
    np.set_printoptions(precision=4, suppress=True)

    viewer = Viewer()
    with open(model_result, 'rb') as f:
        data = pickle.load(f)

    start = 0
    end = len(data)
    dis = 0
    if "output" in data[0]:
        dis = 75
    if "target_motion" in data[0]:
        dis = 100
    for data_idx in range(start, end):
        init_x = -dis
        viewer.init_model()
        viewer.load_file("",
                         pkl_idx=data_idx, model_offset=vector(init_x, 0, 0),
                         is_original_data=True, labelText="original", motion_data=data[data_idx].copy())
        if "target_motion" in data[0]:
            viewer.load_file("",
                             pkl_idx=data_idx, model_offset=vector(init_x+dis, 0, 0),
                             is_original_data=False,labelText="style", motion_data=data[data_idx].copy())
        if "output" in data[0]:
            viewer.load_file("",
                             pkl_idx=data_idx, model_offset=vector(init_x + dis*2, 0, 0),
                             is_original_data=True,is_model=True,denoising=True,labelText="output",motion_data=data[data_idx].copy())
        if "output2" in data[0]:
            viewer.load_file("",
                             pkl_idx=data_idx, model_offset=vector(init_x + dis*3, 0, 0),
                             is_original_data=True,is_model=True,denoising=False,labelText="output2",motion_data=data[data_idx].copy())

        viewer.run_motion()
        time.sleep(1)

def main():
    np.set_printoptions(precision=4, suppress=True)

    viewer = Viewer()
    viewer.init_model()

    with open("data/variable_600_all.pkl", 'rb') as f:
        motion_data = pickle.load(f)
    start = 0
    end = len(motion_data)

    for data_idx in range(start, end,4):
        viewer.init_model()
        data_idx = rand.randint(start ,end)
        #viewer.load_file("C:/Users/user/Desktop/NC/HJK/VAAI_Non_R_12_de_01.bvh",model_offset=vector(-150,0,0))
        #viewer.load_file("C:/Users/user/Desktop/NC/HJK/VAAI_Non_R_12_di_01.bvh", model_offset=vector(150, 0, 0))
        #viewer.load_file("data/fixed_800_all.pkl", pkl_idx=data_idx, model_offset=vector(-150, 0, 0), is_original_data = True)
        #viewer.load_file("data/style_pair_all.pkl", pkl_idx=data_idx, model_offset=vector(-150, 0, 0),is_original_data=True,labelText="original")
        #viewer.load_file("data/style_pair_all.pkl", pkl_idx=data_idx, model_offset=vector(0, 0, 0),is_original_data=False,labelText="styled")
        viewer.load_file("data/variable_600_all.pkl", pkl_idx=0, model_offset=vector(-150, 0, 0), is_original_data=True,
                         labelText="original")
        viewer.load_file("data/decord_result_tVAE_best_64.pkl", pkl_idx=0, model_offset=vector(0, 0, 0),
                         is_original_data=True, labelText="output", denoising=False)
        # viewer.load_file("data/variable_all.pkl", pkl_idx=data_idx, model_offset=vector(0, 0, 0),is_original_data=False,labelText="styled")
        #viewer.load_file("data/fixed_300_all.pkl", pkl_idx=data_idx, model_offset=vector(-150, 0, 0),is_original_data=True,labelText="original")
        #viewer.load_file("data/fixed_300_all.pkl", pkl_idx=data_idx, model_offset=vector(0, 0, 0),is_original_data=False,labelText="styled")
        # viewer.load_file("data/decord_result_tVAE_best_128_8_sequence.pkl", pkl_idx=data_idx, model_offset=vector(0, 0, 0), is_original_data = True, labelText= "output")
        # viewer.load_file("data/decord_result_tVAE_best_64.pkl", pkl_idx=data_idx, model_offset=vector(0, 0, 0),
        #                  is_original_data=True, labelText="output",denoising=False)
        # viewer.load_file("data/decord_result_tVAE_best_64.pkl", pkl_idx=data_idx, model_offset=vector(150, 0, 0),
        #                  is_original_data=True, labelText="output",denoising=True)
        # viewer.load_file("data/decord_result_tVAE_best_128_8_sequence.pkl", pkl_idx=data_idx, model_offset=vector(150, 0, 0),
        #                  is_original_data=True, labelText="output",denoising=True)
        #viewer.load_file("data/decord_result_BaseMST_style_nohand_fixed_all.pkl_10000_2.pkl", pkl_idx=data_idx, model_offset=vector(150, 0, 0), is_original_data = False)

        viewer.run_motion()

    while True:
        time.sleep(5)
        print("End")

if __name__ == "__main__":
    #main()
    #view4style("data/classifier_800_all.pkl")
    #view4style("C:/Users/user/Desktop/NC/git/nc_gesture/simple_rvae/datasets/data/motion_body_hand_KTG.pkl")
    #view_result("data/decord_result_tVAE_best.pkl")
    view_result("data/lhsf_300_fixed_all.pkl")
    #view_result("data/classifier_800_all.pkl")