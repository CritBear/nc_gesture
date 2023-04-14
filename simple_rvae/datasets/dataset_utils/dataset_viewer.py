import vpython.no_notebook
from vpython import *

import numpy as np
import math
import pickle
import time

from bvh_parser import Bvh


class BodyModel:
    def __init__(self, data):
        self.data = data
        self.bones = []
        self.joints = []

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

    def __init__(self, data):
        super().__init__(data)

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

        self.joints[joint_idx].update_obj()

        for child_idx in data['children_indices'][joint_idx]:
            self.update_joint(frame_idx, child_idx)


class BvhModel(BodyModel):

    def __init__(self, data):
        super().__init__(data)

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

        self.joints[joint_idx].update_obj()

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

    def update_obj(self):
        self.obj.pos = vector(*self.global_transform[:3, 3])

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

        self.axis_x = arrow(pos=vector(0, 0, 0), axis=vector(40, 0, 0), shaftwidth=1, color=vpython.color.red)
        self.axis_y = arrow(pos=vector(0, 0, 0), axis=vector(0, 40, 0), shaftwidth=1, color=vpython.color.green)
        self.axis_z = arrow(pos=vector(0, 0, 0), axis=vector(0, 0, 40), shaftwidth=1, color=vpython.color.blue)

    def run_motion(self, path, pkl_idx=5):
        frame_idx = 0
        n_frames = 0
        frame_time = 0

        if 'bvh' in path:
            with open(path) as f:
                bvh = Bvh(f.read())

            model = BvhModel(bvh)

            n_frames = bvh.n_frames
            frame_time = bvh.frame_time
            print(f'Frame length : {n_frames}')

        elif 'pkl' in path:
            with open(path, 'rb') as f:
                pkl_data = pickle.load(f)[pkl_idx]

            model = PklModel(pkl_data)

            n_frames = pkl_data['n_frames']
            frame_time = pkl_data['frame_time']
            file_name = pkl_data['file_name']
            print(f'Frame file name :{file_name}')

        else:
            raise ValueError("file format has to be 'bvh' or 'pkl'.")

        print(f'Frame length : {n_frames}')
        print(f'Frame time : {frame_time}')

        while True:
            time.sleep(frame_time)
            model.update(frame_idx)

            frame_idx += 1
            if frame_idx >= n_frames:
                for obj in self.window.objects:
                    obj.visible = False
                    obj.delete()
                return


def main():
    np.set_printoptions(precision=4, suppress=True)

    viewer = Viewer()
    for i in range(130,200):
       viewer.run_motion("../../decored_data_0414.pkl",pkl_idx=i)
        #viewer.run_motion("../data/motion_body_fixed_HJKKTG.pkl",pkl_idx=i)
    # viewer.run_motion("./motion_data/KTG/VAAI_Non_E_01_de_01.bvh")

    while True:
        time.sleep(5)
        print("End")

main()