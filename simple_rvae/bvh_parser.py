import re
import numpy as np
from math import radians, cos, sin


class BvhNode:
    def __init__(self, joint_name, index, parent_index, parent, is_end_site):
        self.joint_name = joint_name
        self.parent = parent
        self.is_end_site = is_end_site

        # Index of nodes in Bvh class
        self.index = index
        self.parent_index = parent_index

        self.offset = []
        self.channels_size = 0
        self.channels_info = []
        self.is_zxy_channels = False
        self.children = []

        # Lengths of these lists are number of frames
        self.channels = []
        self.position = []
        self.rotation = []

        if self.parent:
            self.parent.add_child(self)

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def tokenize(self, items):
        if items[0] == "OFFSET":
            if len(items) != 4:
                print("ERROR : Offset list length is not invalid.")

            self.offset = np.asarray(items[1:], dtype=float)

        elif items[0] == "CHANNELS":
            #print(f"{self.joint_name} | {items}")
            self.channels_size = int(items[1])
            self.channels_info = items[2:]

            if items[-3] == "Zrotation" and items[-2] == "Xrotation" and items[-1] == "Yrotation":
                self.is_zxy_channels = True

    def stack_frame(self, items):
        self.channels.append(items[:])

        position = np.zeros(3)
        rotation_euler = np.zeros(3)

        for idx, channel_info in enumerate(self.channels_info):
            if channel_info == "Xposition":
                position[0] = items[idx]
            elif channel_info == "Yposition":
                position[1] = items[idx]
            elif channel_info == "Zposition":
                position[2] = items[idx]
            elif channel_info == "Zrotation":
                rotation_euler[0] = items[idx]
            elif channel_info == "Xrotation":
                rotation_euler[1] = items[idx]
            elif channel_info == "Yrotation":
                rotation_euler[2] = items[idx]

        self.position.append(position)

        x_rot = self.x_rot_to_matrix(rotation_euler[1])
        y_rot = self.y_rot_to_matrix(rotation_euler[2])
        z_rot = self.z_rot_to_matrix(rotation_euler[0])

        rotation_matrix = z_rot @ x_rot @ y_rot
        #rotation_matrix = self.zxy_to_matrix(*rotation_euler)
        self.rotation.append(rotation_matrix)

    @staticmethod
    def x_rot_to_matrix(degree):
        x = radians(degree)

        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[0][0] = 1
        rotation_matrix[0][1] = 0
        rotation_matrix[0][2] = 0

        rotation_matrix[1][0] = 0
        rotation_matrix[1][1] = cos(x)
        rotation_matrix[1][2] = -sin(x)

        rotation_matrix[2][0] = 0
        rotation_matrix[2][1] = sin(x)
        rotation_matrix[2][2] = cos(x)

        return rotation_matrix

    @staticmethod
    def y_rot_to_matrix(degree):
        y = radians(degree)

        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[0][0] = cos(y)
        rotation_matrix[0][1] = 0
        rotation_matrix[0][2] = sin(y)

        rotation_matrix[1][0] = 0
        rotation_matrix[1][1] = 1
        rotation_matrix[1][2] = 0

        rotation_matrix[2][0] = -sin(y)
        rotation_matrix[2][1] = 0
        rotation_matrix[2][2] = cos(y)

        return rotation_matrix

    @staticmethod
    def z_rot_to_matrix(degree):
        z = radians(degree)

        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[0][0] = cos(z)
        rotation_matrix[0][1] = -sin(z)
        rotation_matrix[0][2] = 0

        rotation_matrix[1][0] = sin(z)
        rotation_matrix[1][1] = cos(z)
        rotation_matrix[1][2] = 0

        rotation_matrix[2][0] = 0
        rotation_matrix[2][1] = 0
        rotation_matrix[2][2] = 1

        return rotation_matrix

    @staticmethod
    def zxy_to_matrix(z_deg, x_deg, y_deg):
        z = radians(z_deg)
        x = radians(x_deg)
        y = radians(y_deg)

        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[0][0] = cos(y) * cos(z)
        rotation_matrix[0][1] = cos(y) * sin(z) + sin(y) * sin(x) * cos(z)
        rotation_matrix[0][2] = -1 * sin(y) * cos(x)
        rotation_matrix[1][0] = -1 * cos(x) * sin(z)
        rotation_matrix[1][1] = cos(x) * cos(z)
        rotation_matrix[1][2] = sin(x)
        rotation_matrix[2][0] = sin(y) * cos(z) + cos(y) * sin(x) * sin(z)
        rotation_matrix[2][1] = sin(y) * sin(z) - cos(y) * sin(x) * cos(z)
        rotation_matrix[2][2] = cos(y) * cos(x)

        return rotation_matrix


class Bvh:
    def __init__(self, data):
        self.data = data

        self.nodes = []
        self.n_frames = 0
        self.frame_time = 0
        self.frames = []

        self.tokenize()

    def add_node(self, joint_name="", parent_index=-1, is_end_site=False):
        self.nodes.append(
            BvhNode(
                joint_name=(joint_name if not is_end_site else self.nodes[parent_index].joint_name + "_EndSite"),
                index=len(self.nodes),
                parent_index=parent_index,
                parent=(self.nodes[parent_index] if parent_index != -1 else None),
                is_end_site=is_end_site
            )
        )

    def tokenize(self):
        lines = []
        accumulator = ''
        for char in self.data:
            if char not in ('\n', '\r'):
                accumulator += char
            elif accumulator:
                lines.append(re.split('\\s+', accumulator.strip()))
                accumulator = ''

        frame_time_found = False
        node_idx_stack = []

        for items in lines:
            if frame_time_found:
                # DEV | raw frame data is not required.
                # self.frames.append(np.asarray(items, dtype=float))
                channel_index = 0
                for node in self.nodes:
                    node.stack_frame(
                        np.asarray(
                            items[channel_index:channel_index + node.channels_size],
                            dtype=float
                        )
                    )
                    channel_index += node.channels_size
                continue

            if items[0] == "ROOT":
                self.add_node(joint_name=items[1])
            elif items[0] == "JOINT":
                self.add_node(joint_name=items[1], parent_index=node_idx_stack[-1])
            elif items[0] == "End" and items[1] == "Site":
                self.add_node(parent_index=node_idx_stack[-1], is_end_site=True)
            elif items[0] == "{":
                node_idx_stack.append(len(self.nodes) - 1)
            elif items[0] == "}":
                node_idx_stack.pop()
            elif items[0] == "OFFSET" or items[0] == "CHANNELS":
                self.nodes[-1].tokenize(items)
            elif items[0] == "Frames:":
                self.n_frames = int(items[1])
            elif items[0] == "Frame" and items[1] == "Time:":
                self.frame_time = float(items[2])
                frame_time_found = True

