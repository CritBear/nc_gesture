import numpy as np
import os


class NCMocapDataset:
    data_name = 'nc_mocap'

    def __init__(self, data_path='', **kwargs):
        self.data_path = data_path

