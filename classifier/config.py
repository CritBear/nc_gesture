import os
import sys
import torch
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)

class Config():
    data_dir = pjoin(BASEPATH, 'datasets\data')
    expr_dir = BASEPATH
    data_file_name = "motion_body_hand_slow_fast.pkl" #"classifier_900_high.pkl"#"classifier_high_4.pkl"#"classifier_high_all.pkl"#"variable_all.pkl" #"classifier_high_all.pkl"#variable_all.pkl" #classifier_800_all.pkl

    num_classes = 4
    hidden_dim = 16
    num_joints = 74
    num_feats = 6
    num_heads = 4
    num_epochs = 200
    batch_size = 5
    learning_rate = 0.00005