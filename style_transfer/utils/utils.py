import numpy as np
from scipy.signal import savgol_filter

def process_file_name(file_name):
    d = file_name.split('_')
    action =''.join(d[:-2])
    return action

def get_style_from_name(file_name):
    d = file_name.split('_')
    return d[-2]

def to_style_index(file_name):
    style = get_style_from_name(file_name)
    return to_index(style)

def get_onehot_labels(styles):
    labels = []
    for s in styles:
        labels.append(to_style_index(s))
    return labels

def to_index(style):
    if style == "de":
        return 0#[1,0,0,0]
    elif style == "di":
        return 1#[0,1,0,0]
    elif style == "me":
        return 2#[0,0,1,0]
    elif style == "mi":
        return 3#[0,0,0,1]
    else:
        raise ValueError


def motion_denosing(motion,window_size = 60, polygonal = 8,joint_num=26):
    output = motion.reshape(-1, joint_num * 6).transpose().copy()
    for i in range(0, len(output)):
        output[i] = savgol_filter(output[i], window_size, polygonal)
    output = output.transpose()
    return output.reshape(-1, joint_num, 3, 2)



