
import pickle
import os
import random
from nc_gesture.style_transfer.utils.utils import *
import itertools

def get_same_action(action,dataset,usedId = -1):
    sames = []
    for idx, d in enumerate(dataset):
        if idx != usedId and process_file_name(d['file_name']) == action:
            sames.append(dataset[idx])

    if len(sames) == 0:
        return random.choice(dataset)
    return random.choice(sames)


def make_style_dataset(dataset_name):
    with open(dataset_name, 'rb') as f:
        data = pickle.load(f)

    e_data = []
    i_data = []
    style_data = []
    same_style_data = []
    diff_style_data = []
    for d in data:
        if 'e' in d['persona']:
            e_data.append(d)
        else:
            i_data.append(d)

    e_index = 0
    i_index = 0
    for d in data:
        action = process_file_name(d['file_name'])
        if 'e' in d['persona']:
            style_data.append(e_data[e_index])
            same_style_data.append(get_same_action(action,e_data,e_index))
            diff_style_data.append(get_same_action(action, i_data, e_index))
            e_index +=1
        else:
            style_data.append(i_data[i_index])
            same_style_data.append(get_same_action(action,i_data,i_index))
            diff_style_data.append(get_same_action(action, e_data, i_index))
            i_index +=1

    style_dataset = {"origin_dataset_name":dataset_name,
                     "origin_data":data,
                    "style_data":style_data,
                     "same_style_data":same_style_data,
                     "diff_style_data":diff_style_data}


    with open('style_nohand_fixed_all.pkl', 'wb') as f:
        pickle.dump(style_dataset,f)




def refine(data_path):

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    arr = ['de','di','me','mi']
    styleCombo = list(itertools.combinations(arr,2))

    action_style_dict = {}

    actionIdx = 0
    for idx, d in enumerate(data):
        action = process_file_name(d['file_name'])
        data[idx]['action'] = action
        if action in action_style_dict:
            action_style_dict[action][d['persona']] = d
            data[idx]['actionIdx'] = action_style_dict[action]['index']
            if d['n_frames'] > action_style_dict[action]['maxIndex']:
                action_style_dict[action]['maxIndex'] = d['n_frames']
        else:
            action_style_dict[action] = dict()
            action_style_dict[action][d['persona']] = d
            action_style_dict[action]['index'] = actionIdx
            action_style_dict[action]['maxIndex'] = d['n_frames']
            data[idx]['actionIdx'] = action_style_dict[action]['index']
            actionIdx += 1

    content = []
    styled = []
    for action in action_style_dict.keys():
        if(action_style_dict[action]['maxIndex'] < 800):
            for pair in styleCombo:
                content.append(action_style_dict[action][pair[0]])
                styled.append(action_style_dict[action][pair[1]])

    datasets = {"content": content,"styled":styled}
    with open('data/action_style_KTG.pkl', 'wb') as f:
        pickle.dump(datasets,f)

    #dataset1 = ActionStyleDataset(content)
    #dataset2 = ActionStyleDataset(styled)

    #datasets = {'content':dataset1,'styled':dataset2}
    #test(datasets)

if __name__ == '__main__':
    refine("../../simple_rvae/datasets/data/motion_body_HJK.pkl")
    #refine("../../simple_rvae/datasets/data/motion_body_KTG.pkl")

    #make_style_dataset(dataset_name="data/motion_body_fixed_nohand_all.pkl")