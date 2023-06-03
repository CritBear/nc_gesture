
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




def make_pairs(data,max_length,fixed_length):

    arr = ['light','heavy','slow','fast']#['de','di','me','mi']
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

    pairdataset = []
    for action in action_style_dict.keys():
        if(action_style_dict[action]['maxIndex'] < max_length):
            for pair in styleCombo:
                '''
                split data
                '''
                l = min(action_style_dict[action][pair[0]]['n_frames'],action_style_dict[action][pair[1]]['n_frames'])
                for i in range(0,l //fixed_length):
                    s = max((int)(i * fixed_length - fixed_length * 0.1), 0)
                    e = s + fixed_length
                    if (e >= l - 100): break
                    cur = action_style_dict[action][pair[0]].copy()
                    cur['n_frames'] = fixed_length
                    cur['joint_rotation_matrix'] = action_style_dict[action][pair[0]]['joint_rotation_matrix'][s:e]
                    cur['target_style'] = pair[1]
                    cur['target_motion'] = action_style_dict[action][pair[1]]['joint_rotation_matrix'][s:e]
                    pairdataset.append(cur.copy())
    return pairdataset

def refine(data):
    pairdataset = make_pairs(data,800,400)

    # with open('data/style_pair_400_all.pkl', 'wb') as f:
    #     pickle.dump(pairdataset,f)

def for_fixed_recon(data,max_length,fixed_length):
    dataset = []
    for d in data:
        l =  d['n_frames']
        if l < max_length:
            for i in range(l // fixed_length + 1):
                s = max((int)(i * fixed_length - fixed_length*0.1),0)
                e = s + fixed_length
                if(e >= l) :break
                cur = d.copy()
                cur['n_frames'] = fixed_length
                cur['joint_rotation_matrix'] = d['joint_rotation_matrix'][s:e]
                cur['target_motion'] = d['joint_rotation_matrix'][s:e]
                cur['target_style'] = cur['persona']
                dataset.append(cur)


    with open(f'data/fixed_{fixed_length}_all.pkl', 'wb') as f:
        pickle.dump(dataset,f)

def for_variable_recon(data,max_length):
    dataset = []
    for d in data:
        l =  d['n_frames']
        if l < max_length:
            cur = d.copy()
            cur['target_motion'] = d['joint_rotation_matrix']
            cur['target_style'] = cur['persona']
            dataset.append(cur)
        else:
            cur = d.copy()
            cur['n_frames'] = max_length
            cur['target_motion'] = d['joint_rotation_matrix'][:max_length]
            cur['target_style'] = cur['persona']
            dataset.append(cur)
    with open(f'data/lhsf_{max_length}_all.pkl', 'wb') as f:
        pickle.dump(dataset,f)

def for_classifier(data,max_length):
    static = {}
    dataset = []
    fixed_length = 300
    with open("../../classifier/high_action.pkl", 'rb') as f:
        d1 = pickle.load(f)
    for d in data:
        l =  d['n_frames']
        if l < max_length and l > fixed_length:
            for j in range((d["n_frames"] // fixed_length)*2):
                cur = d.copy()
                s = max((int)(j * fixed_length - fixed_length*0.2),0)
                e = s + fixed_length
                if e > d["n_frames"]:
                    break
                cur['n_frames'] = fixed_length
                cur['joint_rotation_matrix'] = d['joint_rotation_matrix'][s:e]
                dataset.append(cur)
    for key,val in static.items():
        if len(val) < 4:
            print(key)
    print(len(d1)*8,len(dataset))
    with open(f'data/classifier_{max_length}_high.pkl', 'wb') as f:
        pickle.dump(dataset,f)

def make_style_pair(data,fixed_length):
    dataset = []
    for d in data:
        l =  d['n_frames']
        if l > fixed_length:
            cur = d.copy()
            target = random.choice(data).copy()
            while True:
                if target['n_frames'] > fixed_length and target['persona'] != d['persona']:
                    break
                else:
                    target = random.choice(data).copy()
            cur['n_frames'] = fixed_length
            cur['joint_rotation_matrix'] = d['joint_rotation_matrix'][:fixed_length]
            cur['target_motion'] = target['joint_rotation_matrix'][:fixed_length]
            cur['target_style'] = target['persona']
            dataset.append(cur)
            #print(cur['persona'],target['persona'])
        else:
            print(d['file_name'])
    #print(len(dataset))
    with open(f'data/lhsf_{fixed_length}_fixed_all.pkl', 'wb') as f:
        pickle.dump(dataset,f)


if __name__ == '__main__':
    f1 = "../../simple_rvae/datasets/data/motion_body_hand_light_heavy.pkl"
    f2 = "../../simple_rvae/datasets/data/motion_body_hand_slow_fast.pkl"
    with open(f1, 'rb') as f:
        d1 = pickle.load(f)
    with open(f2, 'rb') as f:
        d2 = pickle.load(f)
    #for_fixed_recon(d1+d2,800,300)
    #for_variable_recon(d1+d2,800)
    #for_classifier(d1+d2,900)
    make_style_pair(d1+d2,300)
    #refine()
    #refine("../../simple_rvae/datasets/data/motion_body_KTG.pkl")

    #make_style_dataset(dataset_name="data/motion_body_fixed_nohand_all.pkl")