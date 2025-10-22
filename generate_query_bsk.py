import os
import sys
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd
import json
import sys
import random
from scipy.optimize import linear_sum_assignment

def decision_rule(use_query_type, ga_dic, seq_id):
    ga = ga_dic[seq_id]

    if use_query_type == 'l_spike':
        ga_flag = ga == 'l-spike'
        use_query = ga_flag
    elif use_query_type == 'dummy':
        use_query = random.choice([True, False])
    elif use_query_type == '2p-succ.':
        ga_flag = ga == '2p-succ.'
        use_query = ga_flag
    elif use_query_type == '2p-fail.-def.':
        ga_flag = ga == '2p-fail.-def.'
        use_query = ga_flag
    elif use_query_type == '2p-fail.-off.':
        ga_flag = ga == '2p-fail.-off.'
        use_query = ga_flag
    elif use_query_type == '2p-layup-succ.':
        ga_flag = ga == '2p-layup-succ.'
        use_query = ga_flag
    elif use_query_type == '2p-layup-fail.-def.':
        ga_flag = ga == '2p-layup-fail.-def.'
        use_query = ga_flag
    elif use_query_type == '2p-layup-fail.-off.':
        ga_flag = ga == '2p-layup-fail.-off.'
        use_query = ga_flag
    elif use_query_type == '3p-succ.':
        ga_flag = ga == '3p-succ.'
        use_query = ga_flag
    elif use_query_type == '3p-fail.-def.':
        ga_flag = ga == '3p-fail.-def.'
        use_query = ga_flag
    elif use_query_type == '3p-fail.-off.':
        ga_flag = ga == '3p-fail.-off.'
        use_query = ga_flag
    else:
        raise ValueError(f'Invalid use_query_type: {use_query_type}')

    return int(use_query)

def get_user_query(ga_dic, vid_id, seq_id, use_query_type, dataset_name, dataset_dir):
    use_query = decision_rule(use_query_type, ga_dic, seq_id)

    save_img_dir = os.path.join('analysis', 'query_samples', dataset_name, use_query_type)
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    if use_query and len(os.listdir(save_img_dir)) < 16:
        save_img_name_list = [os.path.splitext(img_name)[0] for img_name in os.listdir(save_img_dir)]
        save_img_seq_list = [img_name.split('_')[0] for img_name in save_img_name_list]
        if vid_id in save_img_seq_list:
            return use_query

        img_path = os.path.join(dataset_dir, 'videos', vid_id, seq_id, f'000035.jpg')
        img = cv2.imread(img_path)
        save_img_path = os.path.join(save_img_dir, f'{vid_id}_{seq_id}_000035.jpg')
        cv2.imwrite(save_img_path, img)
    
    return use_query

def generate_query(use_query_type, use_action_type):
    print(f'Generate query for {use_query_type} ({use_action_type})')

    # define the dataset name
    dataset_name = 'BSK'

    if dataset_name == 'BSK':
        ACTIONS = ['blocking', 'digging', 'falling', 'jumping', 'moving', 'setting', 'spiking', 'standing', 'waiting']

    # define the dataset directory
    dataset_dir = os.path.join('data_local', 'NBA_dataset')
    image_dir = os.path.join(dataset_dir, 'videos')
    save_query_dir = os.path.join('data_local', 'basketball_query')
    if not os.path.exists(save_query_dir):
        os.makedirs(save_query_dir)

    # define the query stat directory
    query_stat_dir = os.path.join('analysis', 'query_stats')
    if not os.path.exists(query_stat_dir):
        os.makedirs(query_stat_dir)

    for vid_id in tqdm(sorted(os.listdir(image_dir))):
        if not vid_id.isdigit():
            continue

        # read the annotation file    
        vid_ann = os.path.join(image_dir, vid_id, 'annotations.txt')
        with open(vid_ann, 'r') as f:
            anns = f.readlines()

        ga_dic = {}
        for ann in anns:
            ann = ann.strip().split()
            img_id = os.path.splitext(ann[0])[0]
            img_id = str(int(img_id))
            ga_dic[img_id] = ann[1]

        # get the user query
        save_user_query_vid = {}
        for seq_id in ga_dic.keys():
            use_query = get_user_query(ga_dic, vid_id, seq_id, use_query_type, dataset_name, dataset_dir)
            save_user_query_vid[seq_id] = use_query

        # get the user people query
        save_user_query_people_vid = {}
        for seq_id in ga_dic.keys():
            use_query_people = [0]
            save_user_query_people_vid[seq_id] = use_query_people

        # save the user query
        user_query_vid_dir = os.path.join(save_query_dir, vid_id)
        if not os.path.exists(user_query_vid_dir):
            os.makedirs(user_query_vid_dir)
        user_query_vid_path = os.path.join(user_query_vid_dir, f'query_{use_query_type}_{use_action_type}.txt')
        with open(user_query_vid_path, 'w') as f:
            for seq_id, use_query in save_user_query_vid.items():
                f.write(f'{seq_id} {use_query}\n')
        user_query_people_vid_path = os.path.join(user_query_vid_dir, f'query_people_{use_query_type}_{use_action_type}.txt')
        with open(user_query_people_vid_path, 'w') as f:
            for seq_id, use_query_people in save_user_query_people_vid.items():
                use_query_people = ' '.join(map(lambda x: str(int(x)), use_query_people))
                f.write(f'{seq_id} {use_query_people}\n')

    # count the number of query videos
    query_count_train = []
    query_count_test = []

    train_seqs_path = os.path.join(dataset_dir, 'train_video_ids')
    with open(train_seqs_path, 'r') as f:
        train_seqs = f.readlines()
    train_seqs = train_seqs[0][:-1].split(',')
    train_seqs = list(map(int, train_seqs))
    test_seqs_path = os.path.join(dataset_dir, 'test_video_ids')
    with open(test_seqs_path, 'r') as f:
        test_seqs = f.readlines()
    test_seqs = test_seqs[0][:-1].split(',')
    test_seqs = list(map(int, test_seqs))

    for vid_id in os.listdir(save_query_dir):
        user_query_vid_path = os.path.join(save_query_dir, vid_id, f'query_{use_query_type}_{use_action_type}.txt')
        with open(user_query_vid_path, 'r') as f:
            lines = f.readlines()
        vid_id = int(vid_id)
        for line in lines:
            query_count = int(line.strip().split()[1])
            if vid_id in train_seqs:
                query_count_train.append(query_count)
            elif vid_id in test_seqs:
                query_count_test.append(query_count)

    # write the query statistics to a file
    query_stat_file = os.path.join(query_stat_dir, f'{dataset_name}_{use_query_type}_{use_action_type}.json')
    query_stat = {}

    query_count_train = np.array(query_count_train)
    query_count_test = np.array(query_count_test)
    for data_type in ['train', 'test', 'all']:
        if data_type == 'train':
            query_count = query_count_train
        elif data_type == 'test':
            query_count = query_count_test
        elif data_type == 'all':
            query_count = np.concatenate([query_count_train, query_count_test])
        if not data_type in query_stat.keys():
            query_stat[data_type] = {}

        for mode in ['mean', 'sum', 'total']:
            if mode == 'mean':
                query_stat[data_type][mode] = np.mean(query_count).item()
            elif mode == 'sum':
                query_stat[data_type][mode] = np.sum(query_count).item()
            elif mode == 'total':
                query_stat[data_type][mode] = query_count.size
            else:
                raise ValueError(f'Invalid mode: {mode}')

    # save the query statistics to a file
    with open(query_stat_file, 'w') as f:
        json.dump(query_stat, f, indent=4)
    
    print(f'Query statistics for {use_query_type} is saved to {query_stat_file}')
    print(f'Query statistics for {use_query_type}: {query_stat}')

use_query_type_list = []
use_query_type_list.append('dummy')
use_query_type_list.append('2p-succ.')
use_query_type_list.append('2p-fail.-def.')
use_query_type_list.append('2p-fail.-off.')
use_query_type_list.append('2p-layup-succ.')
use_query_type_list.append('2p-layup-fail.-def.')
use_query_type_list.append('2p-layup-fail.-off.')
use_query_type_list.append('3p-succ.')
use_query_type_list.append('3p-fail.-def.')
use_query_type_list.append('3p-fail.-off.')


use_action_type_list = []
use_action_type_list.append('gt')
use_action_type_list.append('gt_action')

for use_action_type in use_action_type_list:
    for use_query_type in tqdm(use_query_type_list):
        print(f'Generate datast for {use_query_type}')
        generate_query(use_query_type, use_action_type)