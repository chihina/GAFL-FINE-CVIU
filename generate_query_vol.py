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

def decision_rule(use_query_type, ga_dic, ia_dic, people_bbox_dic, seq_id, grouping_dic):
    ga = ga_dic[seq_id]
    ia = ia_dic[seq_id]
    people_bbox = people_bbox_dic[seq_id]
    groupings = grouping_dic[seq_id]

    seq_id_list_before = sorted([int(i) for i in ga_dic.keys() if int(i) < int(seq_id)])
    seq_id_list_after = sorted([int(i) for i in ga_dic.keys() if int(i) > int(seq_id)])
    seq_id_list_before_nearest = max(seq_id_list_before) if seq_id_list_before else None
    seq_id_list_after_nearest = min(seq_id_list_after) if seq_id_list_after else None
    ga_before_nearest = ga_dic[str(seq_id_list_before_nearest)] if seq_id_list_before_nearest else None
    ga_after_nearest = ga_dic[str(seq_id_list_after_nearest)] if seq_id_list_after_nearest else None

    # count the number of blocking actions
    blocking_count = 0
    spiking_count = 0
    for action in ia:
        if action == 'blocking':
            blocking_count += 1
        if action == 'spiking':
            spiking_count += 1

    # label where the person is spiking
    if 'spiking' in ia:
        person_x_min = np.min(people_bbox[:, 0])
        person_y_min = np.min(people_bbox[:, 1])
        person_x_max = np.max(people_bbox[:, 0] + people_bbox[:, 2])
        person_y_max = np.max(people_bbox[:, 1] + people_bbox[:, 3])
        person_x_range = person_x_max - person_x_min
        person_y_range = person_y_max - person_y_min
        spiker_bbox = people_bbox[ia.index('spiking')]
        spiker_x_mid = spiker_bbox[0] + spiker_bbox[2] / 2
        spiker_y_mid = spiker_bbox[1] + spiker_bbox[3] / 2
        spiker_x_ratio = (spiker_x_mid - person_x_min) / person_x_range
        spiker_y_ratio = (spiker_y_mid - person_y_min) / person_y_range
    else:
        spiker_x_ratio= 100
        spiker_y_ratio= 100

    if use_query_type == 'l_spike':
        ga_flag = ga == 'l-spike'
        use_query = ga_flag
    elif use_query_type == 'l_pass':
        ga_flag = ga == 'l-pass'
        use_query = ga_flag
    elif use_query_type == 'l_set':
        ga_flag = ga == 'l_set'
        use_query = ga_flag
    elif use_query_type == 'l_set_jump':
        ga_flag = ga == 'l_set'
        ia_flag = 'jumping' in ia
        use_query = ga_flag and ia_flag
    elif use_query_type == 'l_pass_fall':
        ga_flag = ga == 'l-pass'
        falling_flag = 'falling' in ia
        digging_flag = 'digging' in ia
        ia_flag = falling_flag and not digging_flag
        use_query = ga_flag and ia_flag
    elif use_query_type == 'l_spike_setting':
        ga_flag = ga == 'l-spike'
        setting_flag = 'setting' in ia
        use_query = ga_flag and setting_flag
    elif use_query_type == 'l_spike_back':
        ga_flag = ga == 'l-spike'
        spiker_loc_flag = spiker_x_ratio < 1/5 and spiker_x_ratio > 0
        use_query = ga_flag and spiker_loc_flag
    elif use_query_type == 'l_spike_back_zero_blocker':
        ga_flag = ga == 'l-spike'
        blocking_flag = blocking_count == 0
        spiker_loc_flag = spiker_x_ratio < 1/5 and spiker_x_ratio > 0
        use_query = ga_flag and blocking_flag and spiker_loc_flag
    elif use_query_type == 'l_spike_back_one_blocker':
        ga_flag = ga == 'l-spike'
        blocking_flag = blocking_count == 1
        spiker_loc_flag = spiker_x_ratio < 1/5 and spiker_x_ratio > 0
        use_query = ga_flag and blocking_flag and spiker_loc_flag
    elif use_query_type == 'l_spike_two_blocker':
        ga_flag = ga == 'l-spike'
        blocking_flag = blocking_count == 2
        use_query = ga_flag and blocking_flag
    elif use_query_type == 'l_spike_one_blocker':
        ga_flag = ga == 'l-spike'
        blocking_flag = blocking_count == 1
        use_query = ga_flag and blocking_flag
    elif use_query_type == 'l_spike_zero_blocker':
        ga_flag = ga == 'l-spike'
        blocking_flag = blocking_count == 0
        spiker_flag = spiking_count == 1
        use_query = ga_flag and blocking_flag and spiker_flag
    elif use_query_type == 'l_spike_further':
        ga_flag = ga == 'l-spike'
        spiker_loc_flag = spiker_y_ratio < 1/3 and spiker_y_ratio > 0
        use_query = ga_flag and spiker_loc_flag
    elif use_query_type == 'l_spike_middle':
        ga_flag = ga == 'l-spike'
        spiker_loc_flag = spiker_y_ratio > 1/3 and spiker_y_ratio < 2/3
        use_query = ga_flag and spiker_loc_flag
    elif use_query_type == 'l_spike_closer':
        ga_flag = ga == 'l-spike'
        spiker_loc_flag = spiker_y_ratio > 2/3 and spiker_y_ratio < 1
        use_query = ga_flag and spiker_loc_flag
    elif use_query_type == 'l_set':
        ga_flag = ga == 'l_set'
        use_query = ga_flag
    elif use_query_type == 'l_pass':
        ga_flag = ga == 'l-pass'
        use_query = ga_flag
    elif use_query_type == 'l_pass_spike':
        ga_flag = ga == 'l-pass'
        ga_flag_before = ga_before_nearest == 'r_spike'
        use_query = ga_flag and ga_flag_before
    elif use_query_type == 'l_pass_fall_dig':
        ga_flag = ga == 'l-pass'
        falling_flag = 'falling' in ia
        digging_flag = 'digging' in ia
        use_query = ga_flag and falling_flag and digging_flag
    elif use_query_type == 'l_winpoint':
        ga_flag = ga == 'l_winpoint'
        use_query = ga_flag
    elif use_query_type == 'r_spike':
        ga_flag = ga == 'r_spike'
        use_query = ga_flag
    elif use_query_type == 'r_spike_one_blocker':
        ga_flag = ga == 'r_spike'
        blocking_flag = blocking_count == 1
        use_query = ga_flag and blocking_flag
    elif use_query_type == 'r_set':
        ga_flag = ga == 'r_set'
        use_query = ga_flag
    elif use_query_type == 'r_set_jump':
        ga_flag = ga == 'r_set'
        ia_flag = 'jumping' in ia
        use_query = ga_flag and ia_flag
    elif use_query_type == 'r_pass':
        ga_flag = ga == 'r-pass'
        use_query = ga_flag
    elif use_query_type == 'r_pass_falling':
        ga_flag = ga == 'r-pass'
        falling_flag = 'falling' in ia
        use_query = ga_flag and falling_flag
    elif use_query_type == 'r_pass_spike':
        ga_flag = ga == 'r-pass'
        ga_flag_before = ga_before_nearest == 'l-spike'
        use_query = ga_flag and ga_flag_before
    elif use_query_type == 'r_winpoint':
        ga_flag = ga == 'r_winpoint'
        use_query = ga_flag
    elif use_query_type == 'spike':
        ga_flag = ga in ['l-spike', 'r_spike']
        use_query = ga_flag
    elif use_query_type == 'set':
        ga_flag = ga in ['l_set', 'r_set']
        use_query = ga_flag
    elif use_query_type == 'dummy':
        use_query = random.choice([True, False])
    else:
        raise ValueError(f'Invalid use_query_type: {use_query_type}')

    return int(use_query)

def decision_rule_people(use_query_type, ga, ia_person, people_bbox, person_idx, grouping_person, use_action_type):
    if use_query_type == 'r_winpoint':
        return grouping_person if ga == 'r_winpoint' else 0
    elif use_query_type == 'l_winpoint':
        return grouping_person if ga == 'l_winpoint' else 0
    elif use_query_type == 'l_spike':
        if use_action_type == 'gt_grouping':
            return grouping_person if ga == 'l-spike' else 0
        elif use_action_type in ['gt_action', 'det']:
            return ia_person in ['spiking', 'blocking']
    elif use_query_type == 'l_spike_one_blocker':
        if use_action_type == 'gt_grouping':
            return grouping_person if ga == 'l-spike' else 0
        elif use_action_type in ['gt_action', 'det']:
            return ia_person in ['spiking', 'blocking']
    elif use_query_type == 'r_spike':
        if use_action_type == 'gt_grouping':
            return grouping_person if ga == 'r_spike' else 0
        elif use_action_type in ['gt_action', 'det']:
            return ia_person in ['spiking', 'blocking']
    elif use_query_type == 'r_spike_one_blocker':
        if use_action_type == 'gt_grouping':
            return grouping_person if ga == 'r_spike' else 0
        elif use_action_type in ['gt_action', 'det']:
            return ia_person in ['spiking', 'blocking']
    elif use_query_type == 'l_set':
        if use_action_type == 'gt_grouping':
            return grouping_person if ga == 'l_set' else 0
        elif use_action_type in ['gt_action', 'det']:
            return ia_person in ['setting']
    elif use_query_type == 'l_set_jump':
        if use_action_type == 'gt_grouping':
            return grouping_person if ga == 'l_set' else 0
        elif use_action_type in ['gt_action', 'det']:
            return ia_person in ['setting', 'jumping']
    elif use_query_type == 'r_set':
        if use_action_type == 'gt_grouping':
            return grouping_person if ga == 'r_set' else 0
        elif use_action_type in ['gt_action', 'det']:
            return ia_person in ['setting']
    elif use_query_type == 'r_set_jump':
        if use_action_type == 'gt_grouping':
            return grouping_person if ga == 'r_set' else 0
        elif use_action_type in ['gt_action', 'det']:
            return ia_person in ['setting', 'jumping']
    elif use_query_type == 'l_pass':
        if use_action_type == 'gt_grouping':
            return grouping_person if ga == 'l-pass' else 0
        elif use_action_type in ['gt_action', 'det']:
            return ia_person in ['digging', 'falling']
    elif use_query_type == 'r_pass':
        if use_action_type == 'gt_grouping':
            return grouping_person if ga == 'r-pass' else 0
        elif use_action_type in ['gt_action', 'det']:
            return ia_person in ['digging', 'falling']
    else:
        print(ga)
        print('Not implemented of the query people type')
        return 0

def get_user_query(ga_dic, ia_dic, people_bbox_dic, vid_id, seq_id, use_query_type, dataset_name, dataset_dir, grouping_dic):
    ia = ia_dic[seq_id]
    people_bbox = people_bbox_dic[seq_id]

    use_query = decision_rule(use_query_type, ga_dic, ia_dic, people_bbox_dic, seq_id, grouping_dic)

    save_img_dir = os.path.join('analysis', 'query_samples', dataset_name, use_query_type)
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    if use_query and len(os.listdir(save_img_dir)) < 16:
        save_img_name_list = [os.path.splitext(img_name)[0] for img_name in os.listdir(save_img_dir)]
        save_img_seq_list = [img_name.split('_')[0] for img_name in save_img_name_list]
        if vid_id in save_img_seq_list:
            return use_query

        img_path = os.path.join(dataset_dir, vid_id, seq_id, f'{seq_id}.jpg')
        img = cv2.imread(img_path)
        for person_idx, bbox in enumerate(people_bbox):
            x1, y1, width, height = map(float, bbox)
            x2 = x1 + width
            y2 = y1 + height
            ia_person = ia[person_idx]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, ia_person, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        save_img_path = os.path.join(save_img_dir, f'{vid_id}_{seq_id}.jpg')
        cv2.imwrite(save_img_path, img)
    
    return use_query

def get_user_query_people(ga_dic, ia_dic, people_bbox_dic, grouping_dic, 
                          vid_id, seq_id, use_query_type, dataset_name, dataset_dir,
                          use_action_type):
    ia = ia_dic[seq_id]
    ga = ga_dic[seq_id]
    people_bbox = people_bbox_dic[seq_id]
    groupings = grouping_dic[seq_id]

    user_query_people = np.zeros(len(ia))
    for person_idx, bbox in enumerate(people_bbox):
        ia_person = ia[person_idx]
        grouping_person = groupings[person_idx]
        user_query_people[person_idx] = decision_rule_people(use_query_type, ga, ia_person, people_bbox, 
                                                             person_idx, grouping_person, use_action_type)

    return user_query_people

def generate_query(use_query_type, use_action_type):
    print(f'Generate query for {use_query_type} ({use_action_type})')

    # define the dataset name
    dataset_name = 'VOL'

    if dataset_name == 'VOL':
        ACTIONS = ['blocking', 'digging', 'falling', 'jumping', 'moving', 'setting', 'spiking', 'standing', 'waiting']

    # define the dataset directory
    dataset_dir = os.path.join('data_local', 'videos')
    save_query_dir = os.path.join('data_local', 'volleyball_query')
    if not os.path.exists(save_query_dir):
        os.makedirs(save_query_dir)

    # define the annotation directory
    player_tracking_dir = os.path.join('data_local', 'volleyball_tracking_annotation')

    # define the query stat directory
    query_stat_dir = os.path.join('analysis', 'query_stats')
    if not os.path.exists(query_stat_dir):
        os.makedirs(query_stat_dir)

    for vid_id in tqdm(sorted(os.listdir(dataset_dir))):
        if not vid_id.isdigit():
            continue

        ga_dic = {}
        ia_dic = {}
        people_bbox_dic = {}

        # read the annotation file    
        vid_ann = os.path.join(dataset_dir, vid_id, 'annotations.txt')
        with open(vid_ann, 'r') as f:
            anns = f.readlines()
        for ann in anns:
            ann = ann.strip().split()
            img_id = os.path.splitext(ann[0])[0]
            ga_dic[img_id] = ann[1]

            person_info = ann[2:]
            person_bbox_list = []
            ia_list = []
            for preson_idx in range(len(person_info) // 5):
                ia = person_info[preson_idx * 5 + 4]
                ia_list.append(ia)
                person_bbox = person_info[preson_idx * 5: preson_idx * 5 + 4]
                person_bbox = list(map(float, person_bbox))
                person_bbox_list.append(person_bbox)
            people_bbox_dic[img_id] = np.array(person_bbox_list)
            ia_dic[img_id] = ia_list

        if use_action_type in ['gt_grouping', 'gt_action']:
            pass
        elif use_action_type == 'det':
            rec_act_dir = os.path.join('data_local', 'jae_dataset_bbox_gt')
            for img_id_sample in list(ga_dic.keys()):
                rec_act_path = os.path.join(rec_act_dir, vid_id, img_id_sample, f'{img_id_sample}.json')
                if not os.path.exists(rec_act_path):
                    print(f'Not found: {rec_act_path} {ga_dic[img_id_sample]}')
                    continue
                else:
                    with open(rec_act_path, 'r') as f:
                        rec_act = json.load(f)
                
                # get the people bbox and action in the json file
                people_bbox_sample_json_list = []
                rec_act_json_list = []
                for person_idx in rec_act.keys():
                    person_info = rec_act[person_idx]
                    head_x_center = person_info['head_x_center']
                    head_y_center = person_info['head_y_center']
                    people_bbox_sample_json_list.append([head_x_center, head_y_center])
                    rec_act_json_list.append(person_info['pred_action_num'])
                people_bbox_sample_json = np.array(people_bbox_sample_json_list, dtype=np.float32)
                rec_act_json = np.array(rec_act_json_list, dtype=np.int32)
                
                # get the people bbox and action in the sample
                people_bbox_sample = people_bbox_dic[img_id_sample]
                people_bbox_sample[:, 0] = people_bbox_sample[:, 0] + people_bbox_sample[:, 2] / 2
                people_bbox_sample = people_bbox_sample[:, :2]

                # hungarian matching
                cost_matrix = np.zeros((len(people_bbox_sample), len(people_bbox_sample_json)))
                for i, bbox in enumerate(people_bbox_sample):
                    for j, bbox_json in enumerate(people_bbox_sample_json):
                        cost_matrix[i, j] = np.linalg.norm(bbox - bbox_json)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # update the action based on matching pairs
                ia_sample = ia_dic[img_id_sample]
                for i, person_idx in enumerate(row_ind):
                    pred_act_num = rec_act_json[col_ind[i]]
                    pred_act_name = ACTIONS[pred_act_num]
                    ia_sample[person_idx] = pred_act_name
                ia_dic[img_id_sample] = ia_sample

        grouping_dic = {}
        for seq_id in os.listdir(os.path.join(dataset_dir, vid_id)):
            if not seq_id.isdigit():
                continue
        
            seq_ann = os.path.join(player_tracking_dir, vid_id, seq_id, f'{seq_id}.txt')
            df_tracking_cols = ['person_id', 'xmin', 'ymin', 'xmax', 'ymax', 'img_id', 'a1', 'grouping', 'a3', 'action']
            df_tracking = pd.read_csv(seq_ann, sep=' ', names=df_tracking_cols, index_col=False)
            df_tracking_mid = df_tracking[df_tracking.iloc[:, 5] == int(seq_id)]

            people_bbox_tracks = df_tracking_mid.iloc[:, 1:5].values
            people_bbox_sample = people_bbox_dic[str(seq_id)]

            cost_matrix = np.zeros((len(people_bbox_sample), len(people_bbox_tracks)))
            for i, bbox in enumerate(people_bbox_sample):
                for j, bbox_track in enumerate(people_bbox_tracks):
                    cost_matrix[i, j] = np.linalg.norm(bbox - bbox_track)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            grouping_array = np.zeros(len(people_bbox_sample))
            for i, person_idx in enumerate(row_ind):
                grouping_array[person_idx] = df_tracking_mid.iloc[col_ind[i], 7]
            grouping_dic[seq_id] = grouping_array

        # get the user query
        save_user_query_vid = {}
        for seq_id in os.listdir(os.path.join(dataset_dir, vid_id)):
            if not seq_id.isdigit():
                continue
            use_query = get_user_query(ga_dic, ia_dic, people_bbox_dic, vid_id, seq_id, use_query_type, dataset_name, dataset_dir, grouping_dic)
            save_user_query_vid[seq_id] = use_query

        # get the user people query
        save_user_query_people_vid = {}
        for seq_id in os.listdir(os.path.join(dataset_dir, vid_id)):
            if not seq_id.isdigit():
                continue
            use_query_people = get_user_query_people(ga_dic, ia_dic, people_bbox_dic, grouping_dic, 
                                                     vid_id, seq_id, use_query_type, dataset_name, dataset_dir,
                                                     use_action_type)
            save_user_query_people_vid[seq_id] = use_query_people

        # save the user query
        user_query_vid_dir = os.path.join(save_query_dir, vid_id)
        if not os.path.exists(user_query_vid_dir):
            os.makedirs(user_query_vid_dir)
        # user_query_vid_path = os.path.join(user_query_vid_dir, f'query_{use_query_type}.txt')
        user_query_vid_path = os.path.join(user_query_vid_dir, f'query_{use_query_type}_{use_action_type}.txt')
        with open(user_query_vid_path, 'w') as f:
            for seq_id, use_query in save_user_query_vid.items():
                f.write(f'{seq_id} {use_query}\n')
        # user_query_people_vid_path = os.path.join(user_query_vid_dir, f'query_people_{use_query_type}.txt')
        user_query_people_vid_path = os.path.join(user_query_vid_dir, f'query_people_{use_query_type}_{use_action_type}.txt')
        with open(user_query_people_vid_path, 'w') as f:
            for seq_id, use_query_people in save_user_query_people_vid.items():
                use_query_people = ' '.join(map(lambda x: str(int(x)), use_query_people))
                f.write(f'{seq_id} {use_query_people}\n')

    # count the number of query videos
    query_count_train = []
    query_count_test = []
    train_seqs = [ 1,3,6,7,10,13,15,16,18,22,23,31,32,36,38,39,40,41,42,48,50,52,53,54,
                0,2,8,12,17,19,24,26,27,28,30,33,46,49,51]
    train_seqs = list(map(str, train_seqs))
    test_seqs = [4,5,9,11,14,20,21,25,29,34,35,37,43,44,45,47]
    test_seqs = list(map(str, test_seqs))
    for vid_id in os.listdir(save_query_dir):
        # user_query_vid_path = os.path.join(save_query_dir, vid_id, f'query_{use_query_type}.txt')
        user_query_vid_path = os.path.join(save_query_dir, vid_id, f'query_{use_query_type}_{use_action_type}.txt')
        with open(user_query_vid_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            query_count = int(line.strip().split()[1])
            if vid_id in train_seqs:
                query_count_train.append(query_count)
            elif vid_id in test_seqs:
                query_count_test.append(query_count)

            # if query_count:
                # print(vid_id, line.strip())

    # write the query statistics to a file
    # query_stat_file = os.path.join(query_stat_dir, f'{dataset_name}_{use_query_type}.json')
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

use_query_type_list.append('l_spike')
use_query_type_list.append('l_set')
use_query_type_list.append('l_pass')
use_query_type_list.append('l_winpoint')
use_query_type_list.append('r_spike')
use_query_type_list.append('r_set')
use_query_type_list.append('r_pass')
use_query_type_list.append('r_winpoint')

use_action_type_list = []
# use_action_type_list.append('gt')
use_action_type_list.append('gt_grouping')
use_action_type_list.append('gt_action')
use_action_type_list.append('det')

for use_action_type in use_action_type_list:
    for use_query_type in tqdm(use_query_type_list):
        print(f'Generate datast for {use_query_type} ({use_action_type})')
        generate_query(use_query_type, use_action_type)