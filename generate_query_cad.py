import os
import sys
from tqdm import tqdm
import numpy as np
import cv2
import collections
import json
import random


FRAMES_NUM={1: 302, 2: 347, 3: 194, 4: 257, 5: 536, 6: 401, 7: 968, 8: 221, 9: 356, 10: 302, 
            11: 1813, 12: 1084, 13: 851, 14: 723, 15: 464, 16: 1021, 17: 905, 18: 600, 19: 203, 20: 342, 
            21: 650, 22: 361, 23: 311, 24: 321, 25: 617, 26: 734, 27: 1804, 28: 470, 29: 635, 30: 356, 
            31: 690, 32: 194, 33: 193, 34: 395, 35: 707, 36: 914, 37: 1049, 38: 653, 39: 518, 40: 401, 
            41: 707, 42: 420, 43: 410, 44: 356}

def get_ga_from_ia(ia):
    ia_cnt_dic = collections.Counter(ia)
    counter = ia_cnt_dic.most_common(2)
    group_activity = counter[0][0]-1 if counter[0][0]!=0 else counter[1][0]-1
    group_activity = Activity5to4[group_activity]

    return group_activity

def decision_rule(use_query_type, ia):
    group_activity = get_ga_from_ia(ia)

    if use_query_type == 'Moving':
        use_query = group_activity == ACTIVITIES.index('Moving')
    elif use_query_type == 'Waiting':
        use_query = group_activity == ACTIVITIES.index('Waiting')
    elif use_query_type == 'Queueing':
        use_query = group_activity == ACTIVITIES.index('Queueing')
    elif use_query_type == 'Talking':
        use_query = group_activity == ACTIVITIES.index('Talking')
    elif use_query_type == 'dummy':
        use_query = random.choice([True, False])
    else:
        raise ValueError(f'Invalid use_query_type: {use_query_type}')
    
    return int(use_query)

def decision_rule_people(use_query_type, ia, ia_person):
    group_activity = get_ga_from_ia(ia)
    
    if use_query_type == 'Moving':
        return ia_person == ACTIONS.index('Moving') if group_activity == ACTIVITIES.index('Moving') else False
    elif use_query_type == 'Waiting':
        return ia_person == ACTIONS.index('Waiting') if group_activity == ACTIVITIES.index('Waiting') else False
    elif use_query_type == 'Queueing':
        return ia_person == ACTIONS.index('Queueing') if group_activity == ACTIVITIES.index('Queueing') else False
    elif use_query_type == 'Talking':
        return ia_person == ACTIONS.index('Talking') if group_activity == ACTIVITIES.index('Talking') else False
    elif use_query_type == 'dummy':
        return random.choice([True, False])
    else:
        raise ValueError(f'Invalid use_query_type: {use_query_type}')

def get_user_query(ia_dic, people_bbox_dic, vid_id, seq_id, use_query_type, dataset_name, dataset_dir):
    ia = ia_dic[seq_id]
    people_bbox = people_bbox_dic[seq_id]
    use_query = decision_rule(use_query_type, ia)

    save_img_dir = os.path.join('analysis', 'query_samples', dataset_name, use_query_type)
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    if use_query:
        save_img_name_list = [os.path.splitext(img_name)[0] for img_name in os.listdir(save_img_dir)]
        save_img_seq_list = [img_name.split('_')[0] for img_name in save_img_name_list]
        if vid_id in save_img_seq_list:
            return use_query
    
        img_path = os.path.join(dataset_dir, vid_id, f'frame{str(seq_id).zfill(4)}.jpg')
        img = cv2.imread(img_path)
        for person_idx in range(people_bbox.shape[0]):
            x1, y1, width, height = map(float, people_bbox[person_idx])
            x2 = x1 + width
            y2 = y1 + height
            ia_person_name = ACTIONS[ia[person_idx]]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, ia_person_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        save_img_path = os.path.join(save_img_dir, f'{vid_id}_{seq_id}.jpg')
        cv2.imwrite(save_img_path, img)
    
    return use_query

def get_user_query_people(ia_dic, people_bbox_dic, vid_id, seq_id, 
                          use_query_type, dataset_name, dataset_dir,
                          use_action_type):
    ia = ia_dic[seq_id]
    people_bbox = people_bbox_dic[seq_id]

    user_query_people = np.zeros(len(ia))
    for person_idx, bbox in enumerate(people_bbox):
        ia_person = ia[person_idx]
        ia_person = Action6to5[ia_person]
        user_query_people[person_idx] = decision_rule_people(use_query_type, ia, ia_person)

    return user_query_people

def generate_query(use_query_type, use_action_type):
    # define the dataset name
    dataset_name = 'CAD'

    # define the dataset directory
    dataset_dir = os.path.join('data_local', 'CollectiveActivityDataset')
    save_query_dir = os.path.join('data_local', 'collective_query')
    if not os.path.exists(save_query_dir):
        os.makedirs(save_query_dir)

    # define the query stat directory
    query_stat_dir = os.path.join('analysis', 'query_stats')
    if not os.path.exists(query_stat_dir):
        os.makedirs(query_stat_dir)

    for vid_id in tqdm(sorted(os.listdir(dataset_dir))):
        vid_ann = os.path.join(dataset_dir, vid_id, 'annotations.txt')
        with open(vid_ann, 'r') as f:
            anns = f.readlines()
        ga_dic = {}
        ia_dic = {}
        people_bbox_dic = {}
        for ann in anns:
            ann = ann.strip().split()
            
            # get the id of the image
            seq_id = os.path.splitext(ann[0])[0]

            # follow the original dataloader setting
            frame_id = int(seq_id)
            sid = int(vid_id.replace('seq', ''))
            if frame_id!=None and frame_id%10==1 and frame_id+9<=FRAMES_NUM[sid]:
                pass
            else:
                continue

            if not seq_id in ia_dic.keys():
                ia_dic[seq_id] = []
                people_bbox_dic[seq_id] = []

            # get the action of the person
            ia_dic[seq_id].append(int(ann[5])-1)

            # get the bbox of the person
            person_bbox = list(map(float, ann[1:5]))
            people_bbox_dic[seq_id].append(person_bbox)
        
        for seq_id in people_bbox_dic.keys():
            people_bbox_dic[seq_id] = np.array(people_bbox_dic[seq_id])

        # get the user query
        save_user_query_vid = {}
        for seq_id in ia_dic.keys():
            use_query = get_user_query(ia_dic, people_bbox_dic, vid_id, seq_id, 
                                       use_query_type, dataset_name, dataset_dir)
            save_user_query_vid[seq_id] = use_query
        
        # get the user people query
        save_user_query_people_vid = {}
        for seq_id in ia_dic.keys():
            use_query_people = get_user_query_people(ia_dic, people_bbox_dic, 
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
    test_seqs=[5,6,7,8,9,10,11,15,16,25,28,29]
    train_seqs=[s for s in range(1,45) if s not in test_seqs]
    for vid_id in os.listdir(save_query_dir):
        user_query_vid_path = os.path.join(save_query_dir, vid_id, f'query_{use_query_type}_{use_action_type}.txt')
        with open(user_query_vid_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            query_count = int(line.strip().split()[1])
            vid_id_mod = int(vid_id.replace('seq', ''))
            if vid_id_mod in train_seqs:
                query_count_train.append(query_count)
            elif vid_id_mod in test_seqs:
                query_count_test.append(query_count)
            else:
                assert False, f'Invalid vid_id: {vid_id_mod} {vid_id}'

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

    print(f'Query statistics for {use_query_type}:')
    print(query_stat)
    # save the query statistics to a file
    with open(query_stat_file, 'w') as f:
        json.dump(query_stat, f, indent=4)

ACTIONS = ['NA','Moving','Waiting','Queueing','Talking']
ACTIVITIES = ['Moving','Waiting','Queueing','Talking']
Action6to5 = {0:0, 1:1, 2:2, 3:3, 4:1, 5:4}
Activity5to4 = {0:0, 1:1, 2:2, 3:0, 4:3}

dataset_name = 'CAD'

use_query_type_list = []
use_query_type_list.append('dummy')
use_query_type_list.append('Moving')
use_query_type_list.append('Waiting')
use_query_type_list.append('Queueing')
use_query_type_list.append('Talking')

use_action_type_list = []
use_action_type_list.append('gt')
use_action_type_list.append('gt_action')
use_action_type_list.append('gt_grouping')
use_action_type_list.append('det')

for use_action_type in use_action_type_list:
    for use_query_type in tqdm(use_query_type_list):
        print(f'Generate datast for {use_query_type}')
        generate_query(use_query_type, use_action_type)