'''
    Evaluate the trained model with various metrics and visualization.
'''

import torch
import torch.optim as optim
torch.set_printoptions(sci_mode=False)

import time
import random
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import json
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

import warnings 
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import collections
import itertools
import time

# hungarian algorithm
from torch.utils import data
from config import *
from volleyball import *
from collective import *
from dataset import *
from infer_model import *
from infer_model_prev import *
from infer_model_original import *
from base_model import *
from utils import *

def eval_net(cfg):
    """
    evaluating gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list

    cfg.use_debug = False
    # cfg.use_debug = True

    if cfg.dataset_symbol == 'vol':
        ACTIONS = ['blocking', 'digging', 'falling', 'jumping', 'moving', 'setting', 'spiking', 'standing', 'waiting']
    elif cfg.dataset_symbol == 'cad':
        ACTIONS = ['NA','Moving','Waiting','Queueing','Talking']

    # Show config parameters
    cfg.init_config()
    # cfg.person_size = (368, 368)
    show_config(cfg)
    
    # Reading dataset
    # training_set,validation_set=return_dataset(cfg)
    training_set, validation_set, all_set = return_dataset(cfg)
    
    params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 4, # 4,
    }
    # training_loader=data.DataLoader(training_set,**params)
    params['batch_size']=cfg.test_batch_size
    all_loader = data.DataLoader(all_set, **params)

    # Set random seed
    np.random.seed(cfg.test_random_seed)
    torch.manual_seed(cfg.test_random_seed)
    random.seed(cfg.test_random_seed)
    torch.cuda.manual_seed_all(cfg.test_random_seed)
    torch.cuda.manual_seed(cfg.test_random_seed)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    cfg.device = device
    
    # Build model and Load trained model
    gcnnet_list={'group_relation_volleyball':GroupRelation_volleyball,
                    'group_relation_collective':GroupRelation_volleyball}

    model = gcnnet_list[cfg.inference_module_name](cfg)
    state_dict = torch.load(cfg.stage4model)['state_dict']
    new_state_dict=OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print_log(cfg.log_path, f'Loading stage{cfg.eval_stage} model: ' + cfg.stage4model)

    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)
    model=model.to(device=device)

    # pack models
    models = {'model': model}
    model = models['model']
    model.eval()

    cfg.debug = False
    # cfg.debug = True

    seed_num = cfg.test_random_seed
    seed_num = 15
    print(f'Seed number: {seed_num}')
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.cuda.manual_seed(seed_num)

    # use_set = all_set
    # use_set = validation_set
    use_set = training_set

    queries = torch.tensor(use_set.get_query_list(), device=device)
    query_indices = torch.nonzero(queries).squeeze()
    query_indices = query_indices[torch.randperm(len(query_indices))]
    non_query_indices = torch.nonzero(queries == 0).squeeze()
    non_query_indices = non_query_indices[torch.randperm(len(non_query_indices))]
    
    use_frames = 10
    frames_original = use_set.get_frames_all()
    query_indices_sampled = query_indices[:use_frames]
    non_query_indices_sampled = non_query_indices[:use_frames]
    frames_anchor = []
    for query_index in query_indices_sampled:
        frames_anchor.append(frames_original[query_index])
    # for non_query_index in non_query_indices_sampled:
        # frames_anchor.append(frames_original[non_query_index])

    # with torch.no_grad():
    if True:
        use_set.set_frames(frames_anchor)
        data_loader = data.DataLoader(use_set, **params)
        print(f'Lengh of frames: {len(frames_anchor)} for anchor')
        for batch_idx, batch_data_test in enumerate(tqdm(data_loader)):
            for key in batch_data_test.keys():
                if torch.is_tensor(batch_data_test[key]):
                    batch_data_test[key] = batch_data_test[key].to(device=device)
            images_in = batch_data_test['images_in']
            batch_size, num_frames, _, _, _ = images_in.shape
            actions_in = batch_data_test['actions_in'].reshape((batch_size,num_frames,cfg.num_boxes))
            ret = model(batch_data_test)
            gaf_all = ret['group_feat']
            user_queries_people = batch_data_test['user_queries_people'].long()
            user_queries = batch_data_test['user_queries'].long()

            perturbation_mask = user_queries_people == 0
            batch_data_test['perturbation_mask'] = perturbation_mask
            gaf_key = ret['group_feat']

            anchor_idx = 0
            gaf_key_anchor = gaf_key[anchor_idx]

            print('Train')
            target_idx_start = 1
            target_size = 4
            target_idx_end = target_idx_start + target_size
            gaf_target_all = torch.zeros(target_size, cfg.num_boxes, cfg.num_features_boxes*2, device=device)

            for person_idx in range(cfg.num_boxes):
                perturbation_mask = torch.ones((batch_size, cfg.num_boxes), device=device)
                perturbation_mask[:, person_idx] = 0
                perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes)
                perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
                batch_data_test['perturbation_mask'] = perturbation_mask
                ret_perturb = model(batch_data_test)
                gaf_target_all[:, person_idx] = ret_perturb['group_feat'][target_idx_start:target_idx_end]

            gaf_key_anchor = gaf_key_anchor.view(1, 1, cfg.num_features_boxes*2)
            gaf_key_anchor = gaf_key_anchor.expand(target_size, cfg.num_boxes, cfg.num_features_boxes*2)
            dist_all = ((gaf_key_anchor - gaf_target_all) ** 2) ** 0.5

            reduce_mask = torch.rand(cfg.num_features_boxes*2, device=device, requires_grad=True)
            optimizer = optim.Adam([reduce_mask], lr=0.001)
            gt_user_queries_target = user_queries_people[target_idx_start:target_idx_end, 0]
                for i in tqdm(range(500)):
                    optimizer.zero_grad()
                    loss_all = []
                    for target_idx in range(target_size):
                        dist_all_pos = dist_all[target_idx, gt_user_queries_target[target_idx] == 1]
                        dist_all_neg = dist_all[target_idx, gt_user_queries_target[target_idx] == 0]
                        dist_all_reduce_pos = torch.mean(dist_all_pos * reduce_mask, dim=1)
                        dist_all_reduce_neg = torch.mean(dist_all_neg * reduce_mask, dim=1)
                        loss = torch.mean(dist_all_reduce_pos) - torch.mean(dist_all_reduce_neg)
                        loss_all.append(loss)
                    
                    loss = torch.mean(torch.stack(loss_all))
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    print(f'Loss: {loss.item()}')

            print(f'Reduce mask: {reduce_mask}')
            reduce_mask_opt = reduce_mask
  
            print('Test')
            test_idx_start = target_idx_end + 1
            test_size = 3
            test_idx_end = test_idx_start + test_size
            for test_idx in range(test_idx_start, test_idx_end+1):
                print(f'Test index: {test_idx}')
                gaf_test_all = torch.zeros(cfg.num_boxes, cfg.num_features_boxes*2, device=device)
                dist_list = []
                dist_reduced_list = []
                for person_idx in range(cfg.num_boxes):
                    perturbation_mask = torch.ones((batch_size, cfg.num_boxes), device=device)
                    perturbation_mask[:, person_idx] = 0
                    perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes)
                    perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
                    batch_data_test['perturbation_mask'] = perturbation_mask
                    ret_perturb = model(batch_data_test)
                    gaf_perturb = ret_perturb['group_feat']
                    gaf_test = gaf_perturb[test_idx]
                    gaf_test_all[person_idx] = gaf_test

                    dist_all = ((gaf_key_anchor - gaf_test) ** 2) ** 0.5
                    dist_all_reduced = dist_all * reduce_mask_opt
                    dist_mean = torch.mean(dist_all)
                    dist_mean_reduced = torch.mean(dist_all_reduced)

                    dist_mean = round(dist_mean.item(), 4)
                    dist_mean_reduced = round(dist_mean_reduced.item(), 4)

                    dist_list.append(dist_mean)
                    dist_reduced_list.append(dist_mean_reduced)
                
                for person_idx in range(cfg.num_boxes):
                    action_person = actions_in[test_idx, 0, person_idx]
                    action_person_name = ACTIONS[action_person]

                    dist_mean = dist_list[person_idx]
                    dist_mean_rank = np.argsort(dist_list).tolist().index(person_idx)
                    dist_mean_reduced = dist_reduced_list[person_idx]
                    dist_mean_reduced_rank = np.argsort(dist_reduced_list).tolist().index(person_idx)
                    user_queries_person = user_queries_people[test_idx, 0, person_idx]
                    print(f'Person index: {person_idx} ({action_person_name}, {user_queries_person})')
                    print(f'dist: {dist_mean} ({dist_mean_rank}), dist_reduced: {dist_mean_reduced} ({dist_mean_reduced_rank})')

    return {}