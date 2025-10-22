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
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.manifold import Isomap

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
from scipy.optimize import linear_sum_assignment

from torch.utils import data
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import recall_score, precision_score, f1_score

from config import *
from volleyball import *
from collective import *
from dataset import *
from infer_model import *
from infer_model_prev import *
from infer_model_original import *
from base_model import *
from utils import *
from detect_key_people_utils import *


def detect_key_person(cfg, ret_dic):
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
    elif cfg.dataset_symbol == 'bsk':
        ACTIONS = ['NA']

    # Show config parameters
    cfg.init_config()
    # show_config(cfg)
    
    # Reading dataset
    # cfg.train_seqs = cfg.train_seqs[:2]
    # cfg.test_seqs = cfg.test_seqs[:2]
    training_set, validation_set, all_set = return_dataset(cfg)
    
    params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 4, # 4,
    }
    # training_loader=data.DataLoader(training_set,**params)
    params['batch_size']=cfg.test_batch_size
    all_loader = data.DataLoader(all_set, **params)
    train_loader = data.DataLoader(training_set, **params)
    valid_loader = data.DataLoader(validation_set, **params)

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
    # print_log(cfg.log_path, f'Loading stage{cfg.eval_stage} model: ' + cfg.stage4model)

    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)
    model=model.to(device=device)

    # pack models
    models = {'model': model}
    model = models['model']
    model.eval()

    # get query list
    frames_train = training_set.get_frames_all()
    frames_valid = validation_set.get_frames_all()
    frames_all = all_set.get_frames_all()

    gaf_path_train = os.path.join(cfg.stage2model_dir, 'eval_gaf_train_scene.npy')
    gaf_arr_train = np.load(gaf_path_train)
    gaf_path_test = os.path.join(cfg.stage2model_dir, 'eval_gaf_test_scene.npy')
    gaf_arr_test = np.load(gaf_path_test)
    gaf_arr = np.concatenate((gaf_arr_train, gaf_arr_test), axis=0)
    gaf_tensor = torch.tensor(gaf_arr, device=device)

    # define the debug flag
    cfg.debug = False
    # cfg.debug = True

    # cfg.non_query_init = 20
    cfg.query_sample_num = cfg.query_init

    # load the initial gaf and their corresponding video ids
    # gaf_path_train = os.path.join(cfg.stage2model_dir, 'eval_gaf_train_scene.npy')
    # gaf_arr_train = np.load(gaf_path_train)
    # gaf_path_test = os.path.join(cfg.stage2model_dir, 'eval_gaf_test_scene.npy')
    # gaf_arr_test = np.load(gaf_path_test)
    # gaf_arr = np.concatenate((gaf_arr_train, gaf_arr_test), axis=0)
    vid_id_arr_train = np.load(os.path.join(cfg.stage2model_dir, 'eval_vid_id_train.npy'))
    vid_id_arr_test = np.load(os.path.join(cfg.stage2model_dir, 'eval_vid_id_test.npy'))
    vid_id_arr = np.concatenate((vid_id_arr_train, vid_id_arr_test), axis=0)
    gaf_tensor = torch.tensor(gaf_arr, device=device)

    # split the gaf tensor into training and validation set
    frame_id_list_train = training_set.get_frame_id_list()
    gaf_tensor_train_list = []
    for frame_id in frame_id_list_train:
        gaf_tensor_train_list.append(gaf_tensor[vid_id_arr == frame_id])
    gaf_tensor_train = torch.cat(gaf_tensor_train_list, dim=0)
    frame_id_list_valid = validation_set.get_frame_id_list()
    gaf_tensor_valid_list = []
    for frame_id in frame_id_list_valid:
        gaf_tensor_valid_list.append(gaf_tensor[vid_id_arr == frame_id])
    gaf_tensor_valid = torch.cat(gaf_tensor_valid_list, dim=0)

    # select query samples from the validation set
    queries_train = torch.tensor(training_set.get_query_list(), device=device)
    queries = torch.tensor(validation_set.get_query_list(), device=device)
    query_indices = torch.nonzero(queries).squeeze()
    query_indices_rand_perm = torch.randperm(query_indices.shape[0])
    query_indices_use = query_indices[query_indices_rand_perm[:cfg.query_init]]
    frames_validation = validation_set.get_frames_all()
    frames_query = []
    for query_idx in query_indices_use:
        frames_query.append(frames_validation[query_idx])
    validation_set.set_frames(frames_query)
    # print(f'{len(frames_query)} samples are used as query samples.')

    with torch.no_grad():
        validation_set.set_frames(frames_query)
        data_loader = data.DataLoader(validation_set, **params)
        data_set = validation_set
        individual_features = torch.zeros(len(data_set), cfg.num_boxes, cfg.num_features_boxes*2, device=device)
        group_features = torch.zeros(len(data_set), cfg.num_features_boxes*2, device=device)
        actions_in_all = torch.zeros(len(data_set), cfg.num_boxes, device=device)
        original_features = torch.zeros(len(data_set), cfg.num_frames, cfg.num_boxes, (cfg.emb_features*cfg.crop_size[0]*cfg.crop_size[1]), device=device)
        query_people_labels_gt = torch.zeros(len(data_set), cfg.num_boxes, device=device, dtype=torch.long)
        video_id_all = []
        gaf_query_anchor = torch.zeros(len(data_set), cfg.num_features_boxes*2, device=device)

        for batch_idx, batch_data_test in enumerate(data_loader):
            for key in batch_data_test.keys():
                if torch.is_tensor(batch_data_test[key]):
                    batch_data_test[key] = batch_data_test[key].to(device=device)
            images_in = batch_data_test['images_in']
            batch_size, num_frames, _, _, _ = images_in.shape
            actions_in = batch_data_test['actions_in'].reshape((batch_size,num_frames,cfg.num_boxes))
            actions_in = actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
            ret = model(batch_data_test)

            # save features
            start_batch_idx = batch_idx*cfg.batch_size
            finish_batch_idx = start_batch_idx+batch_size
            individual_features[start_batch_idx:finish_batch_idx] = ret['individual_feat'].view(batch_size, cfg.num_boxes, cfg.num_features_boxes*2)
            group_features[start_batch_idx:finish_batch_idx] = ret['group_feat']
            actions_in_all[start_batch_idx:finish_batch_idx] = actions_in.reshape(batch_size, cfg.num_boxes)
            query_people_labels_gt[start_batch_idx:finish_batch_idx] = batch_data_test['user_queries_people'][:, 0, :]
            video_id_all.extend(batch_data_test['video_id'])

            semi_pruning_list = ['pruning_p2p', 'pruning_p2g', 'pruning_p2g_inner_cos', 'pruning_p2g_inner_euc', 'pruning_p2g_cos']
            if cfg.use_anchor_type in semi_pruning_list:
                vote_history = detect_key_people(cfg, device, data_set, batch_size, model, data_loader, group_features, actions_in_all, ACTIONS, params)

            # vote_history = torch.zeros(len(data_set), cfg.num_boxes, device=device)
            # data_patterns = itertools.permutations(range(len(data_set)), 2)
            # one_mask_anchor_type_list = ['pruning_p2p', 'pruning_p2g_cos', 'pruning_p2g_euc', 'pruning_p2g_inner_cos', 'pruning_p2g_inner_euc']
            # zero_mask_anchor_type_list = ['pruning_p2g_cos_inv', 'pruning_p2g_euc_inv', 'pruning_p2g_inner_cos_inv', 'pruning_p2g_inner_euc_inv']

            # perturbation_features = torch.zeros(batch_size, cfg.num_boxes, cfg.num_features_boxes*2, device=device)
            # for person_index in range(cfg.num_boxes):
            #     if cfg.use_anchor_type in one_mask_anchor_type_list:
            #         perturbation_mask = torch.ones(cfg.num_boxes, device=device, dtype=torch.bool)
            #         perturbation_mask[person_index] = False
            #     else:
            #         perturbation_mask = torch.zeros(cfg.num_boxes, device=device, dtype=torch.bool)
            #         perturbation_mask[person_index] = True
            #     perturbation_mask = perturbation_mask.view(1, 1, cfg.num_boxes)
            #     perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
            #     batch_data_test['perturbation_mask'] = perturbation_mask
            #     ret_purtubation = model(batch_data_test)
            #     perturbation_features[:, person_index, :] = ret_purtubation['group_feat']

            # if cfg.use_anchor_type in ['pruning_p2p']:
            #     for base_index, target_index in data_patterns:
            #         cos_sim_matrix = torch.zeros(cfg.num_boxes, cfg.num_boxes, device=device)
            #         perturbation_features_base = perturbation_features[base_index]
            #         perturbation_features_target = perturbation_features[target_index]
            #         cos_sim_matrix = torch.nn.functional.cosine_similarity(perturbation_features_base.unsqueeze(1), 
            #                                                                 perturbation_features_target.unsqueeze(0), dim=-1)
            #         row_ind, col_ind = linear_sum_assignment(-cos_sim_matrix.cpu().numpy())
            #         # print(f'Base index: {base_index}, Target index: {target_index}')
            #         for row_idx, col_idx in zip(row_ind, col_ind):
            #             cos_sim_best = cos_sim_matrix[row_idx, col_idx]
            #             vote_history[base_index, row_idx] += cos_sim_matrix[row_idx, col_idx]
            #             action_index_base = int(actions_in_all[base_index, row_idx].item())
            #             action_index_target = int(actions_in_all[target_index, col_idx].item())
            #             action_name_base = ACTIONS[action_index_base]
            #             action_name_target = ACTIONS[action_index_target]
            #             # print(f'Person {row_idx}: {cos_sim_best.item()} ({action_name_base}) ({action_name_target})')

            # elif cfg.use_anchor_type in ['pruning_p2g_cos', 'pruning_p2g_euc', 'pruning_p2g_cos_inv', 'pruning_p2g_euc_inv']:
            #     for base_index, target_index in data_patterns:
            #         for base_person_index in range(cfg.num_boxes):
            #             perturbation_features_base = perturbation_features[base_index, base_person_index]
            #             target_features = group_features[target_index]
            #             if cfg.use_anchor_type == 'pruning_p2g_cos':
            #                 kp_score = torch.nn.functional.cosine_similarity(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), dim=-1)
            #             elif cfg.use_anchor_type == 'pruning_p2g_euc':
            #                 kp_score = torch.nn.functional.pairwise_distance(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), p=2) * -1
            #             elif cfg.use_anchor_type == 'pruning_p2g_cos_inv':
            #                 kp_score = torch.nn.functional.cosine_similarity(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), dim=-1) * -1
            #             elif cfg.use_anchor_type == 'pruning_p2g_euc_inv':
            #                 kp_score = torch.nn.functional.pairwise_distance(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), p=2)
            #             vote_history[base_index, base_person_index] += kp_score.item()

            # elif cfg.use_anchor_type in ['pruning_p2g_inner_cos', 'pruning_p2g_inner_euc', 'pruning_p2g_inner_cos_inv', 'pruning_p2g_inner_euc_inv']:
            #     for base_index in range(len(data_set)):
            #         for base_person_index in range(cfg.num_boxes):
            #             perturbation_features_base = perturbation_features[base_index, base_person_index]
            #             target_features = group_features[base_index]
            #             if cfg.use_anchor_type == 'pruning_p2g_inner_cos':
            #                 kp_score = torch.nn.functional.cosine_similarity(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), dim=-1)
            #             elif cfg.use_anchor_type == 'pruning_p2g_inner_euc':
            #                 kp_score = torch.nn.functional.pairwise_distance(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), p=2) * -1
            #             elif cfg.use_anchor_type == 'pruning_p2g_inner_cos_inv':
            #                 kp_score = torch.nn.functional.cosine_similarity(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), dim=-1) * -1
            #             elif cfg.use_anchor_type == 'pruning_p2g_inner_euc_inv':
            #                 kp_score = torch.nn.functional.pairwise_distance(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), p=2)
            #             vote_history[base_index, base_person_index] += kp_score.item()

            for base_index in range(len(data_set)):
                vote_history_base = vote_history[base_index]
                vote_history_base_sorted = torch.argsort(vote_history_base)
                for person_index in vote_history_base_sorted:
                    action_index = int(actions_in_all[base_index, person_index].item())
                    query_labels_gt = query_people_labels_gt[base_index, person_index]
                    if cfg.dataset_symbol in ['vol', 'cad']:
                        action_name = ACTIONS[action_index]
                        # print(f'Person {person_index}: {vote_history_base[person_index].item()} ({action_name}) ({query_labels_gt.item()})')
                        if not action_name in ret_dic['key_people_prob']:
                            ret_dic['key_people_prob'][action_name] = []
                        ret_dic['key_people_prob'][action_name].append(vote_history_base[person_index].item())
                        if not action_name in ret_dic['key_people_gt']:
                            ret_dic['key_people_gt'][action_name] = []
                        ret_dic['key_people_gt'][action_name].append(query_labels_gt.item())
                    elif cfg.dataset_symbol in ['bsk']:
                        assert False, 'NBA dataset is not supported yet.'

    return ret_dic