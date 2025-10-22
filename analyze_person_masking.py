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

    queries = torch.tensor(all_set.get_query_list(), device=device)
    query_indices = torch.nonzero(queries).squeeze()
    query_indices = query_indices[torch.randperm(len(query_indices))]
    non_query_indices = torch.nonzero(queries == 0).squeeze()
    non_query_indices = non_query_indices[torch.randperm(len(non_query_indices))]
    
    use_frames = 1
    trials = 5
    frames_original = all_set.get_frames_all()
    query_indices_sampled = query_indices[:use_frames+1]
    non_query_indices_sampled = non_query_indices[:use_frames]
    frames_anchor = []
    for query_index in query_indices_sampled:
        frames_anchor.append(frames_original[query_index])
    for non_query_index in non_query_indices_sampled:
        frames_anchor.append(frames_original[non_query_index])

    results = torch.zeros(use_frames, trials, cfg.num_boxes-1, 5)
    with torch.no_grad():
        all_set.set_frames(frames_anchor)
        data_loader = data.DataLoader(all_set, **params)
        print(f'Lengh of frames: {len(frames_anchor)} for anchor')
        for batch_idx, batch_data_test in enumerate(tqdm(data_loader)):
            for key in batch_data_test.keys():
                if torch.is_tensor(batch_data_test[key]):
                    batch_data_test[key] = batch_data_test[key].to(device=device)
            images_in = batch_data_test['images_in']
            batch_size, num_frames, _, _, _ = images_in.shape
            actions_in = batch_data_test['actions_in'].reshape((batch_size,num_frames,cfg.num_boxes))
            actions_in = actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
            ret = model(batch_data_test)
            gaf = ret['group_feat']

            gaf_base = gaf[0].unsqueeze(0)
            gaf_query = gaf[1].unsqueeze(0)
            gaf_non_query = gaf[2].unsqueeze(0)
            for seed_num in tqdm(range(trials)):
                print(f'Seed number: {seed_num}')
                torch.manual_seed(seed_num)
                torch.cuda.manual_seed_all(seed_num)
                torch.cuda.manual_seed(seed_num)
                for mask_people_num in range(1, cfg.num_boxes):
                    print(f'Mask people num: {mask_people_num}')
                    mask_people_indices = torch.randperm(cfg.num_boxes)[:mask_people_num]
                    perturbation_mask = torch.zeros(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
                    perturbation_mask[:, mask_people_indices] = True
                    perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes)
                    perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
                    batch_data_test['perturbation_mask'] = perturbation_mask
                    ret_perturbation = model(batch_data_test)
                    gaf_pert = ret_perturbation['group_feat']

                    gaf_pert_base = gaf_pert[0].unsqueeze(0)
                    gaf_pert_query = gaf_pert[1].unsqueeze(0)
                    gaf_pert_non_query = gaf_pert[2].unsqueeze(0)

                    cos_sim_b_pq = torch.nn.functional.cosine_similarity(gaf_base, gaf_pert_query, dim=1).item()
                    cos_sim_b_pb = torch.nn.functional.cosine_similarity(gaf_base, gaf_pert_base, dim=1).item()
                    cos_sim_b_pnq = torch.nn.functional.cosine_similarity(gaf_base, gaf_pert_non_query, dim=1).item()
                    cos_sim_pb_pq = torch.nn.functional.cosine_similarity(gaf_pert_base, gaf_pert_query, dim=1).item()
                    cos_sim_pb_pnq = torch.nn.functional.cosine_similarity(gaf_pert_base, gaf_pert_non_query, dim=1).item()
                    # print(f'B-Q: {cos_sim_b_q:.2f}, B-PQ: {cos_sim_b_pq:.2f}, B-PB: {cos_sim_b_pb:.2f}')
                    # results[batch_idx, seed_num, mask_people_num-1, 0] = cos_sim_b_q
                    # results[batch_idx, seed_num, mask_people_num-1, 1] = cos_sim_b_pq
                    # results[batch_idx, seed_num, mask_people_num-1, 2] = cos_sim_b_pb
                    # results[batch_idx, seed_num, mask_people_num-1, 3] = cos_sim_b_nq
                    # results[batch_idx, seed_num, mask_people_num-1, 4] = cos_sim_b_pnq

                    results[batch_idx, seed_num, mask_people_num-1, 0] = cos_sim_b_pb
                    results[batch_idx, seed_num, mask_people_num-1, 1] = cos_sim_b_pq
                    results[batch_idx, seed_num, mask_people_num-1, 2] = cos_sim_b_pnq
                    results[batch_idx, seed_num, mask_people_num-1, 3] = cos_sim_pb_pq
                    results[batch_idx, seed_num, mask_people_num-1, 4] = cos_sim_pb_pnq

    results_mean = results.mean(dim=(0,1))
    col_names = ['B-PB', 'B-PQ', 'B-PNQ', 'PB-PQ', 'PB-PNQ']
    header_names = [f'Masking {i+1} people' for i in range(cfg.num_boxes-1)]
    df = pd.DataFrame(results_mean.cpu().numpy(), columns=col_names, index=header_names)
    
    save_df_dir = os.path.join('analysis', 'gaf_each_person_num', cfg.model_exp_name)
    if not os.path.exists(save_df_dir):
        os.makedirs(save_df_dir)
    save_df_path = os.path.join(save_df_dir, f'gaf_each_person_num_{cfg.dataset_name}_{cfg.query_type}.xlsx')
    df.to_excel(save_df_path)

    return {}