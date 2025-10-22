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

def detect_key_person(cfg):
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
    # cfg.person_size = (368, 368)
    show_config(cfg)
    
    # Reading dataset
    training_set, validation_set, all_set = return_dataset(cfg)
    training_set.is_training = False
    validation_set.is_training = False
    all_set.is_training = False
    cfg.num_boxes = all_set.get_num_boxes_max()

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
    print_log(cfg.log_path, f'Loading stage{cfg.eval_stage} model: ' + cfg.stage4model)

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

    if cfg.dataset_symbol in ['vol', 'cad']:
        gaf_path_train = os.path.join(cfg.stage2model_dir, 'eval_gaf_train_scene.npy')
        gaf_path_test = os.path.join(cfg.stage2model_dir, 'eval_gaf_test_scene.npy')
    elif cfg.dataset_symbol in ['bsk']:
        gaf_path_train = os.path.join(cfg.stage2model_dir, 'eval_gaf_train_people.npy')
        gaf_path_test = os.path.join(cfg.stage2model_dir, 'eval_gaf_test_people.npy')
    gaf_arr_train = np.load(gaf_path_train)
    gaf_arr_test = np.load(gaf_path_test)
    gaf_arr = np.concatenate((gaf_arr_train, gaf_arr_test), axis=0)
    gaf_tensor = torch.tensor(gaf_arr, device=device)

    # define the debug flag
    cfg.debug = False
    # cfg.debug = True

    # cfg.non_query_init = 20
    cfg.query_sample_num = cfg.query_init

    # make a directory for log
    save_dir = os.path.join('analysis', 'key_person_detection', cfg.model_exp_name, cfg.query_type, 
                            f'query_{cfg.query_sample_num}', cfg.use_anchor_type, cfg.anchor_thresh_type,
                            cfg.key_person_mode, cfg.anchor_agg_mode, str(cfg.test_random_seed))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if (cfg.use_anchor_type in ['pruning_gt_reduction']) or (cfg.key_person_mode in ['mask_one_reduction']):
        reduct_feat_dim = 32
        print(f'Use feature reduction with {reduct_feat_dim} dimensions')
        kpca = KernelPCA(n_components=reduct_feat_dim, kernel='rbf')
        kpca.fit_transform(gaf_tensor.cpu().numpy())

    # load the initial gaf and their corresponding video ids
    vid_id_arr_train = np.load(os.path.join(cfg.stage2model_dir, 'eval_vid_id_train.npy'))
    vid_id_arr_test = np.load(os.path.join(cfg.stage2model_dir, 'eval_vid_id_test.npy'))
    vid_id_arr = np.concatenate((vid_id_arr_train, vid_id_arr_test), axis=0)

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

    # select non-query samples from the training set
    q_to_nq_cos_sim_stack = []
    for query_idx in query_indices_use:
        gaf_query = gaf_tensor_valid[query_idx].unsqueeze(0)
        cos_sim_query = torch.nn.functional.cosine_similarity(gaf_query, gaf_tensor_train, dim=-1)
        q_to_nq_cos_sim_stack.append(cos_sim_query)
    q_to_nq_cos_sim_stack = torch.stack(q_to_nq_cos_sim_stack)
    q_to_nq_cos_sim, _ = q_to_nq_cos_sim_stack.max(dim=0)
    if cfg.sampling_mode == 'rand':
        non_query_indices_use = torch.randperm(len(queries_train))[:int(cfg.non_query_init)]
    elif cfg.sampling_mode == 'near':
        _, non_query_indices_use = torch.topk(q_to_nq_cos_sim, int(cfg.non_query_init), largest=True)
    else:
        _, non_query_indices_use = torch.topk(q_to_nq_cos_sim, int(cfg.non_query_init), largest=True)

    confidence_non_query_use = q_to_nq_cos_sim[non_query_indices_use]

    # calculate the query ratio in non_query_indices_use
    query_ratio = torch.sum(queries_train[non_query_indices_use]).item() / len(non_query_indices_use)
    print(f'Query ratio: {query_ratio}')

    # calculate the query ratio in non_query_indices_use_farther
    _, non_query_indices_use_farther = torch.topk(q_to_nq_cos_sim, int(cfg.non_query_init), largest=False)
    query_ratio_farther = torch.sum(queries_train[non_query_indices_use_farther]).item() / len(non_query_indices_use_farther)
    print(f'Query ratio farther: {query_ratio_farther}')

    # save cosine similarity distribution as a png file
    plt.figure()
    sns.histplot(q_to_nq_cos_sim.cpu().numpy(), bins=100, kde=True)
    plt.xlabel('Cosine similarity')
    plt.ylabel('Frequency')
    plt.title('Cosine similarity distribution')
    plt.savefig(os.path.join(save_dir, 'cosine_similarity.png'))

    # print the 25%, 50%, 75% quantile of the cosine similarity
    q_to_nq_cos_sim_sorted = torch.sort(q_to_nq_cos_sim)
    print(f'25% quantile: {q_to_nq_cos_sim_sorted[0][int(0.25*len(q_to_nq_cos_sim))].item()}')
    print(f'50% quantile: {q_to_nq_cos_sim_sorted[0][int(0.50*len(q_to_nq_cos_sim))].item()}')
    print(f'75% quantile: {q_to_nq_cos_sim_sorted[0][int(0.75*len(q_to_nq_cos_sim))].item()}')

    frames_training = training_set.get_frames_all()
    frames_active = []
    frames_active_inv = []
    non_query_indices_use_pos = []
    non_query_indices_use_neg = []
    for non_query_idx in non_query_indices_use:
        frames_active.append(frames_training[non_query_idx])
        if queries_train[non_query_idx] == 1:
            non_query_indices_use_pos.append(non_query_idx)
        else:
            non_query_indices_use_neg.append(non_query_idx)
    
    for non_query_idx in non_query_indices_use_farther:
        frames_active_inv.append(frames_training[non_query_idx])

    frames_active_pos = []
    for non_query_idx in non_query_indices_use_pos:
        frames_active_pos.append(frames_training[non_query_idx])
    frames_active_neg = []
    for non_query_idx in non_query_indices_use_neg:
        frames_active_neg.append(frames_training[non_query_idx])

    training_set.set_frames(frames_active)
    print(f'{len(frames_active)} samples are used in our fine-tuning.')

    validation_set.set_frames(frames_query)
    print(f'{len(frames_query)} samples are used as query samples.')

    if cfg.use_anchor_type in ['learnable', 'learnable_wei', 'learnable_l2_reg']:
        key_person_estimator = torch.nn.Sequential(
            torch.nn.Linear(cfg.num_features_boxes*2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        ).to(device=device)

        key_person_estimator_optim = optim.Adam(key_person_estimator.parameters(), lr=1e-3)
        key_person_estimator.train()

    # if True:
    with torch.no_grad():
        validation_set.set_frames(frames_query)
        data_loader = data.DataLoader(validation_set, **params)
        data_set = validation_set
        print(f'Lengh of frames: {len(frames_query)} for anchor')
        individual_features = torch.zeros(len(data_set), cfg.num_boxes, cfg.num_features_boxes*2, device=device)
        group_features = torch.zeros(len(data_set), cfg.num_features_boxes*2, device=device)
        actions_in_all = torch.zeros(len(data_set), cfg.num_boxes, device=device)
        original_features = torch.zeros(len(data_set), cfg.num_frames, cfg.num_boxes, (cfg.emb_features*cfg.crop_size[0]*cfg.crop_size[1]), device=device)
        query_people_labels_gt = torch.zeros(len(data_set), cfg.num_boxes, device=device, dtype=torch.long)
        video_id_all = []
        gaf_query_anchor = torch.zeros(len(data_set), cfg.num_features_boxes*2, device=device)

        for batch_idx, batch_data_test in enumerate(tqdm(data_loader)):
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
                for base_index in range(len(data_set)):
                    print(f'Base index: {base_index}')
                    vote_history_base = vote_history[base_index]
                    vote_history_base_sorted = torch.argsort(vote_history_base)
                    for person_index in vote_history_base_sorted:
                        action_index = int(actions_in_all[base_index, person_index].item())
                        query_labels_gt = query_people_labels_gt[base_index, person_index]

                        if cfg.dataset_symbol in ['vol', 'cad']:
                            action_name = ACTIONS[action_index]
                            print(f'Person {person_index}: {vote_history_base[person_index].item()} ({action_name}) ({query_labels_gt.item()})')
                        elif cfg.dataset_symbol in ['bsk']:
                            print(f'Person {person_index}: {vote_history_base[person_index].item()}')

            if cfg.save_image:
                for b_idx in range(batch_size):
                    video_id = batch_data_test['video_id'][b_idx]
                    print(f'Video ID: {video_id}')
                    queries_people_b = batch_data_test['user_queries_people'][b_idx, 0]
                    if cfg.dataset_symbol == 'vol':
                        vid_id, seq_id, img_id = video_id.split('_')
                        img_path = os.path.join('data_local', 'videos', vid_id, seq_id, f'{seq_id}.jpg')
                        video_id_save = video_id
                        boxes = batch_data_test['boxes_wo_norm_in'][b_idx, num_frames//2]
                    elif cfg.dataset_symbol == 'cad':
                        vid_id, seq_id, img_id = video_id.split('_')
                        img_path = os.path.join('data_local', 'collective', f'seq{str(vid_id).zfill(2)}', f'frame{str(seq_id).zfill(4)}.jpg')
                        video_id_save = video_id
                        boxes = batch_data_test['boxes_wo_norm_in'][b_idx, num_frames//2]
                    elif cfg.dataset_symbol == 'bsk':
                        vid_id, seq_id, img_id = video_id.split('_')
                        img_id = 30
                        img_path = os.path.join('data', 'basketball', 'videos', f'{vid_id}', f'{seq_id}', f'{str(img_id).zfill(6)}.jpg')
                        boxes = batch_data_test['boxes_wo_norm_in'][b_idx, num_frames//2-1]
                        video_id_save = f'{vid_id}_{seq_id}_{img_id}'
                    img = cv2.imread(img_path)


                    if cfg.use_anchor_type in semi_pruning_list:
                        vote_history_b = vote_history[b_idx]
                        vote_history_b_norm = (vote_history_b - vote_history_b.min()) / (vote_history_b.max() - vote_history_b.min())
                    for p_idx in range(cfg.num_boxes):
                        action_index = batch_data_test['actions_in'][b_idx, num_frames//2, p_idx]
                        action_name = ACTIONS[int(action_index.item())]
                        x1, y1, x2, y2 = map(int, boxes[p_idx])

                        # if queries_people_b[p_idx] == 1:
                            # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        # else:
                            # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        if cfg.use_anchor_type in semi_pruning_list:
                            cv2.putText(img, f'{vote_history[b_idx, p_idx].item():.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                            vote_history_b_norm_p = vote_history_b_norm[p_idx]
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255-int(vote_history_b_norm_p*255), int(vote_history_b_norm_p*255)), 2)

                    # save the image
                    img_save_path = os.path.join(save_dir, f'{video_id_save}_anchor.jpg')
                    # print(f'Save image: {img_save_path}')
                    cv2.imwrite(img_save_path, img)
            
            use_anchor_type = cfg.use_anchor_type
            print(f'Use anchor type: {use_anchor_type}')
            
            if use_anchor_type == 'normal':
                gaf_query_anchor_b = group_features
            elif use_anchor_type in semi_pruning_list:
                gaf_query_anchor_b = []
                for base_index in range(len(data_set)):

                    if cfg.anchor_thresh_type == 'ratio':
                        rank_list = []
                        vote_history_base = vote_history[base_index]
                        for action_index in range(cfg.num_boxes):
                            vote_history_base_action = vote_history_base[action_index]
                            vote_history_base_action_rank = torch.sum(vote_history_base < vote_history_base_action).item()
                            rank_list.append(vote_history_base_action_rank)
                        rank_list = torch.tensor(rank_list, device=device)
                        rank_list_sorted = torch.argsort(rank_list, descending=False)[int(cfg.num_boxes * cfg.use_pruning_ratio):]
                        perturbation_mask = torch.ones(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
                        for action_index in rank_list_sorted:
                            perturbation_mask[:, action_index] = False
                    elif cfg.anchor_thresh_type == 'val':
                        perturbation_mask = vote_history < cfg.use_pruning_ratio

                    perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes)
                    perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
                    batch_data_test['perturbation_mask'] = perturbation_mask
                    ret_purtubation = model(batch_data_test)
                    gaf_purturbation = ret_purtubation['group_feat'][base_index]
                    gaf_query_anchor_b.append(gaf_purturbation)
                gaf_query_anchor_b = torch.stack(gaf_query_anchor_b, dim=0)

            elif use_anchor_type in ['pruning_gt', 'pruning_gt_reduction']:
                perturbation_mask = torch.ones(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
                for query_index in range(batch_size):
                    query_people_labels_gt_batch = query_people_labels_gt[query_index]
                    query_people_labels_gt_batch = query_people_labels_gt_batch == 1
                    if torch.sum(query_people_labels_gt_batch) != 0:
                        perturbation_mask[query_index, query_people_labels_gt_batch] = False
                    else:
                        perturbation_mask[query_index] = False
                        video_id_batch = video_id_all[query_index]
                        print(f'No query person in video: {video_id_batch}')

                perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes)
                perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
                batch_data_test['perturbation_mask'] = perturbation_mask
                ret_purtubation = model(batch_data_test)
                gaf_purturbation = ret_purtubation['group_feat']
                gaf_query_anchor_b = gaf_purturbation

                perturbation_mask_inv = ~perturbation_mask
                batch_data_test['perturbation_mask'] = perturbation_mask_inv
                ret_purtubation_inv = model(batch_data_test)
                gaf_purturbation_inv = ret_purtubation_inv['group_feat']
                gaf_query_anchor_b_inv = gaf_purturbation_inv

                if use_anchor_type in ['pruning_gt_reduction']:        
                    gaf_query_anchor_b = kpca.transform(gaf_query_anchor_b.cpu().detach().numpy())
                    gaf_query_anchor_b = torch.tensor(gaf_query_anchor_b, device=device)
            gaf_query_anchor[start_batch_idx:finish_batch_idx] = gaf_query_anchor_b
    
    with torch.no_grad():
        print('Extract the group features of the non-query samples farther from the query samples')
        training_set.set_frames(frames_active_inv)
        data_set = training_set
        data_loader = data.DataLoader(training_set, **params)
        group_features_non_query_farther = torch.zeros(len(data_set), cfg.num_features_boxes*2, device=device)
        for batch_idx, batch_data_test in enumerate(tqdm(data_loader)):
            for key in batch_data_test.keys():
                if torch.is_tensor(batch_data_test[key]):
                    batch_data_test[key] = batch_data_test[key].to(device=device)
            images_in = batch_data_test['images_in']
            batch_size, num_frames, _, _, _ = images_in.shape
            actions_in = batch_data_test['actions_in'].reshape((batch_size,num_frames,cfg.num_boxes))
            actions_in = actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
            ret = model(batch_data_test)
            group_features_non_query_farther[batch_idx*cfg.batch_size:(batch_idx+1)*cfg.batch_size] = ret['group_feat']
        if use_anchor_type in ['pruning_gt_reduction']:
            group_features_non_query_farther = kpca.transform(group_features_non_query_farther.cpu().detach().numpy())
            group_features_non_query_farther = torch.tensor(group_features_non_query_farther, device=device)

        action_to_dist_dic = {}
        search_dic = {}
        search_dic['non_anchor'] = non_query_indices_use
        if len(non_query_indices_use_pos) > 0:
            search_dic['non_query_pos'] = non_query_indices_use_pos
        if len(non_query_indices_use_neg) > 0:
            search_dic['non_query_neg'] = non_query_indices_use_neg
        result_dic = {}
        for indices_name, indices in search_dic.items():
            if not indices_name in action_to_dist_dic:
                action_to_dist_dic[indices_name] = {}
            
            frames = []
            for idx in indices:
                frames.append(frames_train[idx])
            training_set.set_frames(frames)
            data_set = training_set
            data_loader = data.DataLoader(training_set, **params)

            dist_purt_agg_list = []
            dist_inner_agg_list = []
            labels_gt_list = []
            actions_gt_list = []
            query_people_labels_det = torch.zeros(len(data_set), cfg.num_boxes, device=device, dtype=torch.long)
            for batch_idx, batch_data_test in enumerate(tqdm(data_loader)):
                for key in batch_data_test.keys():
                    if torch.is_tensor(batch_data_test[key]):
                        batch_data_test[key] = batch_data_test[key].to(device=device)

                images_in = batch_data_test['images_in']
                batch_size, num_frames, _, _, _ = images_in.shape
                start_index = batch_idx*cfg.batch_size
                finish_index = start_index+batch_size
                actions_in = batch_data_test['actions_in'].reshape((batch_size,num_frames,cfg.num_boxes))
                actions_in = actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
                locations_in = batch_data_test['boxes_in'].reshape((batch_size,num_frames,cfg.num_boxes, 4))
                locations_in_x_mid = (locations_in[:, :, :, 0] + locations_in[:, :, :, 2]) / 2
                locations_in_y_mid = (locations_in[:, :, :, 1] + locations_in[:, :, :, 3]) / 2
                query_labels_gt_batch = batch_data_test['user_queries'][:, 0]
                query_people_labels_gt_batch = batch_data_test['user_queries_people'][:, 0, :]

                ret = model(batch_data_test)
                # original_features_batch = ret['original_features'].view(batch_size, cfg.num_frames, cfg.num_boxes, cfg.emb_features*cfg.crop_size[0]*cfg.crop_size[1])

                if 'recog_key_person' in ret:
                    recog_key_person_mid = ret['recog_key_person'][:, 4, :]
                    recog_key_person_arg = torch.argmax(recog_key_person_mid, dim=-1)
                    for data_idx in range(batch_size):
                        print(f'Data index: {data_idx}')
                        print(f'GT query people: {query_people_labels_gt_batch[data_idx]}')
                        # print(f'Det query people: {recog_key_person_mid[data_idx]}')
                        print(f'Det query people: {recog_key_person_arg[data_idx]}')

                key_person_mode = cfg.key_person_mode
                anchor_agg_mode = cfg.anchor_agg_mode
                gaf_original = ret['group_feat']
                if use_anchor_type in ['pruning_gt_reduction']:
                    gaf_original = kpca.transform(gaf_original.cpu().detach().numpy())
                    gaf_original = torch.tensor(gaf_original, device=device)

                
                elif key_person_mode in ['mask_zero', 'mask_one', 'mask_one_farther', 'mask_one_negative', 'mask_one_position', 'mask_one_appearance']:
                    dist_purt_pert_list = []
                    dist_inner_pert_list = []
                    for p_idx in range(cfg.num_boxes):
                        if key_person_mode in ['mask_zero']:
                            perturbation_mask = torch.zeros(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
                            perturbation_mask[:, p_idx] = True
                        elif key_person_mode in ['mask_one', 'mask_one_farther', 'mask_one_negative', 'mask_one_position', 'mask_one_appearance']:
                            perturbation_mask = torch.ones(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
                            perturbation_mask[:, p_idx] = False
                        perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes)
                        perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
                        batch_data_test['perturbation_mask'] = perturbation_mask
                        ret_purtubation = model(batch_data_test)

                        gaf_purturbation = ret_purtubation['group_feat']
                        if use_anchor_type in ['pruning_gt_reduction']:
                            gaf_purturbation = kpca.transform(gaf_purturbation.cpu().detach().numpy())
                            gaf_purturbation = torch.tensor(gaf_purturbation, device=device)

                        feat_dot_purt = gaf_purturbation @ gaf_query_anchor.t()
                        cos_sim = feat_dot_purt / (gaf_purturbation.norm(dim=-1).view(-1, 1) * gaf_query_anchor.norm(dim=-1).view(1, -1))

                        feat_dot_orig = gaf_original @ gaf_query_anchor.t()
                        cos_sim_orig = feat_dot_orig / (gaf_original.norm(dim=-1).view(-1, 1) * gaf_query_anchor.norm(dim=-1).view(1, -1))

                        if key_person_mode in ['mask_one_position']:
                            perturbation_mask_position = torch.zeros(batch_size, cfg.num_frames, cfg.num_boxes, 2, device=device, dtype=torch.float)
                            perturbation_mask_position[:, :, p_idx, 0] = torch.randn(batch_size, cfg.num_frames, device=device) * 30
                            perturbation_mask_position[:, :, p_idx, 1] = torch.randn(batch_size, cfg.num_frames, device=device) * 20
                            batch_data_test['perturbation_mask_position'] = perturbation_mask_position
                        elif key_person_mode in ['mask_one_appearance']:
                            perturbation_mask_apperance = torch.ones(batch_size, cfg.num_frames, cfg.num_boxes, device=device, dtype=torch.float)
                            perturbation_mask_apperance[:, :, p_idx] = torch.randn(batch_size, cfg.num_frames, device=device)
                            batch_data_test['perturbation_mask_apperance'] = perturbation_mask_apperance
                        
                        if key_person_mode in ['mask_one_position']:
                            gaf_purturbation_pos = model(batch_data_test)['group_feat']
                            feat_dot_purt_pos = gaf_purturbation_pos @ gaf_query_anchor.t()
                            cos_sim_pos_purt = feat_dot_purt_pos / (gaf_purturbation_pos.norm(dim=-1).view(-1, 1) * gaf_query_anchor.norm(dim=-1).view(1, -1))
                        elif key_person_mode in ['mask_one_appearance']:
                            gaf_purturbation_app = model(batch_data_test)['group_feat']
                            feat_dot_purt_app = gaf_purturbation_app @ gaf_query_anchor.t()
                            cos_sim_app_purt = feat_dot_purt_app / (gaf_purturbation_app.norm(dim=-1).view(-1, 1) * gaf_query_anchor.norm(dim=-1).view(1, -1))

                        feat_dot_orig_purt = torch.sum(gaf_original*gaf_purturbation, dim=-1)
                        cos_sim_orig_purt = feat_dot_orig_purt / (gaf_original.norm(dim=-1) * gaf_purturbation.norm(dim=-1))

                        feat_dot_purt_n_q_farther = gaf_purturbation @ group_features_non_query_farther.t()
                        cos_sim_n_q_farther = feat_dot_purt_n_q_farther / (gaf_purturbation.norm(dim=-1).view(-1, 1) * group_features_non_query_farther.norm(dim=-1).view(1, -1))
                        cos_sim_n_q_farther = cos_sim_n_q_farther.mean(dim=-1).unsqueeze(1)

                        if key_person_mode == 'mask_zero':
                            cos_sim = cos_sim_orig - cos_sim
                        elif key_person_mode in ['mask_one']:
                            cos_sim = cos_sim
                        elif key_person_mode in ['mask_one_farther']:
                            cos_sim = cos_sim - cos_sim_n_q_farther
                        elif key_person_mode in ['mask_one_negative']:
                            cos_sim = cos_sim - cos_sim_n_q_farther
                        elif key_person_mode in ['mask_one_position']:
                            cos_sim = (cos_sim + cos_sim_pos_purt) / 2
                        elif key_person_mode in ['mask_one_appearance']:
                            cos_sim = (cos_sim + cos_sim_app_purt) / 2

                        if anchor_agg_mode == 'max':
                            cos_sim_agg = cos_sim.max(dim=-1).values
                        elif anchor_agg_mode == 'mean':
                            cos_sim_agg = cos_sim.mean(dim=-1)

                        dist_purt_pert_list.append(cos_sim_agg)
                        dist_inner_pert_list.append(cos_sim_orig_purt)

                    dist_purt_agg = torch.stack(dist_purt_pert_list, dim=1).view(batch_size, cfg.num_boxes)
                    dist_purt_agg_list.append(dist_purt_agg)

                    dist_inner_agg = torch.stack(dist_inner_pert_list, dim=1).view(batch_size, cfg.num_boxes)
                    dist_inner_agg_list.append(dist_inner_agg)

                elif key_person_mode == 'mask_position':
                    dist_purt_pert_list = []
                    for p_idx in range(cfg.num_boxes):
                        perturbation_mask_position = torch.zeros(batch_size, cfg.num_frames, cfg.num_boxes, 2, device=device, dtype=torch.float)
                        perturbation_mask_position[:, :, p_idx, 0] = (30 - locations_in_x_mid[:, :, p_idx]) - locations_in_x_mid[:, :, p_idx]
                        perturbation_mask_position[:, :, p_idx, 1] = (20 - locations_in_y_mid[:, :, p_idx]) - locations_in_y_mid[:, :, p_idx]
                        batch_data_test['perturbation_mask_position'] = perturbation_mask_position
                        ret_purtubation = model(batch_data_test)
                        gaf_purturbation = ret_purtubation['group_feat']
                        feat_dot_purt = gaf_purturbation @ gaf_query_anchor.t()
                        cos_sim_purt = feat_dot_purt / (gaf_purturbation.norm(dim=-1).view(-1, 1) * gaf_query_anchor.norm(dim=-1).view(1, -1))
                        feat_dot_orig = gaf_original @ gaf_query_anchor.t()
                        cos_sim_orig = feat_dot_orig / (gaf_original.norm(dim=-1).view(-1, 1) * gaf_query_anchor.norm(dim=-1).view(1, -1))
                        cos_sim = cos_sim_orig - cos_sim_purt
                        if anchor_agg_mode == 'max':
                            dist_purt_pert_list.append(cos_sim.max(dim=-1).values)
                        elif anchor_agg_mode == 'mean':
                            dist_purt_pert_list.append(cos_sim.mean(dim=-1))
                    dist_purt_agg = torch.stack(dist_purt_pert_list, dim=1).view(batch_size, cfg.num_boxes)
                    dist_purt_agg_list.append(dist_purt_agg)
                
                elif key_person_mode in ['mask_appearance']:
                    dist_purt_pert_list = []
                    for p_idx in range(cfg.num_boxes):
                        perturbation_mask_apperance = torch.ones(batch_size, cfg.num_frames, cfg.num_boxes, device=device, dtype=torch.float)
                        perturbation_mask_apperance[:, :, p_idx] = torch.randn(batch_size, cfg.num_frames, device=device)
                        batch_data_test['perturbation_mask_apperance'] = perturbation_mask_apperance
                        ret_purtubation = model(batch_data_test)
                        gaf_purturbation = ret_purtubation['group_feat']

                        if key_person_mode == 'mask_appearance':
                            feat_dot_purt = gaf_purturbation @ gaf_query_anchor.t()
                            cos_sim_purt = feat_dot_purt / (gaf_purturbation.norm(dim=-1).view(-1, 1) * gaf_query_anchor.norm(dim=-1).view(1, -1))
                            feat_dot_orig = gaf_original @ gaf_query_anchor.t()
                            cos_sim_orig = feat_dot_orig / (gaf_original.norm(dim=-1).view(-1, 1) * gaf_query_anchor.norm(dim=-1).view(1, -1))
                            cos_sim = cos_sim_orig - cos_sim_purt

                        if anchor_agg_mode == 'max':
                            dist_purt_pert_list.append(cos_sim.max(dim=-1).values)
                        elif anchor_agg_mode == 'mean':
                            dist_purt_pert_list.append(cos_sim.mean(dim=-1))
                    dist_purt_agg = torch.stack(dist_purt_pert_list, dim=1).view(batch_size, cfg.num_boxes)
                    dist_purt_agg_list.append(dist_purt_agg)

                elif key_person_mode == 'ind_feat':
                    ind_feature = ret['individual_feat'].view(batch_size*cfg.num_boxes, cfg.num_features_boxes*2)
                    feat_dot = ind_feature @ gaf_query_anchor.t()
                    cos_sim = feat_dot / (ind_feature.norm(dim=-1).view(-1, 1) * gaf_query_anchor.norm(dim=-1).view(1, -1))
                    cos_sim = cos_sim.view(batch_size, cfg.num_boxes, gaf_query_anchor.shape[0])
                    if anchor_agg_mode == 'max':
                        dist_purt_agg = cos_sim.max(dim=-1).values
                    elif anchor_agg_mode == 'mean':
                        dist_purt_agg = cos_sim.mean(dim=-1)
                    dist_purt_agg_list.append(dist_purt_agg)
                elif key_person_mode == 'gt':
                    dist_purt_agg = query_people_labels_gt_batch
                    dist_purt_agg_list.append(dist_purt_agg)

                # replace the Nan values in dist_purt_agg with zero
                dist_purt_agg[torch.isnan(dist_purt_agg)] = 0

                confidence_batch = confidence_non_query_use
                for b_idx in range(batch_size):
                    # print(f'Batch index: {b_idx}')
                    dist_purt_agg_b = dist_purt_agg[b_idx]
                    dist_purt_agg_b_norm = (dist_purt_agg_b - dist_purt_agg_b.min()) / (dist_purt_agg_b.max() - dist_purt_agg_b.min())

                    if cfg.save_image:
                        video_id = batch_data_test['video_id'][b_idx]
                        # print(f'Video ID: {video_id}')

                        if cfg.dataset_symbol == 'vol':
                            vid_id, seq_id, img_id = video_id.split('_')
                            img_path = os.path.join('data_local', 'videos', vid_id, seq_id, f'{seq_id}.jpg')
                        elif cfg.dataset_symbol == 'cad':
                            vid_id, seq_id, img_id = video_id.split('_')
                            img_path = os.path.join('data_local', 'collective', f'seq{str(vid_id).zfill(2)}', f'frame{str(seq_id).zfill(4)}.jpg')
                        elif cfg.dataset_symbol == 'bsk':
                            vid_id, seq_id, _ = video_id.split('_')
                            img_id = 30
                            img_path = os.path.join('data', 'basketball', 'videos', f'{vid_id}', f'{seq_id}', f'{str(img_id).zfill(6)}.jpg')
                        img = cv2.imread(img_path)

                        if cfg.dataset_symbol == 'vol':
                            b_people_num = cfg.num_boxes
                            boxes = batch_data_test['boxes_wo_norm_in'][b_idx, num_frames//2]
                        elif cfg.dataset_symbol == 'cad':
                            bboxes_num = batch_data_test['bboxes_num'].reshape(batch_size, num_frames)
                            b_people_num = bboxes_num[b_idx, num_frames//2].item()
                            boxes = batch_data_test['boxes_wo_norm_in'][b_idx, num_frames//2]
                        elif cfg.dataset_symbol == 'bsk':
                            use_frame_idx = 2
                            bboxes_num = batch_data_test['bboxes_num'].reshape(batch_size, num_frames)
                            b_people_num = bboxes_num[b_idx, use_frame_idx].item()
                            boxes = batch_data_test['boxes_wo_norm_in'][b_idx, use_frame_idx]
                        else:
                            assert False, f'Invalid dataset symbol: {cfg.dataset_symbol}'

                        for p_idx in range(b_people_num):
                            action_index = batch_data_test['actions_in'][b_idx, 0, p_idx]
                            action_name = ACTIONS[int(action_index.item())]
                            x1, y1, x2, y2 = map(int, boxes[p_idx])
                            dist_purt_vis = dist_purt_agg_b[p_idx].item()
                            dist_inner_vis = dist_inner_agg[b_idx, p_idx].item()
                            dist_purt_vis_norm = dist_purt_agg_b_norm[p_idx].item()
                            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255-int(dist_purt_vis_norm*255), int(dist_purt_vis_norm*255)), 2)
                            img = cv2.putText(img, f'{dist_purt_vis:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                        # visualize the confidence score on the upper left corner
                        confidence_global = confidence_batch[b_idx].item()
                        confidence_local_max = dist_purt_agg_b.max().item()
                        confidence_local_mean = dist_purt_agg_b.mean().item()
                        confidence_total_max = (confidence_global + confidence_local_max) / 2
                        confidence_total_mean = (confidence_global + confidence_local_mean) / 2
                        # img = cv2.putText(img, f'{confidence_global:.2f}:{confidence_local:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                        img = cv2.putText(img, f'{confidence_global:.2f}:{confidence_local_max:.2f}:{confidence_local_mean:.2f}:{confidence_total_max:.2f}:{confidence_total_mean:.2f}',
                         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                        # save the image
                        query_labels_gt_batch_b = query_labels_gt_batch[b_idx].long().item()
                        img_save_path = os.path.join(save_dir, f'{video_id}_{query_labels_gt_batch_b}.jpg')
                        cv2.imwrite(img_save_path, img)

                dist_threshold = torch.quantile(dist_purt_agg, cfg.use_key_person_ratio)
                query_people_labels_det_batch = torch.where(dist_purt_agg > dist_threshold, 1, 0)
                query_people_labels_det_batch = query_people_labels_det_batch.view(batch_size, cfg.num_boxes)
                query_people_labels_det[start_index:finish_index] = query_people_labels_det_batch

                # generate labels based on distance in each batch
                # dist_purt_agg = dist_purt_agg.view(batch_size, cfg.num_boxes)
                # cos_sim_people_rank_indices = torch.argsort(dist_purt_agg, dim=1, descending=True)
                # selection_num = int(cfg.use_key_person_ratio*cfg.num_boxes)
                # det_key_person_selected_not = cos_sim_people_rank_indices[:, selection_num:]
                # for b_idx in range(batch_size):
                #     query_people_labels_det[b_idx, det_key_person_selected_not[b_idx]] = 0

                user_queries_people = batch_data_test['user_queries_people'][:, 0, :].to('cpu').flatten().numpy().tolist()
                labels_gt_list.extend(user_queries_people)
                actions_gt_list.extend(actions_in.to('cpu').flatten().numpy().tolist())

                dist_purt_agg = dist_purt_agg.view(batch_size, cfg.num_boxes)
                for b_idx in range(batch_size):
                    dist_purt_agg_b = dist_purt_agg[b_idx]
                    for p_idx in range(cfg.num_boxes):
                        action_index = batch_data_test['actions_in'][b_idx, 0, p_idx]
                        action_name = ACTIONS[int(action_index.item())]
                        dist_purt_agg_b_p = dist_purt_agg_b[p_idx].item()
                        if not action_name in action_to_dist_dic[indices_name]:
                            action_to_dist_dic[indices_name][action_name] = []
                        action_to_dist_dic[indices_name][action_name].append(dist_purt_agg_b_p)

            dist_purt_agg = torch.cat(dist_purt_agg_list, dim=0)
            dist_purt_agg_flatten = dist_purt_agg.view(-1)
            data_size = dist_purt_agg.shape[0]

            # dist_threshold = torch.quantile(dist_purt_agg_flatten, cfg.use_key_person_ratio)
            # query_people_labels_det = torch.where(dist_purt_agg_flatten > dist_threshold, 1, 0)

            # dist_purt_mean_all_rank_indices = torch.argsort(dist_purt_agg, dim=-1, descending=True)
            # selection_num = int(cfg.num_boxes * (1 - cfg.use_key_person_ratio))
            # det_key_person_selected = dist_purt_mean_all_rank_indices[:, :selection_num]
            # query_people_labels_det = torch.zeros(data_size, cfg.num_boxes, device=device, dtype=torch.long)
            # query_people_labels_det[torch.arange(data_size).view(-1, 1), det_key_person_selected] = 1

            dist_det_arr = dist_purt_agg.view(-1).to('cpu').flatten().detach().numpy()
            labels_gt_arr = np.array(labels_gt_list)
            actions_gt_arr = np.array(actions_gt_list)

            dist_det_arr_view = dist_purt_agg.view(-1, cfg.num_boxes)
            actions_gt_arr_view = actions_gt_arr.reshape(-1, cfg.num_boxes)
            labels_gt_arr_view = labels_gt_arr.reshape(-1, cfg.num_boxes)
            # for b_idx in range(data_size):
            #     print(f'Batch index: {b_idx}')
            #     actions_gt_b = actions_gt_arr_view[b_idx]
            #     query_people_labels_det_b = query_people_labels_det[b_idx]
            #     labels_gt_b = labels_gt_arr_view[b_idx]
            #     dist_det_arr_b = dist_det_arr_view[b_idx]
            #     dist_det_arr_b_sorted = torch.argsort(dist_det_arr_b)
            #     for p_idx in dist_det_arr_b_sorted:
            #         action_index = int(actions_gt_b[p_idx])
            #         action_name = ACTIONS[action_index]
            #         label_gt = int(labels_gt_b[p_idx])
            #         dist_det = dist_det_arr_b[p_idx].item()
            #         print(f'Person {p_idx}: {dist_det} ({action_name}) ({label_gt})')

            result_dic[indices_name] = {}
            result_dic[indices_name]['labels_gt'] = labels_gt_arr
            result_dic[indices_name]['labels_det'] = query_people_labels_det.to('cpu').flatten().numpy()
            result_dic[indices_name]['actions_gt'] = actions_gt_arr
            result_dic[indices_name]['dist_det'] = dist_det_arr

        return_dic = {}
        for search_dic_key in search_dic.keys():
            if search_dic_key == 'non_query_neg':
                continue
            print(f'======================={search_dic_key}========================')
            actions_gt_arr = result_dic[search_dic_key]['actions_gt']
            dist_det_arr = result_dic[search_dic_key]['dist_det']
            labels_gt_arr = result_dic[search_dic_key]['labels_gt']
            labels_det_arr = result_dic[search_dic_key]['labels_det']

            # compute recall, precision, f1-score, and AUC
            return_dic_search = {}
            labels_gt_arr = labels_gt_arr.flatten()
            labels_det_arr = labels_det_arr.flatten()
            recall = recall_score(labels_gt_arr, labels_det_arr)
            precision = precision_score(labels_gt_arr, labels_det_arr)
            f1 = f1_score(labels_gt_arr, labels_det_arr)
            ap = average_precision_score(labels_gt_arr, dist_det_arr)
            auc = roc_auc_score(labels_gt_arr, dist_det_arr)
            auc_macro = roc_auc_score(labels_gt_arr, dist_det_arr, average='macro')
            auc_micro = roc_auc_score(labels_gt_arr, dist_det_arr, average='micro')
            auc_weighted = roc_auc_score(labels_gt_arr, dist_det_arr, average='weighted')
            print(f'Recall: {recall:.3f}, Precision: {precision:.3f}, F1-score: {f1:.3f}, AP: {ap:.3f}, AUC: {auc:.3f}')
            # print(f'AUC macro: {auc_macro:.3f}, AUC micro: {auc_micro:.3f}, AUC weighted: {auc_weighted:.3f}')
            return_dic_search['auc'] = auc
            return_dic_search['ap'] = ap

            # compute recall, precision, f1-score for each threshold
            for thresh_quantile_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                labels_gt_arr_thresh = labels_gt_arr.flatten()
                dist_det_arr_thresh = dist_det_arr.flatten()
                labels_det_arr_thresh = np.where(dist_det_arr_thresh > np.quantile(dist_det_arr_thresh, thresh_quantile_ratio), 1, 0)
                recall = recall_score(labels_gt_arr_thresh, labels_det_arr_thresh)
                precision = precision_score(labels_gt_arr_thresh, labels_det_arr_thresh)
                f1 = f1_score(labels_gt_arr_thresh, labels_det_arr_thresh)
                print(f'Thresh: {thresh_quantile_ratio:.1f}, Recall: {recall:.3f}, Precision: {precision:.3f}, F1-score: {f1:.3f}')
                # return_dic[thresh_quantile_ratio] = {'recall': recall, 'precision': precision, 'f1': f1}

            print('Aggregated distance')
            action_to_dist_mean_agg = {}
            for action_name in ACTIONS:
                dist_list = []
                for indices_name, action_to_dist in action_to_dist_dic.items():
                    if action_name in action_to_dist.keys():
                        dist_list += action_to_dist[action_name]
                dist_arr = np.array(dist_list)
                action_to_dist_mean_agg[action_name] = dist_arr.mean()
            action_to_dist_mean_agg_sorted = sorted(action_to_dist_mean_agg.items(), key=lambda x: x[1])
            for action_name, dist_mean in action_to_dist_mean_agg_sorted:
                print(f'Action: {action_name}, distance: {dist_mean}')
            
            return_dic[search_dic_key] = return_dic_search
        
    return return_dic