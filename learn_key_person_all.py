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

def calc_metrics(estimated_key_person, gt_key_person):
    estimated_key_person_label = np.argmax(estimated_key_person, axis=-1)

    # calculate the precision, recall, and f1 score
    precision = precision_score(gt_key_person, estimated_key_person_label)
    recall = recall_score(gt_key_person, estimated_key_person_label)
    f1 = f1_score(gt_key_person, estimated_key_person_label)

    # calculate the AP score
    ap = average_precision_score(gt_key_person, estimated_key_person[:, 1])

    # calculate the AUC score
    auc = roc_auc_score(gt_key_person, estimated_key_person[:, 1])

    met_dic = {}
    met_dic['precision'] = precision
    met_dic['recall'] = recall
    met_dic['f1'] = f1
    met_dic['ap'] = ap
    met_dic['auc'] = auc

    return met_dic

def detect_key_person(cfg):
    """
    evaluating gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list

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

    # select non-query samples from the training set
    cfg.non_query_init = 10
    q_to_nq_cos_sim_stack = []
    for query_idx in query_indices_use:
        gaf_query = gaf_tensor_valid[query_idx].unsqueeze(0)
        cos_sim_query = torch.nn.functional.cosine_similarity(gaf_query, gaf_tensor_train, dim=-1)
        q_to_nq_cos_sim_stack.append(cos_sim_query)
    q_to_nq_cos_sim_stack = torch.stack(q_to_nq_cos_sim_stack)
    q_to_nq_cos_sim, _ = q_to_nq_cos_sim_stack.max(dim=0)
    _, non_query_indices_use = torch.topk(q_to_nq_cos_sim, int(cfg.non_query_init), largest=True)
    query_ratio = torch.sum(queries_train[non_query_indices_use]).item() / len(non_query_indices_use)
    print(f'Query ratio close: {query_ratio}')

    # select non-query samples from the training set
    _, non_query_indices_use_farther = torch.topk(q_to_nq_cos_sim, int(cfg.non_query_init), largest=False)
    query_ratio_farther = torch.sum(queries_train[non_query_indices_use_farther]).item() / len(non_query_indices_use_farther)
    print(f'Query ratio farther: {query_ratio_farther}')

    # select non-query samples from the training set
    cfg.non_query_test_init = 20
    _, non_query_indices_use_test = torch.topk(q_to_nq_cos_sim, int(cfg.non_query_test_init), largest=True)
    query_ratio_test = torch.sum(queries_train[non_query_indices_use_test]).item() / len(non_query_indices_use_test)
    print(f'Query ratio test: {query_ratio_test}')
    
    # set the parameters for the training
    frames_validation = validation_set.get_frames_all()
    frames_query = []
    for query_idx in query_indices_use:
        frames_query.append(frames_validation[query_idx])

    frames_training = training_set.get_frames_all()
    frames_fine_close = []
    for non_query_idx in non_query_indices_use:
        frames_fine_close.append(frames_training[non_query_idx])
    frames_fine_farther = []
    for non_query_idx in non_query_indices_use_farther:
        frames_fine_farther.append(frames_training[non_query_idx])
    frames_fine_test = []
    for non_query_idx in non_query_indices_use_test:
        frames_fine_test.append(frames_training[non_query_idx])

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

    # set the parameters for the training
    learning_rate = 1e-4
    # num_epochs = 1
    num_epochs = 100

    # generate key person estimator
    query_estimator = KeyPersonEstimator(cfg).to(device=device)
    optimizer = optim.Adam(query_estimator.parameters(), lr=learning_rate)

    # set the loss function
    # loss_func = torch.nn.CrossEntropyLoss()
    class_weight = torch.ones(2, device=device)
    # class_weight[0] = 1 / torch.sum(query_people_labels_det_all == 0).float()
    # class_weight[1] = 1 / torch.sum(query_people_labels_det_all == 1).float()
    loss_func = torch.nn.CrossEntropyLoss(weight=class_weight)

    # optimize the key person estimator on the close and farther set in the training set
    frame_fine_all = frames_query + frames_fine_farther
    query_labels_all = torch.cat((torch.ones(len(frames_query)), torch.zeros(len(frames_fine_farther))), dim=0).to(device=device).long()
    print(f'Length of frames fine all: {len(frame_fine_all)}')
    all_set.set_frames(frame_fine_all)
    data_set = all_set
    data_loader = data.DataLoader(all_set, **params)

    gaf_purturbation_all = torch.zeros(len(data_set), cfg.num_boxes, cfg.num_features_boxes*2, device=device)
    actions_in_all = torch.zeros(len(data_set), cfg.num_boxes, device=device)
    for batch_idx, batch_data_test in enumerate(tqdm(data_loader)):
        for key in batch_data_test.keys():
            if torch.is_tensor(batch_data_test[key]):
                batch_data_test[key] = batch_data_test[key].to(device=device)
        images_in = batch_data_test['images_in']
        batch_size, num_frames, _, _, _ = images_in.shape
        start_index = batch_idx*cfg.batch_size
        finish_index = start_index+batch_size
        actions_in_all[start_index:finish_index] = batch_data_test['actions_in'].view(batch_size, num_frames, cfg.num_boxes)[:, 0, :]

        for p_idx in range(cfg.num_boxes):
            perturbation_mask = torch.ones(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
            perturbation_mask[:, p_idx] = False
            perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes)
            perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
            batch_data_test['perturbation_mask'] = perturbation_mask
            ret_purtubation = model(batch_data_test)
            gaf_purturbation = ret_purtubation['group_feat']
            gaf_purturbation_all[start_index:finish_index, p_idx] = gaf_purturbation
        
    print('gaf_purturbation_all:', gaf_purturbation_all.shape)

    gaf_purturbation_all = gaf_purturbation_all.detach()
    for epoch in tqdm(range(num_epochs)):
        loss_epoch = []
        out = query_estimator(gaf_purturbation_all)
        query_est = out['query_est']
        people_weight = out['people_weight']
        
        loss_bce = loss_func(query_est, query_labels_all)
        loss_mask_ent = torch.mean(torch.sum(people_weight * torch.log(people_weight), dim=-1))
        loss = loss_bce

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
        # print(f'Query est: {query_est}')
        # print(f'People weight: {people_weight}')

    query_est = query_est.detach().cpu().numpy()
    people_weight = people_weight.detach().cpu().numpy()

    for b_idx in range(len(frame_fine_all)):
        query_est_b = query_est[b_idx]
        query_label_b = query_labels_all[b_idx].item()
        print(f'Query est: {query_est_b}, Query label: {query_label_b}')
        for p_idx in range(cfg.num_boxes):
            action_b_p = int(actions_in_all[b_idx, p_idx].item())
            action_name = ACTIONS[action_b_p]
            print(f'People weight: {people_weight[b_idx, p_idx]}, Action: {action_name}')

    # evaluate the key person estimator on the test set
    with torch.no_grad():
        cfg.batch_size = 1
        params['batch_size'] = cfg.batch_size
        print(f'{len(frames_fine_test)} frames in the test set')
        training_set.set_frames(frames_fine_test)
        data_set = training_set
        data_loader = data.DataLoader(training_set, **params)

        gaf_purturbation_memory_test = torch.zeros(len(data_set), cfg.num_boxes, cfg.num_features_boxes*2, device=device)
        actions_in_all_test = torch.zeros(len(data_set), cfg.num_boxes, device=device)
        query_all_test = torch.zeros(len(data_set), device=device, dtype=torch.long)
        boxes_wo_norm_in_all_test = torch.zeros(len(data_set), cfg.num_boxes, 4, device=device)
        video_id_test = []
        for batch_idx, batch_data_test in enumerate(tqdm(data_loader)):
            for key in batch_data_test.keys():
                if torch.is_tensor(batch_data_test[key]):
                    batch_data_test[key] = batch_data_test[key].to(device=device)

            images_in = batch_data_test['images_in']
            batch_size, num_frames, _, _, _ = images_in.shape
            start_index = batch_idx*cfg.batch_size
            finish_index = start_index+batch_size
            actions_in_all_test[start_index:finish_index] = batch_data_test['actions_in'].view(batch_size, num_frames, cfg.num_boxes)[:, 0, :]
            video_id_test.extend(batch_data_test['video_id'])
            boxes_wo_norm_in_all_test[start_index:finish_index] = batch_data_test['boxes_wo_norm_in'].view(batch_size, num_frames, cfg.num_boxes, 4)[:, num_frames//2, :]

            ret = model(batch_data_test)
            query_all_test[start_index:finish_index] = batch_data_test['user_queries'][:, 0]
            for p_idx in range(cfg.num_boxes):
                perturbation_mask = torch.ones(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
                perturbation_mask[:, p_idx] = False
                perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes)
                perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
                batch_data_test['perturbation_mask'] = perturbation_mask
                ret_purtubation = model(batch_data_test)
                gaf_purturbation = ret_purtubation['group_feat']
                gaf_purturbation_memory_test[start_index:finish_index, p_idx] = gaf_purturbation
            
        out = query_estimator(gaf_purturbation_memory_test)
        query_est = out['query_est'].detach().cpu().numpy()
        people_weight = out['people_weight'].detach().cpu().numpy()

        save_dir = os.path.join('analysis', 'weakly_key_person', cfg.model_exp_name, cfg.query_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        for b_idx in range(len(frames_fine_test)):
            query_est_b = query_est[b_idx]
            query_label_b = query_all_test[b_idx].item()
            print(f'Query est: {query_est_b}, Query label: {query_label_b}')
            for p_idx in range(cfg.num_boxes):
                action_b_p = int(actions_in_all_test[b_idx, p_idx].item())
                action_name = ACTIONS[action_b_p]
                print(f'People weight: {people_weight[b_idx, p_idx]}, Action: {action_name}')

            if cfg.save_image:
                video_id = video_id_test[b_idx]
                # print(f'Video ID: {video_id}')

                if cfg.dataset_symbol == 'vol':
                    vid_id, seq_id, img_id = video_id.split('_')
                    img_path = os.path.join('data_local', 'videos', vid_id, seq_id, f'{seq_id}.jpg')
                elif cfg.dataset_symbol == 'cad':
                    vid_id, seq_id, img_id = video_id.split('_')
                    img_path = os.path.join('data_local', 'collective', f'seq{str(vid_id).zfill(2)}', f'frame{str(seq_id).zfill(4)}.jpg')
                img = cv2.imread(img_path)
                # boxes = batch_data_test['boxes_wo_norm_in'][b_idx, num_frames//2]
                boxes = boxes_wo_norm_in_all_test[b_idx]
                for p_idx in range(cfg.num_boxes):
                    action_index = actions_in_all_test[b_idx, p_idx]
                    action_name = ACTIONS[int(action_index.item())]
                    x1, y1, x2, y2 = map(int, boxes[p_idx])
                    dist_purt_vis_norm = people_weight[b_idx, p_idx]
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255-int(dist_purt_vis_norm*255), int(dist_purt_vis_norm*255)), 2)
                    img = cv2.putText(img, f'{dist_purt_vis_norm:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            save_path = os.path.join(save_dir, f'{video_id}_{query_label_b}.jpg')
            cv2.imwrite(save_path, img)

    return {}


class KeyPersonEstimator(torch.nn.Module):
    def __init__(self, cfg):
        super(KeyPersonEstimator, self).__init__()
        self.cfg = cfg
        self.latent_dim = 32
        self.emb_nn = torch.nn.Sequential(
            torch.nn.Linear(cfg.num_features_boxes*2, self.latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.weight_est_nn = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 1),
        )

        self.query_est_nn = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 2),
        )

    def forward(self, gaf_purturbation_all):
        batch_size, num_boxes, num_features = gaf_purturbation_all.shape
        gaf_emb = self.emb_nn(gaf_purturbation_all)

        gaf_emb_weight = self.weight_est_nn(gaf_emb)
        gaf_emb_weight_sm = torch.nn.functional.softmax(gaf_emb_weight, dim=-2)
        gaf_emb_agg = torch.sum(gaf_emb_weight_sm * gaf_emb, dim=-2)
        query_est = self.query_est_nn(gaf_emb_agg)

        out = {}
        out['query_est'] = query_est
        out['people_weight'] = gaf_emb_weight_sm.view(batch_size, num_boxes)

        return out