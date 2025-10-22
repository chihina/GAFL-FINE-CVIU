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

    gaf_path_train = os.path.join(cfg.stage2model_dir, 'eval_gaf_train.npy')
    gaf_arr_train = np.load(gaf_path_train)
    gaf_path_test = os.path.join(cfg.stage2model_dir, 'eval_gaf_test.npy')
    gaf_arr_test = np.load(gaf_path_test)
    gaf_arr = np.concatenate((gaf_arr_train, gaf_arr_test), axis=0)
    gaf_tensor = torch.tensor(gaf_arr, device=device)

    # load the initial gaf and their corresponding video ids
    gaf_path_train = os.path.join(cfg.stage2model_dir, 'eval_gaf_train.npy')
    gaf_arr_train = np.load(gaf_path_train)
    gaf_path_test = os.path.join(cfg.stage2model_dir, 'eval_gaf_test.npy')
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
    cfg.non_query_init = 30
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

    with torch.no_grad():
        # >=== obtain the gaf of the query samples
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
            original_features[start_batch_idx:finish_batch_idx] = ret['original_features'].view(batch_size, cfg.num_frames, cfg.num_boxes, cfg.emb_features*cfg.crop_size[0]*cfg.crop_size[1])            
            query_people_labels_gt[start_batch_idx:finish_batch_idx] = batch_data_test['user_queries_people'][:, 0, :]
            video_id_all.extend(batch_data_test['video_id'])

        use_anchor_type = cfg.use_anchor_type
        print(f'Use anchor type: {use_anchor_type}')
        perturbation_mask = torch.ones(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
        for query_index in range(len(data_set)):
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
        gaf_query_anchor = gaf_purturbation

        # generate pseudo key person labels on the close set in the training set
        training_set.set_frames(frames_fine_close)
        data_set = training_set
        data_loader = data.DataLoader(training_set, **params)

        query_people_labels_gt_close = torch.zeros(len(data_set), cfg.num_boxes, device=device, dtype=torch.long)
        query_people_labels_det_close = torch.zeros(len(data_set), cfg.num_boxes, device=device, dtype=torch.long)
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

            anchor_agg_mode = cfg.anchor_agg_mode
            gaf_original = ret['group_feat']

            dist_purt_pert_list = []
            for p_idx in range(cfg.num_boxes):
                perturbation_mask = torch.ones(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
                perturbation_mask[:, p_idx] = False
                perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes)
                perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
                batch_data_test['perturbation_mask'] = perturbation_mask
                ret_purtubation = model(batch_data_test)
                gaf_purturbation = ret_purtubation['group_feat']
                feat_dot_purt = gaf_purturbation @ gaf_query_anchor.t()
                cos_sim = feat_dot_purt / (gaf_purturbation.norm(dim=-1).view(-1, 1) * gaf_query_anchor.norm(dim=-1).view(1, -1))
                cos_sim = cos_sim
                cos_sim_agg = cos_sim.mean(dim=-1)
                dist_purt_pert_list.append(cos_sim_agg)

            dist_purt_agg = torch.stack(dist_purt_pert_list, dim=1).view(batch_size, cfg.num_boxes)
            dist_threshold = torch.quantile(dist_purt_agg, cfg.use_key_person_ratio)
            query_people_labels_det_batch = torch.where(dist_purt_agg > dist_threshold, 1, 0)
            query_people_labels_det_batch = query_people_labels_det_batch.view(batch_size, cfg.num_boxes)
            query_people_labels_det_close[start_index:finish_index] = query_people_labels_det_batch
            query_people_labels_gt_close[start_index:finish_index] = batch_data_test['user_queries_people'][:, 0, :]

        # generate pseudo key person labels on the farther set in the training set
        query_people_labels_det_farther = torch.zeros(len(data_set), cfg.num_boxes, device=device, dtype=torch.long)

    # set the parameters for the training
    learning_rate = 1e-4
    # num_epochs = 1
    # num_epochs = 100
    num_epochs = 500

    # generate key person estimator
    key_person_estimator = KeyPersonEstimator(cfg).to(device=device)
    key_person_estimator_optim = optim.Adam(key_person_estimator.parameters(), lr=learning_rate)
    key_person_estimator.train()

    # optimize the key person estimator on the close and farther set in the training set
    frame_fine_all = frames_query + frames_fine_farther
    query_people_labels_det_all = torch.cat((query_people_labels_gt, query_people_labels_det_farther), dim=0)
    query_labels_all = torch.cat((torch.ones(len(frames_query)), torch.zeros(len(frames_fine_farther))), dim=0).to(device=device).long()
    print(f'Length of frames fine all: {len(frame_fine_all)}')
    all_set.set_frames(frame_fine_all)
    data_set = all_set
    data_loader = data.DataLoader(all_set, **params)
    gaf_purturbation_memory = torch.zeros(len(data_set), cfg.num_boxes, cfg.num_features_boxes*2, device=device)
    for batch_idx, batch_data_test in enumerate(tqdm(data_loader)):
        for key in batch_data_test.keys():
            if torch.is_tensor(batch_data_test[key]):
                batch_data_test[key] = batch_data_test[key].to(device=device)
        images_in = batch_data_test['images_in']
        batch_size, num_frames, _, _, _ = images_in.shape
        start_index = batch_idx*cfg.batch_size
        finish_index = start_index+batch_size
        for p_idx in range(cfg.num_boxes):
            perturbation_mask = torch.ones(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
            perturbation_mask[:, p_idx] = False
            perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes)
            perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
            batch_data_test['perturbation_mask'] = perturbation_mask
            ret_purtubation = model(batch_data_test)
            gaf_purturbation = ret_purtubation['group_feat']
            gaf_purturbation_memory[start_index:finish_index, p_idx] = gaf_purturbation
            gaf_purturbation = gaf_purturbation_memory[start_index:finish_index, p_idx]

    # loss_func = torch.nn.CrossEntropyLoss()
    class_weight = torch.zeros(2, device=device)
    class_weight[0] = 1 / torch.sum(query_people_labels_det_all == 0).float()
    class_weight[1] = 1 / torch.sum(query_people_labels_det_all == 1).float()
    loss_func = torch.nn.CrossEntropyLoss(weight=class_weight)
    gt_key_person = query_people_labels_det_all

    for epoch in tqdm(range(num_epochs)):
        loss_epoch = []
        estimated_key_person_out = key_person_estimator(gaf_purturbation_memory)
        estimated_key_person_final = estimated_key_person_out['estimated_key_person_final']
        loss_final = loss_func(estimated_key_person_final.view(-1, 2), gt_key_person.view(-1))

        estimated_key_person_final_sm = estimated_key_person_final.softmax(dim=-1)
        estimated_key_person_final_pos = estimated_key_person_final_sm[:, :, 1]
        estimated_key_person_final_pos_seq = torch.sum(estimated_key_person_final_pos, dim=-1)
        gt_key_person_num_avg = gt_key_person.float().sum(dim=-1)
        loss_key_person_num = torch.mean((gt_key_person_num_avg - estimated_key_person_final_pos_seq).abs())
        loss = loss_final
        key_person_estimator_optim.zero_grad()
        loss.backward(retain_graph=True)
        key_person_estimator_optim.step()
        loss_epoch.append(loss.item())
        print(f'Epoch: {epoch}')
        print(f'Loss: {loss.item()}, Loss final: {loss_final.item()}, Loss key person num: {loss_key_person_num.item()}')
    
    # release the gpu memory
    del gaf_purturbation_memory
    torch.cuda.empty_cache()

    # evaluate the key person estimator on the test set
    with torch.no_grad():
        cfg.batch_size = 2
        params['batch_size'] = cfg.batch_size
        print(f'{len(frames_fine_test)} frames in the test set')
        training_set.set_frames(frames_fine_test)
        data_set = training_set
        data_loader = data.DataLoader(training_set, **params)

        gt_key_person_all_test = torch.zeros(len(data_set), cfg.num_boxes, device=device, dtype=torch.long)
        estimated_key_person_prob_all_test = torch.zeros(len(data_set), cfg.num_boxes, 2, device=device)
        query_all_test = torch.zeros(len(data_set))
        gaf_purturbation_memory_test = torch.zeros(len(data_set), cfg.num_boxes, cfg.num_features_boxes*2, device=device)
        for batch_idx, batch_data_test in enumerate(tqdm(data_loader)):
            for key in batch_data_test.keys():
                if torch.is_tensor(batch_data_test[key]):
                    batch_data_test[key] = batch_data_test[key].to(device=device)

            images_in = batch_data_test['images_in']
            batch_size, num_frames, _, _, _ = images_in.shape
            start_index = batch_idx*cfg.batch_size
            finish_index = start_index+batch_size

            ret = model(batch_data_test)
            gt_key_person_all = batch_data_test['user_queries_people'][:, 0, :]
            gt_key_person_all_test[start_index:finish_index] = gt_key_person_all
            query_all_test[start_index:finish_index] = batch_data_test['user_queries'][:, 0]

            for p_idx in range(cfg.num_boxes):
                perturbation_mask = torch.ones(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
                perturbation_mask[:, p_idx] = False
                perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes)
                perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
                batch_data_test['perturbation_mask'] = perturbation_mask
                ret_purtubation = model(batch_data_test)
                gaf_purturbation = ret_purtubation['group_feat']
                # estimated_key_person_prob_all_test_b_p = key_person_estimator(gaf_purturbation)['estimated_key_person_final']
                # estimated_key_person_prob_all_test[start_index:finish_index, p_idx] = estimated_key_person_prob_all_test_b_p
                gaf_purturbation_memory_test[start_index:finish_index, p_idx] = gaf_purturbation
        estimated_key_person_prob_all_test = key_person_estimator(gaf_purturbation_memory_test)['estimated_key_person_final']

        for b_idx in range(len(frames_fine_test)):
            gt_key_person = gt_key_person_all_test[b_idx].cpu().numpy()
            estimated_key_person_prob = estimated_key_person_prob_all_test[b_idx]
            estimated_key_person_prob_sm = estimated_key_person_prob.softmax(dim=-1)
            estimated_key_person_prob = estimated_key_person_prob.cpu().numpy()
            estimated_key_person = np.argmax(estimated_key_person_prob, axis=-1)
            print(f'GT: {gt_key_person}, Estimated: {estimated_key_person}')
            # print(f'Prob: {estimated_key_person_prob_sm[:, 1]}')

        # convert the tensors to numpy arrays
        met_dic = calc_metrics(estimated_key_person_prob_all_test.view(-1, 2).cpu().numpy(), gt_key_person_all_test.view(-1).cpu().numpy())
        print(f'All: {met_dic} ({len(frames_fine_test)})')

        # calculate the metrics for the test set where the positive samples are only included
        positive_indices = torch.nonzero(query_all_test).squeeze()
        gt_key_person_positive = gt_key_person_all_test[positive_indices]
        estimated_key_person_positive = estimated_key_person_prob_all_test[positive_indices]
        met_dic_positive = calc_metrics(estimated_key_person_positive.view(-1, 2).cpu().numpy(), gt_key_person_positive.view(-1).cpu().numpy())
        print(f'positive: {met_dic_positive} ({len(positive_indices)})')

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
        self.key_person_head_mid = torch.nn.Linear(self.latent_dim, 2)

    def forward(self, gaf_purturbation_memory):
        feat_emb = self.emb_nn(gaf_purturbation_memory)
        est_key_person_mid = self.key_person_head_mid(feat_emb)

        out = {}
        out['estimated_key_person_final'] = est_key_person_mid

        return out