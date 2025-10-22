'''
    Train the model with person action/appearance features.
'''

import torch
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler

import time
import random
import os
import sys
import wandb
from tqdm import tqdm
from collections import OrderedDict
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import itertools
import json
import collections
from sklearn.cluster import KMeans

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
from sampling_key_people_utils import *
from sampling_formation_utils import *
from sampling_pert_utils import *
from sampling_unce_utils import *

torch.autograd.set_detect_anomaly(True)


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
            
def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def generate_all_loader(cfg, params, all_set, loader_type):
    if loader_type == 'weighted':
        query_list = [all_set.get_query(frame) for frame in all_set.get_frames_all()]
        query_list_count = dict(collections.Counter(query_list))
        query_count_inv = {k:1/v for k, v in query_list_count.items()}
        sample_weights = [query_count_inv[query] for query in query_list]
        sampler = WeightedRandomSampler(sample_weights, cfg.update_iter*params['batch_size'], replacement=True)
        all_loader = data.DataLoader(all_set, sampler=sampler, batch_size=params['batch_size'], num_workers=params['num_workers'])
        print(f'Query count: {query_list_count}')
    elif loader_type == 'normal':
        all_loader = data.DataLoader(all_set, **params)

    return all_loader

def train_net(cfg):
    """
    training gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list
    
    # Show config parameters
    cfg.init_config()
    show_config(cfg)

    # Save config parameters
    cfg_save_path = os.path.join(cfg.result_path, 'cfg.pickle')
    with open(cfg_save_path, "wb") as f:
        pickle.dump(cfg, f)
    
    # Reading dataset
    training_set, validation_set, all_set = return_dataset(cfg)
    cfg.num_boxes = all_set.get_num_boxes_max()
        
    params = {
        'batch_size': cfg.batch_size,
        # 'shuffle': False,
        'shuffle': True,
        'num_workers': 4,
        # 'num_workers': 1,
        'pin_memory': False,
        # 'pin_memory': True,
    }
    
    # set parameters of torch to be optimized
    torch.backends.cudnn.benchmark = True

    # Set random seed
    random.seed(cfg.train_random_seed)
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    torch.cuda.manual_seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    cfg.device = device
    
    # Build model and optimizer
    basenet_list = {'volleyball':Basenet_volleyball, 'collective':Basenet_collective}
    gcnnet_list = {'group_relation_volleyball':GroupRelation_volleyball,
                   'group_relation_collective':GroupRelation_volleyball,}
    
    # build main GAFL network
    GCNnet = gcnnet_list[cfg.inference_module_name]
    model = GCNnet(cfg)
    state_dict = torch.load(cfg.stage2model)['state_dict']
    new_state_dict=OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    
    if cfg.load_backbone_stage4:
        model.load_state_dict(new_state_dict, strict=False)
        print_log(cfg.log_path, 'Loading stage2 model for stage4: ' + cfg.stage2model)

    # move models to gpu
    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)
    model=model.to(device=device)

    # set mode of models
    if cfg.set_bn_eval:
        model.apply(set_bn_eval)

    # set parameters to be optimized
    optimizer_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer=optim.Adam(optimizer_params, lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)

    models = {'model': model}
    train_list={'volleyball':train_collective, 'collective':train_collective, 'basketball':train_collective}
    test_list={'volleyball':test_collective, 'collective':test_collective, 'basketball':test_collective}
    train=train_list[cfg.dataset_name]
    test=test_list[cfg.dataset_name]

    if cfg.dataset_name in ['volleyball']:
        gaf_mode = 'scene'
    elif cfg.dataset_name in ['collective']:
        # gaf_mode = 'people'
        gaf_mode = 'scene'
    elif cfg.dataset_name in ['basketball']:
        gaf_mode = 'scene'
    else:
        assert False, 'Not implemented the dataset name.'        

    gaf_path_train = os.path.join(cfg.stage2model_dir, f'eval_gaf_train_{gaf_mode}.npy')
    gaf_arr_train = np.load(gaf_path_train)
    gaf_path_test = os.path.join(cfg.stage2model_dir, f'eval_gaf_test_{gaf_mode}.npy')
    gaf_arr_test = np.load(gaf_path_test)
    gaf_arr = np.concatenate((gaf_arr_train, gaf_arr_test), axis=0)
    gaf_tensor = torch.tensor(gaf_arr, device=device)
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
    queries = torch.tensor(validation_set.get_query_list(), device=device)
    query_indices = torch.nonzero(queries).squeeze()
    randperm_g = torch.Generator()
    randperm_g.manual_seed(cfg.train_random_seed)
    query_indices_rand_perm = torch.randperm(query_indices.shape[0], generator=randperm_g)
    query_indices_use = query_indices[query_indices_rand_perm[:cfg.query_init]]
    frames_validation = validation_set.get_frames_all()
    frames_query = []
    for query_idx in query_indices_use:
        frames_query.append(frames_validation[query_idx])

    # select non-query samples from the training set
    frames_training = training_set.get_frames_all()

    # sampling videos for fine-tuning
    if cfg.non_query_init == 'all':
        cfg.non_query_init = len(frames_training)
    else:
        cfg.non_query_init = int(cfg.non_query_init)

    if cfg.sampling_mode == 'rand':
        non_query_indices_use = torch.randperm(len(frames_training))[:cfg.non_query_init]
    elif cfg.sampling_mode == 'near':
        if cfg.train_smp_type == 'max':
            q_to_nq_cos_sim_stack = []
            for query_idx in query_indices_use:
                gaf_query = gaf_tensor_valid[query_idx].unsqueeze(0)
                cos_sim_query = torch.nn.functional.cosine_similarity(gaf_query, gaf_tensor_train, dim=-1)
                q_to_nq_cos_sim_stack.append(cos_sim_query)
            q_to_nq_cos_sim_stack = torch.stack(q_to_nq_cos_sim_stack)
            q_to_nq_cos_sim, _ = q_to_nq_cos_sim_stack.max(dim=0)
            _, non_query_indices_use = torch.topk(q_to_nq_cos_sim, int(cfg.non_query_init), largest=True)
        elif cfg.train_smp_type == 'each':
            non_query_indices_use = []
            non_query_indices_mask = torch.zeros(len(frames_training), device=device, dtype=torch.bool)
            for query_idx in query_indices_use:
                gaf_query = gaf_tensor_valid[query_idx].unsqueeze(0)
                cos_sim_query = torch.nn.functional.cosine_similarity(gaf_query, gaf_tensor_train, dim=-1)
                cos_sim_query[non_query_indices_mask] = -1
                _, non_query_indices_use_query = torch.topk(cos_sim_query, int(cfg.non_query_init / len(query_indices_use)), largest=True)
                non_query_indices_mask[non_query_indices_use_query] = True
                non_query_indices_use.extend(non_query_indices_use_query)
            non_query_indices_use = torch.tensor(non_query_indices_use, device=device).view(-1)
    else:
        if cfg.train_smp_type == 'all':
            frame_training_sample = frames_training
            non_query_indices_use = torch.arange(len(frame_training_sample), device=device)
        elif 'each' in cfg.train_smp_type:
            each_num = int(cfg.train_smp_type.split('_')[-1])

            non_query_indices_use = []
            non_query_indices_mask = torch.zeros(len(frames_training), device=device, dtype=torch.bool)

            if 'pert' in cfg.train_smp_type:
                pert_coef = [i for i in cfg.train_smp_type.split('_') if 'pert' in i][0].split('pert')[-1]
                pert_coef = float(pert_coef)

                # update GAFs of query videos with perturbation
                inp_dic_pert = {'frames_query': frames_query, 'models': models, 'device': device, 
                                'cfg': cfg, 'params': params, 'validation_set': validation_set}
                query_gaf_valid_pert = sampling_pert(inp_dic_pert)

                pert_trials = query_gaf_valid_pert.shape[1]
                for query_idx in range(query_gaf_valid_pert.shape[0]):
                    # compute the cosine similarity of the query GAFs with the training GAFs
                    gaf_query = gaf_tensor_valid[query_indices_use[query_idx]].unsqueeze(0)
                    cos_sim_query = torch.nn.functional.cosine_similarity(gaf_query, gaf_tensor_train, dim=-1)

                    # compute the uncertainty of the query GAFs
                    cos_sim_query_pert_all = torch.zeros(pert_trials, gaf_tensor_train.shape[0], device=device)
                    for pert_idx in range(pert_trials):
                        gaf_query_pert = query_gaf_valid_pert[query_idx, pert_idx].unsqueeze(0)
                        cos_sim_query_pert = torch.nn.functional.cosine_similarity(gaf_query_pert, gaf_tensor_train, dim=-1)
                        cos_sim_query_pert_all[pert_idx] = cos_sim_query_pert
                    cos_sim_query_pert_std = torch.std(cos_sim_query_pert_all, dim=0)

                    # add the uncertainty to the cosine similarity
                    cos_sim_query = cos_sim_query + pert_coef * cos_sim_query_pert_std
                    _, non_query_indices_use_query = torch.topk(cos_sim_query, int(cfg.non_query_init / len(query_indices_use) * each_num), largest=True)
                    non_query_indices_mask[non_query_indices_use_query] = True
                    non_query_indices_use.extend(non_query_indices_use_query)
            else:
                for query_idx in query_indices_use:
                    gaf_query = gaf_tensor_valid[query_idx].unsqueeze(0)
                    cos_sim_query = torch.nn.functional.cosine_similarity(gaf_query, gaf_tensor_train, dim=-1)
                    cos_sim_query[non_query_indices_mask] = -1
                    _, non_query_indices_use_query = torch.topk(cos_sim_query, int(cfg.non_query_init / len(query_indices_use) * each_num) , largest=True)
                    non_query_indices_mask[non_query_indices_use_query] = True
                    non_query_indices_use.extend(non_query_indices_use_query)
                
            non_query_indices_use = torch.tensor(non_query_indices_use, device=device).view(-1)
            frame_training_sample = []
            for non_query_idx in non_query_indices_use:
                frame_training_sample.append(frames_training[non_query_idx])
        
        inp_dic = {'frames_query': frames_query, 'frames_training': frame_training_sample,
                'models': models, 'device': device, 'cfg': cfg,
                'params': params, 'training_set': training_set, 'validation_set': validation_set}

        if cfg.sampling_mode in ['unce']:
            key_peopleness_ent = sampling_unce(inp_dic)
            selected_indices = torch.topk(key_peopleness_ent, cfg.non_query_init, largest=True)[1]
        elif cfg.sampling_mode in ['earan']:
            # randomly select non-query samples
            selected_indices = torch.randperm(len(non_query_indices_use))[:cfg.non_query_init]
        # elif 'kmeans' in cfg.sampling_mode:
        elif ('kmeans' in cfg.sampling_mode) or ('coreset' in cfg.sampling_mode):
            # extract decision features with GAF
            if 'gaf' in cfg.sampling_mode:
                gaf_tensor_train_list = []
                for non_query_idx in non_query_indices_use:
                    gaf_tensor_train_list.append(gaf_tensor_train[non_query_idx].unsqueeze(0))
                gaf_tensor_train_sampled = torch.cat(gaf_tensor_train_list, dim=0)
                decision_features_gaf = gaf_tensor_train_sampled.cpu().numpy()

            # extract decision features with formation
            if 'form' in cfg.sampling_mode:
                formation_cost = sampling_formation_people(inp_dic)
                decision_features_form = formation_cost.cpu().numpy()

            # normalize the decision features
            if 'gaf_innorm' in cfg.sampling_mode:
                decision_features_gaf = decision_features_gaf / np.linalg.norm(decision_features_gaf, axis=1, keepdims=True)
            if 'form_innorm' in cfg.sampling_mode:
                decision_features_form = decision_features_form / np.linalg.norm(decision_features_form, axis=1, keepdims=True)

            if ('gaf' in cfg.sampling_mode) and ('form' in cfg.sampling_mode):
                decision_features = np.concatenate((decision_features_gaf, decision_features_form), axis=1)
            elif 'gaf' in cfg.sampling_mode:
                decision_features = decision_features_gaf
            elif 'form' in cfg.sampling_mode:
                decision_features = decision_features_form

            if 'kmeans' in cfg.sampling_mode:
                # apply k-means clustering
                kmeans = KMeans(n_clusters=cfg.non_query_init, random_state=cfg.train_random_seed)
                kmeans.fit(decision_features)

                # pick up the one sample which is the close to the cluster
                cluster_idxs = kmeans.labels_
                centers = kmeans.cluster_centers_[cluster_idxs]
                dist = (decision_features - centers)**2
                dist = dist.sum(axis=1)
                selected_indices = torch.tensor(np.array([np.arange(len(non_query_indices_use))[cluster_idxs==i][dist[cluster_idxs==i].argmin()] for i in range(cfg.non_query_init)]), device=device)
            
            elif 'coreset' in cfg.sampling_mode:
                # randomly select one sample
                init_sample_index = random.randint(0, len(non_query_indices_use)-1)
                selected_indices = np.array([init_sample_index], dtype=np.int64)

                for i in range(cfg.non_query_init-1):
                    # compute distance matrix
                    if ('gaf' in cfg.sampling_mode):
                        decision_features_gaf_selected = decision_features_gaf[selected_indices]
                        dist_matrix_gaf = np.linalg.norm(decision_features_gaf[:, np.newaxis] - decision_features_gaf_selected, axis=2)
                    if ('form' in cfg.sampling_mode): 
                        decision_features_form_selected = decision_features_form[selected_indices]
                        dist_matrix_form = np.linalg.norm(decision_features_form[:, np.newaxis] - decision_features_form_selected, axis=2)
                    if ('gaf' in cfg.sampling_mode) and ('form' in cfg.sampling_mode):
                        decision_features_gaf_form_selected = np.concatenate((decision_features_gaf_selected, decision_features_form_selected), axis=1)
                        decision_features_gaf_form = np.concatenate((decision_features_gaf, decision_features_form), axis=1)
                        dist_matrix_gaf_form = np.linalg.norm(decision_features_gaf_form[:, np.newaxis] - decision_features_gaf_form_selected, axis=2)

                    if ('gaf' in cfg.sampling_mode) and ('form' in cfg.sampling_mode):
                        coef_form = float(cfg.sampling_mode.split('_')[-1])
                        dist_matrix = coef_form * dist_matrix_gaf + (1 - coef_form) * dist_matrix_form
                    elif ('gaf' in cfg.sampling_mode):
                        dist_matrix = dist_matrix_gaf
                    elif ('form' in cfg.sampling_mode):
                        dist_matrix = dist_matrix_form

                    # compute the pairwise distance
                    dist_min = np.min(dist_matrix, axis=1)

                    # select the sample with the maximum distance
                    dist_min_argmax = np.argmax(dist_min)
                    selected_indices = np.append(selected_indices, dist_min_argmax)

                # convert to tensor
                selected_indices = torch.tensor(selected_indices, device=device)

        non_query_indices_use_update = []
        for non_query_idx in selected_indices:
            non_query_indices_use_update.append(non_query_indices_use[non_query_idx])
        non_query_indices_use = torch.tensor(non_query_indices_use_update, device=device).view(-1)

    # compuate the target GA ratio in the selected samples
    training_set.set_frames(frames_training)
    queries_fine = torch.tensor(training_set.get_query_list(), device=device)
    non_query_flag = []
    for non_query_idx in non_query_indices_use:
        non_query_flag.append(queries_fine[non_query_idx])
    non_query_flag = torch.tensor(non_query_flag, device=device, dtype=torch.float)
    non_query_flag_mean = non_query_flag.mean()
    print(f'Target GA ratio in {len(non_query_indices_use)} samples: {non_query_flag_mean.item()}')

    sampling_json_path = os.path.join(cfg.result_path, 'sampling.json')
    sampling_info_dic = {}
    sampling_info_dic['query_indices'] = query_indices_use.tolist()
    sampling_info_dic['non_query_indices_num'] = len(non_query_indices_use)
    sampling_info_dic['non_query_indices'] = non_query_indices_use.tolist()
    sampling_info_dic['non_query_indices_mean'] = non_query_flag_mean.item()
    with open(sampling_json_path, 'w') as f:
        json.dump(sampling_info_dic, f, indent=4)

    frames_active = []
    for non_query_idx in non_query_indices_use:
        frames_active.append(frames_training[non_query_idx])

    training_set.set_frames(frames_active)
    print(f'{len(frames_active)} samples are used in our fine-tuning.')

    validation_set.set_frames(frames_query)
    print(f'{len(frames_query)} samples are used as query samples.')

    # initialize detection of key person
    print(f'Initialize the tensor to store the key people labeling result.')
    gt_key_person_all = torch.zeros(len(training_set), cfg.num_frames, cfg.num_boxes, device=device, dtype=torch.float)

    # Training iteration
    best_result = {'epoch':0, 'activities_acc':0, 'actions_acc':0, 'loss':100000000000000}

    batch_size_original = cfg.batch_size
    for epoch in range(cfg.max_epoch):
    # for epoch in range(1, cfg.max_epoch+1):
        if epoch % cfg.key_person_det_interval == 0:
            cfg.batch_size = 4
            params['batch_size'] = cfg.batch_size
        else:
            params['batch_size'] = batch_size_original

        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])
        
        if (cfg.pruning_mode == 'dynamic') and (((epoch+1) % cfg.pruning_interval) == 0):
            all_pruning_ratio = cfg.all_pruning_ratio + cfg.pruning_decay
            if all_pruning_ratio <= cfg.pruning_ratio_max:
                cfg.all_pruning_ratio = all_pruning_ratio
                cfg.pruning_ratio = cfg.all_pruning_ratio
                cfg.use_pruning_ratio = cfg.all_pruning_ratio
                print(f'Pruning ratio is updated to {cfg.pruning_ratio} at epoch #{epoch}')

        inp_dic = {'frames_active': frames_active, 'frames_query': frames_query, 'frames_training': frames_training,
                   'models': models, 'device': device, 'optimizer': optimizer, 'epoch': epoch, 'cfg': cfg,
                   'params': params, 'training_set': training_set, 'validation_set': validation_set,
                   'gt_key_person_all': gt_key_person_all,
                   'gaf_tensor_train': gaf_tensor_train, 'gaf_tensor_valid': gaf_tensor_valid,
                   'vid_id_arr_train': vid_id_arr_train, 'vid_id_arr_test': vid_id_arr_test}
        all_info = train(inp_dic)
        for wandb_loss_name in cfg.wandb_loss_list:
            wandb.log({f"All {wandb_loss_name}": all_info[wandb_loss_name]}, step=epoch)

        if all_info['loss']<best_result['loss'] and (epoch % cfg.save_epoch_interval == 0):
            best_result=all_info
            state = {
                # 'epoch': epoch,
                'state_dict': model.state_dict(),
                # 'optimizer': optimizer.state_dict(),
            }
            torch.save(state, cfg.result_path+'/best_model.pth')
            print('Best model saved.')
        print_log(cfg.log_path, 'Best loss: %.2f%% at epoch #%d.'%(best_result['loss'], best_result['epoch']))

        # update the key person detection result
        inp_dic['gt_key_person_all'] = all_info['gt_key_person_all']

    return cfg

def train_collective(inp_dic):
    frames_active = inp_dic['frames_active']
    frames_query = inp_dic['frames_query']
    models = inp_dic['models']
    device = inp_dic['device']
    optimizer = inp_dic['optimizer']
    epoch = inp_dic['epoch']
    cfg = inp_dic['cfg']
    params = inp_dic['params']
    training_set = inp_dic['training_set']
    validation_set = inp_dic['validation_set']
    gt_key_person_all = inp_dic['gt_key_person_all']

    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    loss_act_meter=AverageMeter()
    loss_recon_meter=AverageMeter()
    loss_jae_meter=AverageMeter()
    loss_query_meter=AverageMeter()
    loss_proximity_meter=AverageMeter()
    loss_distant_meter=AverageMeter()
    loss_distill_meter=AverageMeter()
    loss_recon_key_meter=AverageMeter()
    loss_recon_non_key_meter=AverageMeter()
    loss_key_cos_sim_meter=AverageMeter()
    loss_key_recog_meter=AverageMeter()
    loss_ga_recog_meter=AverageMeter()
    loss_ia_recog_meter=AverageMeter()
    loss_maintain_meter=AverageMeter()
    loss_metric_lr_meter=AverageMeter()
    loss_recon_pose_feat_meter=AverageMeter()
    loss_recon_pose_coord_meter=AverageMeter()
    activities_conf = ConfusionMeter(cfg.num_activities)
    actions_conf = ConfusionMeter(cfg.num_actions)
    epoch_timer=Timer()
    model = models['model']

    # generate group features of query samples
    validation_set.set_frames(frames_query)
    query_loader = generate_all_loader(cfg, params, validation_set, 'normal')

    data_set = validation_set
    if cfg.feature_adapt_type in ['line', 'mlp']:
        individual_features = torch.zeros(len(data_set), cfg.num_boxes, cfg.num_features_boxes*2, device=device)
        group_features = torch.zeros(len(data_set), cfg.feature_adapt_dim, device=device)
    else:
        individual_features = torch.zeros(len(data_set), cfg.num_boxes, cfg.num_features_boxes*2, device=device)
        group_features = torch.zeros(len(data_set), cfg.num_features_boxes*2, device=device)
    
    with torch.no_grad():
        for query_idx, query_data in enumerate(tqdm(query_loader)):
            for key in query_data.keys():
                if torch.is_tensor(query_data[key]):
                    query_data[key] = query_data[key].to(device=device)

            images_in = query_data['images_in']
            batch_size, num_frames, _, _, _ = images_in.shape
            ret = model(query_data)
            start_batch_idx = query_idx*params['batch_size']
            finish_batch_idx = start_batch_idx+batch_size
            group_features[start_batch_idx:finish_batch_idx] = ret['group_feat']
            individual_features[start_batch_idx:finish_batch_idx] = ret['individual_feat'].view(batch_size, cfg.num_boxes, -1)

    # skip the key person detection if the epoch is not the key person detection interval
    if epoch % cfg.key_person_det_interval == 0:
        data_set = validation_set
        if cfg.feature_adapt_type in ['line', 'mlp']:
            individual_features = torch.zeros(len(data_set), cfg.num_boxes, cfg.num_features_boxes*2, device=device)
            group_features_prune = torch.zeros(len(data_set), cfg.feature_adapt_dim, device=device)
        else:
            individual_features = torch.zeros(len(data_set), cfg.num_boxes, cfg.num_features_boxes*2, device=device)
            group_features_prune = torch.zeros(len(data_set), cfg.num_features_boxes*2, device=device)

        # off the random perturbation
        model.module.eval_only = True

        # obtain the group activity features for the query samples
        validation_set.set_frames(frames_query)
        query_loader = generate_all_loader(cfg, params, validation_set, 'normal')
        data_set = validation_set

        actions_in_all = torch.zeros(len(data_set), cfg.num_boxes, device=device)
        query_people_labels_gt = torch.zeros(len(data_set), cfg.num_boxes, device=device, dtype=torch.long)
        for query_idx, query_data in enumerate(tqdm(query_loader)):
            for key in query_data.keys():
                if torch.is_tensor(query_data[key]):
                    query_data[key] = query_data[key].to(device=device)
            images_in = query_data['images_in']
            batch_size, num_frames, _, _, _ = images_in.shape
            actions_in = query_data['actions_in'].reshape((batch_size,num_frames,cfg.num_boxes))
            actions_in = actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
            ret = model(query_data)

            if cfg.use_anchor_type == 'pruning_gt':
                perturbation_mask = torch.ones(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
                for query_idx_b in range(batch_size):
                    query_people_labels_gt_batch = query_data['user_queries_people'][query_idx_b, 0, :]
                    query_people_labels_gt_batch = query_people_labels_gt_batch == 1
                    if torch.sum(query_people_labels_gt_batch) != 0:
                        perturbation_mask[query_idx_b, query_people_labels_gt_batch] = False
                    else:
                        perturbation_mask[query_idx_b] = False
                perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes)
                perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
                query_data['perturbation_mask'] = perturbation_mask

            # forward
            ret_prune = model(query_data)

            # save features
            start_batch_idx = query_idx*params['batch_size']
            finish_batch_idx = start_batch_idx+batch_size
            individual_features[start_batch_idx:finish_batch_idx] = ret_prune['individual_feat'].view(batch_size, cfg.num_boxes, -1)
            group_features_prune[start_batch_idx:finish_batch_idx] = ret_prune['group_feat']
            actions_in_all[start_batch_idx:finish_batch_idx] = actions_in.view(batch_size, cfg.num_boxes)
            query_people_labels_gt[start_batch_idx:finish_batch_idx] = query_data['user_queries_people'][:, 0, :]

        if cfg.use_anchor_type == 'normal':
            gaf_query_anchor = group_features_prune
        elif cfg.use_anchor_type == 'pruning_gt':
            gaf_query_anchor = group_features_prune
        else:
            # define the action set for the dataset
            if cfg.dataset_name == 'volleyball':
                ACTIONS = ['blocking', 'digging', 'falling', 'jumping', 'moving', 'setting', 'spiking', 'standing', 'waiting']
            elif cfg.dataset_name == 'collective':
                ACTIONS = ['NA','Moving','Waiting','Queueing','Talking']
            elif cfg.dataset_name == 'basketball':
                ACTIONS = ['NA']
            else:
                assert False, 'Not implemented the dataset name.'
            vote_history = detect_key_people(cfg, device, data_set, batch_size, model, query_loader, group_features, actions_in_all, ACTIONS, params)

            # generate key people mask in the query samples
            if cfg.anchor_thresh_type == 'ratio':
                pseudo_key_people_mask_query = torch.ones(len(data_set), cfg.num_boxes, device=device, dtype=torch.bool)
                for base_index in range(len(data_set)):
                    vote_history_base = vote_history[base_index]
                    vote_history_base_sorted = torch.argsort(vote_history_base)
                    vote_history_base_sorted = vote_history_base_sorted[:int(cfg.num_boxes * cfg.use_pruning_ratio)]
                    for person_index in vote_history_base_sorted:
                        pseudo_key_people_mask_query[base_index, person_index] = False
            elif cfg.anchor_thresh_type == 'val':
                pseudo_key_people_mask_query = vote_history < cfg.use_pruning_ratio
            else:
                pass

            # extract the group activity features for the query samples
            gaf_query_anchor = torch.zeros(len(data_set), cfg.feature_adapt_dim, device=device)
            for query_idx, query_data in enumerate(tqdm(query_loader)):
                for key in query_data.keys():
                    if torch.is_tensor(query_data[key]):
                        query_data[key] = query_data[key].to(device=device)
                images_in = query_data['images_in']
                batch_size, num_frames, _, _, _ = images_in.shape
                start_batch_idx = query_idx*params['batch_size']
                finish_batch_idx = start_batch_idx+batch_size

                # perturbation_mask = pseudo_key_people_mask_query.view(batch_size, 1, cfg.num_boxes).expand(batch_size, cfg.num_frames, cfg.num_boxes)
                perturbation_mask = pseudo_key_people_mask_query[start_batch_idx:finish_batch_idx]
                perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes).expand(batch_size, cfg.num_frames, cfg.num_boxes)
                query_data['perturbation_mask'] = perturbation_mask
                ret_prune = model(query_data)

                # gaf_query_anchor = ret_prune['group_feat']
                gaf_query_anchor[start_batch_idx:finish_batch_idx] = ret_prune['group_feat']
        
        # detach the tensor for the following loss functions
        gaf_query_anchor = gaf_query_anchor.detach()
        
        # on the random perturbation
        model.module.eval_only = False

    # obtain the group activity features for all samples
    time_start = time.time()
    training_set.set_frames(frames_active)
    all_loader = generate_all_loader(cfg, params, training_set, 'normal')
    data_set = training_set
    model.train()

    for batch_idx, batch_data in enumerate(tqdm(all_loader)):
        if batch_idx % 100 == 0 and batch_idx > 0:
            print('Training in processing {}/{}, Loss: {:.4f}'.format(batch_idx, len(all_loader), loss_meter.avg))

        if cfg.set_bn_eval:
            model.apply(set_bn_eval)
    
        # prepare batch data
        for key in batch_data.keys():
            if torch.is_tensor(batch_data[key]):
                batch_data[key] = batch_data[key].to(device=device)

        images_in = batch_data['images_in']
        batch_size, num_frames, _, _, _ = images_in.shape
        actions_in = batch_data['actions_in'].reshape((batch_size, num_frames, cfg.num_boxes))
        activities_in = batch_data['activities_in'].reshape((batch_size, num_frames))
        activities_in = activities_in[:, 0].reshape((batch_size,))
        query_labels = batch_data['user_queries'][:, 0]
        query_people_labels = batch_data['user_queries_people'][:, 0]
        bboxes_num = batch_data['bboxes_num'].reshape(batch_size, num_frames)
        video_ids = batch_data['video_id']
        input_data = batch_data

        # forward
        ret = model(input_data)

        # define list for various losses
        loss_list = []

        actions_in_nopad=[]
        for b in range(batch_size):
            N = bboxes_num[b][0]
            actions_in_nopad.append(actions_in[b][0][:N])
        actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)
        
        if cfg.use_key_recog_loss or cfg.recon_loss_type == 'key' or cfg.use_disentangle_loss:
            if cfg.use_key_person_type == 'gt':
                gt_key_person = batch_data['user_queries_people']
                gt_key_person = gt_key_person.view(-1)
            elif cfg.use_key_person_type in ['det', 'det_semi']:
                if epoch % cfg.key_person_det_interval == 0:
                    # off the random perturbation
                    model.module.eval_only = True

                    gaf_original = ret['group_feat']
                    if cfg.key_person_mode in ['mask_zero', 'mask_one', 'mask_zero_diff', 'mask_one_diff']:
                        cos_sim_persons = []
                        for p_idx in range(cfg.num_boxes):
                            if cfg.key_person_mode in ['mask_zero', 'mask_zero_diff']:
                                perturbation_mask = torch.zeros(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
                                perturbation_mask[:, p_idx] = True
                            elif cfg.key_person_mode in ['mask_one', 'mask_one_diff']:
                                perturbation_mask = torch.ones(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
                                perturbation_mask[:, p_idx] = False
                            perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes)
                            perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
                            batch_data['perturbation_mask'] = perturbation_mask
                            ret_purtubation = model(batch_data)
                            gaf_purturbation = ret_purtubation['group_feat']
                            feat_dot_purt = gaf_purturbation @ gaf_query_anchor.t()
                            cos_sim_purt = feat_dot_purt / (gaf_purturbation.norm(dim=-1).view(-1, 1) * gaf_query_anchor.norm(dim=-1).view(1, -1))
                            feat_dot_orig = gaf_original @ gaf_query_anchor.t()
                            cos_sim_orig = feat_dot_orig / (gaf_original.norm(dim=-1).view(-1, 1) * gaf_query_anchor.norm(dim=-1).view(1, -1))

                            if cfg.key_person_mode == 'mask_one':
                                cos_sim = cos_sim_purt
                            elif cfg.key_person_mode == 'mask_one_diff':
                                cos_sim = cos_sim_purt - cos_sim_orig
                            elif cfg.key_person_mode == 'mask_zero':
                                cos_sim = (1 - cos_sim_purt)
                            elif cfg.key_person_mode == 'mask_zero_diff':
                                cos_sim = cos_sim_orig - cos_sim_purt
                            
                            if cfg.anchor_agg_mode == 'max':
                                cos_sim_agg = cos_sim.max(dim=1).values
                            elif cfg.anchor_agg_mode == 'mean':
                                cos_sim_agg = cos_sim.mean(dim=1)
                            cos_sim_persons.append(cos_sim_agg)
                        cos_sim_people = torch.stack(cos_sim_persons, dim=1)
                    elif cfg.key_person_mode == 'ind_feat':
                        individual_features = ret['individual_feat'].view(batch_size, cfg.num_boxes, -1)
                        feat_dot = individual_features @ gaf_query_anchor.t()
                        cos_sim = feat_dot / (individual_features.norm(dim=-1).view(batch_size, cfg.num_boxes, 1) * gaf_query_anchor.norm(dim=-1).view(1, -1))
                        cos_sim_people = cos_sim.mean(dim=-1)
                    else:
                        assert False, 'Not implemented the key person detection mode.'

                    if cfg.use_key_person_loss_func in ['bce', 'wbce']:
                        # generate binary labels in which (cfg.use_key_person_ratio*cfg.num_boxes) key people are selected
                        query_people_labels_det = torch.zeros(batch_size, cfg.num_boxes, device=device)
                        for b in range(batch_size):
                            N = bboxes_num[b][0]
                            cos_sim_people_b = cos_sim_people[b][:N]
                            use_people_num = int(cfg.use_key_person_ratio * cfg.num_boxes)
                            use_people_num = min(use_people_num, N)
                            _, indices = torch.topk(cos_sim_people_b, use_people_num)
                            query_people_labels_det[b][indices] = 1
                    elif cfg.use_key_person_loss_func == 'mse':
                        query_people_labels_det = cos_sim_people.view(batch_size, cfg.num_boxes)

                    query_people_labels_det = query_people_labels_det.contiguous().view(batch_size, 1, cfg.num_boxes)
                    query_people_labels_det = query_people_labels_det.expand(batch_size, cfg.num_frames, cfg.num_boxes)
                    query_people_labels_det = query_people_labels_det.contiguous().view(-1)

                    if cfg.use_key_person_type == 'det':
                        # gt_key_person = query_people_labels_det.detach()
                        gt_key_person = query_people_labels_det
                    elif cfg.use_key_person_type == 'det_semi':
                        gt_key_person = batch_data['user_queries_people']
                        det_key_person = query_people_labels_det.view(batch_size, cfg.num_frames, cfg.num_boxes)
                        gt_key_person[query_labels == 1] = det_key_person[query_labels == 1]
                        gt_key_person = gt_key_person.view(-1).detach()
                    
                    # save the key person detection result
                    frame_indices = [frames_active.index((sid, src_fid)) for sid, src_fid in zip(batch_data['frame'][0], batch_data['frame'][1])]
                    gt_key_person_all[frame_indices] = gt_key_person.view(batch_size, cfg.num_frames, cfg.num_boxes)
                    
                    # on the random perturbation
                    model.module.eval_only = False
                else:
                    # skip the key person detection if the epoch is not the key person detection interval
                    frame_indices = [frames_active.index((sid, src_fid)) for sid, src_fid in zip(batch_data['frame'][0], batch_data['frame'][1])]
                    gt_key_person = gt_key_person_all[frame_indices].view(-1)

        if cfg.use_key_recog_loss:
            # get the recognized key person
            recog_key_person = ret['recog_key_person']
            recog_key_person = recog_key_person.view(-1, recog_key_person.shape[-1])

            # remove padding parts
            recog_key_person = recog_key_person.view(batch_size, cfg.num_frames, cfg.num_boxes, 2)
            recog_key_person_nopad = []
            gt_key_person_recog = gt_key_person.view(batch_size, cfg.num_frames, cfg.num_boxes)
            gt_key_person_nopad = []
            N_sum = 0
            for b in range(batch_size):
                N = bboxes_num[b][0]
                N_sum += N
                recog_key_person_nopad.append(recog_key_person[b][:, :N].reshape(-1, recog_key_person.shape[-1]))
                gt_key_person_nopad.append(gt_key_person_recog[b][:, :N].reshape(-1,))
            recog_key_person = torch.cat(recog_key_person_nopad, dim=0).reshape(-1, recog_key_person.shape[-1])
            gt_key_person_recog = torch.cat(gt_key_person_nopad, dim=0).reshape(-1,)

            # calculate the key person recognition loss
            if cfg.use_key_person_loss_func == 'bce':
                recog_key_person_loss = F.cross_entropy(recog_key_person, gt_key_person.long())
            elif cfg.use_key_person_loss_func == 'wbce':
                gt_key_person_one = torch.sum(gt_key_person_recog)
                gt_key_person_zero = torch.sum(gt_key_person_recog == 0)
                weight = torch.tensor([1/gt_key_person_zero, 1/gt_key_person_one], device=device)
                recog_key_person_loss = F.cross_entropy(recog_key_person, gt_key_person_recog.long(), weight=weight)
            elif cfg.use_key_person_loss_func == 'mse':
                recog_key_person = recog_key_person[:, 1]
                recog_key_person_loss = F.mse_loss(recog_key_person, gt_key_person_recog.float())

            recog_key_person_loss = recog_key_person_loss * 1e1

            loss_list.append(recog_key_person_loss)
            loss_key_recog_meter.update(recog_key_person_loss, batch_size)
        
        if cfg.use_query_classfication_loss:
            query_labels = batch_data['user_queries'][:, 0]
            recog_query = ret['recog_query']
            recog_query = recog_query.view(-1, recog_query.shape[-1])
            query_loss = F.cross_entropy(recog_query, query_labels.long())
            loss_list.append(query_loss)
            loss_query_meter.update(query_loss, batch_size)

        # if cfg.use_disentangle_loss:
        #     gt_key_person = gt_key_person.view(batch_size, cfg.num_frames, cfg.num_boxes)
        #     perturbation_mask = ~gt_key_person.bool()
        #     batch_data['perturbation_mask'] = perturbation_mask
        #     ret_disentangle = model(batch_data)
        #     group_feat_disent = ret_disentangle['group_feat']
        #     group_feat_fine = ret['group_feat']
        #     query_labels = batch_data['user_queries'][:, 0]
        #     group_feat_disent_pos = group_feat_disent[query_labels == 1]
        #     group_feat_pos = group_feat_fine[query_labels == 1]
        #     loss_disentangle = F.mse_loss(group_feat_disent_pos.detach(), group_feat_pos)
        #     loss_list.append(loss_disentangle)
        #     loss_distill_meter.update(loss_disentangle, batch_size)

        # if cfg.use_disentangle_loss:
        #     group_feat_fine = ret['group_feat']
        #     gt_key_person = gt_key_person.view(batch_size, cfg.num_frames, cfg.num_boxes)
        #     perturbation_mask = ~gt_key_person.bool()

        #     batch_data['perturbation_mask'] = perturbation_mask
        #     ret_dil = model(batch_data)
        #     group_feat_dil = ret_dil['group_feat']
        #     loss_dil = F.mse_loss(group_feat_dil.detach(), group_feat_fine)
        #     loss_list.append(loss_dil)
        #     loss_distill_meter.update(loss_dil, batch_size)

        # if cfg.use_disentangle_loss:
        #     gt_key_person = gt_key_person.view(batch_size, cfg.num_frames, cfg.num_boxes)
        #     gt_key_person_user = batch_data['user_queries_people']
        #     loss_dil = F.binary_cross_entropy_with_logits(gt_key_person, gt_key_person_user)
        #     loss_list.append(loss_dil)
        #     loss_distill_meter.update(loss_dil, batch_size)

        if cfg.use_metric_lr_loss:
            query_labels = batch_data['user_queries'][:, 0]
            group_feat_fine = ret['group_feat']
            group_feat_fine_pos = group_feat_fine[query_labels == 1]
            group_feat_fine_neg = group_feat_fine[query_labels == 0]
            
            if cfg.use_disentangle_loss:
                gt_key_person = gt_key_person.view(batch_size, cfg.num_frames, cfg.num_boxes)
                perturbation_mask = ~gt_key_person.bool()
                batch_data['perturbation_mask'] = perturbation_mask
                ret_dil = model(batch_data)
                group_feat_dil = ret_dil['group_feat']
                group_feat_fine_pos = group_feat_dil[query_labels == 1]
                group_features = gaf_query_anchor

                # group_feat_dil_pos = group_feat_dil[query_labels == 1]
                # group_feat_dil_neg = group_feat_dil[query_labels == 0]
                # loss_dil_pos = F.mse_loss(group_feat_dil_pos.detach(), group_feat_fine_pos)
                # loss_dil_neg = F.mse_loss(group_feat_dil_neg.detach(), group_feat_fine_neg)
                # loss_dil = (loss_dil_pos+loss_dil_neg)/2
                # loss_list.append(loss_dil)
                # loss_distill_meter.update(loss_dil, batch_size)

            # sample features
            anchor_num = group_features.shape[0]
            positive_num = group_feat_fine_pos.shape[0]
            negative_num = group_feat_fine_neg.shape[0]

            if positive_num != 0 and negative_num != 0:
                anc_to_pos_dist = torch.cdist(group_features.detach(), group_feat_fine_pos, p=2)
                anc_to_neg_dist = torch.cdist(group_features.detach(), group_feat_fine_neg, p=2)
                triplet_loss = anc_to_pos_dist.unsqueeze(2) - anc_to_neg_dist.unsqueeze(1) + cfg.metric_lr_margin
                triplet_loss = torch.clamp(triplet_loss, min=0)
                loss_metric_lr = triplet_loss.mean()
                loss_list.append(loss_metric_lr)
                loss_metric_lr_meter.update(loss_metric_lr, anchor_num*positive_num*negative_num)
            else:
                # add a dummy loss to avoid the backword error
                print(f'anchor_num: {anchor_num}, positive_num: {positive_num}, negative_num: {negative_num}')

        if cfg.use_ga_recog_loss:
            recognized_ga = ret['recognized_ga']
            recognized_ga_labels = torch.argmax(recognized_ga, dim=1)
            ga_recog_loss = F.cross_entropy(recognized_ga, activities_in)
            loss_list.append(ga_recog_loss)
            loss_ga_recog_meter.update(ga_recog_loss, batch_size)
        
        if cfg.use_ia_recog_loss:
            recognized_ia = ret['recognized_ia']
            recognized_ia = recognized_ia[:, 0, :, :]
            recognized_ia_no_pad = []
            for b in range(batch_size):
                N = bboxes_num[b][0]
                recognized_ia_no_pad.append(recognized_ia[b, :N, :].reshape(-1, recognized_ia.shape[-1]))
            recognized_ia = torch.cat(recognized_ia_no_pad, dim=0)
            ia_recog_loss = F.cross_entropy(recognized_ia, actions_in)
            loss_list.append(ia_recog_loss)
            loss_ia_recog_meter.update(ia_recog_loss, batch_size)

        if cfg.use_maintain_loss:
            group_feat_ori = inp_dic['gaf_tensor_train']
            frames_training = inp_dic['frames_training']
            frame_indices = [frames_training.index((sid, src_fid)) for sid, src_fid in zip(batch_data['frame'][0], batch_data['frame'][1])]
            group_feat_ori_filter = group_feat_ori[frame_indices]
            
            if cfg.dataset_name in ['volleyball', 'basketball']:
                group_feat = ret['group_feat']
            elif cfg.dataset_name in ['collective']:
                group_feat = ret['group_feat']
            else:
                assert False, 'Not implemented the dataset name.'
            
            loss_maintain = F.mse_loss(group_feat, group_feat_ori_filter)
            loss_list.append(loss_maintain)
            loss_maintain_meter.update(loss_maintain, batch_size)

        if 'actions' in list(ret.keys()):
            actions_scores = ret['actions'].reshape(batch_size, cfg.num_boxes, -1)
            actions_scores_nopad=[]
            for b in range(batch_size):
                N = bboxes_num[b][0]
                actions_scores_nopad.append(actions_scores[b][:N])
            actions_scores=torch.cat(actions_scores_nopad,dim=0).reshape(-1, cfg.num_actions)
            actions_loss=F.cross_entropy(actions_scores,actions_in)
            loss_list.append(actions_loss)
            actions_labels=torch.argmax(actions_scores,dim=1)
            actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
            actions_accuracy = actions_correct.item() / actions_scores.shape[0]
            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            actions_conf.add(actions_labels, actions_in)

        if 'recon_features' in list(ret.keys()):
            recon_features = ret['recon_features']
            original_features = ret['original_features']
            
            recon_features_nopad=[]
            original_features_nopad=[]
            gt_key_person_nopad = []
            for b in range(batch_size):
                N = bboxes_num[b][0]
                recon_features_b = recon_features[b][:, :N, :].reshape(cfg.num_frames*N, -1)
                original_features_b = original_features[b][:, :N, :].reshape(cfg.num_frames*N, -1)
                recon_features_nopad.append(recon_features_b)
                original_features_nopad.append(original_features_b)
                # if cfg.use_disentangle_loss:
                    # gt_key_person = gt_key_person.view(batch_size, cfg.num_frames, cfg.num_boxes)
                    # gt_key_person_b = gt_key_person[b][:, :N].reshape(cfg.num_frames*N)
                    # gt_key_person_nopad.append(gt_key_person_b)

            recon_features_nopad=torch.cat(recon_features_nopad,dim=0)
            original_features_nopad=torch.cat(original_features_nopad,dim=0)
            # if cfg.use_disentangle_loss:
                # gt_key_person_nopad = torch.cat(gt_key_person_nopad, dim=0)
                # soft_weight = gt_key_person_nopad
                # recon_loss = F.mse_loss(recon_features_nopad, original_features_nopad, reduction='none')
                # recon_loss = recon_loss.mean(dim=1)
                # recon_loss = recon_loss * soft_weight
                # recon_loss = recon_loss.mean(dim=0)
            # else:
                # recon_loss = F.mse_loss(recon_features_nopad, original_features_nopad)

            recon_loss = F.mse_loss(recon_features_nopad, original_features_nopad)
            loss_list.append(recon_loss)
            loss_recon_meter.update(recon_loss.item(), batch_size)

        # Total loss
        total_loss = sum(loss_list)
        loss_meter.update(total_loss, batch_size)

        # print(f'===> [{epoch}] [{batch_idx}/{len(data_loader)}]: {total_loss.item():.4f}')

        # if total_loss is int type, continue
        if isinstance(total_loss, int):
            continue
        else:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # if batch_idx > 10:
            # break

    train_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'loss_act':loss_act_meter.avg,
        'loss_recon':loss_recon_meter.avg,
        'loss_recon_pose_feat':loss_recon_pose_feat_meter.avg,
        'loss_recon_pose_coord':loss_recon_pose_coord_meter.avg,
        'loss_jae':loss_jae_meter.avg,
        'loss_query':loss_query_meter.avg,
        'loss_proximity':loss_proximity_meter.avg,
        'loss_distant':loss_distant_meter.avg,
        'loss_distill':loss_distill_meter.avg,
        'loss_recon_key':loss_recon_key_meter.avg,
        'loss_recon_non_key':loss_recon_non_key_meter.avg,
        'loss_key_cos_sim':loss_key_cos_sim_meter.avg,
        'loss_key_recog':loss_key_recog_meter.avg,
        'loss_ga_recog':loss_ga_recog_meter.avg,
        'loss_ia_recog':loss_ia_recog_meter.avg,
        'loss_maintain':loss_maintain_meter.avg,
        'loss_metric_lr':loss_metric_lr_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'activities_conf':activities_conf.value(),
        'activities_MPCA':MPCA(activities_conf.value()),
        'actions_acc':actions_meter.avg*100,
        'actions_conf':actions_conf.value(),
        'actions_MPCA':MPCA(actions_conf.value()),
        'gt_key_person_all':gt_key_person_all,
    }  

    return train_info
        
# def test_volleyball(data_loader, models, device, epoch, cfg, group_features):
#     epoch_timer=Timer()
#     model = models['model']
#     model.eval()

#     with torch.no_grad():
#         for batch_idx, batch_data_test in enumerate(tqdm(data_loader)):
#             for key in batch_data_test.keys():
#                 if torch.is_tensor(batch_data_test[key]):
#                     batch_data_test[key] = batch_data_test[key].to(device=device)

#             # forward
#             batch_size, num_frames, _, _, _ = batch_data_test['images_in'].shape
#             actions_in = batch_data_test['actions_in'].reshape((batch_size, num_frames, cfg.num_boxes))
#             activities_in = batch_data_test['activities_in'].reshape((batch_size, num_frames))
#             actions_in = actions_in[:, 0, :].reshape((batch_size * cfg.num_boxes,))
#             activities_in = activities_in[:, 0].reshape((batch_size,))
#             input_data = batch_data_test
#             ret= model(input_data)

#             group_feat = ret['group_feat']
#             start_batch_idx = batch_idx*cfg.batch_size
#             finish_batch_idx = start_batch_idx+batch_size
#             group_features[start_batch_idx:finish_batch_idx] = group_feat

#             # if batch_idx > 1:
#                 # break
    
#     test_info={
#         'time':epoch_timer.timeit(),
#         'epoch':epoch,
#         'group_features': group_features,
#     }
    
#     return test_info

def test_collective(data_loader, models, device, epoch, cfg, group_features):
    epoch_timer=Timer()
    model = models['model']
    model.eval()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader)):
            for key in batch_data.keys():
                if torch.is_tensor(batch_data[key]):
                    batch_data[key] = batch_data[key].to(device=device)

            batch_size, num_frames, _, _, _ = batch_data['images_in'].shape
            input_data = batch_data
            ret= model(input_data)
            group_feat = ret['group_feat']
            start_batch_idx = batch_idx*cfg.batch_size
            finish_batch_idx = start_batch_idx+batch_size
            group_features[start_batch_idx:finish_batch_idx] = group_feat

            # if batch_idx > 1:
                # break

    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'group_features': group_features,
    }

    return test_info