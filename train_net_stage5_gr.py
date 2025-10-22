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

    # calculate the cosine similarity between query and non-query samples
    q_to_nq_cos_sim_stack = []
    for query_idx in query_indices_use:
        gaf_query = gaf_tensor_valid[query_idx].unsqueeze(0)
        cos_sim_query = torch.nn.functional.cosine_similarity(gaf_query, gaf_tensor_train, dim=-1)
        q_to_nq_cos_sim_stack.append(cos_sim_query)
    q_to_nq_cos_sim_stack = torch.stack(q_to_nq_cos_sim_stack)
    q_to_nq_cos_sim, _ = q_to_nq_cos_sim_stack.max(dim=0)

    if cfg.non_query_init == 'all':
        cfg.non_query_init = len(frames_training)
    else:
        cfg.non_query_init = int(cfg.non_query_init)

    if cfg.sampling_mode == 'rand':
        non_query_indices_use = torch.randperm(len(frames_training))[:cfg.non_query_init]
    elif cfg.sampling_mode == 'near':
        _, non_query_indices_use = torch.topk(q_to_nq_cos_sim, int(cfg.non_query_init), largest=True)
    elif 'conf' in cfg.sampling_mode:
        # obtain indices of samples with high cosine similarity greater than the threshold
        conf_thresh = float(cfg.sampling_mode.split('_')[1])
        non_query_indices_use = torch.nonzero(q_to_nq_cos_sim > conf_thresh).squeeze()
    elif 'clustering' in cfg.sampling_mode:
        print(f'Clustering mode is used for sampling.')
        cluster_num = int(cfg.non_query_init)
        gaf_arr_train_np = gaf_tensor_train.cpu().numpy()
        kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(gaf_arr_train_np)
        cluster_centers = kmeans.cluster_centers_
        non_query_indices_use = []
        for cluster_idx in range(cluster_num):
            cluster_center = cluster_centers[cluster_idx]
            cluster_center_tensor = torch.tensor(cluster_center, device=device)
            cos_sim_cluster = torch.nn.functional.cosine_similarity(cluster_center_tensor, gaf_tensor_train, dim=-1)
            _, non_query_indices_use_cluster = torch.topk(cos_sim_cluster, int(cfg.non_query_init//cluster_num), largest=True)
            non_query_indices_use.append(non_query_indices_use_cluster)
        non_query_indices_use = torch.cat(non_query_indices_use)
    else:
        assert False, 'Not implemented the sampling mode.'

    # compuate the target GA ratio in the selected samples
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
    print(f'Initialize the tensor to store the key person detection result.')
    gt_key_person_all = torch.zeros(len(training_set), cfg.num_frames, cfg.num_boxes, device=device, dtype=torch.float)

    # Training iteration
    best_result = {'epoch':0, 'activities_acc':0, 'actions_acc':0, 'loss':100000000000000}

    for epoch in range(cfg.max_epoch):
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
    gaf_query = torch.zeros(len(data_set), cfg.num_features_boxes*2, device=device)
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
            gaf_query[start_batch_idx:finish_batch_idx] = ret['group_feat']

    # obtain the group activity features for all samples
    training_set.set_frames(frames_active)
    all_loader = generate_all_loader(cfg, params, training_set, 'normal')
    data_set = training_set
    model.train()

    if cfg.use_key_person_type in ['det', 'det_semi']:
        group_features = torch.zeros(len(data_set), cfg.feature_adapt_dim, device=device)
        actions_in = torch.zeros(len(data_set), cfg.num_boxes, device=device)
        vote_history_all = detect_key_people(cfg, device, data_set, params['batch_size'], model, all_loader, group_features, actions_in, ACTIONS, params)

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
        start_batch_idx = batch_idx*params['batch_size']
        finish_batch_idx = start_batch_idx+batch_size

        # forward
        input_data = batch_data
        ret = model(input_data)
        group_features = ret['group_feat']

        # define list for various losses
        loss_list = []

        actions_in_nopad=[]
        for b in range(batch_size):
            N = bboxes_num[b][0]
            actions_in_nopad.append(actions_in[b][0][:N])
        actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)

        pseudo_key_people_mask_query = torch.ones(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
        vote_history = vote_history_all[start_batch_idx:finish_batch_idx]
        for base_index in range(batch_size):
            vote_history_base = vote_history[base_index]
            vote_history_base_sorted = torch.argsort(vote_history_base)
            vote_history_base_sorted = vote_history_base_sorted[:int(cfg.num_boxes * cfg.use_pruning_ratio)]
            for person_index in vote_history_base_sorted:
                pseudo_key_people_mask_query[base_index, person_index] = False
        
        key_mask = pseudo_key_people_mask_query.view(batch_size, 1, cfg.num_boxes)
        key_mask = key_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
        batch_data['perturbation_mask'] = key_mask
        ret_key = model(batch_data)
        gaf_key = ret_key['group_feat']

        if cfg.use_metric_lr_loss:
            query_labels = batch_data['user_queries'][:, 0]
            positive_num = torch.sum(query_labels == 1).item()
            negative_num = torch.sum(query_labels == 0).item()

            metric_lr_loss_list = []
            if positive_num != 0:
                gaf_pos = group_features[query_labels == 1]
                gaf_pos_key = gaf_key[query_labels == 1].detach()
                gaf_pos_key_inner_dist = F.pairwise_distance(gaf_pos, gaf_pos_key, p=2)
                loss_key_pos_inner_dist = torch.mean(gaf_pos_key_inner_dist)
                metric_lr_loss_list.append(loss_key_pos_inner_dist)

            if negative_num != 0:
                gaf_neg = group_features[query_labels == 0]
                gaf_neg_key = gaf_key[query_labels == 0].detach()
                gaf_neg_key_inner_dist = F.pairwise_distance(gaf_neg, gaf_neg_key, p=2)
                loss_key_neg_inner_dist = torch.mean(gaf_neg_key_inner_dist)
                metric_lr_loss_list.append(loss_key_neg_inner_dist)

            if positive_num != 0 and negative_num != 0:
                anc_to_pos_dist = torch.cdist(gaf_query.detach(), gaf_pos, p=2)
                anc_to_neg_dist = torch.cdist(gaf_query.detach(), gaf_neg, p=2)
                triplet_loss = anc_to_pos_dist.unsqueeze(2) - anc_to_neg_dist.unsqueeze(1) + cfg.metric_lr_margin
                triplet_loss = torch.clamp(triplet_loss, min=0)
                loss_metric_lr = triplet_loss.mean()
                metric_lr_loss_list.append(loss_metric_lr)
            
            if len(metric_lr_loss_list) == 0:
                loss_metric_lr = torch.tensor(0.0, device=device)
            else:
                loss_metric_lr = torch.mean(torch.stack(metric_lr_loss_list))

            loss_list.append(loss_metric_lr)

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
                if cfg.use_disentangle_loss:
                    gt_key_person = gt_key_person.view(batch_size, cfg.num_frames, cfg.num_boxes)
                    gt_key_person_b = gt_key_person[b][:, :N].reshape(cfg.num_frames*N)
                    gt_key_person_nopad.append(gt_key_person_b)

            recon_features_nopad=torch.cat(recon_features_nopad,dim=0)
            original_features_nopad=torch.cat(original_features_nopad,dim=0)
            if cfg.use_disentangle_loss:
                # generate weight
                gt_key_person_nopad = torch.cat(gt_key_person_nopad, dim=0)
                soft_weight = gt_key_person_nopad
                
                # weighted recon loss
                recon_loss = F.mse_loss(recon_features_nopad, original_features_nopad, reduction='none')
                recon_loss = recon_loss.mean(dim=1)
                recon_loss = recon_loss * soft_weight
                recon_loss = recon_loss.mean(dim=0)
            else:
                recon_loss = F.mse_loss(recon_features_nopad, original_features_nopad)

            loss_list.append(recon_loss)
            loss_recon_meter.update(recon_loss.item(), batch_size)

        # Total loss
        total_loss = sum(loss_list)
        loss_meter.update(total_loss.item(), batch_size)

        # print(f'===> [{epoch}] [{batch_idx}/{len(data_loader)}]: {total_loss.item():.4f}')

        # Optim
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