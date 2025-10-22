import torch
import itertools
from tqdm import tqdm

from torch.utils import data
from torch.utils.data import WeightedRandomSampler

import collections
import lap
import sys

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

# def sampling_formation_people(inp_dic):
#     print('Computing formation similarity for effient sampling...')
#     frames_query = inp_dic['frames_query']
#     frmaes_training = inp_dic['frames_training']
#     model = inp_dic['models']['model']
#     device = inp_dic['device']
#     cfg = inp_dic['cfg']
#     params = inp_dic['params']
#     training_set = inp_dic['training_set']
#     validation_set = inp_dic['validation_set']

#     validation_set.set_frames(frames_query)
#     query_loader = generate_all_loader(cfg, params, validation_set, 'normal')
#     data_set = validation_set
#     group_features = torch.zeros(len(data_set), cfg.num_features_boxes*2, device=device)
#     boxes_in_query = torch.zeros(len(data_set), cfg.num_frames, cfg.num_boxes, 4, device=device)

#     with torch.no_grad():
#         for query_idx, query_data in enumerate(tqdm(query_loader)):
#             for key in query_data.keys():
#                 if torch.is_tensor(query_data[key]):
#                     query_data[key] = query_data[key].to(device=device)
#             images_in = query_data['images_in']
#             batch_size, num_frames, _, _, _ = images_in.shape
#             ret = model(query_data)
#             start_batch_idx = query_idx*params['batch_size']
#             finish_batch_idx = start_batch_idx+batch_size
#             group_features[start_batch_idx:finish_batch_idx] = ret['group_feat']
#             boxes_in_query[start_batch_idx:finish_batch_idx] = query_data['boxes_in'].view(batch_size, num_frames, cfg.num_boxes, 4)
#     gaf_query_anchor = group_features.detach()
#     boxes_in_query = boxes_in_query.detach()

#     print('Extracting GAF of training videos...')
#     training_set.set_frames(frmaes_training)
#     all_loader = generate_all_loader(cfg, params, training_set, 'normal')
#     data_set = training_set
#     boxes_in_ret = torch.zeros(len(training_set), cfg.num_frames, cfg.num_boxes, 4, device=device)
#     for frame_index in tqdm(range(len(training_set.frames))):
#         sample = training_set.get_bboxs(frame_index)
#         boxes_in_b = sample['boxes_in'].view(1, cfg.num_frames, cfg.num_boxes, 4).to(device=device)
#         boxes_in_ret[frame_index] = boxes_in_b

#     frame_mid = cfg.num_frames // 2
#     boxes_in_ret = boxes_in_ret[:, frame_mid, :, :]
#     boxes_in_query = boxes_in_query[:, frame_mid, :, :]

#     # compute the formation cost matrix
#     formation_cost = torch.zeros(len(training_set), len(validation_set), device=device)
#     for query_idx in tqdm(range(len(validation_set))):
#         p_query_batch = boxes_in_query[query_idx].unsqueeze(0).unsqueeze(2) # Shape: (1, num_boxes, 1, D)
#         diffs = p_query_batch - boxes_in_ret.unsqueeze(1)
#         cost_matrices_batch = torch.norm(diffs, dim=-1)
#         for ret_idx in range(len(training_set)):
#             current_cost_matrix = cost_matrices_batch[ret_idx].cpu().numpy()
#             cost, _, _ = lap.lapjv(current_cost_matrix, extend_cost=True)
#             formation_cost[ret_idx, query_idx] = cost

#     return formation_cost

def sampling_formation_people(inp_dic):
    print('Computing formation similarity for effient sampling...')
    frames_query = inp_dic['frames_query']
    frmaes_training = inp_dic['frames_training']
    model = inp_dic['models']['model']
    device = inp_dic['device']
    cfg = inp_dic['cfg']
    params = inp_dic['params']
    training_set = inp_dic['training_set']
    validation_set = inp_dic['validation_set']

    print('Extracting informataion of training videos...')
    training_set.set_frames(frmaes_training)
    all_loader = generate_all_loader(cfg, params, training_set, 'normal')
    data_set = training_set
    boxes_in_ret = torch.zeros(len(training_set), cfg.num_frames, cfg.num_boxes, 4, device=device)
    for frame_index in tqdm(range(len(training_set.frames))):
        sample = training_set.get_bboxs(frame_index)
        boxes_in_b = sample['boxes_in'].view(1, cfg.num_frames, cfg.num_boxes, 4).to(device=device)
        boxes_in_ret[frame_index] = boxes_in_b
    frame_mid = cfg.num_frames // 2
    boxes_in_ret = boxes_in_ret[:, frame_mid, :, :]

    # compute the formation cost matrix
    formation_cost = torch.zeros(len(training_set), len(training_set), device=device)
    for ret_idx_base in tqdm(range(len(training_set))):
        p_ret_base = boxes_in_ret[ret_idx_base].unsqueeze(2) # Shape: (num_boxes, 1, D)
        for ret_idx_target in range(len(training_set)):
            p_ret_target = boxes_in_ret[ret_idx_target].unsqueeze(1) # Shape: (1, num_boxes, D)
            diffs = p_ret_base - p_ret_target
            cost_matrices_pairs = torch.norm(diffs, dim=-1)
            current_cost_matrix = cost_matrices_pairs.cpu().numpy()
            cost, _, _ = lap.lapjv(current_cost_matrix, extend_cost=True)
            formation_cost[ret_idx_base, ret_idx_target] = cost

    return formation_cost