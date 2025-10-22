import torch
import itertools
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# def detect_key_people(cfg, device, data_set, batch_size, model, 
                    # query_loader, group_features, actions_in_all, ACTIONS, params):
    
    # vote_history = torch.zeros(len(data_set), cfg.num_boxes, device=device)
    # data_patterns = itertools.permutations(range(len(data_set)), 2)
    # perturbation_features = torch.zeros(len(data_set), cfg.num_boxes, cfg.num_features_boxes*2, device=device)
    # for query_idx, query_data in enumerate(query_loader):
    #     for key in query_data.keys():
    #         if torch.is_tensor(query_data[key]):
    #             query_data[key] = query_data[key].to(device=device)

    #     images_in = query_data['images_in']
    #     batch_size, num_frames, _, _, _ = images_in.shape
    #     actions_in = query_data['actions_in'].reshape((batch_size,num_frames,cfg.num_boxes))
    #     actions_in = actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))

    #     ret = model(query_data)
    #     start_batch_idx = query_idx*params['batch_size']
    #     finish_batch_idx = start_batch_idx+batch_size

        # if cfg.use_anchor_type in ['pruning_p2p', 'pruning_p2g']:
        #     for person_index in range(cfg.num_boxes):
        #         perturbation_mask = torch.ones(cfg.num_boxes, device=device, dtype=torch.bool)
        #         perturbation_mask[person_index] = False
        #         perturbation_mask = perturbation_mask.view(1, 1, cfg.num_boxes)
        #         perturbation_mask = perturbation_mask.expand(batch_size, num_frames, cfg.num_boxes)
        #         query_data['perturbation_mask'] = perturbation_mask
        #         ret_purtubation = model(query_data)
        #         perturbation_features[start_batch_idx:finish_batch_idx, person_index, :] = ret_purtubation['group_feat']

    # if cfg.use_anchor_type in ['pruning_p2p']:
    #     for base_index, target_index in data_patterns:
    #         cos_sim_matrix = torch.zeros(cfg.num_boxes, cfg.num_boxes, device=device)
    #         perturbation_features_base = perturbation_features[base_index]
    #         perturbation_features_target = perturbation_features[target_index]
    #         cos_sim_matrix = torch.nn.functional.cosine_similarity(perturbation_features_base.unsqueeze(1), 
    #                                                                 perturbation_features_target.unsqueeze(0), dim=-1)
    #         row_ind, col_ind = linear_sum_assignment(-cos_sim_matrix.cpu().numpy())
    #         for row_idx, col_idx in zip(row_ind, col_ind):
    #             cos_sim_best = cos_sim_matrix[row_idx, col_idx]
    #             vote_history[base_index, row_idx] += cos_sim_matrix[row_idx, col_idx]
    #             action_index_base = int(actions_in_all[base_index, row_idx].item())
    #             action_index_target = int(actions_in_all[target_index, col_idx].item())
    #             action_name_base = ACTIONS[action_index_base]
    #             action_name_target = ACTIONS[action_index_target]
    #             # print(f'Person {row_idx}: {cos_sim_best.item()} ({action_name_base}) ({action_name_target})')
    #     vote_history = vote_history / (len(data_set)-1)
    # elif cfg.use_anchor_type in ['pruning_p2g']:
    #     for base_index, target_index in data_patterns:
    #         for base_person_index in range(cfg.num_boxes):
    #             perturbation_features_base = perturbation_features[base_index, base_person_index]
    #             target_features = group_features[target_index]
    #             cos_sim = torch.nn.functional.cosine_similarity(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), dim=-1)
    #             # print(f'{base_index}, {target_index}, {base_person_index}: {cos_sim.item()}')
    #             vote_history[base_index, base_person_index] += cos_sim.item()
    #     vote_history = vote_history / (len(data_set)-1)

def detect_key_people(cfg, device, data_set, batch_size, model, 
                    query_loader, group_features, actions_in_all, ACTIONS, params):
    
    vote_history_all = torch.zeros(len(data_set), cfg.num_boxes, device=device)

    for query_idx, query_data in enumerate(query_loader):
        for key in query_data.keys():
            if torch.is_tensor(query_data[key]):
                query_data[key] = query_data[key].to(device=device)

        images_in = query_data['images_in']
        batch_size, num_frames, _, _, _ = images_in.shape
        actions_in = query_data['actions_in'].reshape((batch_size,num_frames,cfg.num_boxes))
        actions_in = actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))

        ret = model(query_data)
        start_batch_idx = query_idx*params['batch_size']
        finish_batch_idx = start_batch_idx+batch_size
        group_features = ret['group_feat']

        vote_history = torch.zeros(batch_size, cfg.num_boxes, device=device)
        data_patterns = itertools.permutations(range(batch_size), 2)
        one_mask_anchor_type_list = ['pruning_p2p', 'pruning_p2g_cos', 'pruning_p2g_euc', 'pruning_p2g_inner_cos', 'pruning_p2g_inner_euc']
        zero_mask_anchor_type_list = ['pruning_p2g_cos_inv', 'pruning_p2g_euc_inv', 'pruning_p2g_inner_cos_inv', 'pruning_p2g_inner_euc_inv']

        perturbation_features = torch.zeros(batch_size, cfg.num_boxes, cfg.num_features_boxes*2, device=device)
        for person_index in range(cfg.num_boxes):
            if cfg.use_anchor_type in one_mask_anchor_type_list:
                perturbation_mask = torch.ones(cfg.num_boxes, device=device, dtype=torch.bool)
                perturbation_mask[person_index] = False
            else:
                perturbation_mask = torch.zeros(cfg.num_boxes, device=device, dtype=torch.bool)
                perturbation_mask[person_index] = True
            perturbation_mask = perturbation_mask.view(1, 1, cfg.num_boxes)
            perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
            query_data['perturbation_mask'] = perturbation_mask
            ret_purtubation = model(query_data)
            perturbation_features[:, person_index, :] = ret_purtubation['group_feat']

        if cfg.use_anchor_type in ['pruning_p2p']:
        #     for base_index, target_index in data_patterns:
        #         cos_sim_matrix = torch.zeros(cfg.num_boxes, cfg.num_boxes, device=device)
        #         perturbation_features_base = perturbation_features[base_index]
        #         perturbation_features_target = perturbation_features[target_index]
        #         cos_sim_matrix = torch.nn.functional.cosine_similarity(perturbation_features_base.unsqueeze(1), 
        #                                                                 perturbation_features_target.unsqueeze(0), dim=-1)
        #         row_ind, col_ind = linear_sum_assignment(-cos_sim_matrix.cpu().numpy())
                # print(f'Base index: {base_index}, Target index: {target_index}')
                # for row_idx, col_idx in zip(row_ind, col_ind):
                #     cos_sim_best = cos_sim_matrix[row_idx, col_idx]
                #     vote_history[base_index, row_idx] += cos_sim_matrix[row_idx, col_idx]
                #     action_index_base = int(actions_in[base_index, row_idx].item())
                #     action_index_target = int(actions_in[target_index, col_idx].item())
                #     action_name_base = ACTIONS[action_index_base]
                #     action_name_target = ACTIONS[action_index_target]
                    # print(f'Person {row_idx}: {cos_sim_best.item()} ({action_name_base}) ({action_name_target})')
            vote_history = vote_history / (len(data_set)-1)

        elif cfg.use_anchor_type in ['pruning_p2g_cos', 'pruning_p2g_euc', 'pruning_p2g_cos_inv', 'pruning_p2g_euc_inv']:
            for base_index, target_index in data_patterns:
                for base_person_index in range(cfg.num_boxes):
                    perturbation_features_base = perturbation_features[base_index, base_person_index]
                    target_features = group_features[target_index]
                    if cfg.use_anchor_type == 'pruning_p2g_cos':
                        kp_score = torch.nn.functional.cosine_similarity(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), dim=-1)
                    elif cfg.use_anchor_type == 'pruning_p2g_euc':
                        kp_score = torch.nn.functional.pairwise_distance(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), p=2) * -1
                    elif cfg.use_anchor_type == 'pruning_p2g_cos_inv':
                        kp_score = torch.nn.functional.cosine_similarity(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), dim=-1) * -1
                    elif cfg.use_anchor_type == 'pruning_p2g_euc_inv':
                        kp_score = torch.nn.functional.pairwise_distance(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), p=2)
                    vote_history[base_index, base_person_index] += kp_score.item()
            vote_history = vote_history / (len(data_set)-1)

        elif cfg.use_anchor_type in ['pruning_p2g_inner_cos', 'pruning_p2g_inner_euc', 'pruning_p2g_inner_cos_inv', 'pruning_p2g_inner_euc_inv']:
            for base_index in range(batch_size):
                for base_person_index in range(cfg.num_boxes):
                    perturbation_features_base = perturbation_features[base_index, base_person_index]
                    target_features = group_features[base_index]
                    if cfg.use_anchor_type == 'pruning_p2g_inner_cos':
                        kp_score = torch.nn.functional.cosine_similarity(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), dim=-1)
                    elif cfg.use_anchor_type == 'pruning_p2g_inner_euc':
                        kp_score = torch.nn.functional.pairwise_distance(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), p=2) * -1
                    elif cfg.use_anchor_type == 'pruning_p2g_inner_cos_inv':
                        kp_score = torch.nn.functional.cosine_similarity(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), dim=-1) * -1
                    elif cfg.use_anchor_type == 'pruning_p2g_inner_euc_inv':
                        kp_score = torch.nn.functional.pairwise_distance(perturbation_features_base.unsqueeze(0), target_features.unsqueeze(0), p=2)
                    vote_history[base_index, base_person_index] += kp_score.item()
        else:
            assert False, f'Unknown anchor type: {cfg.use_anchor_type}'

        vote_history_all[start_batch_idx:finish_batch_idx] = vote_history
    
    return vote_history_all