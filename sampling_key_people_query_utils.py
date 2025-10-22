import torch
import itertools
from tqdm import tqdm

from torch.utils import data
from torch.utils.data import WeightedRandomSampler

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

def sampling_key_people(inp_dic):
    frames_query = inp_dic['frames_query']
    frmaes_training = inp_dic['frames_training']
    model = inp_dic['models']['model']
    device = inp_dic['device']
    cfg = inp_dic['cfg']
    params = inp_dic['params']
    training_set = inp_dic['training_set']
    validation_set = inp_dic['validation_set']

    # set the model to eval mode
    model.eval()

    # extract the GAF of query videos
    validation_set.set_frames(frames_query)
    query_loader = generate_all_loader(cfg, params, validation_set, 'normal')
    data_set = validation_set
    group_features = torch.zeros(len(data_set), cfg.num_features_boxes*2, device=device)
    print('Extracting GAF of query videos...')
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
    gaf_query_anchor = group_features.detach()

    print('Extracting GAF of training videos...')
    training_set.set_frames(frmaes_training)
    all_loader = generate_all_loader(cfg, params, training_set, 'normal')
    data_set = training_set
    # key_peopleness = torch.zeros(len(data_set), cfg.num_boxes, device=device)
    key_peopleness = torch.zeros(len(training_set), len(validation_set), cfg.num_boxes, device=device)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(all_loader)):
            for key in batch_data.keys():
                if torch.is_tensor(batch_data[key]):
                    batch_data[key] = batch_data[key].to(device=device)

            images_in = batch_data['images_in']
            batch_size, num_frames, _, _, _ = images_in.shape
            video_ids = batch_data['video_id']
            input_data = batch_data

            # forward
            ret = model(input_data)
            start_batch_idx = query_idx*params['batch_size']
            finish_batch_idx = start_batch_idx+batch_size

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

                    key_peopleness[start_batch_idx:finish_batch_idx, :, p_idx] = cos_sim
            else:
                assert False, 'Not implemented the key person detection mode.'
            
            # key_peopleness[start_batch_idx:finish_batch_idx] = cos_sim_people
    
    # normalize the key peopleness in each video ranged from 0 to 1
    key_peopleness = key_peopleness.view(len(training_set)*len(validation_set), cfg.num_boxes)
    key_peopleness = key_peopleness - key_peopleness.min(dim=-1, keepdim=True).values
    key_peopleness = key_peopleness / (key_peopleness.max(dim=-1, keepdim=True).values + 1e-10)

    # compute the entropy of the key peopleness
    key_peopleness_ent = key_peopleness.softmax(dim=-1)
    key_peopleness_ent = -torch.sum(key_peopleness_ent * torch.log(key_peopleness_ent + 1e-10), dim=-1)
    key_peopleness_ent = key_peopleness_ent.view(len(training_set), len(validation_set))

    if cfg.anchor_agg_mode == 'max':
        key_peopleness_ent = key_peopleness_ent.max(dim=-1).values
    elif cfg.anchor_agg_mode == 'mean':
        key_peopleness_ent = key_peopleness_ent.mean(dim=-1)
    
    return key_peopleness_ent