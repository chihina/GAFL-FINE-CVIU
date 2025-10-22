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

def sampling_pert(inp_dic):
    frames_query = inp_dic['frames_query']
    model = inp_dic['models']['model']
    device = inp_dic['device']
    cfg = inp_dic['cfg']
    params = inp_dic['params']
    validation_set = inp_dic['validation_set']

    # set the model to eval mode
    model.eval()

    # extract the GAF of query videos
    validation_set.set_frames(frames_query)
    query_loader = generate_all_loader(cfg, params, validation_set, 'normal')
    data_set = validation_set

    # generate the random people mask based on the given masking ratio
    num_boxes = cfg.num_boxes
    num_masked_boxes_list = [i for i in cfg.train_smp_type.split('_') if 'mk' in i][0].split('mk')[-1]
    num_masked_boxes = int(num_masked_boxes_list)

    masking_trials = 5
    group_features = torch.zeros(len(data_set), masking_trials, cfg.num_features_boxes*2, device=device)
    print('Extracting GAF of query videos...')
    with torch.no_grad():
        for query_idx, query_data in enumerate(tqdm(query_loader)):
            for key in query_data.keys():
                if torch.is_tensor(query_data[key]):
                    query_data[key] = query_data[key].to(device=device)
            images_in = query_data['images_in']
            batch_size, num_frames, _, _, _ = images_in.shape
            start_batch_idx = query_idx*params['batch_size']
            finish_batch_idx = start_batch_idx+batch_size
            # bboxes_num = query_data['images_in']

            model.module.eval_only = True
            for trial_idx in range(masking_trials):
                # randomly sample the people to mask
                mask_p_indices = torch.randperm(num_boxes)[:num_masked_boxes]
                perturbation_mask = torch.zeros(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
                perturbation_mask[:, mask_p_indices] = True
                perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes)
                perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
                query_data['perturbation_mask'] = perturbation_mask
                ret_purtubation = model(query_data)
                gaf_purturbation = ret_purtubation['group_feat']
                group_features[start_batch_idx:finish_batch_idx, trial_idx, :] = gaf_purturbation.detach()

    return group_features