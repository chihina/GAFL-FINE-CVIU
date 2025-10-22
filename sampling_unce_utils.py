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

def sampling_unce(inp_dic):
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

    print('Computing uncertainty of GAF')
    training_set.set_frames(frmaes_training)
    all_loader = generate_all_loader(cfg, params, training_set, 'normal')
    data_set = training_set
    
    uncertainty = torch.zeros(len(training_set), device=device)
    
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
            start_batch_idx = batch_idx*params['batch_size']
            finish_batch_idx = start_batch_idx+batch_size

            # off the random perturbation
            model.module.eval_only = True
            gaf_original = ret['group_feat']

            # generate the random people mask based on the given masking ratio
            masking_ratio = 0.3
            num_boxes = cfg.num_boxes
            num_masked_boxes = int(num_boxes * masking_ratio)
            masking_trials = 3
            gaf_trial = torch.zeros(batch_size, masking_trials, gaf_original.shape[-1], device=device)

            for trial_idx in range(masking_trials):
                # randomly sample the people to mask
                mask_p_indices = torch.randperm(num_boxes)[:num_masked_boxes]
                perturbation_mask = torch.zeros(batch_size, cfg.num_boxes, device=device, dtype=torch.bool)
                perturbation_mask[:, mask_p_indices] = True
                perturbation_mask = perturbation_mask.view(batch_size, 1, cfg.num_boxes)
                perturbation_mask = perturbation_mask.expand(batch_size, cfg.num_frames, cfg.num_boxes)
                batch_data['perturbation_mask'] = perturbation_mask
                ret_purtubation = model(batch_data)
                gaf_purturbation = ret_purtubation['group_feat']
                gaf_trial[:, trial_idx, :] = gaf_purturbation
            gaf_trial_variance = gaf_trial.var(dim=1, unbiased=False)
            uncertainty[start_batch_idx:finish_batch_idx] = gaf_trial_variance.mean(dim=1)
    
    return uncertainty