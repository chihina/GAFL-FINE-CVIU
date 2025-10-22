import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('gpu', help='gpu_number')
args = parser.parse_args()
device_list = str(args.gpu)

sys.path.append(".")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import pickle
import json

from detect_key_person_all import *
from config_utils import update_config_all

# model_exp_name = '[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>'
# model_exp_name = '[GR ours recon feat random mask 0_stage2]<2023-10-18_22-30-29>'
# model_exp_name = '[GR ours recon feat random mask 6_stage2]<2023-10-25_22-26-38>'
model_exp_name = '[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>'

stage4model = f'result/{model_exp_name}/best_model.pth'
cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
with open(cfg_pickle_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.mode = 'PAF'

cfg.net_det_dir = os.path.join('data_local', 'volleyball_net_detection')
cfg.line_det_dir = os.path.join('data_local', 'volleyball_line_detection')

# all_pruning_ratio = 0.2
# all_pruning_ratio = 0.3
# all_pruning_ratio = 0.4
# all_pruning_ratio = 0.5
# all_pruning_ratio = 0.6
all_pruning_ratio = 0.7
cfg.use_pruning_ratio = all_pruning_ratio
cfg.use_key_person_ratio = all_pruning_ratio

cfg.use_individual_action_type = 'gt_action'
cfg.model_exp_name = model_exp_name
cfg.stage4model = stage4model
cfg.stage2model_dir = os.path.join('result', model_exp_name)
cfg = update_config_all(cfg)
cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.image_size = 320, 640
cfg.eval_only = True
cfg.eval_stage = 4
cfg.batch_size = 16
cfg.test_batch_size = cfg.batch_size
cfg.num_frames = 10
cfg.eval_mask_num = 0
cfg.eval_mask_action = False
cfg.dataset_symbol = 'vol'
cfg.test_random_seed = 3
cfg.save_image = False
cfg.sampling_mode = 'near'

cfg.query_init = 3
cfg.query_sample_num = cfg.query_init
cfg.non_query_init = 20

use_anchor_type_list = [
                        # 'normal', 
                        # 'pruning_gt',
                        # 'pruning_p2p',
                        # 'pruning_p2g',
                        'pruning_p2g_cos',
                        # 'pruning_p2g_inner_cos',
                        ]

# cfg.anchor_thresh_type = 'ratio'
cfg.anchor_thresh_type = 'val'

key_person_mode_list = [
                        'mask_one',
                        # 'mask_one_farther',
                        # 'mask_one_appearance',
                        # 'mask_one_negative'
                        # 'mask_one_diff',
                        # 'mask_one_reduction',
                        # 'mask_zero',
                        # 'mask_zero_diff',
                        # 'mask_position', 'mask_appearance', 'mask_appearance_dist',
                        #  'gt',
                        #  'ind_feat'
                        # 'learnable'
                         ]

# anchor_agg_mode_list = ['max', 'mean']
anchor_agg_mode_list = ['mean']
# anchor_agg_mode_list = ['max']

# query_type_list = ['l_spike', 'l_set', 'l_pass', 'l_winpoint', 'r_spike', 'r_set', 'r_pass', 'r_winpoint']
# query_type_list = ['l_spike', 'l_set', 'l_pass', 'l_winpoint']
# query_type_list = ['r_spike', 'r_set', 'r_pass', 'r_winpoint']
# query_type_list = ['l_spike', 'r_spike']
# query_type_list = ['l_set', 'r_set']
# query_type_list = ['l_pass', 'r_pass']
query_type_list = ['l_winpoint', 'r_winpoint']

save_json_dir = os.path.join('analysis', 'key_person_detection_all', model_exp_name)
if not os.path.exists(save_json_dir):
    os.makedirs(save_json_dir)

p = itertools.product(query_type_list, anchor_agg_mode_list, use_anchor_type_list, key_person_mode_list)
trial_cnt = 0
seed_trial = 3
total_trial_cnt = len(query_type_list) * len(anchor_agg_mode_list) * len(use_anchor_type_list) * len(key_person_mode_list)
for query_type, anchor_agg_mode, anchor_type, key_person_mode in p:
    trial_cnt += 1
    print(f'Trial {trial_cnt}/{total_trial_cnt}')
    cfg.query_type = query_type
    cfg.anchor_agg_mode = anchor_agg_mode
    cfg.use_anchor_type = anchor_type
    cfg.key_person_mode = key_person_mode

    ret_dic_all = {}
    for test_random_seed in range(seed_trial):
        cfg.test_random_seed = test_random_seed
        ret_dic = detect_key_person(cfg)
        ret_dic_all[test_random_seed] = ret_dic

    save_dic = {}        
    for key in ret_dic_all[0].keys():
        print(key)
        save_dic[key] = {}
    
        # obtain average results
        ret_dic_avg = {}
        use_metrics = ['auc', 'ap']
        for use_metric in use_metrics:
            ret_dic_avg[use_metric] = 0
            for ret_dic_child in ret_dic_all.values():
                # ret_dic_avg[use_metric] += ret_dic_child[use_metric]
                ret_dic_avg[use_metric] += ret_dic_child[key][use_metric]
            ret_dic_avg[use_metric] /= len(ret_dic_all)
            save_dic[key][use_metric] = ret_dic_avg[use_metric]

    # save_json_dir_query_type = os.path.join(save_json_dir, query_type)
    save_json_dir_query_type = os.path.join(save_json_dir, cfg.query_type, f'query_{cfg.query_sample_num}', 
                                            cfg.use_anchor_type, cfg.anchor_thresh_type, cfg.key_person_mode, cfg.anchor_agg_mode)

    if not os.path.exists(save_json_dir_query_type):
        os.makedirs(save_json_dir_query_type)
    save_json_name = f'{anchor_agg_mode}_{anchor_type}_{key_person_mode}.json'
    save_json_path = os.path.join(save_json_dir_query_type, save_json_name)
    with open(save_json_path, 'w') as f:
        json.dump(save_dic, f, indent=4)
    print(save_dic)
