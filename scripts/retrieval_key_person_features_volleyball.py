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

from retrieval_key_person_features import detect_key_person
from config_utils import update_config_all

# model_exp_name = '[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>'
# model_exp_name = '[VOL_GAFL_PAF_RP_NET_mask 6 1024_stage2]<2025-01-17_11-41-43>'
# model_exp_name = '[GR ours recon feat random mask 0_stage2]<2023-10-18_22-30-29>'
model_exp_name = '[GR ours recon feat random mask 6_stage2]<2023-10-25_22-26-38>'

stage4model = f'result/{model_exp_name}/best_model.pth'
cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
with open(cfg_pickle_path, 'rb') as f:
    cfg = pickle.load(f)
cfg.stage2model_dir = os.path.join('result', model_exp_name)

cfg.net_det_dir = os.path.join('data_local', 'volleyball_net_detection')

# all_pruning_ratio = 0.2
# all_pruning_ratio = 0.3
# all_pruning_ratio = 0.4
# all_pruning_ratio = 0.5
# all_pruning_ratio = 0.6
all_pruning_ratio = 0.7
cfg.use_pruning_ratio = all_pruning_ratio
cfg.use_key_person_ratio = all_pruning_ratio

cfg.save_image = True

cfg.test_random_seed = 25
cfg.query_init = 1
cfg.non_query_init = 100

# cfg.sampling_mode = 'rand'
cfg.sampling_mode = 'near'

# cfg.query_type = 'l_spike'
cfg.query_type = 'l_set'
# cfg.query_type = 'l_pass'
# cfg.query_type = 'l_winpoint'
# cfg.query_type = 'r_spike'
# cfg.query_type = 'r_set'
# cfg.query_type = 'r_pass'
# cfg.query_type = 'r_winpoint'

# cfg.query_type = 'l_set_jump'
# cfg.query_type = 'spike'

cfg.use_feat_reduction = False
# cfg.use_feat_reduction = True

# cfg.use_anchor_type = 'normal'
# cfg.use_anchor_type = 'pruning'
cfg.use_anchor_type = 'pruning_gt'
# cfg.use_anchor_type = 'pruning_gt_reduction'
# cfg.use_anchor_type = 'learnable'
# cfg.use_anchor_type = 'learnable_l2_reg'

# cfg.key_person_mode = 'learnable'
# cfg.key_person_mode = 'mask_zero'
cfg.key_person_mode = 'mask_one'
# cfg.key_person_mode = 'mask_one_farther'
# cfg.key_person_mode = 'mask_one_negative'
# cfg.key_person_mode = 'mask_one_position'
# cfg.key_person_mode = 'mask_one_appearance'
# cfg.key_person_mode = 'mask_position'
# cfg.key_person_mode = 'mask_appearance'
# cfg.key_person_mode = 'gt'
# cfg.key_person_mode = 'ind_feat'

# cfg.anchor_agg_mode = 'max'
cfg.anchor_agg_mode = 'mean'

cfg.use_individual_action_type = 'gt'

cfg.model_exp_name = model_exp_name
cfg.stage4model = stage4model

cfg = update_config_all(cfg)

cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.image_size = 320, 640
cfg.eval_only = True
cfg.eval_stage = 4
# cfg.train_seqs = [1]
# cfg.test_seqs = [4]
# cfg.old_act_rec = False
# cfg.old_act_rec = True
# cfg.eval_mask_num = float(args.eval_mask_num)
# cfg.eval_mask_action = args.eval_mask_action

cfg.batch_size = 16
cfg.test_batch_size = cfg.batch_size
cfg.num_frames = 10

cfg.eval_mask_num = 0
cfg.eval_mask_action = False

cfg.dataset_symbol = 'vol'
ret_dic = detect_key_person(cfg)

print(ret_dic)