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

model_exp_name = '[BSK_GAFL_PAF_mask_6_hbb_vgg_0.0001_wo_TB_15_6_stage2]<2025-04-29_14-26-23>'

stage4model = f'result/{model_exp_name}/best_model.pth'
cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
with open(cfg_pickle_path, 'rb') as f:
    cfg = pickle.load(f)
cfg.stage2model_dir = os.path.join('result', model_exp_name)

cfg.data_path = 'data/basketball/videos'  # data path for the basketball dataset
train_seq_path = os.path.join(cfg.data_path, 'train_video_ids')
with open(train_seq_path, 'r') as f:
    cfg.train_seqs = list(map(int, f.readlines()[0].strip().split(',')[:-1]))
cfg.train_seqs = cfg.train_seqs[:4]
test_seq_path = os.path.join(cfg.data_path, 'test_video_ids')
with open(test_seq_path, 'r') as f:
    cfg.test_seqs = list(map(int, f.readlines()[0].strip().split(',')[:-1]))
cfg.test_seqs = cfg.test_seqs[:4]

# all_pruning_ratio = 0.2
# all_pruning_ratio = 0.3
# all_pruning_ratio = 0.4
# all_pruning_ratio = 0.5
# all_pruning_ratio = 0.6
all_pruning_ratio = 0.7
cfg.use_pruning_ratio = all_pruning_ratio
cfg.use_key_person_ratio = all_pruning_ratio

cfg.save_image = True

cfg.test_random_seed = 11
cfg.query_init = 1
cfg.non_query_init = 10

# cfg.sampling_mode = 'rand'
cfg.sampling_mode = 'near'

# cfg.query_type = '2p-succ.'
# cfg.query_type = '2p-fail.-def.'
# cfg.query_type = '2p-fail.-off.'
# cfg.query_type = '2p-layup-succ.'
# cfg.query_type = '2p-layup-fail.-def.'
cfg.query_type = '2p-layup-fail.-off.'
# cfg.query_type = '3p-succ.'
# cfg.query_type = '3p-fail.-def.'
# cfg.query_type = '3p-fail.-off.'

cfg.use_feat_reduction = False
# cfg.use_feat_reduction = True

# cfg.use_anchor_type = 'normal'
# cfg.use_anchor_type = 'pruning_p2g'
cfg.use_anchor_type = 'pruning_p2g_inner_cos'

# cfg.anchor_thresh_type = 'ratio'
cfg.anchor_thresh_type = 'val'

# cfg.key_person_mode = 'mask_zero'
cfg.key_person_mode = 'mask_one'

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

cfg.batch_size = 8
cfg.test_batch_size = cfg.batch_size
# cfg.num_frames = 10

cfg.eval_mask_num = 0
cfg.eval_mask_action = False

cfg.is_training = False

cfg.dataset_symbol = 'bsk'
ret_dic = detect_key_person(cfg)
print(ret_dic)