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

from detect_key_person_all import *
from config_utils import update_config_all

model_exp_name = '[CAD GA ours recon rand mask 6_stage2]<2023-10-26_16-18-47>'
stage4model = f'result/{model_exp_name}/best_model.pth'
cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
with open(cfg_pickle_path, 'rb') as f:
    cfg = pickle.load(f)
cfg.stage2model_dir = os.path.join('result', model_exp_name)

cfg.mode = 'PAF'

# cfg.query_type = 'Moving'
# cfg.query_type = 'Queueing'
cfg.query_type = 'Talking'
# cfg.query_type = 'Waiting'

cfg.query_dir = os.path.join('data_local', 'collective_query')

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

all_pruning_ratio = 0.7
cfg.use_pruning_ratio = all_pruning_ratio
cfg.use_key_person_ratio = all_pruning_ratio

cfg.batch_size = 32
cfg.test_batch_size = cfg.batch_size
cfg.num_frames = 10

cfg.eval_mask_num = 0
cfg.eval_mask_action = False

cfg.test_random_seed = 11
cfg.query_init = 1
cfg.non_query_init = 10

cfg.use_anchor_type = 'normal'
# cfg.use_anchor_type = 'pruning'
# cfg.use_anchor_type = 'pruning_gt'
# cfg.use_anchor_type = 'pruning_gt_reduction'
# cfg.use_anchor_type = 'pruning_p2g_inner_cos'

# cfg.anchor_thresh_type = 'ratio'
cfg.anchor_thresh_type = 'val'

# cfg.key_person_mode = 'mask_zero'
cfg.key_person_mode = 'mask_one'

# cfg.anchor_agg_mode = 'max'
cfg.anchor_agg_mode = 'mean'

# cfg.sampling_mode = 'rand'
cfg.sampling_mode = 'near'

cfg.save_image = True

cfg.dataset_symbol = 'cad'
ret_dic = detect_key_person(cfg)
print(ret_dic)