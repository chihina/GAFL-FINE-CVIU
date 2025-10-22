import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('gpu', help='gpu_number')
parser.add_argument('model_exp_name', help='model_exp_name')

args = parser.parse_args()
device_list = str(args.gpu)

sys.path.append(".")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import pickle

from eval_net_stage4_gr import *
from config_utils import update_config_all

model_exp_name = str(args.model_exp_name)

stage4model = f'result/{model_exp_name}/best_model.pth'
cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
with open(cfg_pickle_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.query_dir = os.path.join('data_local', 'collective_query')
# cfg.query_type = 'talking_three'
# cfg.use_query_loss = True

cfg.model_exp_name = model_exp_name
cfg.stage4model = stage4model

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

# cfg.use_debug = True

cfg.dataset_symbol = 'cad'
eval_net(cfg)