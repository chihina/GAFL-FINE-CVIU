import sys
import glob
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

def run_eval(model_exp_name):
    stage4model = f'result/{model_exp_name}/best_model.pth'
    cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
    with open(cfg_pickle_path, 'rb') as f:
        cfg = pickle.load(f)

    cfg.model_exp_name = model_exp_name

    cfg.stage4model = stage4model
    cfg = update_config_all(cfg)
    cfg.device_list = device_list
    cfg.use_gpu = True
    cfg.use_multi_gpu = True
    cfg.image_size = 320, 640
    cfg.eval_only = True
    cfg.eval_stage = 4

    dataset = 'basketball'
    cfg_original=Config(dataset)
    cfg.train_seqs = cfg_original.train_seqs
    cfg.test_seqs = cfg_original.test_seqs

    cfg.batch_size = 16
    cfg.test_batch_size = cfg.batch_size
    # cfg.num_frames = 10

    cfg.eval_mask_num = 0
    cfg.eval_mask_action = False

    # cfg.use_debug = True

    cfg.dataset_symbol = 'bsk'
    eval_net(cfg)

# model_exp_name = str(args.model_exp_name)
# model_exp_name_list = [model_exp_name]

model_exp_name_list = glob.glob(os.path.join('result', '*BSK_GAFL_PAF_2p*'))
model_exp_name_list = [os.path.basename(i) for i in model_exp_name_list]

for model_exp_name in model_exp_name_list:
    run_eval(model_exp_name)