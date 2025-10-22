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

from analyze_person_masking_one_sample import *
from config_utils import update_config_all

query_type_list = []
query_type_list.append('l_spike')
# query_type_list.append('l_pass')
# query_type_list.append('l_set')
# query_type_list.append('l_winpoint')

# model_exp_name = '[GR ours recon feat random mask 0_stage2]<2023-10-18_22-30-29>'
model_exp_name = '[GR ours recon feat random mask 6_stage2]<2023-10-25_22-26-38>'
for query_type in query_type_list:
    stage4model = f'result/{model_exp_name}/best_model.pth'
    cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
    with open(cfg_pickle_path, 'rb') as f:
        cfg = pickle.load(f)

    cfg.net_det_dir = os.path.join('data_local', 'volleyball_net_detection')

    cfg.test_random_seed = 2
    cfg.model_exp_name = model_exp_name
    cfg.stage4model = stage4model
    cfg.query_type = query_type
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
    ret_dic = eval_net(cfg)
    print(ret_dic)