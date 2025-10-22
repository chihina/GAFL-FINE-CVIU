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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from gen_key_people_query import *
from config_utils import update_config_all

# model_exp_name = '[GR ours recon feat random mask 0_stage2]<2023-10-18_22-30-29>'
model_exp_name = '[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>'

stage4model = f'result/{model_exp_name}/best_model.pth'
cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
with open(cfg_pickle_path, 'rb') as f:
    cfg = pickle.load(f)
cfg.stage2model_dir = os.path.join('result', model_exp_name)

cfg.mode = 'PAF'

cfg.net_det_dir = os.path.join('data_local', 'volleyball_net_detection')
cfg.line_det_dir = os.path.join('data_local', 'volleyball_line_detection')

cfg.use_individual_action_type = 'gt_action'

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

cfg.dataset_symbol = 'vol'

# cfg.query_init = 5
cfg.query_init = 3
cfg.trial_num = 5

query_type_list = [
    # 'l_spike',
    # 'l_set',
    # 'l_pass',
    # 'l_winpoint',
    # 'r_spike',
    # 'r_set',
    # 'r_pass',
    'r_winpoint'
]

use_anchor_type_list = [
    # 'normal',
    'pruning_p2g_cos',
    # 'pruning_p2g_euc',
    # 'pruning_p2g_inner_cos',
    # 'pruning_p2g_inner_euc',
    # 'pruning_p2g_cos_inv',
    # 'pruning_p2g_euc_inv',
    # 'pruning_p2g_inner_cos_inv',
    # 'pruning_p2g_inner_euc_inv',
]

for query_type in query_type_list:
    for use_anchor_type in use_anchor_type_list:
        print(f'query_type: {query_type}, use_anchor_type: {use_anchor_type}')
        cfg.query_type = query_type
        cfg.use_anchor_type = use_anchor_type

        ret_dic = {}
        ret_dic['key_people_prob'] = {}
        ret_dic['key_people_gt'] = {}
        for test_random_seed in range(cfg.trial_num):
            cfg.test_random_seed = test_random_seed
            ret_dic = detect_key_person(cfg, ret_dic)

        # save the pseudo key people generation result
        save_dir = os.path.join('analysis', 'gen_key_people_query', model_exp_name, cfg.query_type,
                                f'query_{cfg.query_init}', cfg.use_anchor_type, f'trial{cfg.trial_num}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save the key people probability
        save_stats_all = {}
        key_people_prob = ret_dic['key_people_prob']
        key_people_gt = ret_dic['key_people_gt']

        # visualize the histogram of the key people probability
        key_people_val_all = []
        for action_name, prob_list in key_people_prob.items():
            key_people_val_all.extend(prob_list)
        print(f'key people probability all: {len(key_people_val_all)}')
        key_people_val_all = np.array(key_people_val_all)
        key_people_val_max = np.max(key_people_val_all)
        key_people_val_min = np.min(key_people_val_all)
        pad_val = 0.1
        bins_num = 20
        print(f'key people probability max: {key_people_val_max}, min: {key_people_val_min}')
        for action_name, prob_list in key_people_prob.items():
            fig, ax = plt.subplots()
            ax.hist(prob_list, bins=bins_num, range=(0, 1), density=True)
            ax.set_title(f'{action_name} key people probability')
            ax.set_xlabel('key people probability')
            ax.set_ylabel('count')
            ax.set_xlim([key_people_val_min - pad_val, key_people_val_max + pad_val])
            save_dir_action = os.path.join(save_dir, action_name)
            if not os.path.exists(save_dir_action):
                os.makedirs(save_dir_action)
            plt.savefig(os.path.join(save_dir_action, 'key_people_prob_hist.png'))
            plt.close()

            # calculate the mean and std of the key people probability
            mean_prob = np.mean(prob_list)
            std_prob = np.std(prob_list)
            save_stats_action = {}
            save_stats_action['mean_prob'] = mean_prob
            save_stats_action['std_prob'] = std_prob
            save_stats_all[action_name] = {'mean_prob': mean_prob, 'std_prob': std_prob}

        # calculate the auc of the key people probability in all actions
        key_people_prob_all = []
        key_people_gt_all = []
        for action_name, prob_list in key_people_prob.items():
            key_people_prob_all.extend(prob_list)
            key_people_gt_all.extend(key_people_gt[action_name])
        fpr, tpr, _ = roc_curve(key_people_gt_all, key_people_prob_all)
        auc_val = auc(fpr, tpr)
        save_stats_all['all'] = {'auc': auc_val}

        # save the mean and std of the key people probability
        with open(os.path.join(save_dir, 'key_people_prob_stats_all.json'), 'w') as f:
            json.dump(save_stats_all, f, indent=4)

        # visualize the histogram of the key people probability in all actions with their names
        fig, ax = plt.subplots()
        for action_name, prob_list in key_people_prob.items():
            ax.hist(prob_list, bins=bins_num, range=(0, 1), alpha=0.5, label=action_name, density=True)
        ax.set_title('key people probability')
        ax.set_xlabel('key people probability')
        ax.set_ylabel('count')
        ax.set_xlim([key_people_val_min - pad_val, key_people_val_max + pad_val])
        ax.legend()
        plt.savefig(os.path.join(save_dir, 'key_people_prob_hist_all.png'))
        plt.close()