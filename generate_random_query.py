import sys
sys.path.append(".")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', help='gpu_number', default='0')
parser.add_argument('-seed_num', help='seed_number', default=0)
parser.add_argument('-query', help='query_type', default='l_spike')
parser.add_argument('-max_epoch', help='max_epoch', default=100)
parser.add_argument('-use_key_person_type', help='use_key_person_type', default='gt')
parser.add_argument('-all_pruning_ratio', help='all_pruning_ratio', default=0.5)
parser.add_argument('-model_exp_name_stage2', help='model_exp_name_stage2')
parser.add_argument('-use_key_person_loss_func', help='use_key_person_loss_func', default='bce_we')
parser.add_argument('-use_individual_action_type', help='use_individual_action_type', default='gt')
parser.add_argument('-query_init', help='non_query_init', default=5)
parser.add_argument('-non_query_init', help='non_query_init', default=50)
parser.add_argument('-use_recon_loss', help='use_recon_loss', default=True)
parser.add_argument('-feature_adapt_type', help='feature_adapt_type', default='ft')
parser.add_argument('-freeze_backbone_stage4', help='freeze_back_bone_stage4', default=False)
parser.add_argument('-recon_loss_type', help='recon_loss_type', default='all')
parser.add_argument('-use_key_recog_loss', help='use_key_recog_loss', default=True)
parser.add_argument('-use_disentangle_loss', help='use_disentangle_loss', default=False)
parser.add_argument('-sampling_mode', help='nn_sampling_mode', default='close')
parser.add_argument('-train_learning_rate', help='train_learning_rate', default='1e-4')
parser.add_argument('-key_person_det_interval', help='key_person_det_interval', default=10)
parser.add_argument('-save_epoch_interval', help='save_epoch_interval', default=1)
parser.add_argument('-anchor_agg_mode', help='anchor_agg_mode', default='mean')
parser.add_argument('-key_recog_feat_type', help='key_recog_feat_type', default='gaf')
parser.add_argument('-use_random_mask', help='use_random_mask', default=False)
parser.add_argument('-use_proximity_loss', help='use_proximity_loss', default=False)
parser.add_argument('-feature_adapt_dim', help='feature_adapt_dim', default=2048)
parser.add_argument('-use_query_classfication_loss', help='use_query_classfication_loss', default=False)
parser.add_argument('-use_anchor_type', help='use_anchor_type', default='pruning_gt')
parser.add_argument('-use_recon_loss_key', help='use_recon_loss_key', default=False)
parser.add_argument('-use_ga_recog_loss', help='use_ga_recog_loss', default=False)
parser.add_argument('-use_ia_recog_loss', help='use_ia_recog_loss', default=False)
parser.add_argument('-use_maintain_loss', help='use_maintain_loss', default=False)
parser.add_argument('-use_metric_lr_loss', help='use_metric_lr_loss', default=False)
parser.add_argument('-metric_lr_margin', help='metric_lr_margin', default=1.0)

args = parser.parse_args()

import torch
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler

import time
import random
import os
import sys
import wandb
from tqdm import tqdm
from collections import OrderedDict
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import itertools
import json
import collections
from sklearn.cluster import KMeans

from config import *
from volleyball import *
from collective import *
from dataset import *
from infer_model import *
from infer_model_prev import *
from infer_model_original import *
from base_model import *
from utils import *
from detect_key_people_utils import *

def train_net(cfg):
    # Reading dataset
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list
    device = torch.device('cuda')
    training_set, validation_set, all_set = return_dataset(cfg)
    cfg.num_boxes = all_set.get_num_boxes_max()
        
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': False,
    }
    
    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    torch.cuda.manual_seed(cfg.train_random_seed)

    # select query samples from the validation set
    queries = torch.tensor(validation_set.get_query_list(), device=device)
    query_indices = torch.nonzero(queries).squeeze()
    query_indices_rand_perm = torch.randperm(query_indices.shape[0])
    query_indices_use = query_indices[query_indices_rand_perm[:cfg.query_init]]
    print('query_indices_use', query_indices_use)

dataset = 'volleyball'
cfg=Config(dataset)
cfg.dataset = dataset

if dataset == 'volleyball':
    cfg.query_dir = os.path.join('data_local', 'volleyball_query')
    cfg.jae_ann_dir = os.path.join('data_local', 'vatic_ball_annotation', 'annotation_data_sub')
    cfg.net_det_dir = os.path.join('data_local', 'volleyball_net_detection')
    cfg.line_det_dir = os.path.join('data_local', 'volleyball_line_detection')
elif dataset == 'collective':
    cfg.query_dir = os.path.join('data_local', 'collective_query')

cfg.train_random_seed = int(args.seed_num)
cfg.max_epoch = int(args.max_epoch)
cfg.use_key_person_type = args.use_key_person_type
cfg.all_pruning_ratio = float(args.all_pruning_ratio)
cfg.model_exp_name_stage2 = args.model_exp_name_stage2
cfg.use_key_person_loss_func = args.use_key_person_loss_func
cfg.use_individual_action_type = args.use_individual_action_type
cfg.query_init = int(args.query_init)
cfg.non_query_init = args.non_query_init
cfg.use_recon_loss = args.use_recon_loss == 'True'
cfg.feature_adapt_type = args.feature_adapt_type
cfg.freeze_backbone_stage4 = args.freeze_backbone_stage4 == 'True'
cfg.recon_loss_type = args.recon_loss_type
cfg.use_key_recog_loss = args.use_key_recog_loss == 'True'
cfg.use_disentangle_loss = args.use_disentangle_loss == 'True'
cfg.sampling_mode = args.sampling_mode
cfg.train_learning_rate = float(args.train_learning_rate)
cfg.key_person_det_interval = int(args.key_person_det_interval)
cfg.save_epoch_interval = int(args.save_epoch_interval)
cfg.anchor_agg_mode = args.anchor_agg_mode
cfg.key_recog_feat_type = args.key_recog_feat_type
cfg.use_random_mask = args.use_random_mask == 'True'
cfg.use_proximity_loss = args.use_proximity_loss == 'True'
cfg.feature_adapt_dim = int(args.feature_adapt_dim)
cfg.use_query_classfication_loss = args.use_query_classfication_loss == 'True'
cfg.use_anchor_type = args.use_anchor_type
cfg.use_recon_loss_key = args.use_recon_loss_key == 'True'
cfg.use_ga_recog_loss = args.use_ga_recog_loss == 'True'
cfg.use_ia_recog_loss = args.use_ia_recog_loss == 'True'
cfg.use_maintain_loss = args.use_maintain_loss == 'True'
cfg.use_metric_lr_loss = args.use_metric_lr_loss == 'True'
cfg.metric_lr_margin = float(args.metric_lr_margin)
cfg.query_type = args.query

cfg.use_act_loss = False
cfg.use_pose_loss = False
cfg.use_recon_diff_loss = False
cfg.use_jae_loss = False
cfg.use_recon_pose_feat_loss = False
cfg.use_recon_pose_coord_loss = False

cfg = train_net(cfg)