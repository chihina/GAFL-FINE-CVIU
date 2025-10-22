'''
    Evaluate the trained model with various metrics and visualization.
'''

import torch
import torch.optim as optim

import time
import random
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import json
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import average_precision_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

import warnings 
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import *
from volleyball import *
from collective import *
from dataset import *
from infer_model import *
from infer_model_prev import *
from infer_model_original import *
from base_model import *
from utils import *

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors

@torch.no_grad()
def eval_net(cfg):
    """
    evaluating gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list

    cfg.use_debug = False
    # cfg.use_debug = True

    # Show config parameters
    cfg.init_config()
    # cfg.person_size = (368, 368)
    show_config(cfg)
    
    # Reading dataset
    training_set,validation_set,all_set=return_dataset(cfg)
    cfg.num_boxes = all_set.get_num_boxes_max()
    training_set_len = len(training_set)
    validation_set_len = len(validation_set)
    all_set_len = len(all_set)
    
    params = {
        'shuffle': False,
        'num_workers': 4, # 4,
    }
    cfg.batch_size = 1
    cfg.test_batch_size = cfg.batch_size
    params['batch_size'] = cfg.batch_size
    training_loader=data.DataLoader(training_set,**params)
    validation_loader=data.DataLoader(validation_set,**params)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    cfg.device = device
    
    # Build model and Load trained model
    basenet_list={'group_relation_volleyball':Basenet_volleyball,
                  'group_relation_collective':Basenet_collective,
                  }
    gcnnet_list={
                 'group_activity_volleyball':GroupActivity_volleyball,
                 'group_relation_volleyball':GroupRelation_volleyball,
                 'group_relation_higcin_volleyball':GroupRelation_HiGCIN_volleyball,
                 'group_relation_din_volleyball':GroupRelation_DIN_volleyball,
                 'group_relation_ident_volleyball':GroupRelationIdentity_volleyball,
                 'group_relation_ae_volleyball':GroupRelationAutoEncoder_volleyball,
                 'group_relation_hrn_volleyball':GroupRelationHRN_volleyball,
                 'group_activity_collective':GroupActivity_volleyball,
                 'group_relation_collective':GroupRelation_volleyball,
                 'group_relation_higcin_collective':GroupRelation_HiGCIN_volleyball,
                 'group_relation_din_collective':GroupRelation_DIN_volleyball,
                 'group_relation_ident_collective':GroupRelationIdentity_volleyball,
                 'group_relation_ae_collective':GroupRelationAutoEncoder_volleyball,
                 'group_relation_hrn_collective':GroupRelationHRN_volleyball,
                 'dynamic_volleyball':Dynamic_volleyball,
                 'dynamic_collective':Dynamic_volleyball,
                 'higcin_volleyball':HiGCIN_volleyball,
                 'higcin_collective':HiGCIN_volleyball,
                 'person_action_recognizor':PersonAction_volleyball,
                 'single_branch_transformer':PersonActionSigleBranch_volleyball,
                 'single_branch_transformer_wo_spatial':PersonActionSigleBranchTemporal_volleyball,
                }

    if cfg.eval_stage == 1:
        model = basenet_list[cfg.inference_module_name](cfg)
        model.loadmodel(cfg.stage1model)
        print_log(cfg.log_path, f'Loading stage{cfg.eval_stage} model: ' + cfg.stage1model)
    elif cfg.eval_stage == 2:
        model = gcnnet_list[cfg.inference_module_name](cfg)
        if 'group_relation_ident' in cfg.inference_module_name:
            if cfg.load_backbone_stage2:
                model.loadmodel(cfg.stage1_model_path)
            else:
                pass
        else:
            state_dict = torch.load(cfg.stage2model)['state_dict']
            new_state_dict=OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] 
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print_log(cfg.log_path, f'Loading stage{cfg.eval_stage} model: ' + cfg.stage2model)
    else:
        assert False, 'cfg.eval_stage should be 1 or 2'

    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)
    model=model.to(device=device)

    """
    マルチラベル retrieval task 評価：
      - 学習・テストデータから動画特徴量と multi-hot ラベルを抽出
      - k 最近傍を探索
      - IoU@k を計算（各サンプル上位 k の近傍との最大 IoU の平均）
      - IoU しきい値ごとの Hit@k も計算（0.3, 0.5, 0.7, 1.0）
      - 必要に応じてマルチラベル分類の precision/recall/F1 も計算
    """

    # set the parent directory of cfg.stage2model as save_path
    save_path = os.path.dirname(cfg.stage2model)

    model.eval()

    # --- 特徴量・マルチラベル収集（学習データ） ---
    train_feats, train_labels = [], []
    for sample in tqdm(training_loader, desc="Extracting Train Features"):

        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                sample[key] = sample[key].to(device=device)

        activities = sample['activities_all_in']
        ret = model(sample)
        train_feats.append(ret['group_feat'].cpu().numpy())         # (B, D)
        train_labels.append(activities[:, 0, :].cpu().numpy())         # (B, C)
    train_features = np.vstack(train_feats)   # (N_train, D)
    train_labels   = np.vstack(train_labels)  # (N_train, C)

    # --- 特徴量・マルチラベル収集（テストデータ） ---
    test_feats, test_labels = [], []
    for sample in tqdm(validation_loader, desc="Extracting Test Features"):

        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                sample[key] = sample[key].to(device=device)

        activities = sample['activities_all_in']
        ret = model(sample)
        test_feats.append(ret['group_feat'].cpu().numpy())
        test_labels.append(activities[:, 0, :].cpu().numpy())
    test_features = np.vstack(test_feats)   # (N_test, D)
    test_labels   = np.vstack(test_labels)  # (N_test, C)

    # --- k 最近傍探索 ---
    k_list = [1, 2, 3, 4, 5]
    max_k = max(k_list)
    nbrs = NearestNeighbors(n_neighbors=max_k, algorithm='brute').fit(train_features)
    _, indices = nbrs.kneighbors(test_features)  # indices: (N_test, max_k)

    # --- IoU@k の計算 & しきい値別 Hit@k のカウント準備 ---
    thresholds = [0.3, 0.5, 0.7, 1.0]
    iou_sums       = {k: 0.0 for k in k_list}
    thr_hit_counts = {thr: {k: 0 for k in k_list} for thr in thresholds}

    for i, neigh_idxs in enumerate(indices):
        y_true = test_labels[i]  # (C,)
        for k in k_list:
            neigh_labels = train_labels[neigh_idxs[:k]]  # (k, C)
            # 各近傍との IoU を計算し、最大値を取る
            max_iou = 0.0
            for y_pred in neigh_labels:
                inter = np.logical_and(y_true, y_pred).sum()
                union = np.logical_or(y_true, y_pred).sum()
                iou = inter / union if union > 0 else 0.0
                if iou > max_iou:
                    max_iou = iou
            # IoU@k の累積
            iou_sums[k] += max_iou
            # 各しきい値でヒット判定
            for thr in thresholds:
                if max_iou >= thr:
                    thr_hit_counts[thr][k] += 1

    n_test = len(test_labels)
    # 平均 IoU@k
    iou_at_k = {k: iou_sums[k] / n_test for k in k_list}
    # しきい値別 Hit@k 比率 (%)
    thr_hit_rates = {
        thr: {k: thr_hit_counts[thr][k] / n_test * 100.0 for k in k_list}
        for thr in thresholds
    }

    # --- 結果表示 ---
    print("Retrieval evaluation (IoU@k):")
    for k in k_list:
        print(f"  IoU@{k}: {iou_at_k[k]:.4f}")

    print("\nHit@k for various IoU thresholds:")
    for thr in thresholds:
        print(f" IoU ≥ {thr}:")
        for k in k_list:
            print(f"   Hit@{k}: {thr_hit_rates[thr][k]:.2f}%")

    # --- オプション：マルチラベル分類として precision/recall/F1 も出す場合 ---
    y_pred1 = train_labels[indices[:,0]]  # (N_test, C)
    precision = precision_score(test_labels, y_pred1, average='samples') * 100
    recall    = recall_score   (test_labels, y_pred1, average='samples') * 100
    f1        = f1_score       (test_labels, y_pred1, average='samples') * 100
    print("\nAs multi-label classification (using nearest neighbor):")
    print(f"  Precision (samples avg): {precision:.2f}%")
    print(f"  Recall    (samples avg): {recall:.2f}%")
    print(f"  F1-score  (samples avg): {f1:.2f}%")

    # --- 結果をファイルにも保存 ---
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "retrieval_iou.txt"), "w") as f:
        f.write("IoU@k:\n")
        for k in k_list:
            f.write(f"IoU@{k}: {iou_at_k[k]:.4f}\n")
        f.write("\nHit@k for IoU thresholds:\n")
        for thr in thresholds:
            f.write(f"IoU ≥ {thr}:\n")
            for k in k_list:
                f.write(f"  Hit@{k}: {thr_hit_rates[thr][k]:.2f}%\n")
        f.write("\nMulti-label classification (k=1 neighbor):\n")
        f.write(f"Precision: {precision:.2f}%\n")
        f.write(f"Recall   : {recall:.2f}%\n")
        f.write(f"F1-score : {f1:.2f}%\n")

    # print(f"\nResults saved to {save_path}/retrieval_iou.txt")