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

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
            
def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def evaluate_iar_metrics(info_train, info_test, metrics_dic, cfg):
    iar_metrics_dic = {}
    get_actions_list={'volleyball':volley_get_actions, 'collective':collective_get_actions}
    get_actions = get_actions_list[cfg.dataset_name]

    print("=========> Calculate IAR accuracy")
    iar_acc_array = info_test['actions_in_all'] == info_test['actions_labels_all']
    iar_mca = torch.mean(iar_acc_array.float()).to('cpu').detach().item()
    iar_acc_mean_mpca_list = []
    for iar_idx, iar_name in enumerate(get_actions()):
        iar_acc_array_class = iar_acc_array[info_test['actions_in_all']==iar_idx]
        iar_acc_mean_class = torch.mean(iar_acc_array_class.float()).to('cpu').detach().item()
        metrics_dic[f'IAR accuracy {iar_name}'] = iar_acc_mean_class
        print(f'IAR accuracy {iar_name}', iar_acc_mean_class)
        iar_acc_mean_mpca_list.append(iar_acc_mean_class)
    iar_mpca = sum(iar_acc_mean_mpca_list)/len(iar_acc_mean_mpca_list)
    metrics_dic['IAR MCA'] = iar_mca
    metrics_dic['IAR MPCA'] = iar_mpca
    print('IAR MCA', iar_mca)
    print('IAR MPCA', iar_mpca)

    if cfg.dataset_name != 'volleyball':
        return

    if not cfg.use_debug:
        print("=========> Calculate confusion matrix")
        cm_iar = confusion_matrix(info_test['actions_in_all'].view(-1).cpu().numpy(), info_test['actions_labels_all'].view(-1).cpu().numpy(), normalize='true')
        cm_iar = pd.DataFrame(cm_iar, columns=get_actions(), index=get_actions())
        plt.figure(figsize=(10, 10))
        plt.tight_layout()
        s = sns.heatmap(cm_iar, annot=True, cmap='Blues', fmt=".2f", annot_kws={"size": 25}, vmin=0, vmax=1)
        s.set(xlabel='Pr', ylabel='GT')
        save_cm_path = os.path.join(cfg.result_path, 'confusion_matrix_iar.png')
        plt.savefig(save_cm_path)

def plot_action_iou_histgram(mode, cfg, action_iou_type, action_iou_array):
    print(f"===> Plot histgram {mode} {action_iou_type}")
    action_iou_array_np = action_iou_array.cpu().numpy()
    plt.figure()
    plt.hist(action_iou_array_np)
    save_action_iou_path = os.path.join(cfg.result_path, f'hist_{action_iou_type}_{mode}.png')
    plt.savefig(save_action_iou_path)
    save_action_iou_np_path = os.path.join(cfg.result_path, f'hist_{action_iou_type}_{mode}')
    np.save(save_action_iou_np_path, action_iou_array_np)

# def calc_action_iou_labels(actions_cnt_cat, action_iou_thresh, actions_weights):
#     action_cnt_cat_max, _ = actions_cnt_cat.max(dim=3)
#     action_cnt_cat_min, _ = actions_cnt_cat.min(dim=3)
#     action_cnt_cat_max_weighted = action_cnt_cat_max * actions_weights
#     action_cnt_cat_min_weighted = action_cnt_cat_min * actions_weights
#     action_iou_weighted = action_cnt_cat_min_weighted.sum(dim=2) / action_cnt_cat_max_weighted.sum(dim=2)
#     action_iou_weighted_labels = action_iou_weighted > action_iou_thresh

#     return action_iou_weighted, action_iou_weighted_labels

def calc_action_iou_labels(actions_cnt_cat, action_iou_thresh, mode='jaccard'):
    print(f'action iou {mode}')
    test_num, train_val_num, actions_num, _ = actions_cnt_cat.shape
    actions_cnt_cat_union, _ = actions_cnt_cat.max(dim=3)
    actions_cnt_cat_common, _ = actions_cnt_cat.min(dim=3)
    actions_cnt_cat_sum_train_val = actions_cnt_cat[:, :, :, 0].sum(dim=2)
    actions_cnt_cat_sum_test = actions_cnt_cat[:, :, :, 1].sum(dim=2)
    actions_cnt_cat_sum_common = torch.min(actions_cnt_cat_sum_train_val, actions_cnt_cat_sum_test)

    if mode == 'jaccard':
        action_iou = actions_cnt_cat_common.sum(dim=2) / actions_cnt_cat_union.sum(dim=2)
    elif mode == 'dice':
        action_iou = (2*actions_cnt_cat_common.sum(dim=2)) / (actions_cnt_cat_sum_train_val+actions_cnt_cat_sum_test)
    elif mode == 'simpson':
        action_iou = actions_cnt_cat_common.sum(dim=2) / actions_cnt_cat_sum_common
    elif mode == 'tfidf':
        actions_cnt_cat_tf_train_val = actions_cnt_cat[:, :, :, 0] / actions_cnt_cat_sum_train_val.view(test_num, train_val_num, 1)
        actions_cnt_cat_tf_train_val = actions_cnt_cat_tf_train_val.view(test_num, train_val_num, actions_num, 1)
        actions_cnt_cat_tf_test = actions_cnt_cat[:, :, :, 1] / actions_cnt_cat_sum_test.view(test_num, train_val_num, 1)
        actions_cnt_cat_tf_test = actions_cnt_cat_tf_test.view(test_num, train_val_num, actions_num, 1)
        actions_cnt_cat_tf = torch.cat([actions_cnt_cat_tf_train_val, actions_cnt_cat_tf_test], dim=-1)
        actions_cnt_train_val_freq = actions_cnt_cat[0, :, :, 0]
        actions_cnt_train_val_freq_flag = actions_cnt_train_val_freq > 0
        actions_cnt_train_val_freq_flag_sum = actions_cnt_train_val_freq_flag.sum(dim=0)
        actions_cnt_cat_idf =  torch.log(train_val_num/(actions_cnt_train_val_freq_flag_sum+1e-10))
        actions_cnt_cat_tfidf = actions_cnt_cat_idf.view(1, 1, actions_num, 1) * actions_cnt_cat_tf
        actions_cnt_cat_tfidf = actions_cnt_cat_tfidf / (torch.norm(actions_cnt_cat_tfidf, dim=2, keepdim=True))
        action_iou = (actions_cnt_cat_tfidf[:, :, :, 0] * actions_cnt_cat_tfidf[:, :, :, 1]).sum(dim=2)

    action_iou_labels = action_iou > action_iou_thresh

    return action_iou, action_iou_labels

def calc_hit_at_n(mode, cfg, action_iou_type, action_iou_labels_all, match_score, test_num, metrics_dic):
    print(f"======> Calculate hit@n ({action_iou_type}) ({mode})")
    match_score_argsort = torch.argsort(match_score, dim=1)
    hit_num = 5
    action_iou_flag = torch.zeros(test_num, hit_num, device=match_score.device)
    for hit_idx in range(hit_num):
        action_iou_labels_hit = action_iou_labels_all[torch.arange(test_num, device=match_score.device), match_score_argsort[:, hit_idx]]
        action_iou_flag[:, hit_idx] = action_iou_labels_hit
    for hit_idx in range(hit_num):
        action_iou_flag_hit = action_iou_flag[:, :(hit_idx+1)]
        action_iou_flag_hit_sum = action_iou_flag_hit.sum(dim=1) > 0
        action_iou_hit_score = torch.mean(action_iou_flag_hit_sum.float()).to('cpu').detach().item()
        metrics_dic[f'GAR accuracy hit@{hit_idx+1} ({action_iou_type}) ({mode})'] = action_iou_hit_score
        print(f'GAR accuracy hit@{hit_idx+1} ({action_iou_type}) ({mode})', action_iou_hit_score)

def calc_map(mode, cfg, action_iou_type, action_iou_labels_all, match_score, metrics_dic):
    print(f"======> Calculate mAP ({action_iou_type}) ({mode})")
    map_pred_conf = -1 * match_score.cpu()
    map_action_iou = average_precision_score(action_iou_labels_all.cpu().numpy(), map_pred_conf.numpy())
    metrics_dic[f'mAP ({action_iou_type}) ({mode})'] = map_action_iou
    print(f'mAP ({action_iou_type}) ({mode})', map_action_iou)
    map_action_iou_ranking = label_ranking_average_precision_score(action_iou_labels_all.cpu().numpy(), map_pred_conf.numpy())
    metrics_dic[f'mAP rank ({action_iou_type}) ({mode})'] = map_action_iou_ranking
    print(f'mAP rank ({action_iou_type}) ({mode})', map_action_iou_ranking)

def visualize_features_query(mode, cfg, info_test, test_tsne, use_legend, query_type, alpha = 0.8):
    plt.figure()
    feat_non_query = test_tsne[query_all_test.cpu().numpy()==0, :]
    feat_query = test_tsne[query_all_test.cpu().numpy()==1, :]
    plt.scatter(feat_non_query[:, 0], feat_non_query[:, 1], alpha=alpha, label='non-query')
    plt.scatter(feat_query[:, 0], feat_query[:, 1], alpha=alpha, label='query')
    if use_legend:
        plt.legend()
        plt.savefig(os.path.join(cfg.result_path, f'tsne_{mode}_{query_type}.png'))
    else:
        plt.savefig(os.path.join(cfg.result_path, f'tsne_{mode}_{query_type}_wo_legend.png'))

def visualize_features(mode, cfg, info_test, test_tsne, use_legend, alpha = 0.8):
    get_activities_list={'volleyball':volley_get_activities, 'collective':collective_get_activities}
    get_activities = get_activities_list[cfg.dataset_name]
    plt.figure()
    for gar_idx, gar_name in enumerate(get_activities()):
        plot_feat = test_tsne[info_test['activities_in_all'].cpu().numpy()==gar_idx, :]
        plt.scatter(plot_feat[:, 0], plot_feat[:, 1], label=gar_name, alpha=alpha)
    
    if use_legend:
        plt.legend()
        plt.savefig(os.path.join(cfg.result_path, f'tsne_{mode}.png'))
    else:
        plt.savefig(os.path.join(cfg.result_path, f'tsne_{mode}_wo_legend.png'))

def visualize_features_wo_ga(mode, cfg, info_test, test_tsne, alpha = 0.7):
    get_activities_list={'volleyball':volley_get_activities, 'collective':collective_get_activities}
    get_activities = get_activities_list[cfg.dataset_name]
    plt.figure()
    plt.scatter(test_tsne[:, 0], test_tsne[:, 1], alpha=alpha, color='k')
    plt.savefig(os.path.join(cfg.result_path, f'tsne_{mode}_wo_ga.png'))

def save_visualized_features(mode, cfg, info_test, test_tsn):
    test_gr = info_test['activities_in_all'].cpu().numpy().reshape(-1, 1)
    test_vis_array = np.concatenate([test_tsn, test_gr], axis=1)
    df_vis_results = pd.DataFrame(test_vis_array, info_test['video_id_all'], ['dim1', 'dim2', 'GA'])
    save_excel_file_path = os.path.join(cfg.result_path, f'test_tsne_{mode}.xlsx')
    df_vis_results.to_excel(save_excel_file_path, sheet_name='all')

def save_clustering_results(mode, cfg, info_test, kmeans_labels):
    test_gr = info_test['activities_in_all'].cpu().numpy().reshape(-1, 1)
    df_cluster_array = np.concatenate([kmeans_labels, test_gr], axis=1)
    df_cluster_results = pd.DataFrame(df_cluster_array, info_test['video_id_all'], ['cluster_id', 'GA'])
    save_excel_file_path = os.path.join(cfg.result_path, f'test_kmeans_{mode}.xlsx')
    df_cluster_results.to_excel(save_excel_file_path, sheet_name='all')

def save_nn_video(mode, cfg, info_all, match_idx):
    video_id_all = info_all['video_id_all']
    activities_in_all = info_all['activities_in_all'].cpu().numpy()
    query_all = info_all['query_all'].cpu().numpy()
    print(len(video_id_all))
    print('match_idx', match_idx)
    print('match_idx', match_idx.shape)
    video_id_all_nn = video_id_all[match_idx].cpu().numpy()
    activities_in_all_nn = activities_in_all[match_idx]
    query_all_nn = query_all[match_idx]

    # pack data
    nn_results_dic = {}
    nn_results_dic['nn_id'] = video_id_all_nn
    nn_results_dic['nn_ga'] = activities_in_all_nn
    nn_results_dic['nn_query'] = query_all_nn

    nn_results_dic['gt_ga'] = activities_in_all
    nn_results_dic['fl_ga'] = activities_in_all == activities_in_all_nn
    nn_results_dic['gt_query'] = query_all
    nn_results_dic['fl_query'] = query_all == query_all_nn

    # save data as a excel file
    nn_results_array = np.array(list(nn_results_dic.values())).transpose(1, 0)
    data_name_list = list(nn_results_dic.keys())
    df_nn_results = pd.DataFrame(nn_results_array, video_id_all, data_name_list)
    df_nn_results.to_excel(os.path.join(cfg.result_path, f'nn_video_{mode}.xlsx'), sheet_name='all')

def evaluate_gar_metrics(mode, info_train, info_test, metrics_dic, cfg):

    get_activities_list={'volleyball':volley_get_activities, 
                         'collective':collective_get_activities,
                         'basketball':basketball_get_activities,
                         }
    get_activities = get_activities_list[cfg.dataset_name]

    print("===> Calculate distance between train and test features")
    feat_dic = {'people':'individual_features', 'scene':'group_features'}
    match_score = torch.cdist(info_test[feat_dic[mode]], info_train[feat_dic[mode]], p=2)
    match_score_argsort = torch.argsort(match_score, dim=1)
    match_idx = torch.argmin(match_score, dim=-1)
    all_num = match_score.shape[0]

    print("===> Save a nearest neighbor video id")
    # save_nn_video(mode, cfg, info_all, match_idx)

    print("===> Set a threshold of action iou")
    action_iou_thresh = 0.5

    print("===> Calculate match label of action iou")
    actions_cnt_all = torch.zeros(all_num, cfg.num_actions, device=match_score.device)
    for action_idx in range(cfg.num_actions):
        actions_cnt_all[:, action_idx] = (info_test['actions_in_all'] == action_idx).sum(dim=1)
    actions_cnt_all_expand = actions_cnt_all.view(all_num, 1, -1, 1).expand(all_num, all_num, cfg.num_actions, 1)
    actions_cnt_cat = torch.cat([actions_cnt_all_expand, actions_cnt_all_expand], dim=3)
    action_iou_jac, action_iou_labels_jac = calc_action_iou_labels(actions_cnt_cat, action_iou_thresh, 'jaccard')
    action_iou_dice, action_iou_labels_dice = calc_action_iou_labels(actions_cnt_cat, action_iou_thresh, 'dice')
    action_iou_simp, action_iou_labels_simp = calc_action_iou_labels(actions_cnt_cat, action_iou_thresh, 'simpson')
    action_iou_tfidf, action_iou_labels_tfidf = calc_action_iou_labels(actions_cnt_cat, action_iou_thresh, 'tfidf')
    # plot_action_iou_histgram(mode, cfg, 'action iou jaccard', action_iou_jac[torch.arange(all_num), match_idx])
    # plot_action_iou_histgram(mode, cfg, 'action iou dice', action_iou_dice[torch.arange(all_num), match_idx])
    # plot_action_iou_histgram(mode, cfg, 'action iou simpson', action_iou_simp[torch.arange(all_num), match_idx])
    # plot_action_iou_histgram(mode, cfg, 'action iou tfidf', action_iou_tfidf[torch.arange(all_num), match_idx])

    print("===> Calculate hit@n")
    # calc_hit_at_n(mode, cfg, 'action iou jaccard', action_iou_labels_jac, match_score, all_num, metrics_dic)
    # calc_hit_at_n(mode, cfg, 'action iou dice', action_iou_labels_dice, match_score, all_num, metrics_dic)
    # calc_hit_at_n(mode, cfg, 'action iou simpson', action_iou_labels_simp, match_score, all_num, metrics_dic)
    # calc_hit_at_n(mode, cfg, 'action iou tfidf', action_iou_labels_tfidf, match_score, all_num, metrics_dic)

    print("===> Calculate mAP")
    # calc_map(mode, cfg, 'action iou jaccard', action_iou_labels_jac, match_score, metrics_dic)
    # calc_map(mode, cfg, 'action iou dice', action_iou_labels_dice, match_score, metrics_dic)
    # calc_map(mode, cfg, 'action iou simpson', action_iou_labels_simp, match_score, metrics_dic)
    # calc_map(mode, cfg, 'action iou tfidf', action_iou_labels_tfidf, match_score, metrics_dic)

    print(f"======> Calculate GAR accuracy (group activity) ({mode})")
    gar_acc_array = info_test['activities_in_all'] == info_train['activities_in_all'][match_idx]
    gar_acc_mean = torch.mean(gar_acc_array.float()).to('cpu').detach().item()
    metrics_dic[f'GAR accuracy (group activity) ({mode})'] = gar_acc_mean
    print(f'GAR accuracy (group activity) ({mode})', gar_acc_mean)
    for gar_idx, gar_name in enumerate(get_activities()):
        gar_acc_array_class = gar_acc_array[info_test['activities_in_all']==gar_idx]
        gar_acc_mean_class = torch.mean(gar_acc_array_class.float()).to('cpu').detach().item()
        metrics_dic[f'GAR accuracy {gar_name} (group activity) ({mode})'] = gar_acc_mean_class
        print(f'GAR accuracy {gar_name} (group activity) (scene)', gar_acc_mean_class)
    print(f"======> Calculate hit@n (group activity) ({mode})")
    hit_num = 10
    hit_num_list = [i for i in range(1, hit_num+1)]
    for hit_num_add in range(10, 101, 5):
        hit_num_list.append(hit_num_add)
    for hit_idx in hit_num_list:
        gar_label_array_pred_hit = info_train['activities_in_all'][match_score_argsort[:, :hit_idx]]
        gar_label_array_test_hit = info_test['activities_in_all'].view(-1, 1)
        gar_label_array_test_hit = gar_label_array_test_hit.expand(info_test['activities_in_all'].shape[0], hit_idx)
        gar_acc_array_hit = gar_label_array_test_hit == gar_label_array_pred_hit
        gar_acc_array_hit_sum = torch.sum(gar_acc_array_hit, dim=-1)
        gar_acc_array_hit_sum_over_one = gar_acc_array_hit_sum >= 1
        gar_accuracy_hit = torch.mean(gar_acc_array_hit_sum_over_one.float()).to('cpu').detach().item()
        metrics_dic[f'GAR accuracy hit@{hit_idx} (group activity) ({mode})'] = gar_accuracy_hit
        print(f'GAR accuracy hit@{hit_idx} (group activity) ({mode})', gar_accuracy_hit)
    # calc_map(mode, cfg, 'group activity', torch.cdist(info_test['activities_in_all'].view(-1, 1), info_test['activities_in_all'].view(-1, 1), p=2) == 0, match_score, metrics_dic)

    print(f"======> Calculate KNN (group activity) ({mode})")
    # k_max = 10
    # if cfg.use_debug:
    #     k_max = 2
    # for k in range(1, k_max+1):
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     knn_feat_train, knn_feat_test = info_train[feat_dic[mode]].cpu().numpy(), info_test[feat_dic[mode]].cpu().numpy()
    #     gar_labels_train, gar_labels_test = info_train['activities_in_all'].cpu().numpy(), info_test['activities_in_all'].cpu().numpy()
    #     knn.fit(knn_feat_train, gar_labels_train)
    #     gar_acc_knn = knn.score(knn_feat_test, gar_labels_test)
    #     metrics_dic[f'GAR accuracy {k}NN (group activity) ({mode})'] = gar_acc_knn
    #     print(f'GAR accuracy {k}NN (group activity) ({mode})', gar_acc_knn)

    print(f"======> Save confusion matrix ({mode})")
    annot_kws_size = 20
    if cfg.dataset_symbol == 'cad':
        annot_kws_size = 35

    # if not cfg.use_debug:
    #     cm_gar_scene = confusion_matrix(info_test['activities_in_all'].cpu().numpy(), info_train['activities_in_all'][match_idx].cpu().numpy(), normalize='true')
    #     cm_gar_scene = pd.DataFrame(cm_gar_scene, columns=get_activities(), index=get_activities())
    #     plt.figure(figsize=(10, 10))
    #     plt.tight_layout()
    #     s = sns.heatmap(cm_gar_scene, annot=True, cmap='Blues', fmt=".2f", annot_kws={"size": annot_kws_size}, vmin=0, vmax=1)
    #     s.set(xlabel='Pr', ylabel='GT')
    #     save_cm_scene_path = os.path.join(cfg.result_path, f'confusion_matrix_gar_{mode}.png')
    #     plt.savefig(save_cm_scene_path)

    # load sampling information
    sampling_path = os.path.join(cfg.result_path, 'sampling.json')
    with open(sampling_path, 'r') as f:
        sampling_dic = json.load(f)
    query_indices = sampling_dic['query_indices']
    print('query_indices', query_indices)
    non_query_indices = sampling_dic['non_query_indices']
    print('non_query_indices', non_query_indices)

    print(f"======> Save Query retrieval accuracy ({mode})")
    query_all_test = info_test['query_all']
    print(match_idx)

    # exploit match_score without fine-tuning samples
    match_score_wo_fns = torch.cdist(info_test[feat_dic[mode]], info_train[feat_dic[mode]], p=2)
    match_score_wo_fns[:, non_query_indices] = torch.max(match_score)
    match_score_wo_fns_argsort = torch.argsort(match_score_wo_fns, dim=1)
    match_idx_wo_fns = torch.argmin(match_score_wo_fns, dim=-1)

    base_key = 'Query retrieval accuracy'
    retrieval_db = {
        '': match_score,
        ' wofns': match_score_wo_fns,
    }
    queries_train = info_train['query_all']
    queries_test = info_test['query_all']
    hit_value = 10
    hit_num_list.append(int(queries_train.sum()))
    one_tensor = torch.ones(1, device=match_score.device)
    for db_key, retrieval_score in retrieval_db.items():
        print(f'{base_key}{db_key} ({mode})')
        match_idx = torch.argmin(retrieval_score, dim=-1)
        match_score_argsort = torch.argsort(retrieval_score, dim=-1)

        retrieval_score_query = retrieval_score[query_indices]
        retrieval_score_query_max = torch.max(retrieval_score_query, dim=0)[0].unsqueeze(0)
        retrieval_score_query_mean = torch.mean(retrieval_score_query, dim=0).unsqueeze(0)

        match_idx_query_max = torch.argmin(retrieval_score_query_max, dim=-1)
        match_idx_query_mean = torch.argmin(retrieval_score_query_mean, dim=-1)

        match_score_argsort_query_max = torch.argsort(retrieval_score_query_max, dim=-1)
        match_score_argsort_query_mean = torch.argsort(retrieval_score_query_mean, dim=-1)

        acc_all = queries_test == queries_train[match_idx]
        acc_query = queries_test[query_indices] == queries_train[match_idx][query_indices]
        acc_query_max = one_tensor == queries_train[match_idx_query_max]
        acc_query_mean = one_tensor == queries_train[match_idx_query_mean]

        query_acc_results = {
            '': acc_all,
            ' user query': acc_query,
            ' user query fused': acc_query_max,
            ' user query fused (max)': acc_query_max,
            ' user query fused (mean)': acc_query_mean,
        }

        for key, query_acc_array in query_acc_results.items():
            query_acc_mean = torch.mean(query_acc_array.float()).to('cpu').detach().item()            
            metrics_dic[f'{base_key}{db_key}{key} ({mode})'] = query_acc_mean
            print(f'{base_key}{db_key}{key} ({mode})', query_acc_mean)

            if key == ' user query':
                queries_test_mode = queries_test[query_indices]
            elif key in [' user query fused (max)', ' user query fused (mean)', ' user query fused']:
                queries_test_mode = one_tensor
            else:
                queries_test_mode = queries_test

            for query_idx, query_name in enumerate([' non-query', ' query']):
                query_acc_array_class = query_acc_array[queries_test_mode == query_idx]
                query_acc_mean_class = torch.mean(query_acc_array_class.float()).to('cpu').detach().item()
                metrics_dic[f'{base_key} {db_key}{key}{query_name} ({mode})'] = query_acc_mean_class
                print(f'{base_key} {db_key}{key}{query_name} ({mode})', query_acc_mean_class)

        for hit_idx in hit_num_list:
            acc_all_hit = queries_test.view(-1, 1).expand(queries_test.shape[0], hit_idx) == queries_train[match_score_argsort[:, :hit_idx]]
            acc_query_hit = queries_test[query_indices].view(-1, 1).expand(len(query_indices), hit_idx) == queries_train[match_score_argsort[query_indices, :hit_idx]]
            # acc_query_fused_hit = one_tensor.view(-1, 1).expand(1, hit_idx) == queries_train[match_score_argsort_query_max[0, :hit_idx]]
            acc_query_max_hit = one_tensor.view(-1, 1).expand(1, hit_idx) == queries_train[match_score_argsort_query_max[0, :hit_idx]]
            acc_query_mean_hit = one_tensor.view(-1, 1).expand(1, hit_idx) == queries_train[match_score_argsort_query_mean[0, :hit_idx]]

            query_acc_results_hit = {
                '': acc_all_hit,
                ' user query': acc_query_hit,
                ' user query fused': acc_query_max_hit,
                ' user query fused (max)': acc_query_max_hit,
                ' user query fused (mean)': acc_query_mean_hit,
            }
            for key, query_acc_array_hit in query_acc_results_hit.items():
                query_acc_array_hit_sum = torch.sum(query_acc_array_hit, dim=-1)
                query_accuracy_hit = torch.mean((query_acc_array_hit_sum >= 1).float()).to('cpu').detach().item()
                metrics_dic[f'{base_key} hit@{hit_idx}{db_key}{key} ({mode})'] = query_accuracy_hit
                print(f'{base_key} hit@{hit_idx}{db_key}{key} ({mode})', query_accuracy_hit)

                query_accuracy_hit_sum = torch.mean(query_acc_array_hit_sum.float()).to('cpu').detach().item()
                metrics_dic[f'{base_key} hit@{hit_idx} sum{db_key}{key} ({mode})'] = query_accuracy_hit_sum
                print(f'{base_key} hit@{hit_idx} sum{db_key}{key} ({mode})', query_accuracy_hit_sum)

                if key == ' user query':
                    queries_test_mode = queries_test[query_indices]
                # elif key == ' user query fused':
                elif key in [' user query fused (max)', ' user query fused (mean)', ' user query fused']:
                    queries_test_mode = one_tensor
                else:
                    queries_test_mode = queries_test

                for query_idx, query_name in enumerate(['non-query', 'query']):
                    query_acc_array_hit_query = query_acc_array_hit[queries_test_mode == query_idx]
                    query_accuracy_hit_query = torch.mean((torch.sum(query_acc_array_hit_query, dim=-1) >= 1).float()).to('cpu').detach().item()
                    metrics_dic[f'{base_key} hit@{hit_idx}{db_key}{key} {query_name} ({mode})'] = query_accuracy_hit_query
                    print(f'{base_key} hit@{hit_idx}{db_key}{key} {query_name} ({mode})', query_accuracy_hit_query)
                    query_acc_array_hit_sum_query = query_acc_array_hit_sum[queries_test_mode == query_idx]
                    query_accuracy_hit_sum_query = torch.mean(query_acc_array_hit_sum_query.float()).to('cpu').detach().item()
                    metrics_dic[f'{base_key} hit@{hit_idx} sum{db_key}{key} {query_name} ({mode})'] = query_accuracy_hit_sum_query
                    print(f'{base_key} hit@{hit_idx} sum{db_key}{key} {query_name} ({mode})', query_accuracy_hit_sum_query)


    # =================================================================================================================

    # query_acc_array = query_all_test == info_train['query_all'][match_idx]
    # query_acc_mean = torch.mean(query_acc_array.float()).to('cpu').detach().item()
    # metrics_dic[f'Query retrieval accuracy ({mode})'] = query_acc_mean
    # print(f'Query retrieval accuracy ({mode})', query_acc_mean)

    # query_acc_array_user_query = query_acc_array[query_indices_sample]
    # query_acc_mean_user_query = torch.mean(query_acc_array_user_query.float()).to('cpu').detach().item()
    # metrics_dic[f'Query retrieval accuracy user query ({mode})'] = query_acc_mean_user_query
    # print(f'Query retrieval accuracy user query ({mode})', query_acc_mean_user_query)

    # query_acc_array_wo_fns = query_all_test == info_train['query_all'][match_idx_wo_fns]
    # query_acc_mean_wo_fns = torch.mean(query_acc_array_wo_fns.float()).to('cpu').detach().item()
    # metrics_dic[f'Query retrieval accuracy wofns ({mode})'] = query_acc_mean_wo_fns
    # print(f'Query retrieval accuracy wo fns ({mode})', query_acc_mean_wo_fns)

    # query_acc_array_wo_fns_user_query = query_acc_array_wo_fns[query_indices_sample]
    # query_acc_mean_wo_fns_user_query = torch.mean(query_acc_array_wo_fns_user_query.float()).to('cpu').detach().item()
    # metrics_dic[f'Query retrieval accuracy wofns user query ({mode})'] = query_acc_mean_wo_fns_user_query
    # print(f'Query retrieval accuracy wofns user query ({mode})', query_acc_mean_wo_fns_user_query)

    # for query_idx, query_name in enumerate(['non-query', 'query']):
    #     query_acc_array_class = query_acc_array[query_all_test==query_idx]
    #     query_acc_mean_class = torch.mean(query_acc_array_class.float()).to('cpu').detach().item()
    #     metrics_dic[f'Query retrieval accuracy {query_name} ({mode})'] = query_acc_mean_class
    #     print(f'Query retrieval accuracy {query_name} ({mode})', query_acc_mean_class)
    #     query_acc_array_user_query_class = query_acc_array_user_query[query_all_test[query_indices_sample]==query_idx]
    #     query_acc_mean_user_query_class = torch.mean(query_acc_array_user_query_class.float()).to('cpu').detach().item()
    #     metrics_dic[f'Query retrieval accuracy {query_name} user query ({mode})'] = query_acc_mean_user_query_class
    #     print(f'Query retrieval accuracy {query_name} user query ({mode})', query_acc_mean_user_query_class)

    #     query_acc_array_wo_fns_class = query_acc_array_wo_fns[query_all_test==query_idx]
    #     query_acc_mean_wo_fns_class = torch.mean(query_acc_array_wo_fns_class.float()).to('cpu').detach().item()
    #     metrics_dic[f'Query retrieval accuracy {query_name} wofns ({mode})'] = query_acc_mean_wo_fns_class
    #     print(f'Query retrieval accuracy {query_name} wofns ({mode})', query_acc_mean_wo_fns_class)
    #     query_acc_array_wo_fns_user_query_class = query_acc_array_wo_fns_user_query[query_all_test[query_indices_sample]==query_idx]
    #     query_acc_mean_wo_fns_user_query_class = torch.mean(query_acc_array_wo_fns_user_query_class.float()).to('cpu').detach().item()
    #     metrics_dic[f'Query retrieval accuracy {query_name} wofns user query ({mode})'] = query_acc_mean_wo_fns_user_query_class
    #     print(f'Query retrieval accuracy {query_name} wofns user query ({mode})', query_acc_mean_wo_fns_user_query_class)

    # hit_num_list.append(int(info_train['query_all'].sum()))
    # for hit_idx in hit_num_list:
    #     query_label_array_test_hit = query_all_test.view(-1, 1)
    #     query_label_array_test_hit = query_label_array_test_hit.expand(query_all_test.shape[0], hit_idx)

    #     query_label_array_pred_hit = info_train['query_all'][match_score_argsort[:, :hit_idx]]
    #     query_label_array_pred_wo_fns_hit = info_train['query_all'][match_score_wo_fns_argsort[:, :hit_idx]]
        
    #     query_acc_array_hit = query_label_array_test_hit == query_label_array_pred_hit
    #     query_acc_array_hit_sum = torch.sum(query_acc_array_hit, dim=-1)
    #     query_acc_array_wo_fns_hit = query_label_array_test_hit == query_label_array_pred_wo_fns_hit
    #     query_acc_array_wo_fns_hit_sum = torch.sum(query_acc_array_wo_fns_hit, dim=-1)

    #     # all test samples v.s all train samples
    #     query_acc_array_hit_sum_over_one = query_acc_array_hit_sum >= 1
    #     query_accuracy_hit = torch.mean(query_acc_array_hit_sum_over_one.float()).to('cpu').detach().item()
    #     metrics_dic[f'Query retrieval accuracy hit@{hit_idx} ({mode})'] = query_accuracy_hit
    #     print(f'Query retrieval accuracy hit@{hit_idx} ({mode})', query_accuracy_hit)
    #     query_accuracy_hit_sum = torch.mean(query_acc_array_hit_sum.float()).to('cpu').detach().item()
    #     metrics_dic[f'Query retrieval accuracy hit@{hit_idx} sum ({mode})'] = query_accuracy_hit_sum
    #     print(f'Query retrieval accuracy hit@{hit_idx} sum ({mode})', query_accuracy_hit_sum)

    #     # all test samples v.s all train samples without fine-tuning samples
    #     query_acc_array_wo_fns_hit_sum_over_one = query_acc_array_wo_fns_hit_sum >= 1
    #     query_accuracy_wo_fns_hit = torch.mean(query_acc_array_wo_fns_hit_sum_over_one.float()).to('cpu').detach().item()
    #     metrics_dic[f'Query retrieval accuracy hit@{hit_idx} wofns ({mode})'] = query_accuracy_wo_fns_hit
    #     print(f'Query retrieval accuracy hit@{hit_idx} wofns ({mode})', query_accuracy_wo_fns_hit)
    #     query_accuracy_wo_fns_hit_sum = torch.mean(query_acc_array_wo_fns_hit_sum.float()).to('cpu').detach().item()
    #     metrics_dic[f'Query retrieval accuracy hit@{hit_idx} sum wofns ({mode})'] = query_accuracy_wo_fns_hit_sum
    #     print(f'Query retrieval accuracy hit@{hit_idx} sum wofns ({mode})', query_accuracy_wo_fns_hit_sum)

    #     # query test samples v.s all train samples
    #     query_acc_array_hit_sum_over_one_user_query = query_acc_array_hit_sum_over_one[query_indices_sample]
    #     query_accuracy_hit_user_query = torch.mean(query_acc_array_hit_sum_over_one_user_query.float()).to('cpu').detach().item()
    #     metrics_dic[f'Query retrieval accuracy hit@{hit_idx} user query ({mode})'] = query_accuracy_hit_user_query
    #     print(f'Query retrieval accuracy hit@{hit_idx} user query ({mode})', query_accuracy_hit_user_query)
    #     query_acc_array_hit_sum_user_query = query_acc_array_hit_sum[query_indices_sample]
    #     query_accuracy_hit_sum_user_query = torch.mean(query_acc_array_hit_sum_user_query.float()).to('cpu').detach().item()
    #     metrics_dic[f'Query retrieval accuracy hit@{hit_idx} sum user query ({mode})'] = query_accuracy_hit_sum_user_query
    #     print(f'Query retrieval accuracy hit@{hit_idx} sum user query ({mode})', query_accuracy_hit_sum_user_query)

    #     # query test samples v.s all train samples without fine-tuning samples
    #     query_acc_array_wo_fns_hit_sum_over_one_user_query = query_acc_array_wo_fns_hit_sum_over_one[query_indices_sample]
    #     query_accuracy_wo_fns_hit_user_query = torch.mean(query_acc_array_wo_fns_hit_sum_over_one_user_query.float()).to('cpu').detach().item()
    #     metrics_dic[f'Query retrieval accuracy hit@{hit_idx} wofns user query ({mode})'] = query_accuracy_wo_fns_hit_user_query
    #     print(f'Query retrieval accuracy hit@{hit_idx} wofns user query ({mode})', query_accuracy_wo_fns_hit_user_query)
    #     query_acc_array_wo_fns_hit_sum_user_query = query_acc_array_wo_fns_hit_sum[query_indices_sample]
    #     query_accuracy_wo_fns_hit_sum_user_query = torch.mean(query_acc_array_wo_fns_hit_sum_user_query.float()).to('cpu').detach().item()
    #     metrics_dic[f'Query retrieval accuracy hit@{hit_idx} sum wofns user query ({mode})'] = query_accuracy_wo_fns_hit_sum_user_query
    #     print(f'Query retrieval accuracy hit@{hit_idx} sum wofns user query ({mode})', query_accuracy_wo_fns_hit_sum_user_query)

    #     for query_idx, query_name in enumerate(['non-query', 'query']):
    #         # all test samples v.s all train samples
    #         query_acc_array_hit_sum_over_one_query = query_acc_array_hit_sum_over_one[query_all_test==query_idx]
    #         query_accuracy_hit_query = torch.mean(query_acc_array_hit_sum_over_one_query.float()).to('cpu').detach().item()
    #         metrics_dic[f'Query retrieval accuracy hit@{hit_idx} {query_name} ({mode})'] = query_accuracy_hit_query
    #         print(f'Query retrieval accuracy hit@{hit_idx} {query_name} ({mode})', query_accuracy_hit_query)
    #         query_acc_array_hit_sum_query = query_acc_array_hit_sum[query_all_test==query_idx]
    #         query_accuracy_hit_sum_query = torch.mean(query_acc_array_hit_sum_query.float()).to('cpu').detach().item()
    #         metrics_dic[f'Query retrieval accuracy hit@{hit_idx} sum {query_name} ({mode})'] = query_accuracy_hit_sum_query
    #         print(f'Query retrieval accuracy hit@{hit_idx} sum {query_name} ({mode})', query_accuracy_hit_sum_query)

    #         # all test samples v.s all train samples without fine-tuning samples
    #         query_acc_array_wo_fns_hit_sum_over_one_query = query_acc_array_wo_fns_hit_sum_over_one[query_all_test==query_idx]
    #         query_accuracy_wo_fns_hit_query = torch.mean(query_acc_array_wo_fns_hit_sum_over_one_query.float()).to('cpu').detach().item()
    #         metrics_dic[f'Query retrieval accuracy hit@{hit_idx} {query_name} wofns ({mode})'] = query_accuracy_wo_fns_hit_query
    #         print(f'Query retrieval accuracy hit@{hit_idx} {query_name} wofns ({mode})', query_accuracy_wo_fns_hit_query)
    #         query_acc_array_wo_fns_hit_sum_query = query_acc_array_wo_fns_hit_sum[query_all_test==query_idx]
    #         query_accuracy_wo_fns_hit_sum_query = torch.mean(query_acc_array_wo_fns_hit_sum_query.float()).to('cpu').detach().item()
    #         metrics_dic[f'Query retrieval accuracy hit@{hit_idx} sum {query_name} wofns ({mode})'] = query_accuracy_wo_fns_hit_sum_query
    #         print(f'Query retrieval accuracy hit@{hit_idx} sum {query_name} wofns ({mode})', query_accuracy_wo_fns_hit_sum_query)

    #         # query test samples v.s all train samples
    #         query_acc_array_hit_sum_over_one_user_query_query = query_acc_array_hit_sum_over_one_user_query[query_all_test[query_indices_sample]==query_idx]
    #         query_accuracy_hit_user_query_query = torch.mean(query_acc_array_hit_sum_over_one_user_query_query.float()).to('cpu').detach().item()
    #         metrics_dic[f'Query retrieval accuracy hit@{hit_idx} user query {query_name} ({mode})'] = query_accuracy_hit_user_query_query
    #         print(f'Query retrieval accuracy hit@{hit_idx} user query {query_name} ({mode})', query_accuracy_hit_user_query_query)
    #         query_acc_array_hit_sum_user_query_query = query_acc_array_hit_sum_user_query[query_all_test[query_indices_sample]==query_idx]
    #         query_accuracy_hit_sum_user_query_query = torch.mean(query_acc_array_hit_sum_user_query_query.float()).to('cpu').detach().item()
    #         metrics_dic[f'Query retrieval accuracy hit@{hit_idx} sum user query {query_name} ({mode})'] = query_accuracy_hit_sum_user_query_query
    #         print(f'Query retrieval accuracy hit@{hit_idx} sum user query {query_name} ({mode})', query_accuracy_hit_sum_user_query_query)

    #         # query test samples v.s all train samples without fine-tuning samples
    #         query_acc_array_wo_fns_hit_sum_over_one_user_query_query = query_acc_array_wo_fns_hit_sum_over_one_user_query[query_all_test[query_indices_sample]==query_idx]
    #         query_accuracy_wo_fns_hit_user_query_query = torch.mean(query_acc_array_wo_fns_hit_sum_over_one_user_query_query.float()).to('cpu').detach().item()
    #         metrics_dic[f'Query retrieval accuracy hit@{hit_idx} wofns user query {query_name} ({mode})'] = query_accuracy_wo_fns_hit_user_query_query
    #         print(f'Query retrieval accuracy hit@{hit_idx} wofns user query {query_name} ({mode})', query_accuracy_wo_fns_hit_user_query_query)
    #         query_acc_array_wo_fns_hit_sum_user_query_query = query_acc_array_wo_fns_hit_sum_user_query[query_all_test[query_indices_sample]==query_idx]
    #         query_accuracy_wo_fns_hit_sum_user_query_query = torch.mean(query_acc_array_wo_fns_hit_sum_user_query_query.float()).to('cpu').detach().item()
    #         metrics_dic[f'Query retrieval accuracy hit@{hit_idx} sum wofns user query {query_name} ({mode})'] = query_accuracy_wo_fns_hit_sum_user_query_query
    #         print(f'Query retrieval accuracy hit@{hit_idx} sum wofns user query {query_name} ({mode})', query_accuracy_wo_fns_hit_sum_user_query_query)

    print(f"======> Calculate KNN (query) ({mode})")
    # for k in range(1, k_max+1):
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     knn_feat_train, knn_feat_test = info_train[feat_dic[mode]].cpu().numpy(), info_test[feat_dic[mode]].cpu().numpy()
    #     query_labels_train, query_labels_test = info_train['query_all'].cpu().numpy(), query_all_test.cpu().numpy()
    #     knn.fit(knn_feat_train, query_labels_train)
    #     query_acc_knn = knn.score(knn_feat_test, query_labels_test)
    #     metrics_dic[f'Query retrieval accuracy {k}NN ({mode})'] = query_acc_knn
    #     print(f'Query retrieval accuracy {k}NN ({mode})', query_acc_knn)
    
    print(f"======> Save confusion matrix (query) ({mode})")
    if not cfg.use_debug:
        cm_query_scene = confusion_matrix(query_all_test.cpu().numpy(), info_train['query_all'][match_idx].cpu().numpy(), normalize='true')
        cm_query_scene = pd.DataFrame(cm_query_scene, columns=['non-query', 'query'], index=['non-query', 'query'])
        plt.figure(figsize=(10, 10))
        plt.tight_layout()
        s = sns.heatmap(cm_query_scene, annot=True, cmap='Blues', fmt=".2f", annot_kws={"size": annot_kws_size}, vmin=0, vmax=1)
        s.set(xlabel='Pr', ylabel='GT')
        save_cm_query_path = os.path.join(cfg.result_path, f'confusion_matrix_query_{mode}.png')
        plt.savefig(save_cm_query_path)

    # print("===> Evaluate clustering")
    # print(f'======> K-means clsutering ({mode})')
    # kmeans = KMeans(n_clusters=30, n_init="auto")
    # kmeans.fit(info_all[feat_dic[mode]].cpu().numpy())
    # save_clustering_results(mode, cfg, info_all, kmeans.labels_.reshape(-1, 1))

    # print("===> Evaluate 2D visualization")
    # feat_dic = {'people':'individual_features', 'scene':'group_features'}
    # mode_list = ['scene']
    # for mode in mode_list:
    #     print(f'======> TSNE analysis ({mode})')
    #     tsne = TSNE(n_components=2, random_state=41)
    #     test_tsne = tsne.fit_transform(info_all[feat_dic[mode]].cpu().numpy())
    #     query_type = cfg.query_type
    #     visualize_features_query(mode, cfg, info_all, test_tsne, True, query_type)
    #     visualize_features_query(mode, cfg, info_all, test_tsne, False, query_type)
    #     visualize_features(mode, cfg, info_all, test_tsne, True)
    #     visualize_features(mode, cfg, info_all, test_tsne, False)
    #     visualize_features_wo_ga(mode, cfg, info_all, test_tsne)
    #     save_visualized_features(mode, cfg, info_all, test_tsne)

def evaluate_jae_metrics(mode, info_train, info_test, metrics_dic, cfg):
    print("===> Calculate distance between train and test features")
    feat_dic = {'people':'individual_features', 'scene':'group_features'}
    match_score = torch.cdist(info_test[feat_dic[mode]], info_train[feat_dic[mode]], p=2)
    match_idx = torch.argmin(match_score, dim=-1)

    print("===> Calculate the joint attention distance between matched samples")
    gt_ja_all_train = info_train['gt_ja_all']
    gt_ja_all_test = info_test['gt_ja_all']
    gt_ja_all_matched = gt_ja_all_train[match_idx]
    ja_dist_matched = F.l1_loss(gt_ja_all_matched, gt_ja_all_test)
    ja_dist_key = f'JA matching error ({mode})'
    metrics_dic[ja_dist_key] = ja_dist_matched.item()
    print(f'{ja_dist_key}', ja_dist_matched.item())

    print("===> Calculate distance between estimated and groun-truth joint attention")
    if 'estimated_ja_all' in info_test:
        estimated_ja_all = info_test['estimated_ja_all']
        ja_dist = F.l1_loss(estimated_ja_all, gt_ja_all_test)
        ja_dist_key = f'JA estimation error ({mode})'
        metrics_dic[ja_dist_key] = ja_dist.item()
        print(f'{ja_dist_key}', ja_dist.item())

def save_gafs(info_all, cfg, data_mode='all'):
    info_all_gaf_scene = info_all['group_features']
    save_gaf_path_scene = os.path.join(cfg.result_path, f'eval_gaf_{data_mode}_scene.npy')
    np.save(save_gaf_path_scene, info_all_gaf_scene.cpu().numpy())

    info_all_gaf_people = info_all['individual_features']
    save_gaf_path_people = os.path.join(cfg.result_path, f'eval_gaf_{data_mode}_people.npy')
    np.save(save_gaf_path_people, info_all_gaf_people.cpu().numpy())

    info_all_vid_id = info_all['video_id_all']
    save_vid_id_path = os.path.join(cfg.result_path, f'eval_vid_id_{data_mode}.npy')
    np.save(save_vid_id_path, info_all_vid_id)

def eval_net(cfg):
    """
    evaluating gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list

    cfg.use_debug = False
    # cfg.use_debug = True

    # Show config parameters
    cfg.init_config()
    show_config(cfg)
    
    # Reading dataset
    training_set, validation_set, all_set = return_dataset(cfg)
    cfg.num_boxes = all_set.get_num_boxes_max()
    
    params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 4, # 4,
    }
    params['batch_size']=cfg.test_batch_size
    training_loader=data.DataLoader(training_set, **params)
    validation_loader=data.DataLoader(validation_set, **params)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    cfg.device = device
    
    # Build model and Load trained model
    gcnnet_list={
                 'group_relation_volleyball':GroupRelation_volleyball,
                 'group_relation_collective':GroupRelation_volleyball,
                }
    model = gcnnet_list[cfg.inference_module_name](cfg)
    state_dict = torch.load(cfg.stage4model)['state_dict']

    new_state_dict=OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print_log(cfg.log_path, f'Loading stage{cfg.eval_stage} model: ' + cfg.stage4model)

    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)
    model=model.to(device=device)

    # pack models
    models = {'model': model}
    
    test_list={'volleyball':test_volleyball, 'collective':test_collective, 'basketball':test_collective}
    test=test_list[cfg.dataset_name]
    # info_all = test(all_set, all_loader, models, device, 0, cfg)
    info_train = test(training_set, training_loader, models, device, 0, cfg)
    info_test = test(validation_set, validation_loader, models, device, 0, cfg)

    print("======> Evaluation")
    cfg.eval_mask_num = int(cfg.eval_mask_num)

    print("======> Save GAFs")
    # save_gafs(info_all, cfg, 'all')
    save_gafs(info_train, cfg, 'train')
    save_gafs(info_test, cfg, 'test')

    print("======> Evaluate gar metrics")
    gar_metrics_dic = {}
    evaluate_gar_metrics('people', info_train, info_test, gar_metrics_dic, cfg)
    evaluate_gar_metrics('scene', info_train, info_test, gar_metrics_dic, cfg)

    print("======> Save gar metrics")
    save_gar_metrics_path = os.path.join(cfg.result_path, f'eval_gar_metrics_{cfg.query_type}.json')
    with open(save_gar_metrics_path, 'w') as f:
        json.dump(gar_metrics_dic, f, indent=4)

def test_volleyball(data_set, data_loader, models, device, epoch, cfg):
    model = models['model']
    model.eval()

    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    activities_conf = ConfusionMeter(cfg.num_activities)
    actions_conf = ConfusionMeter(cfg.num_actions)
    epoch_timer=Timer()

    actions_in_all = torch.zeros(len(data_set), cfg.num_boxes, device=device)
    actions_labels_all = torch.zeros(len(data_set), cfg.num_boxes, device=device)
    activities_in_all = torch.zeros(len(data_set), device=device)
    locations_in_all = torch.zeros(len(data_set), cfg.num_boxes, 2, device=device)
    video_id_all = ['0' for i in range(len(data_set))]
    estimated_ja_all = torch.zeros(len(data_set), cfg.num_frames, 2, device=device)
    gt_ja_all = torch.zeros(len(data_set), cfg.num_frames, 2, device=device)
    query_all = torch.zeros(len(data_set), device=device)
    query_people_all = torch.zeros(len(data_set), cfg.num_boxes, device=device)
    query_people_all_rec = torch.zeros(len(data_set), cfg.num_boxes, 2, device=device)

    if cfg.feature_adapt_type in ['line', 'mlp']:
        individual_features = torch.zeros(len(data_set), cfg.num_boxes*cfg.num_features_boxes*2, device=device)
        group_features = torch.zeros(len(data_set), cfg.feature_adapt_dim, device=device)
    else:
        individual_features = torch.zeros(len(data_set), cfg.num_boxes*cfg.num_features_boxes*2, device=device)
        group_features = torch.zeros(len(data_set), cfg.num_features_boxes*2, device=device)

    if cfg.feature_adapt_type == 'disent':
        group_features_key = torch.zeros(len(data_set), cfg.feature_adapt_dim, device=device)
        group_features_non_key = torch.zeros(len(data_set), cfg.feature_adapt_dim, device=device)
    
    with torch.no_grad():
        for batch_idx, batch_data_test in enumerate(tqdm(data_loader)):
            for key in batch_data_test.keys():
                if torch.is_tensor(batch_data_test[key]):
                    batch_data_test[key] = batch_data_test[key].to(device=device)

            images_in = batch_data_test['images_in']
            batch_size, num_frames, _, _, _ = images_in.shape
            locations_in = batch_data_test['boxes_in'].reshape((batch_size,num_frames,cfg.num_boxes, 4))
            actions_in = batch_data_test['actions_in'].reshape((batch_size,num_frames,cfg.num_boxes))
            activities_in = batch_data_test['activities_in'].reshape((batch_size,num_frames))
            actions_in=actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
            activities_in=activities_in[:,0].reshape((batch_size,))
            query_labels = batch_data_test['user_queries'][:, 0]
            query_people_labels = batch_data_test['user_queries_people'][:, 0]
            input_data = batch_data_test
            ret = model(input_data)

            # save features
            start_batch_idx = batch_idx*cfg.batch_size
            finish_batch_idx = start_batch_idx+batch_size
            individual_features[start_batch_idx:finish_batch_idx] = ret['individual_feat'].reshape(batch_size, -1)
            group_features[start_batch_idx:finish_batch_idx] = ret['group_feat'].reshape(batch_size, -1)
            actions_in_all[start_batch_idx:finish_batch_idx] = actions_in.reshape(batch_size, cfg.num_boxes)
            activities_in_all[start_batch_idx:finish_batch_idx] = activities_in.reshape(batch_size)
            locations_in_all[start_batch_idx:finish_batch_idx, :, 0] = (locations_in[0, num_frames//2, :, 0]+locations_in[0, num_frames//2, :, 2])/2
            locations_in_all[start_batch_idx:finish_batch_idx, :, 1] = (locations_in[0, num_frames//2, :, 1]+locations_in[0, num_frames//2, :, 3])/2
            video_id_all[start_batch_idx:finish_batch_idx] = batch_data_test['video_id']
            query_all[start_batch_idx:finish_batch_idx] = query_labels
            query_people_all[start_batch_idx:finish_batch_idx] = query_people_labels
            print(f'start_batch_idx: {start_batch_idx}, finish_batch_idx: {finish_batch_idx}')
            if cfg.use_debug and batch_idx > 3:
                break
            
    test_info={
        'time':epoch_timer.timeit(),
        'activities_acc':activities_meter.avg*100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
        'actions_acc':actions_meter.avg*100,
        'actions_conf':actions_conf.value(),
        'actions_MPCA':MPCA(actions_conf.value()),
        'individual_features':individual_features,
        'group_features':group_features,
        'actions_in_all':actions_in_all,
        'actions_labels_all':actions_labels_all,
        'activities_in_all':activities_in_all,
        'locations_in_all':locations_in_all,
        'video_id_all':video_id_all,
        'estimated_ja_all': estimated_ja_all,
        'gt_ja_all': gt_ja_all,
        'query_all': query_all,
        'query_people_all': query_people_all,
    }

    return test_info

def test_collective(data_set, data_loader, models, device, epoch, cfg):
    model = models['model']
    model.eval()
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    
    epoch_timer=Timer()
    activities_conf = ConfusionMeter(cfg.num_activities)
    actions_conf = ConfusionMeter(cfg.num_actions)

    actions_in_all = torch.zeros(len(data_set), cfg.num_boxes, device=device)
    actions_labels_all = torch.zeros(len(data_set), cfg.num_boxes, device=device)
    activities_in_all = torch.zeros(len(data_set), device=device)
    locations_in_all = torch.zeros(len(data_set), cfg.num_boxes, 2, device=device)
    video_id_all = ['0' for i in range(len(data_set))]
    individual_features = torch.zeros(len(data_set), cfg.num_features_boxes*2*cfg.num_boxes, device=device)
    group_features = torch.zeros(len(data_set), cfg.num_features_boxes*2, device=device)
    query_all = torch.zeros(len(data_set), device=device)

    with torch.no_grad():
        for batch_idx, batch_data_test in enumerate(tqdm(data_loader)):
            for key in batch_data_test.keys():
                if torch.is_tensor(batch_data_test[key]):
                    batch_data_test[key] = batch_data_test[key].to(device=device)

            images_in = batch_data_test['images_in']
            batch_size, num_frames, _, _, _ = images_in.shape
            locations_in = batch_data_test['boxes_in'].reshape((batch_size,num_frames,cfg.num_boxes, 4))
            actions_in = batch_data_test['actions_in'].reshape((batch_size,num_frames,cfg.num_boxes))
            activities_in = batch_data_test['activities_in'].reshape((batch_size,num_frames))
            bboxes_num = batch_data_test['bboxes_num']
            input_data = batch_data_test
            ret = model(input_data)

            actions_in_nopad=[]
            for b in range(batch_size):
                N=bboxes_num[b][0]
                actions_in_nopad.append(actions_in[b][0][:N])
            actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            activities_in=activities_in[:,0].reshape(batch_size,)

            # save features
            start_batch_idx = batch_idx*cfg.batch_size
            finish_batch_idx = start_batch_idx+batch_size
            individual_features[start_batch_idx:finish_batch_idx] = ret['individual_feat']
            group_features[start_batch_idx:finish_batch_idx] = ret['group_feat']
            actions_in_all[start_batch_idx:finish_batch_idx] = batch_data_test['actions_in'].reshape((batch_size,num_frames,cfg.num_boxes))[:,0,:].reshape((batch_size, cfg.num_boxes,))
            activities_in_all[start_batch_idx:finish_batch_idx] = activities_in.reshape(batch_size,)
            locations_in_all[start_batch_idx:finish_batch_idx, :, 0] = (locations_in[0, num_frames//2, :, 0]+locations_in[0, num_frames//2, :, 2])/2
            locations_in_all[start_batch_idx:finish_batch_idx, :, 1] = (locations_in[0, num_frames//2, :, 1]+locations_in[0, num_frames//2, :, 3])/2
            video_id_all[start_batch_idx:finish_batch_idx] = batch_data_test['video_id']
            query_all[start_batch_idx:finish_batch_idx] = batch_data_test['user_queries'][:, 0]
            print(f'start_batch_idx: {start_batch_idx}, finish_batch_idx: {finish_batch_idx}')

            # if 'recognized_ga' in ret.keys():
                # recognized_ga_prob = ret['recognized_ga']
                # recognized_ga_label = recognized_ga_prob.argmax(dim=-1)
                # print(f'recognized_ga_label: {recognized_ga_label}')
                # print(f'activities_in: {activities_in}')

            # if 'recog_query' in ret.keys():
            #     recognized_query_prob = ret['recog_query']
            #     recognized_query_label = recognized_query_prob.argmax(dim=-1)
            #     print(f'recognized_query_label: {recognized_query_label}')
            #     print(f'query_all: {query_all[start_batch_idx:finish_batch_idx]}')

            # if 'estimated_ja' in ret.keys():
                # estimated_ja = ret['estimated_ja']
                # gt_ja = batch_data_test['gt_ja']
                # print('estimated_ja', estimated_ja.shape)
                # print('gt_ja', gt_ja.shape)

            # if start_batch_idx > 20:
                # break

            # batch_idx += 1
            if cfg.use_debug and batch_idx > 3:
                break

    test_info={
        'time':epoch_timer.timeit(),
        'activities_acc':activities_meter.avg*100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
        'actions_acc':actions_meter.avg*100,
        'actions_conf':actions_conf.value(),
        'actions_MPCA':MPCA(actions_conf.value()),
        'individual_features':individual_features,
        'group_features':group_features,
        'actions_in_all':actions_in_all,
        'actions_labels_all':actions_labels_all,
        'activities_in_all':activities_in_all,
        'locations_in_all':locations_in_all,
        'video_id_all':video_id_all,
        'query_all': query_all,
    }

    return test_info