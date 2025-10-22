import openpyxl
from openpyxl.styles import PatternFill
import pandas as pd
import numpy as np
import os
import sys
import json
import shutil
import glob

def extract_values(use_col_names, use_vals, saved_result_dir, model_exp_name):
    eval_json_path = os.path.join(saved_result_dir, model_exp_name, f'eval_gar_metrics.json')
    with open(eval_json_path, 'r') as f:
        results = json.load(f)

    use_vals_model = []
    for use_col_name in use_col_names:
        use_vals_model.append(results[use_col_name])
    return use_vals_model

def gen_col_names(use_mode, use_k_max, dataset_name='VOL'):
    use_col_names = []

    if dataset_name != 'BSK':
        # add action IoU
        use_col_names_action_iou = [f'GAR accuracy hit@{i} (action iou jaccard) ({use_mode})' for i in range(1, use_k_max + 1)]
        use_col_names.extend(use_col_names_action_iou)
        use_col_names.append(f'mAP (action iou jaccard) ({use_mode})')
        
        use_col_names_action_iou_tfidf = [f'GAR accuracy hit@{i} (action iou tfidf) ({use_mode})' for i in range(1, use_k_max + 1)]
        use_col_names.extend(use_col_names_action_iou_tfidf)
        use_col_names.append(f'mAP (action iou tfidf) ({use_mode})')

    # add GA accuracy
    use_col_names_gar = [f'GAR accuracy hit@{i} (group activity) ({use_mode})' for i in range(1, use_k_max + 1)]
    use_col_names.extend(use_col_names_gar)

    if dataset_name == 'BSK':
        use_col_names_gar = [f'GAR accuracy precision@{i} (group activity) ({use_mode})' for i in range(1, use_k_max + 1)]
        use_col_names.extend(use_col_names_gar)

    return use_col_names

def gen_col_names_save(use_k_max, dataset_name='VOL'):
    use_col_names_save = []

    if dataset_name != 'BSK':
        # action IoU (save)
        use_col_names_action_iou_save = [f'Hit@{i} (action IoU)' for i in range(1, use_k_max + 1)]
        use_col_names_save.extend(use_col_names_action_iou_save)
        use_col_names_save.append(f'mAP (action IoU)')
        # action IoU tfidf (save)
        use_col_names_action_iou_tfidf_save = [f'Hit@{i} (action IoU tfidf)' for i in range(1, use_k_max + 1)]
        use_col_names_save.extend(use_col_names_action_iou_tfidf_save)
        use_col_names_save.append(f'mAP (action IoU tfidf)')

    # GA accuracy (save)
    use_col_names_gar_save = [f'Hit@{i} (GA)' for i in range(1, use_k_max + 1)]
    use_col_names_save.extend(use_col_names_gar_save)

    if dataset_name == 'BSK':
        # GA accuracy precision (save)
        use_col_names_gar_save = [f'Precision@{i} (GA)' for i in range(1, use_k_max + 1)]
        use_col_names_save.extend(use_col_names_gar_save)

    return use_col_names_save