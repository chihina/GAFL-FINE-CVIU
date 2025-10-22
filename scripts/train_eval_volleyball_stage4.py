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
parser.add_argument('-key_person_det_interval', help='key_person_det_interval', default=None)
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
parser.add_argument('-key_person_mode', help='key_person_mode', default='mask_one')
parser.add_argument('-anchor_thresh_type', help='anchor_thresh_type', default='ratio')
parser.add_argument('-train_smp_type', help='training_sampling_type', default='max')
parser.add_argument('-load_backbone_stage4', help='load_backbone_stage4', default=True)

args = parser.parse_args()
device_list = str(args.gpu)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb
import pickle

from train_net_stage4_gr import train_net, Config
from eval_net_stage4_gr import *
from detect_key_person_all import detect_key_person
from config_utils import update_config_all

dataset = 'volleyball'
cfg=Config(dataset)
cfg.dataset = dataset

cfg.use_app_feat_type = 'vgg'
cfg.use_ind_feat = 'loc_and_app'
cfg.use_grp_feat = 'hbb'

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
cfg.metric_lr_margin = int(args.metric_lr_margin)
cfg.key_person_mode = args.key_person_mode
cfg.anchor_thresh_type = args.anchor_thresh_type
cfg.train_smp_type = args.train_smp_type
cfg.load_backbone_stage4 = args.load_backbone_stage4 == 'True'

cfg.jae_ann_dir = os.path.join('data_local', 'vatic_ball_annotation', 'annotation_data_sub')
cfg.net_det_dir = os.path.join('data_local', 'volleyball_net_detection')
cfg.line_det_dir = os.path.join('data_local', 'volleyball_line_detection')
cfg.use_act_loss = False
cfg.use_pose_loss = False
cfg.use_recon_diff_loss = False
cfg.use_jae_loss = False
cfg.use_jae_type = 'gt'
cfg.use_recon_pose_feat_loss = False
cfg.use_recon_pose_coord_loss = False
cfg.use_traj_loss = False
cfg.use_person_num_loss = False
cfg.model_exp_name = 'train_volleyball_stage4'

# define the setting for query
cfg.query_dir = os.path.join('data_local', 'volleyball_query')
cfg.query_type = args.query

# define the setting for active learning
cfg.use_active_ln = False
# cfg.use_active_ln = True

# define the setting for network and training
if cfg.feature_adapt_dim == 2048:
    cfg.feature_adapt_residual = True
else:
    cfg.feature_adapt_residual = False

cfg.final_head_mid_adapt_num = 2
# cfg.final_head_mid_adapt_num = 0

if cfg.key_person_mode == 'mask_one':
    cfg.key_person_mode_symbol = 'mo'
elif cfg.key_person_mode == 'mask_zero':
    cfg.key_person_mode_symbol = 'mz'

if cfg.use_anchor_type == 'normal':
    cfg.use_anchor_type_symbol = 'na'
elif cfg.use_anchor_type == 'pruning':
    cfg.use_anchor_type_symbol = 'pr'
elif cfg.use_anchor_type == 'pruning_p2p':
    cfg.use_anchor_type_symbol = 'prp2p'
elif cfg.use_anchor_type == 'pruning_p2g':
    cfg.use_anchor_type_symbol = 'prp2g'
elif cfg.use_anchor_type == 'pruning_p2g_cos':
    cfg.use_anchor_type_symbol = 'prp2g'
elif cfg.use_anchor_type == 'pruning_p2g_inner_cos':
    cfg.use_anchor_type_symbol = 'prp2g_in'
elif cfg.use_anchor_type == 'pruning_gt':
    cfg.use_anchor_type_symbol = 'prgt'
else:
    assert False, 'Not implemente use_anchor_type'

cfg.use_pruning_ratio = cfg.all_pruning_ratio
cfg.use_key_person_ratio = cfg.all_pruning_ratio

cfg.pruning_mode = 'static'
# cfg.pruning_mode = 'dynamic'

cfg.pruning_decay = 0.1
cfg.pruning_interval = 10
cfg.pruning_ratio_max = 0.6

# additional setup for stage4
# cfg = update_config_all(cfg)
model_exp_name_stage2 = cfg.model_exp_name_stage2
cfg.stage2model = os.path.join('result', model_exp_name_stage2, 'best_model.pth')
cfg.stage2model_dir = os.path.join('result', model_exp_name_stage2)

cfg_stage2_pickle_path = os.path.join('result', model_exp_name_stage2, 'cfg.pickle')
print(f"===> Load cfg from {cfg_stage2_pickle_path}")
with open(cfg_stage2_pickle_path, 'rb') as f:
    cfg_stage2 = pickle.load(f)

if 'mode' in dir(cfg_stage2):
    cfg.mode = cfg_stage2.mode
else:
    cfg.mode = 'PAF'

if cfg.mode == 'PAC':
    cfg.train_backbone = True
    cfg.use_act_loss = True
    cfg.use_pos_type = 'absolute'
    cfg.batch_size = 16
    mk_num = 5
elif cfg.mode == 'PAF':
    cfg.train_backbone = False
    cfg.use_pos_type = 'absolute'
    cfg.batch_size = 16
    mk_num = 6
elif cfg.mode == 'PPF_JAE':
    cfg.train_backbone = False
    cfg.use_pos_type = 'absolute'
    cfg.batch_size = 4
    mk_num = 6
    cfg.use_jae_type = 'pred'
    cfg.use_grp_feat = 'hbb_whole'
    cfg.use_ind_feat_crop = 'roi_multi'
    cfg.jae_ann_dir = os.path.join('data_local', 'vatic_ball_annotation', 'annotation_data_sub')
    cfg.jae_pred_dir = os.path.join('data_local', 'volleyball_wasb')
    pose_model_name = 'pose_hrnet_w32_256x192.pth'
    pose_model_inp_size = pose_model_name.split('.')[0].split('_')[-1].split('x')
    cfg.person_size = int(pose_model_inp_size[0]), int(pose_model_inp_size[1])
    cfg.pose_model_path = os.path.join('weights', pose_model_name)
    cfg.pose_cfg_path = os.path.join('weights', 'inference-config.yaml')

cfg.random_mask_type = f'random_to_{mk_num}'
cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 4
cfg.test_before_train = False
cfg.test_interval_epoch = 3
cfg.image_size = 320, 640
cfg.wandb_loss_list = ['activities_acc', 'activities_conf', 'activities_MPCA',
                        'actions_acc', 'actions_conf', 'actions_MPCA', 
                        'loss', 
                        'loss_act', 'loss_recon',
                        'loss_recon_pose_feat', 'loss_recon_pose_coord',
                        'loss_query',
                        'loss_proximity', 'loss_distant',
                        'loss_distill',
                        'loss_recon_key', 'loss_recon_non_key', 'loss_key_cos_sim',
                        'loss_key_recog',
                        'loss_ga_recog', 'loss_ia_recog',
                        'loss_maintain',
                        'loss_metric_lr',
                        ]

# vgg16 setup
cfg.backbone = 'vgg16'
cfg.stage1_model_path = 'result/[GR ours_stage1]<2023-07-07_23-22-44>/stage1_epoch10_0.59%.pth'
cfg.out_size = 10, 20
cfg.emb_features = 512

cfg.eval_only = False
cfg.eval_mask_num = 0
cfg.old_act_rec = False
cfg.test_batch_size = cfg.batch_size
# cfg.num_frames = 10

cfg.lr_plan = {500: 3e-5}
cfg.actions_weights = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]

# stage2 setup
cfg.use_res_connect = False
# cfg.use_trans = False
cfg.use_trans = True
cfg.use_same_enc_dual_path = False
# cfg.use_same_enc_dual_path = True
cfg.trans_head_num = 1
cfg.trans_layer_num = 1

cfg.use_ind_feat_crop = 'roi_multi'
cfg.use_ind_feat = 'loc_and_app'
cfg.people_pool_type = 'max'
# cfg.people_pool_type = 'we_mean'
# cfg.pool_weight_loss_coef = 5e-2

cfg.use_pos_cond = True
cfg.use_tmp_cond = False
# cfg.use_tmp_cond = True
cfg.final_head_mid_num = 2
cfg.use_gen_iar = False
cfg.gen_iar_ratio = 0.0

# Dual AI setup
cfg.inference_module_name = 'group_relation_volleyball'

cfg.exp_note = f'VOL_GAFL_{cfg.mode}'
cfg.exp_note += f'_{cfg.query_type}_{cfg.query_init}'
cfg.exp_note += f'_int_{cfg.key_person_det_interval}'
cfg.exp_note += f'_{cfg.sampling_mode}_{cfg.train_smp_type}_{cfg.non_query_init}'
cfg.exp_note += f'_sd{cfg.train_random_seed}_{cfg.feature_adapt_type}_{cfg.train_learning_rate}_ep_{cfg.max_epoch}'

if cfg.use_random_mask:
    cfg.exp_note += f'_w_{cfg.random_mask_type}'

if cfg.feature_adapt_type in ['line', 'mlp']:
    cfg.exp_note += f'_{cfg.feature_adapt_dim}'
    if cfg.feature_adapt_residual:
        cfg.exp_note += '_w_res'

if cfg.use_recon_loss:
    cfg.exp_note += f'_w_RE_{cfg.recon_loss_type}'
    if cfg.feature_adapt_type == 'disent':
        if cfg.use_recon_loss_key:
            cfg.exp_note += '_w_KL'
        else:
            cfg.exp_note += '_wo_KL'
else:
    cfg.exp_note += '_wo_RE'

if cfg.use_act_loss:
    cfg.exp_note += '_w_RE_act'

if cfg.use_key_recog_loss or cfg.recon_loss_type == 'key' or cfg.use_disentangle_loss:
    cfg.exp_note += f'_w_KD'
    if cfg.use_key_recog_loss:
        cfg.exp_note += f'PRED'
    elif cfg.recon_loss_type == 'key':
        cfg.exp_note += f'KEY'
    elif cfg.use_disentangle_loss:
        cfg.exp_note += f'DIL'

    if cfg.use_key_person_type in ['det', 'det_semi']:
        if cfg.use_key_person_type == 'det':
            cfg.exp_note += f'DET_{cfg.key_person_mode_symbol}'
        elif cfg.use_key_person_type == 'det_semi':
            cfg.exp_note += f'DET_SEMI_{cfg.key_person_mode_symbol}'
        elif cfg.use_key_person_type == 'gt':
            cfg.exp_note += f'GT_{cfg.key_person_mode_symbol}'
        
        if cfg.use_key_person_loss_func in ['bce', 'bce_we']:
            cfg.exp_note += f'_{cfg.use_key_person_ratio}'
        if 'pruning' in cfg.use_anchor_type:
            cfg.exp_note += f'_{cfg.use_anchor_type_symbol}_{cfg.anchor_thresh_type}'
            if cfg.use_anchor_type in ['pruning_p2p', 'pruning_p2g', 'pruning_p2g_cos', 'pruning_p2g_inner_cos']:
                if cfg.pruning_mode == 'dynamic':
                    cfg.exp_note += f'd_{cfg.use_pruning_ratio}_{cfg.pruning_ratio_max}_{cfg.pruning_decay}_{cfg.pruning_interval}'
                elif cfg.pruning_mode == 'static':
                    cfg.exp_note += f's_{cfg.use_pruning_ratio}'
        elif 'normal' in cfg.use_anchor_type:
            cfg.exp_note += f'_{cfg.use_anchor_type_symbol}_{cfg.use_pruning_ratio}'

    elif cfg.use_key_person_type == 'gt':
        cfg.exp_note += 'GT'
        if cfg.use_individual_action_type == 'det':
            cfg.exp_note += '_DetIA'
        elif cfg.use_individual_action_type == 'gt_action':
            cfg.exp_note += '_IA'
        elif cfg.use_individual_action_type == 'gt_grouping':
            cfg.exp_note += '_GR'

if cfg.use_key_recog_loss:
    cfg.exp_note += f'_w_KDL_{cfg.use_key_person_loss_func}_{cfg.key_recog_feat_type}'

if cfg.use_proximity_loss:
    cfg.exp_note += '_w_PR'

if cfg.use_query_classfication_loss:
    cfg.exp_note += '_w_QCL'

if cfg.use_ga_recog_loss:
    cfg.exp_note += '_w_GAR'

if cfg.use_ia_recog_loss:
    cfg.exp_note += '_w_IAR'

if cfg.use_maintain_loss:
    cfg.exp_note += '_w_ML'

if cfg.use_metric_lr_loss:
    cfg.exp_note += f'_w_METL_{cfg.metric_lr_margin}'

if cfg.load_backbone_stage4 and cfg.freeze_backbone_stage4:
    cfg.exp_note += '_w_BBF'
elif cfg.load_backbone_stage4 and not cfg.freeze_backbone_stage4:
    cfg.exp_note += '_w_BB'
elif not cfg.load_backbone_stage4 and cfg.freeze_backbone_stage4:
    assert False, 'Not implemented tge combination of not load and freeze'
else:
    cfg.exp_note += '_wo_BB'

print("===> Generate wandb system")
wandb.login()
wandb.init(project=f"DIN-Group-Activity-Recognition-Benchmark-{dataset}-Active",
            name=f'{cfg.exp_note}_stage4', 
            config=cfg,
            settings=wandb.Settings(start_method='fork'),
            )

cfg = train_net(cfg)
print('===> Finish training stage4')

print('===> Start evaluating stage4')
trained_stage4model = os.path.join(cfg.result_path, 'best_model.pth')
cfg_pickle_path = os.path.join(cfg.result_path, 'cfg.pickle')
with open(cfg_pickle_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.model_exp_name = cfg.result_path.split('/')[-1]
cfg.stage4model = trained_stage4model
cfg = update_config_all(cfg)
cfg.eval_only = True
cfg.eval_stage = 4

dataset = 'volleyball'
cfg_original=Config(dataset)
cfg.train_seqs = cfg_original.train_seqs
cfg.test_seqs = cfg_original.test_seqs

cfg.batch_size = 16
cfg.test_batch_size = cfg.batch_size
cfg.num_frames = 10
cfg.eval_mask_num = 0
cfg.eval_mask_action = False
cfg.dataset_symbol = 'vol'
eval_net(cfg)
print('===> Finish evaluating stage4')