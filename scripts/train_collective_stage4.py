import sys
sys.path.append(".")
device_list = '0'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb
import pickle

from train_net_stage4_gr import train_net, Config

dataset = 'collective'
cfg=Config(dataset)

cfg.jae_ann_dir = os.path.join('data_local', 'vatic_ball_annotation', 'annotation_data_sub')
cfg.use_recon_loss = False
cfg.use_act_loss = False
cfg.use_pose_loss = False
cfg.use_recon_diff_loss = False
cfg.use_jae_loss = False

# define the setting for query
cfg.query_dir = os.path.join('data_local', 'collective_query')
# cfg.query_type = 'walking_three'
cfg.query_type = 'Walking'

cfg.use_query_loss = False
# cfg.use_query_loss = True
cfg.query_margin = 10

# define the setting for sampling
cfg.query_init = 3
# cfg.query_init = 10
cfg.non_query_init = 0
# cfg.non_query_init = 3
cfg.use_sampling = False
# cfg.use_sampling = True
# cfg.sampling_strategy = 'rand'
cfg.sampling_strategy = 'dist'
cfg.sampling_num_max = 30
cfg.sampling_num_iter = 5

# define the setting for active learning
cfg.use_active_ln = False
# cfg.use_active_ln = True

# define the setting for network and training
cfg.feature_adapt_type = 'ft'
# cfg.feature_adapt_type = 'fix_line'

# define the setting for person appearance prediction
cfg.use_recon_loss = False
# cfg.use_recon_loss = True

cfg.use_recon_mask = False
# cfg.use_recon_mask = True
cfg.recon_mask_no_action = ['spiking']

# cfg.load_backbone_stage4 = False
cfg.load_backbone_stage4 = True

cfg.update_iter = 50

# define the setting for propose grouping loss
# cfg.use_proximity_loss = False
cfg.use_proximity_loss = True
cfg.proximity_person_num = 2

cfg.use_distant_loss = False
# cfg.use_distant_loss = True

# additional setup for stage4
model_exp_name = '[CAD GR ours recon rand mask 0_stage2]<2023-10-19_22-26-11>'
cfg.stage2model = os.path.join('result', model_exp_name, 'best_model.pth')

cfg_stage2_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
print(f"===> Load cfg from {cfg_stage2_pickle_path}")
with open(cfg_stage2_pickle_path, 'rb') as f:
    cfg_stage2 = pickle.load(f)

if 'mode' in dir(cfg_stage2):
    cfg.mode = cfg_stage2.mode
else:
    cfg.mode = 'PAF'

if cfg.mode == 'PAC':
    cfg.train_backbone = True
    cfg.load_backbone_stage2 = True
    cfg.use_act_loss = True
    cfg.max_epoch = 300
    cfg.batch_size = 8
elif cfg.mode == 'PAF':
    cfg.train_backbone = False
    cfg.load_backbone_stage2 = False
    cfg.max_epoch = 100
    cfg.batch_size = 16
    cfg.use_pos_type = 'absolute'

cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 4
cfg.test_before_train = False
cfg.test_interval_epoch = 3
cfg.image_size = 240, 360
cfg.wandb_loss_list = ['activities_acc', 'activities_conf', 'activities_MPCA',
                        'actions_acc', 'actions_conf', 'actions_MPCA', 
                        'loss', 
                        'loss_act', 'loss_recon',
                        'loss_query',
                        'loss_proximity', 'loss_distant',
                        ]

# vgg16 setup
cfg.backbone = 'inv3'
cfg.stage1_model_path = 'result/[CAD GR ours_stage1]<2023-07-09_21-47-14>/stage1_epoch17_0.92%.pth'
cfg.out_size=57,87

cfg.eval_only = False
cfg.old_act_rec = False
# cfg.test_batch_size = 1
cfg.test_batch_size = cfg.batch_size
cfg.num_boxes = 13
cfg.num_actions = 5
cfg.num_activities = 4
cfg.num_frames = 10

cfg.train_learning_rate = 1e-4
# cfg.train_learning_rate = 1e-5
# cfg.train_learning_rate = 1e-3
cfg.lr_plan = {500: 3e-5}

# stage2 setup
cfg.use_res_connect = False
cfg.use_trans = True
cfg.use_same_enc_dual_path = False
cfg.trans_head_num = 1
cfg.trans_layer_num = 1

cfg.use_ind_feat_crop = 'roi_multi'
cfg.use_ind_feat = 'loc_and_app'
cfg.people_pool_type = 'max'
cfg.use_pos_cond = True
cfg.use_tmp_cond = False
# cfg.use_tmp_cond = True
cfg.final_head_mid_num = 2
cfg.use_gen_iar = False
cfg.gen_iar_ratio = 0.0

cfg.use_random_mask = False
# cfg.use_random_mask = True
mk_num = 6
cfg.random_mask_type = f'random_to_{mk_num}'

cfg.inference_module_name = 'group_relation_collective'

cfg.exp_note = f'CAD_GAFL_PAF'
cfg.exp_note += f'_{cfg.feature_adapt_type}_{cfg.query_type}_{cfg.query_init}_{cfg.non_query_init}_{cfg.update_iter}'

if cfg.use_recon_loss:
    cfg.exp_note += f'_w_RE'
    if cfg.use_recon_mask:
        cfg.exp_note += '_FI'
else:
    cfg.exp_note += '_wo_RE'

if cfg.load_backbone_stage4:
    cfg.exp_note += '_w_bblr'
else:
    cfg.exp_note += '_wo_bblr'

if cfg.use_sampling:
    cfg.exp_note += f'_w_S_{cfg.sampling_num_max}_{cfg.sampling_num_iter}_{cfg.sampling_strategy}'
else:
    cfg.exp_note += '_wo_S'

if cfg.use_query_loss:
    cfg.exp_note += f'_w_QL_{cfg.query_margin}'
else:
    cfg.exp_note += '_wo_QL'

if cfg.use_proximity_loss:
    cfg.exp_note += f'_w_PL_{cfg.proximity_person_num}'
else:
    cfg.exp_note += '_wo_PL'

if cfg.use_distant_loss:
    cfg.exp_note += '_w_DL'
else:
    cfg.exp_note += '_wo_DL'

if cfg.use_active_ln:
    cfg.exp_note += f'_w_AL'
else:
    cfg.exp_note += '_wo_AL'

print("===> Generate wandb system")
wandb.login()
wandb.init(project=f"DIN-Group-Activity-Recognition-Benchmark-{dataset}-Active",
            name=f'{cfg.exp_note}_stage4', 
            config=cfg,
            settings=wandb.Settings(start_method='fork'),
            )

train_net(cfg)
