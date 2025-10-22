import sys
sys.path.append(".")
device_list = '7'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb

from train_net_stage2_gr import *

# mode = 'PAF'

dataset = 'jrdb'
cfg=Config(dataset)

mode = 'PAF'
cfg.mode = mode
cfg.use_pos_type = 'absolute'
cfg.use_recon_loss = False
cfg.use_act_loss = False
cfg.use_activity_loss = False
cfg.use_pose_loss = False
cfg.use_recon_diff_loss = False
cfg.use_jae_loss = False
cfg.use_jae_type = 'gt'
cfg.use_query_loss = False
cfg.use_key_recog_loss = False
cfg.use_recon_loss_key = False
cfg.use_recon_loss_non_key = False
cfg.use_recon_pose_feat_loss = False
cfg.use_recon_pose_coord_loss = False
cfg.use_traj_loss = False
cfg.use_person_num_loss = False
cfg.query_type = 'dummy'
cfg.feature_adapt_type = 'ft'
cfg.freeze_backbone_stage4 = False
cfg.use_individual_action_type = 'gt'
cfg.use_query_classfication_loss = False
cfg.use_ga_recog_loss = False
cfg.use_ia_recog_loss = False

cfg.num_features_boxes = 1024
# cfg.num_features_boxes = 256
# cfg.num_features_boxes = 64
# cfg.num_features_boxes = 32

cfg.use_app_feat_type = 'vgg'
cfg.use_ind_feat = 'loc_and_app'
cfg.use_grp_feat = 'hbb'

# pose_model_name = 'pose_hrnet_w32_384x288.pth'
pose_model_name = 'pose_hrnet_w32_256x192.pth'
pose_model_inp_size = pose_model_name.split('.')[0].split('_')[-1].split('x')
cfg.person_size = int(pose_model_inp_size[0]), int(pose_model_inp_size[1])
cfg.pose_model_path = os.path.join('weights', pose_model_name)
cfg.pose_cfg_path = os.path.join('weights', 'inference-config.yaml')

cfg.load_stage2model = False
# cfg.load_stage2model = True
model_name = '[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>'
cfg.stage2model = f'result/{model_name}/best_model.pth'

if mode == 'PAC':
    cfg.train_backbone = True
    cfg.load_backbone_stage2 = True
    cfg.use_act_loss = True
    cfg.max_epoch = 300
    cfg.batch_size = 8
    cfg.use_ind_feat_crop = 'roi_multi'
elif mode == 'PAF':
    cfg.train_backbone = False
    cfg.load_backbone_stage2 = False
    cfg.use_recon_loss = True
    cfg.max_epoch = 500
    cfg.use_ind_feat_crop = 'roi_multi'
    cfg.batch_size = 16
    cfg.use_grp_feat = 'hbb_whole'
    cfg.batch_size = 8
    # cfg.use_app_feat_type = 'vgg_pose'
else:
    cfg.train_backbone = False
    cfg.load_backbone_stage2 = True
    cfg.use_recon_loss = True
    cfg.use_act_loss = True
    cfg.max_epoch = 100
    cfg.batch_size = 16

cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 2
cfg.test_before_train = False
cfg.test_interval_epoch = 3
# cfg.image_size = 320, 640
cfg.image_size = 112, 882
cfg.wandb_loss_list = ['activities_acc', 'activities_conf', 'activities_MPCA',
                        'actions_acc', 'actions_conf', 'actions_MPCA', 
                        'loss', 
                        'loss_act', 'loss_recon',
                        'loss_recon_pose_feat', 'loss_recon_pose_coord',
                        'loss_jae',
                        'loss_traj',
                        ]

# vgg16 setup
cfg.backbone = 'vgg16'
cfg.stage1_model_path = 'result/[GR ours_stage1]<2023-07-07_23-22-44>/stage1_epoch10_0.59%.pth'
cfg.out_size = 10, 20
cfg.emb_features = 512

cfg.eval_only = False
cfg.eval_mask_num = 0
cfg.old_act_rec = False
cfg.test_batch_size = 1

cfg.train_learning_rate = 1e-4
cfg.lr_plan = {}
cfg.actions_weights = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]

# stage2 setup
cfg.use_res_connect = False
cfg.use_trans = True
cfg.use_same_enc_dual_path = False
cfg.trans_head_num = 1
cfg.trans_layer_num = 1

cfg.num_before = 0
cfg.num_after = 0
cfg.num_frames = 1
cfg.random_sampling = False

cfg.num_actions = 27
cfg.num_activities = 7
cfg.num_social_activities = 32
cfg.num_boxes = 54

cfg.people_pool_type = 'max'
# cfg.use_pos_cond = False
cfg.use_pos_cond = True
# cfg.use_tmp_cond = False
cfg.use_tmp_cond = True
cfg.final_head_mid_num = 2
cfg.use_gen_iar = False
cfg.gen_iar_ratio = 0.0

cfg.use_random_mask = True
if cfg.use_random_mask:
    if mode == 'PAC':
        mk_num = 5
    elif mode == 'PAF':
        mk_num = 6
else:
    mk_num = 0

cfg.random_mask_type = f'random_to_{mk_num}'

# cfg.inference_module_name = 'group_activity_volleyball'

# Dual AI setup
cfg.inference_module_name = 'group_relation_volleyball'

# HIGCIN Inference setup
# cfg.inference_module_name = 'group_relation_higcin_volleyball'
# cfg.crop_size = 7, 7

# Dynamic Inference setup
# cfg.inference_module_name = 'group_relation_din_volleyball'
# cfg.group = 1
# cfg.stride = 1
# cfg.ST_kernel_size = [(3, 3)] #[(3, 3),(3, 3),(3, 3),(3, 3)]
# cfg.dynamic_sampling = True
# cfg.sampling_ratio = [1]
# cfg.lite_dim = 128 # None # 128
# cfg.scale_factor = True
# cfg.beta_factor = False
# cfg.hierarchical_inference = False
# cfg.parallel_inference = False
# cfg.num_DIM = 1
# cfg.train_dropout_prob = 0.3

# cfg.exp_note = f'JRDB_GAFL_{mode}_mask_{mk_num}_{cfg.use_grp_feat}_{cfg.use_app_feat_type}'
cfg.exp_note = f'JRDB_GAFL_{mode}'

print("===> Generate wandb system")
wandb.login()
wandb.init(project=f"DIN-Group-Activity-Recognition-Benchmark-JRDB",
            name=f'{cfg.exp_note}_stage2', 
            config=cfg,
            settings=wandb.Settings(start_method='fork'),
            )

cfg = train_net(cfg)