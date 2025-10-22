import sys
sys.path.append(".")
device_list = '5'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb

from train_net_stage2_gr import *

# mode = 'PAC'
mode = 'PAC_PNM'
# mode = 'PAF'
# mode = 'PPF'
# mode = 'PAF_PPF'
# mode = 'PPC'
# mode = 'PAF_PTR'
# mode = 'PAF_PNM'
# mode = 'PAF_PPF_PNM'

dataset = 'collective'
cfg=Config(dataset)

cfg.mode = mode
cfg.use_pos_type = 'absolute'
cfg.use_recon_loss = False
cfg.use_act_loss = False
cfg.use_activity_loss = False
cfg.use_pose_loss = False
cfg.use_recon_diff_loss = False
cfg.use_jae_loss = False
cfg.use_query_loss = False
cfg.use_key_recog_loss = False
cfg.use_recon_loss_key = False
cfg.use_recon_loss_non_key = False
cfg.use_recon_pose_feat_loss = False
cfg.use_recon_pose_coord_loss = False
cfg.use_traj_loss = False
cfg.use_person_num_loss = False
cfg.query_dir = os.path.join('data_local', 'collective_query')
cfg.query_type = 'dummy'
cfg.feature_adapt_type = 'ft'
cfg.freeze_backbone_stage4 = False
cfg.use_individual_action_type = 'gt'
cfg.use_query_classfication_loss = False
cfg.use_ga_recog_loss = False
cfg.use_ia_recog_loss = False

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
model_name = '[CAD_GAFL_PAC_mask_6_hbb_vgg_w_TB_w_loc_guide_stage2]<2025-05-11_00-03-03>'
cfg.stage2model = f'result/{model_name}/best_model.pth'

if mode == 'PAC':
    cfg.train_backbone = True
    cfg.load_backbone_stage2 = True
    cfg.use_act_loss = True
    cfg.max_epoch = 300
    cfg.batch_size = 8
    cfg.use_ind_feat_crop = 'roi_multi'
elif mode == 'PAC_PNM':
    cfg.train_backbone = True
    cfg.load_backbone_stage2 = True
    cfg.use_act_loss = True
    cfg.use_person_num_loss = True
    cfg.max_epoch = 300
    cfg.batch_size = 8
    cfg.use_ind_feat_crop = 'roi_multi'
    cfg.use_grp_feat = 'hbb'
    # cfg.use_grp_feat = 'hbb_whole'
elif mode == 'PAF':
    cfg.train_backbone = False
    cfg.load_backbone_stage2 = False
    cfg.use_recon_loss = True
    cfg.max_epoch = 500
    cfg.use_ind_feat_crop = 'roi_multi'
    cfg.batch_size = 8
    cfg.use_grp_feat = 'hbb'
    # cfg.use_grp_feat = 'hbb_whole'
elif mode == 'PPF':
    # cfg.train_backbone = False
    cfg.train_backbone = True
    cfg.load_backbone_stage2 = False
    cfg.use_recon_pose_feat_loss = True
    cfg.max_epoch = 500
    cfg.batch_size = 4
    cfg.use_ind_feat_crop = 'roi_multi'
    # cfg.use_app_feat_type = 'pose'
    # cfg.use_ind_feat_crop = 'crop_single'
    # cfg.batch_size = 8
elif mode == 'PAF_PPF':
    cfg.train_backbone = False
    cfg.load_backbone_stage2 = False
    cfg.use_recon_loss = True
    cfg.use_recon_pose_feat_loss = True
    cfg.max_epoch = 500
    cfg.use_ind_feat_crop = 'roi_multi'
    # cfg.batch_size = 8
    cfg.batch_size = 4
    cfg.use_app_feat_type = 'vgg'
    # cfg.use_app_feat_type = 'vgg_pose'
elif mode == 'PAF_PPF_PNM':
    cfg.train_backbone = False
    cfg.load_backbone_stage2 = False
    cfg.use_recon_loss = True
    cfg.use_recon_pose_feat_loss = True
    cfg.use_person_num_loss = True
    cfg.max_epoch = 500
    cfg.use_ind_feat_crop = 'roi_multi'
    cfg.batch_size = 4
    cfg.use_grp_feat = 'hbb_whole'
    cfg.use_app_feat_type = 'vgg'
elif mode == 'PPC':
    cfg.train_backbone = False
    cfg.load_backbone_stage2 = False
    cfg.use_recon_pose_coord_loss = True
    cfg.max_epoch = 500
    cfg.use_ind_feat_crop = 'roi_multi'
    # cfg.use_app_feat_type = 'pose'
    # cfg.use_app_feat_type = 'vgg_pose'
    # cfg.use_ind_feat_crop = 'crop_single'
    cfg.batch_size = 4
    cfg.num_features_boxes = 36
elif mode == 'PAF_PTR':
    cfg.train_backbone = False
    cfg.load_backbone_stage2 = False
    cfg.use_recon_loss = True
    cfg.use_traj_loss = True
    cfg.max_epoch = 300
    cfg.batch_size = 16
    cfg.use_ind_feat_crop = 'roi_multi'
elif mode == 'PAF_PNM':
    cfg.train_backbone = False
    cfg.load_backbone_stage2 = False
    cfg.use_recon_loss = True
    cfg.use_person_num_loss = True
    cfg.max_epoch = 500
    cfg.batch_size = 8
    cfg.use_ind_feat_crop = 'roi_multi'

cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 2
cfg.test_before_train = False
cfg.test_interval_epoch = 3
cfg.image_size = 240, 360
cfg.wandb_loss_list = ['activities_acc', 'activities_conf', 'activities_MPCA',
                        'actions_acc', 'actions_conf', 'actions_MPCA', 
                        'loss', 
                        'loss_act', 'loss_recon',
                        'loss_recon_pose_feat', 'loss_recon_pose_coord',
                        'loss_jae',
                        'loss_traj',
                        'loss_person_num',
                        ]

# vgg16 setup
cfg.backbone = 'inv3'
cfg.stage1_model_path = 'result/[CAD GR ours_stage1]<2023-07-09_21-47-14>/stage1_epoch17_0.92%.pth'
cfg.out_size=57,87

cfg.eval_only = False
cfg.eval_mask_num = 0
cfg.old_act_rec = False
cfg.test_batch_size = 1
cfg.num_boxes = 13
cfg.num_actions = 5
cfg.num_activities = 4
cfg.num_frames = 10

cfg.train_learning_rate = 5e-5
cfg.train_dropout_prob = 0.5
cfg.weight_decay = 1e-4
cfg.lr_plan = {}
# cfg.max_epoch = 100

# stage2 setup
cfg.use_res_connect = False
cfg.use_trans = True
cfg.use_same_enc_dual_path = False
cfg.trans_head_num = 1
cfg.trans_layer_num = 1

cfg.people_pool_type = 'max'
# cfg.use_pos_cond = False
cfg.use_pos_cond = True
# cfg.use_tmp_cond = False
cfg.use_tmp_cond = True
cfg.final_head_mid_num = 2
cfg.use_gen_iar = False
cfg.gen_iar_ratio = 0.0

cfg.use_random_mask = True
mk_num = 6
cfg.random_mask_type = f'random_to_{mk_num}'

# cfg.inference_module_name = 'group_activity_collective'
cfg.inference_module_name = 'group_relation_collective'

# HIGCIN INference setup
# cfg.inference_module_name = 'group_relation_higcin_collective'
# cfg.crop_size = 7, 7

# Dynamic Inference setup
# cfg.inference_module_name = 'group_relation_din_collective'
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

cfg.exp_note = f'CAD_GAFL_{mode}_mask_{mk_num}_{cfg.use_grp_feat}_{cfg.use_app_feat_type}'

if cfg.train_backbone:
    cfg.exp_note += '_w_TB'
else:
    cfg.exp_note += '_wo_TB'

if cfg.use_pos_cond:
    cfg.exp_note += '_w_loc_guide'
else:
    cfg.exp_note += '_wo_loc_guide'

print("===> Generate wandb system")
wandb.login()
wandb.init(project=f"DIN-Group-Activity-Recognition-Benchmark-{dataset}",
            name=f'{cfg.exp_note}_stage2', 
            config=cfg,
            settings=wandb.Settings(start_method='fork'),
            )

cfg = train_net(cfg)