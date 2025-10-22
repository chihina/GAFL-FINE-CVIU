from backbone.backbone import *
from backbone.backbone_kinetics import My3DResNet18
from utils import *
from roi_align.roi_align import RoIAlign      # RoIAlign module
import collections
import sys
import os
import cv2
import numpy as np
import pandas as pd
import torchvision
import torchvision.models as models
from torchvision.utils import save_image
from torchvision import transforms as pth_transforms

from sub_model import get_pose_net, get_activation

import yaml

from infer_module.higcin_infer_module import CrossInferBlock
from infer_module.dynamic_infer_module import Dynamic_Person_Inference, Hierarchical_Dynamic_Inference, Multi_Dynamic_Inference

from volleyball import volley_get_actions

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class GroupRelation_volleyball(nn.Module):
    """
    main module of GR learning for the volleyball dataset
    """

    def __init__(self, cfg):
        super(GroupRelation_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        H, W = self.cfg.image_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        self.backbone_type = self.cfg.backbone

        backbone_pretrain = True
        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=backbone_pretrain)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=backbone_pretrain)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=backbone_pretrain)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=backbone_pretrain)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=backbone_pretrain)
        elif cfg.backbone == '3d_res18_kinetics':
            self.backbone = My3DResNet18(pretrained=backbone_pretrain)
        elif cfg.backbone == 'c3d_ucf101':
            from backbone.backbone_ucf101 import C3D
            self.backbone = C3D(pretrained=backbone_pretrain)
        else:
            assert False
        
        if not cfg.train_backbone:
            if cfg.backbone == 'vgg16_ucf101':
                for p in self.backbone.collect_params().values():
                    p.grad_req = 'null'
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)

        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # replace feature extractor with pre-trained pose estimator
        if ('PPF' in cfg.mode) or ('PPC' in cfg.mode):
            pose_cfg_path = self.cfg.pose_cfg_path
            pose_cfg = yaml.load(open(pose_cfg_path, 'r'), Loader=yaml.FullLoader)
            self.backbone_pose = get_pose_net(pose_cfg)
            pose_weight_path = self.cfg.pose_model_path
            self.backbone_pose.load_state_dict(torch.load(pose_weight_path))
            for p in self.backbone_pose.parameters():
                p.requires_grad = False

        if ('PPF' in cfg.mode):        
            self.fc_emb_pose_feat = nn.Linear(K * K * 256, NFB)
        elif ('PPC' in cfg.mode):
            self.fc_emb_pose_feat = nn.Linear(17 * 2, NFB)

        self.use_ind_feat_crop = self.cfg.use_ind_feat_crop

        if self.use_ind_feat_crop == 'roi_multi':
            self.ind_ext_feat_dim = K * K * D
        elif self.use_ind_feat_crop == 'crop_single':
            self.ind_ext_feat_dim = NFB

        # define linear layer to embed image features
        if 'whole' in self.cfg.use_grp_feat:
            if 'whole_dino' in self.cfg.use_grp_feat:
                self.image_size_dino = 224
                self.dino_out_size = 384
                self.whole_img_dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
                self.transform_dino = pth_transforms.Compose([
                    pth_transforms.Resize(self.image_size_dino),
                    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
                self.whole_img_dino_fc_emb = nn.Sequential(
                    nn.Linear(self.dino_out_size, NFB*2),
                    nn.ReLU(),
                    nn.Linear(NFB*2, NFB*2),
                )
            else:
                self.whole_img_fc_emb = nn.Sequential(
                    # nn.Linear(D*cfg.out_size[0]*cfg.out_size[1], NFB*2),
                    nn.Linear(D, NFB*2),
                    nn.ReLU(),
                    nn.Linear(NFB*2, NFB*2),
                )

        # define the dimension of the pose feature
        self.ind_ext_feat_dim_pose_feat = int(K * K * 512 / 2)

        # addtional parameters
        self.eval_only = self.cfg.eval_only
        self.eval_mask_num = self.cfg.eval_mask_num
        self.use_random_mask = self.cfg.use_random_mask
        if self.use_random_mask:
            self.random_mask_type = self.cfg.random_mask_type
        self.use_ind_feat = self.cfg.use_ind_feat
        self.use_trans = self.cfg.use_trans
        self.use_same_enc_dual_path = self.cfg.use_same_enc_dual_path
        self.trans_head_num = self.cfg.trans_head_num
        self.trans_layer_num = self.cfg.trans_layer_num
        self.people_pool_type = self.cfg.people_pool_type
        self.use_pos_cond = self.cfg.use_pos_cond
        self.use_tmp_cond = self.cfg.use_tmp_cond
        self.final_head_mid_num = self.cfg.final_head_mid_num
        self.use_recon_loss = self.cfg.use_recon_loss
        self.use_recon_diff_loss = self.cfg.use_recon_diff_loss
        self.use_act_loss = self.cfg.use_act_loss
        self.use_pose_loss = self.cfg.use_pose_loss
        self.use_jae_loss = self.cfg.use_jae_loss
        self.use_old_act_rec = self.cfg.old_act_rec
        self.use_res_connect = self.cfg.use_res_connect
        self.use_gen_iar = self.cfg.use_gen_iar
        if self.use_gen_iar:
            self.gen_iar_ratio = self.cfg.gen_iar_ratio

        # transformer
        pos_enc_ind = positionalencoding2d(NFB, H, W)
        self.pos_enc_ind = pos_enc_ind.to(device=self.cfg.device)
        tem_enc_ind = positionalencoding1d(NFB, 100)
        self.tem_enc_ind = tem_enc_ind.to(device=self.cfg.device)
        self.people_tem = torch.arange(T, device=self.cfg.device, dtype=torch.long)
        box_size_enc_ind = positionalencoding1d(NFB, H*W)
        self.box_size_enc_ind = box_size_enc_ind.to(device=self.cfg.device)

        if self.use_trans:
            self.temporal_transformer_encoder = nn.ModuleList([
                nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=self.trans_head_num, batch_first=True, dropout=0.0), num_layers=self.trans_layer_num),
                nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=self.trans_head_num, batch_first=True, dropout=0.0), num_layers=self.trans_layer_num),
            ])
            self.spatial_transformer_encoder = nn.ModuleList([
                nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=self.trans_head_num, batch_first=True, dropout=0.0), num_layers=self.trans_layer_num),
                nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=self.trans_head_num, batch_first=True, dropout=0.0), num_layers=self.trans_layer_num)
            ])
        
        self.temporal_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])
        self.spatial_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])

        # IAR with locations
        if self.cfg.feature_adapt_type in ['line', 'mlp']:
            pos_enc_iar = positionalencoding2d(self.cfg.feature_adapt_dim, H, W)
            tem_enc_iar = positionalencoding1d(self.cfg.feature_adapt_dim, 100)
        else:
            pos_enc_iar = positionalencoding2d(2*NFB, H, W)
            tem_enc_iar = positionalencoding1d(2*NFB, 100)

        self.pos_enc_iar = pos_enc_iar.to(device=self.cfg.device)
        self.tem_enc_iar = tem_enc_iar.to(device=self.cfg.device)

        if self.cfg.feature_adapt_type == 'disent':
            pos_enc_iar_disentangle = positionalencoding2d(self.cfg.feature_adapt_dim, H, W)
            self.pos_enc_iar_disentangle = pos_enc_iar_disentangle.to(device=self.cfg.device)
            tem_enc_iar_disentangle = positionalencoding1d(self.cfg.feature_adapt_dim, 100)
            self.tem_enc_iar_disentangle = tem_enc_iar_disentangle.to(device=self.cfg.device)

        if self.people_pool_type == 'max':
            pass
        elif self.people_pool_type == 'mean':
            pass
        elif self.people_pool_type == 'we_mean':
            self.mask_fc = nn.Sequential(
                nn.Linear(2*NFB, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        # recog_head_mid_num
        if self.final_head_mid_num == 0:
            self.fc_actions_with_gr = nn.Sequential(
                nn.Linear(NFB*2, self.cfg.num_actions),
            )
            if self.use_gen_iar:
                self.fc_actions_gen_iar = nn.Sequential(
                    nn.Linear(NFB, self.cfg.num_actions),
                )

            if self.cfg.use_recon_loss:
                self.fc_recon_with_gr = nn.Sequential(
                    nn.Linear(NFB, self.ind_ext_feat_dim),
                )
                if self.feature_adapt_type in ['line', 'mlp']:
                    self.fc_recon_with_gr_adapt = nn.Sequential(
                        nn.Linear(self.cfg.feature_adapt_dim, self.ind_ext_feat_dim),
                    )
            if self.cfg.use_recon_loss_key:
                self.fc_recon_with_gr_key = nn.Sequential(
                    nn.Linear(self.cfg.feature_adapt_dim, self.ind_ext_feat_dim),
                )
            if self.cfg.use_recon_loss_non_key:
                self.fc_recon_with_gr_non_key = nn.Sequential(
                    nn.Linear(self.cfg.feature_adapt_dim, self.ind_ext_feat_dim),
                    )
        else:
            if self.use_old_act_rec:
                self.fc_actions_with_gr = nn.Sequential(
                        nn.Linear(NFB*2, NFB),
                        nn.ReLU(),
                        nn.Linear(NFB, NFB),
                        nn.ReLU(),
                        nn.Linear(NFB, self.cfg.num_actions),
                )

                if self.cfg.use_recon_loss:
                    self.fc_recon_with_gr = nn.Sequential(
                        nn.Linear(NFB*2, NFB),
                        nn.ReLU(),
                        nn.Linear(NFB, NFB),
                        nn.ReLU(),
                        nn.Linear(NFB, self.ind_ext_feat_dim),
                    )

            else:
                self.final_head_mid = nn.Sequential()
                for i in range(self.final_head_mid_num):
                    if i == 0:
                        self.final_head_mid.add_module('fc%d' % i, nn.Linear(NFB*2, NFB))
                    else:
                        self.final_head_mid.add_module('fc%d' % i, nn.Linear(NFB, NFB))
                    self.final_head_mid.add_module('relu%d' % i, nn.ReLU())

                if self.cfg.use_recon_pose_feat_loss:
                    self.final_head_mid_pose_feat = nn.Sequential()
                    for i in range(self.final_head_mid_num):
                        if i == 0:
                            self.final_head_mid_pose_feat.add_module('fc%d' % i, nn.Linear(NFB*2, NFB))
                        else:
                            self.final_head_mid_pose_feat.add_module('fc%d' % i, nn.Linear(NFB, NFB))
                        self.final_head_mid_pose_feat.add_module('relu%d' % i, nn.ReLU())

                if self.cfg.use_recon_pose_coord_loss:
                    self.final_head_mid_pose_coord = nn.Sequential()
                    for i in range(self.final_head_mid_num):
                        if i == 0:
                            self.final_head_mid_pose_coord.add_module('fc%d' % i, nn.Linear(NFB*2, NFB))
                        else:
                            self.final_head_mid_pose_coord.add_module('fc%d' % i, nn.Linear(NFB, NFB))
                        self.final_head_mid_pose_coord.add_module('relu%d' % i, nn.ReLU())

                if self.cfg.feature_adapt_type in ['line', 'mlp']:
                    self.final_head_mid_adapt = nn.Sequential()
                    self.final_head_mid_adapt_act = nn.Sequential()
                    self.final_head_mid_adapt_key_people = nn.Sequential()
                    self.final_head_mid_adapt_binary_cls = nn.Sequential()
                    self.final_head_mid_adapt_gar = nn.Sequential()
                    self.final_head_mid_adapt_iar = nn.Sequential()
                    for i in range(self.cfg.final_head_mid_adapt_num):
                        if i == 0:
                            self.final_head_mid_adapt.add_module('fc%d' % i, nn.Linear(self.cfg.feature_adapt_dim, self.cfg.feature_adapt_dim))
                            self.final_head_mid_adapt_act.add_module('fc%d' % i, nn.Linear(self.cfg.feature_adapt_dim, self.cfg.feature_adapt_dim))
                            self.final_head_mid_adapt_key_people.add_module('fc%d' % i, nn.Linear(self.cfg.feature_adapt_dim, self.cfg.feature_adapt_dim))
                            self.final_head_mid_adapt_binary_cls.add_module('fc%d' % i, nn.Linear(self.cfg.feature_adapt_dim, self.cfg.feature_adapt_dim))
                            self.final_head_mid_adapt_gar.add_module('fc%d' % i, nn.Linear(self.cfg.feature_adapt_dim, self.cfg.feature_adapt_dim))
                            self.final_head_mid_adapt_iar.add_module('fc%d' % i, nn.Linear(self.cfg.feature_adapt_dim, self.cfg.feature_adapt_dim))
                        else:
                            self.final_head_mid_adapt.add_module('fc%d' % i, nn.Linear(self.cfg.feature_adapt_dim, self.cfg.feature_adapt_dim))
                            self.final_head_mid_adapt_act.add_module('fc%d' % i, nn.Linear(self.cfg.feature_adapt_dim, self.cfg.feature_adapt_dim))
                            self.final_head_mid_adapt_key_people.add_module('fc%d' % i, nn.Linear(self.cfg.feature_adapt_dim, self.cfg.feature_adapt_dim))
                            self.final_head_mid_adapt_binary_cls.add_module('fc%d' % i, nn.Linear(self.cfg.feature_adapt_dim, self.cfg.feature_adapt_dim))
                            self.final_head_mid_adapt_gar.add_module('fc%d' % i, nn.Linear(self.cfg.feature_adapt_dim, self.cfg.feature_adapt_dim))
                            self.final_head_mid_adapt_iar.add_module('fc%d' % i, nn.Linear(self.cfg.feature_adapt_dim, self.cfg.feature_adapt_dim))

                        self.final_head_mid_adapt.add_module('relu%d' % i, nn.ReLU())
                        self.final_head_mid_adapt_act.add_module('relu%d' % i, nn.ReLU())
                        self.final_head_mid_adapt_key_people.add_module('relu%d' % i, nn.ReLU())
                        self.final_head_mid_adapt_binary_cls.add_module('relu%d' % i, nn.ReLU())
                        self.final_head_mid_adapt_gar.add_module('relu%d' % i, nn.ReLU())
                        self.final_head_mid_adapt_iar.add_module('relu%d' % i, nn.ReLU())

                # if self.use_gen_iar:
                #     self.final_head_mid_gen = nn.Sequential()
                #     for i in range(self.final_head_mid_num):
                #         self.final_head_mid_gen.add_module('fc%d' % i, nn.Linear(NFB, NFB))
                #         self.final_head_mid_gen.add_module('relu%d' % i, nn.ReLU())

                #     self.fc_actions_gen_iar = nn.Sequential(
                #             self.final_head_mid_gen,
                #             nn.Linear(NFB, self.cfg.num_actions),
                #     )

                if '2023' in self.cfg.model_exp_name:
                    self.fc_actions_with_gr = nn.Sequential(
                            self.final_head_mid,
                            nn.Linear(NFB, self.cfg.num_actions),
                    )

                if self.cfg.use_act_loss:
                    self.fc_actions_with_gr = nn.Sequential(
                            self.final_head_mid,
                            nn.Linear(NFB, self.cfg.num_actions),
                    )
                    if self.cfg.feature_adapt_type in ['line', 'mlp']:
                        self.fc_actions_with_gr_adapt = nn.Sequential(
                            self.final_head_mid_adapt_act,
                            nn.Linear(self.cfg.feature_adapt_dim, self.cfg.num_actions),
                        )

                if self.cfg.use_recon_loss:
                    self.fc_recon_with_gr = nn.Sequential(
                            self.final_head_mid,
                            nn.Linear(NFB, self.ind_ext_feat_dim),
                    )
                    if self.cfg.feature_adapt_type in ['line', 'mlp']:
                        self.fc_recon_with_gr_adapt = nn.Sequential(
                            self.final_head_mid_adapt,
                            nn.Linear(self.cfg.feature_adapt_dim, self.ind_ext_feat_dim),
                        )

                if self.cfg.use_key_recog_loss:
                    if self.cfg.feature_adapt_type in ['line', 'mlp']:
                        self.fc_key_recog_with_gr = nn.Sequential(
                            self.final_head_mid_adapt_key_people,
                            nn.Linear(self.cfg.feature_adapt_dim, 2),
                        )
                    else:
                        self.fc_key_recog_with_gr = nn.Sequential(
                                self.final_head_mid,
                                nn.Linear(NFB, 2),
                        )
                
                if self.cfg.use_query_classfication_loss:
                    self.cfg.num_query_class = 2
                    if self.cfg.feature_adapt_type in ['line', 'mlp']:
                        self.fc_query_classfication_with_gr = nn.Sequential(
                            # self.final_head_mid_adapt,
                            self.final_head_mid_adapt_binary_cls,
                            nn.Linear(self.cfg.feature_adapt_dim, self.cfg.num_query_class),
                        )
                    else:
                        self.fc_query_classfication_with_gr = nn.Sequential(
                                self.final_head_mid,
                                nn.Linear(NFB, self.cfg.num_query_class),
                        )
                
                if self.cfg.use_ga_recog_loss:
                    if self.cfg.feature_adapt_type in ['line', 'mlp']:
                        self.fc_ga_recog_with_gr = nn.Sequential(
                            self.final_head_mid_adapt_gar,
                            nn.Linear(self.cfg.feature_adapt_dim, self.cfg.num_activities),
                        )
                    else:
                        self.fc_ga_recog_with_gr = nn.Sequential(
                                self.final_head_mid,
                                nn.Linear(NFB, self.cfg.num_activities),
                        )
                
                if self.cfg.use_ia_recog_loss:
                    if self.cfg.feature_adapt_type in ['line', 'mlp']:
                        self.fc_ia_recog_with_gr = nn.Sequential(
                            self.final_head_mid_adapt_iar,
                            nn.Linear(self.cfg.feature_adapt_dim, self.cfg.num_actions),
                        )
                    else:
                        self.fc_ia_recog_with_gr = nn.Sequential(
                                self.final_head_mid,
                                nn.Linear(NFB, self.cfg.num_actions),
                        )


        self.training_stage = self.cfg.training_stage

        if self.training_stage == 2:
            if self.cfg.use_activity_loss:
                self.fc_activities = nn.Sequential(
                    nn.Linear(NFB, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, cfg.num_activities),
                )

        # GAR with group relation
        if self.training_stage == 3:
            self.fc_activities = nn.Sequential(
                nn.Linear(NFB*2, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, cfg.num_activities),
            )
            self.fc_actions = nn.Sequential(
                    nn.Linear(NFB*2, cfg.num_actions),
                    # nn.Linear(NFB*2, NFB),
                    # nn.ReLU(),
                    # nn.Linear(NFB, cfg.num_actions),
            )
        
        if self.cfg.use_jae_loss:
            self.fc_jae = nn.Sequential(
                nn.Linear(NFB*2, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                )
        
        if self.cfg.use_traj_loss:
            self.fc_traj = nn.Sequential(
                nn.Linear(NFB*2, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                )
        
        if self.cfg.use_person_num_loss:
            self.fc_person_num = nn.Sequential(
                nn.Linear(NFB*2, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                )

        if self.cfg.use_recon_pose_feat_loss:
            self.fc_recon_pose_feat_with_gr = nn.Sequential(
                self.final_head_mid_pose_feat,
                nn.Linear(NFB, self.ind_ext_feat_dim_pose_feat),
                )
            
        if self.cfg.use_recon_pose_coord_loss:
            self.fc_recon_pose_coord_with_gr = nn.Sequential(
                self.final_head_mid_pose_coord,
                nn.Linear(NFB, 17*2),
                )

        if self.cfg.feature_adapt_type == 'ft':
            pass
        elif self.cfg.feature_adapt_type == 'line':
            print('Build linear layer for feature adaptation')
            self.fc_adaptation_layer = nn.Linear(NFB*2, self.cfg.feature_adapt_dim)
        elif self.cfg.feature_adapt_type == 'mlp':
            self.fc_adaptation_layer = nn.Sequential(
                nn.Linear(NFB*2, self.cfg.feature_adapt_dim),
                nn.ReLU(),
                nn.Linear(self.cfg.feature_adapt_dim, self.cfg.feature_adapt_dim),
                nn.ReLU(),
            )
        elif self.cfg.feature_adapt_type == 'disent':
            print('Build disentangle layer for feature adaptation')
            self.key_head = nn.Sequential(
                nn.Linear(NFB*2, self.cfg.feature_adapt_dim),
                nn.ReLU(),
                nn.Linear(self.cfg.feature_adapt_dim, self.cfg.feature_adapt_dim),
                nn.ReLU(),
            )
            self.non_key_head = nn.Sequential(
                nn.Linear(NFB*2, self.cfg.feature_adapt_dim),
                nn.ReLU(),
                nn.Linear(self.cfg.feature_adapt_dim, self.cfg.feature_adapt_dim),
                nn.ReLU(),
            )
        elif self.cfg.feature_adapt_type == 'adaptor':
            print('Build adaptor layer for feature adaptation in transformer')
            adapt_dim = 64 * 4
            self.spatial_transformer_encoder_adaptor = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(NFB, adapt_dim), nn.GELU(), nn.Linear(adapt_dim, NFB),
                )
                for _ in range (len(self.spatial_transformer_encoder))
            ])
            self.temporal_transformer_encoder_adaptor = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(NFB, adapt_dim), nn.GELU(), nn.Linear(adapt_dim, NFB),
                )
                for _ in range (len(self.temporal_transformer_encoder))
            ])
            self.freeze_gafl_extractor_wo_adaptor()

        if self.cfg.freeze_backbone_stage4:
            self.freeze_gafl_extractor()
        
    # def freeze_gafl_extractor(self):
        # print('Freeze model except for the linear layer')
        # non_freeze_layers = ['fc_adaptation_layer']
        # for name, param in self.named_parameters():
        #     param.requires_grad = False
        #     print(name)
        #     for non_freeze_layer in non_freeze_layers:
        #         if non_freeze_layer in name:
        #             print('Non-freeze layer:')
        #             param.requires_grad = True
        #             break

    def freeze_gafl_extractor(self):
        freeze_layers = ['backbone', 'fc_emb_1', 'nl_emb_1', 
                         'temporal_transformer_encoder', 'spatial_transformer_encoder',
                         'temporal_transformer_mlp', 'spatial_transformer_mlp',]
        for name, param in self.named_parameters():
            param.requires_grad = True
            for freeze_layer in freeze_layers:
                if freeze_layer in name:
                    param.requires_grad = False
                    break
            # print(f'{name}: {param.requires_grad}')

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])

    def get_max_preds_pose(self, batch_heatmaps):
        '''
        Get predictions from score maps using PyTorch tensors.
        batch_heatmaps: torch.Tensor([batch_size, num_joints, height, width])
        '''
        assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be a torch.Tensor'
        assert batch_heatmaps.ndim == 4, 'batch_heatmaps should be 4-dimensional'
        
        batch_size, num_joints, height, width = batch_heatmaps.shape
        heatmaps_reshaped = batch_heatmaps.view(batch_size, num_joints, -1)
        
        maxvals, idx = torch.max(heatmaps_reshaped, dim=2, keepdim=True)
        preds = idx.repeat(1, 1, 2).float()
        
        preds[..., 0] = preds[..., 0] % width  # x-coordinate
        preds[..., 1] = torch.floor(preds[..., 1] / width)  # y-coordinate
        
        pred_mask = (maxvals > 0).expand(-1, -1, 2).float()
        preds *= pred_mask
        
        return preds

    def transform_pose(self, img):
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])
        img = transform(img)
        return img

    def forward(self, batch_data):
        start_time = time.time()
        # images_in, boxes_in, images_person_in = batch_data
        images_in = batch_data['images_in']
        boxes_in = batch_data['boxes_in']

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        PH, PW = self.cfg.person_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4
        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        if self.use_ind_feat_crop in ['roi_multi']:
            # Use backbone to extract features of images_in
            # Pre-precess first
            images_in_flat = prep_images(images_in_flat)
            outputs = self.backbone(images_in_flat)

            # Build  features
            features_multiscale = []
            for features in outputs:
                if features.shape[2:4] != torch.Size([OH, OW]):
                    features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
                features_multiscale.append(features)
            features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW
            features_multiscale = features_multiscale.contiguous()

            # RoI Align
            boxes_in_flat.requires_grad = False
            boxes_idx_flat.requires_grad = False
            boxes_features = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*N, D, K, K,
            boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

            # Embedding
            boxes_features_emb = self.fc_emb_1(boxes_features)  # B,T,N, NFB
            boxes_features_emb = self.nl_emb_1(boxes_features_emb)
            boxes_features_emb = F.relu(boxes_features_emb)

        # for 3d resnet
        if ('PPF' in self.cfg.mode) or ('PPC' in self.cfg.mode):
            images_person_in = batch_data['images_person_in']
            images_person_in_flat = images_person_in.view(B*T*N, 3, PH, PW)
            pose_feat_final, pose_feat_mid = self.backbone_pose(images_person_in_flat)
            if ('PPF' in self.cfg.mode):
                pose_feat_rH, pose_feat_rW = self.cfg.crop_size
                pose_feat_mid = F.interpolate(pose_feat_mid, size=(pose_feat_rH, pose_feat_rW), mode='bilinear', align_corners=True)
                boxes_features_emb_pose = pose_feat_mid.view(B, T, N, -1)
                boxes_features_emb_pose = self.fc_emb_pose_feat(boxes_features_emb_pose)
            if ('PPC' in self.cfg.mode):
                pose_feat_coords = self.get_max_preds_pose(pose_feat_final)
                pose_feat_coords = pose_feat_coords.view(B, T, N, -1)
                # for n_idx in range(N):
                    # print(pose_feat_coords[0, 0, n_idx].view(17, 2))
                boxes_features_emb_pose = pose_feat_coords
                boxes_features_emb_pose = self.fc_emb_pose_feat(boxes_features_emb_pose)
            # boxes_features_emb = boxes_features
            # elif self.backbone_type == 'c3d_ucf101':
            #     # (B, T, N, 3, H, W)
            #     images_person_in_flat = prep_images_c3d_ucf101(images_person_in)
            #     _, _, _, _, RPH, RPW = images_person_in_flat.shape
            #     images_person_in_flat = images_person_in_flat.permute(0, 2, 3, 1, 4, 5)
            #     images_person_in_flat = images_person_in_flat.reshape(B*N, 3, T, RPH, RPW)
            #     outputs = self.backbone(images_person_in_flat)
            #     outputs = outputs.reshape(B, N, 1, -1)
            #     outputs = outputs.permute(0, 2, 1, 3)
            #     boxes_features = outputs.repeat(1, T, 1, 2)
            #     boxes_features_emb = boxes_features
            # else:
            #     # (B, T, 3, N, 3, H, W)
            #     images_person_in_pad = torch.zeros(B, T, 3, N, 3, PH, PW).to(device=images_person_in.device)
            #     for t_idx in range(1, T-1):
            #         images_person_in_cut = images_person_in[:, t_idx-1:t_idx+2]
            #         images_person_in_pad[:, t_idx, :, :, :, :, :] = images_person_in_cut
            #     images_person_in_pad[:, 0, :, :, :, :, :] = images_person_in[:, 0:3]
            #     images_person_in_pad[:, T-1, :, :, :, :, :] = images_person_in[:, T-3:T]
            #     images_person_in_flat = prep_images_3dresnet(images_person_in_pad)
            #     _, _, _, _, _, RPH, RPW = images_person_in_flat.shape
            #     images_person_in_flat = images_person_in_flat.permute(0, 1, 3, 4, 2, 5, 6)
            #     images_person_in_flat = images_person_in_flat.reshape(B*T*N, 3, 3, RPH, RPW)
            #     outputs = self.backbone(images_person_in_flat)
            #     outputs = outputs.reshape(B, T, N, -1)
            #     boxes_features = outputs.repeat(1, 1, 1, 2)
            #     boxes_features_emb = boxes_features

        if self.cfg.use_app_feat_type == 'vgg':
            pass
        elif 'pose' in self.cfg.use_app_feat_type:
            if self.cfg.use_app_feat_type == 'vgg_pose':
                boxes_features_emb = boxes_features_emb_pose + boxes_features_emb
            elif self.cfg.use_app_feat_type == 'pose':
                boxes_features_emb = boxes_features_emb_pose
        else:
            assert False, 'Unknown appearance feature type'

        # encode position infromation
        pos_enc_ind = self.pos_enc_ind

        if self.cfg.use_pos_type == 'absolute':
            boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
            boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
            boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
            boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        elif self.cfg.use_pos_type == 'relative_vid':
            boxes_wo_norm_in = batch_data['boxes_wo_norm_in']
            boxes_in_x_center = (boxes_wo_norm_in[:, :, :, 0]+boxes_wo_norm_in[:, :, :, 2])/2
            boxes_in_y_center = (boxes_wo_norm_in[:, :, :, 1]+boxes_wo_norm_in[:, :, :, 3])/2
            bbox_region_vid = batch_data['bbox_region_vid']
            bbox_region_vid_x_min, bbox_region_vid_y_min = bbox_region_vid[:, :, 0].view(B, T, 1), bbox_region_vid[:, :, 1].view(B, T, 1)
            bbox_region_vid_x_max, bbox_region_vid_y_max = bbox_region_vid[:, :, 2].view(B, T, 1), bbox_region_vid[:, :, 3].view(B, T, 1)
            boxes_in_x_center_norm = (boxes_in_x_center - bbox_region_vid_x_min) / (bbox_region_vid_x_max - bbox_region_vid_x_min)
            boxes_in_y_center_norm = (boxes_in_y_center - bbox_region_vid_y_min) / (bbox_region_vid_y_max - bbox_region_vid_y_min)
            boxes_in_x_center_view = boxes_in_x_center_norm.view(B*T*N) * W
            boxes_in_y_center_view = boxes_in_y_center_norm.view(B*T*N) * H
        elif self.cfg.use_pos_type == 'relative_img':
            boxes_wo_norm_in = batch_data['boxes_wo_norm_in']
            boxes_in_x_center = (boxes_wo_norm_in[:, :, :, 0]+boxes_wo_norm_in[:, :, :, 2])/2
            boxes_in_y_center = (boxes_wo_norm_in[:, :, :, 1]+boxes_wo_norm_in[:, :, :, 3])/2
            boxes_in_x_center_min, _ = torch.min(boxes_in_x_center, dim=-1, keepdim=True)
            boxes_in_x_center_max, _ = torch.max(boxes_in_x_center, dim=-1, keepdim=True)
            boxes_in_x_center_norm = (boxes_in_x_center - boxes_in_x_center_min) / (boxes_in_x_center_max - boxes_in_x_center_min)
            boxes_in_y_center_min, _ = torch.min(boxes_in_y_center, dim=-1, keepdim=True)
            boxes_in_y_center_max, _ = torch.max(boxes_in_y_center, dim=-1, keepdim=True)
            boxes_in_y_center_norm = (boxes_in_y_center - boxes_in_y_center_min) / (boxes_in_y_center_max - boxes_in_y_center_min)
            boxes_in_x_center_view = boxes_in_x_center_norm.contiguous().view(B*T*N) * W
            boxes_in_y_center_view = boxes_in_y_center_norm.contiguous().view(B*T*N) * H
            boxes_in_x_center_view = torch.clamp(boxes_in_x_center_view, 0, W-1)
            boxes_in_y_center_view = torch.clamp(boxes_in_y_center_view, 0, H-1)
        elif self.cfg.use_pos_type == 'relative_net':
            bbox_wo_norm_in = batch_data['boxes_wo_norm_in']
            boxes_in_x_center = (bbox_wo_norm_in[:, :, :, 0]+bbox_wo_norm_in[:, :, :, 2])/2
            boxes_in_y_center = (bbox_wo_norm_in[:, :, :, 1]+bbox_wo_norm_in[:, :, :, 3])/2

            images_info = batch_data['images_info']
            img_orig_h = images_info[:, :, 0].view(B, T, 1)
            img_orig_w = images_info[:, :, 1].view(B, T, 1)

            bbox_region_net = batch_data['bbox_region_net']
            bbox_region_net_x_min, bbox_region_net_y_min = bbox_region_net[:, :, 0].view(B, T, 1), bbox_region_net[:, :, 1].view(B, T, 1)
            bbox_region_net_x_max, bbox_region_net_y_max = bbox_region_net[:, :, 2].view(B, T, 1), bbox_region_net[:, :, 3].view(B, T, 1)
            bbox_region_net_x_mid = (bbox_region_net_x_min + bbox_region_net_x_max) / 2
            bbox_region_net_y_mid = (bbox_region_net_y_min + bbox_region_net_y_max) / 2
            bbox_region_net_y_len = bbox_region_net_y_max - bbox_region_net_y_min

            # transform x coordinates
            # print(torch.min(bbox_region_net_y_len), torch.max(bbox_region_net_y_len))
            boxes_in_x_center_norm = (boxes_in_x_center - bbox_region_net_x_mid)
            boxes_in_x_center_norm = boxes_in_x_center_norm / bbox_region_net_y_len * (0.55 * img_orig_h)
            boxes_in_x_center_norm = boxes_in_x_center_norm + img_orig_w
            boxes_in_x_center_norm = boxes_in_x_center_norm / (2*img_orig_w)
            # print('boxes_in_x_center_norm', torch.min(boxes_in_x_center_norm), torch.max(boxes_in_x_center_norm))

            # transform y coordinates
            boxes_in_y_center_norm = (boxes_in_y_center - bbox_region_net_y_mid)
            boxes_in_y_center_norm = boxes_in_y_center_norm / bbox_region_net_y_len * (0.55 * img_orig_h)
            boxes_in_y_center_norm = boxes_in_y_center_norm + img_orig_h
            boxes_in_y_center_norm = boxes_in_y_center_norm / (2*img_orig_h)
            # print('boxes_in_y_center_norm', torch.min(boxes_in_y_center_norm), torch.max(boxes_in_y_center_norm))

            # rescale x and y coordinates to the range of [0, W] and [0, H]
            boxes_in_x_center_view = boxes_in_x_center_norm.view(B*T*N) * W
            boxes_in_y_center_view = boxes_in_y_center_norm.view(B*T*N) * H
            boxes_in_x_center_view = torch.clamp(boxes_in_x_center_view, 0, W-1)
            boxes_in_y_center_view = torch.clamp(boxes_in_y_center_view, 0, H-1)
        elif self.cfg.use_pos_type == 'relative_bev':
            boxes_wo_norm_bev_foot = batch_data['boxes_wo_norm_bev_foot']
            boxes_in_x_center = boxes_wo_norm_bev_foot[:, :, :, 0]
            boxes_in_y_center = boxes_wo_norm_bev_foot[:, :, :, 1]

            boxes_in_x_center = torch.clamp(boxes_in_x_center, 0, W-1)
            boxes_in_y_center = torch.clamp(boxes_in_y_center, 0, H-1)

            if torch.min(boxes_in_x_center) < 0 or torch.max(boxes_in_x_center) > W:
                print('boxes_in_x_center', torch.min(boxes_in_x_center), torch.max(boxes_in_x_center), W)
                sys.exit('Not implemented')
            if torch.min(boxes_in_y_center) < 0 or torch.max(boxes_in_y_center) > H:
                print('boxes_in_y_center', torch.min(boxes_in_y_center), torch.max(boxes_in_y_center), H)
                sys.exit('Not implemented')
            boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
            boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
            # sys.exit('Not implemented')
        else:
            assert False, 'use_pos_type error'

        # purturbed position 
        if 'perturbation_mask_position' in batch_data.keys():
            perturbation_mask_position = batch_data['perturbation_mask_position']
            boxes_in_x_center_view = boxes_in_x_center_view + perturbation_mask_position[:, :, :, 0].view(B*T*N)
            boxes_in_y_center_view = boxes_in_y_center_view + perturbation_mask_position[:, :, :, 1].view(B*T*N)
        
        if 'perturbation_mask_apperance' in batch_data.keys():
            perturbation_mask_apperance = batch_data['perturbation_mask_apperance']
            perturbation_mask_apperance =perturbation_mask_apperance.view(B, T, N, 1)
            boxes_features_emb = boxes_features_emb * perturbation_mask_apperance

        ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        ind_loc_feat = ind_loc_feat.view(B, T, N, NFB)

        # encode temporal infromation
        tem_enc_ind = self.tem_enc_ind
        people_tem = self.people_tem
        people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
        ind_tem_feat = tem_enc_ind[people_temp, :].view(B, T, N, NFB)

        # encode box size information
        boxes_size_in = (boxes_in[:, :, :, 2]-boxes_in[:, :, :, 0])*(boxes_in[:, :, :, 3]-boxes_in[:, :, :, 1])
        boxes_size_in = boxes_size_in.view(B*T*N)
        box_size_enc_ind = self.box_size_enc_ind
        ind_box_size_feat = box_size_enc_ind[boxes_size_in.long()].view(B, T, N, NFB)

        # generate individual features
        if self.use_ind_feat == 'loc_and_app_box':
            ind_feat_set = boxes_features_emb + ind_loc_feat + ind_tem_feat + ind_box_size_feat
        elif self.use_ind_feat == 'loc_and_app':
            ind_feat_set = boxes_features_emb + ind_loc_feat + ind_tem_feat
        elif self.use_ind_feat == 'loc':
            ind_feat_set = ind_loc_feat + ind_tem_feat
        elif self.use_ind_feat == 'app':
            ind_feat_set = boxes_features_emb + ind_tem_feat
        
        # generate padding masks for transformer
        people_pad_mask = (torch.sum(boxes_in, dim=-1)==0).bool()

        # reshape individual features
        ind_feat_set_upper = ind_feat_set.view(B*T, N, NFB)
        ind_feat_set_bottom = torch.transpose(ind_feat_set, 1, 2).contiguous().view(B*N, T, NFB)

        # mask during training
        self.adapt_mask = (self.use_random_mask) and (not self.eval_only) and (self.training_stage == 2 or self.training_stage == 4 or self.training_stage == 5)
        if self.adapt_mask:
            if 'active' in self.random_mask_type:
                actions_in = batch_data['actions_in'].reshape(B, T, N)
                if self.random_mask_type == 'active':
                    random_mask = actions_in!=7
                elif self.random_mask_type == 'active_inv':
                    random_mask = actions_in==7
                else:
                    random_max_num = int(self.random_mask_type.split('_')[-1])
                    random_mask = actions_in==random_max_num
            else:
                random_max_num = int(self.random_mask_type.split('_')[-1])
                random_mask = torch.zeros(B, T, N, device=people_pad_mask.device)
                # for b in range(B):
                #     for t in range(T):
                #         people_pad_mask_b_t = people_pad_mask[b, t]
                #         people_pad_mask_b_t_idx = torch.where(people_pad_mask_b_t==0)[0]
                #         people_pad_mask_b_t_idx_shuffle = people_pad_mask_b_t_idx[torch.randperm(people_pad_mask_b_t_idx.shape[0])]
                #         random_max_num_update = int((random_max_num/N)*torch.sum(people_pad_mask_b_t==0))
                #         mask_people_num = people_pad_mask_b_t_idx_shuffle[:random_max_num_update]
                #         random_mask[b, t, mask_people_num] = True
                for b in range(B):
                    people_pad_mask_b = people_pad_mask[b, 0]
                    people_pad_mask_b_idx = torch.where(people_pad_mask_b==0)[0]
                    people_pad_mask_b_idx_shuffle = people_pad_mask_b_idx[torch.randperm(people_pad_mask_b_idx.shape[0])]
                    random_max_num_update = int((random_max_num/N)*torch.sum(people_pad_mask_b==0))
                    mask_people_num = people_pad_mask_b_idx_shuffle[:random_max_num_update]
                    random_mask[b, :, mask_people_num] = True

            people_pad_mask_jud = torch.logical_or(people_pad_mask.bool(), random_mask.bool()).bool()
            all_mask_flag = torch.sum(people_pad_mask_jud==0, dim=(-1))==0
            people_pad_mask[~all_mask_flag] = torch.logical_or(people_pad_mask[~all_mask_flag].bool(), random_mask[~all_mask_flag].bool()).bool()
        
        # mask during inference
        if self.eval_only:
            random_max_num = self.eval_mask_num
            # random_mask = torch.zeros(B, N).to(device=people_pad_mask.device)
            # for b in range(B):
            #     people_pad_mask_b = people_pad_mask[b, 0]
            #     people_pad_mask_b_idx = torch.where(people_pad_mask_b==0)[0]
            #     people_pad_mask_b_idx_shuffle = people_pad_mask_b_idx[torch.randperm(people_pad_mask_b_idx.shape[0])]
            #     random_max_num_update = int((random_max_num/N)*torch.sum(people_pad_mask_b==0))
            #     mask_people_num = people_pad_mask_b_idx_shuffle[:random_max_num_update]
            #     random_mask[b, mask_people_num] = True
            # random_mask = random_mask.view(B, 1, N).expand(B, T, N)
            random_mask = torch.zeros(B, T, N).to(device=people_pad_mask.device)
            for b in range(B):
                for t in range(T):
                    people_pad_mask_b_t = people_pad_mask[b, t]
                    people_pad_mask_b_t_idx = torch.where(people_pad_mask_b_t==0)[0]
                    people_pad_mask_b_t_idx_shuffle = people_pad_mask_b_t_idx[torch.randperm(people_pad_mask_b_t_idx.shape[0])]
                    random_max_num_update = int((random_max_num/N)*torch.sum(people_pad_mask_b_t==0))
                    mask_people_num = people_pad_mask_b_t_idx_shuffle[:random_max_num_update]
                    random_mask[b, t, mask_people_num] = True

            people_pad_mask_jud = torch.logical_or(people_pad_mask.bool(), random_mask.bool()).bool()
            all_mask_flag = torch.sum(people_pad_mask_jud==0, dim=(-1))==0
            people_pad_mask[~all_mask_flag] = torch.logical_or(people_pad_mask[~all_mask_flag].bool(), random_mask[~all_mask_flag].bool()).bool()

        # mask people to detect key-person in a cluster composed of query samples
        if 'perturbation_mask' in batch_data.keys():
            perturbation_mask = batch_data['perturbation_mask']

            # avoid masking all people in an image
            people_pad_mask_jud = torch.logical_or(people_pad_mask.bool(), perturbation_mask.bool()).bool()
            all_mask_flag = torch.sum(people_pad_mask_jud==0, dim=(-1))==0
            people_pad_mask[~all_mask_flag] = torch.logical_or(people_pad_mask[~all_mask_flag].bool(), perturbation_mask[~all_mask_flag].bool()).bool()

        people_pad_mask_spatial = people_pad_mask.contiguous().view(B*T, N)
        people_pad_mask_temporal = torch.transpose(people_pad_mask, 1, 2).contiguous().view(B*N, T)

        if self.use_trans:
            # upper branch
            ind_feat_enc_upper = self.spatial_transformer_encoder[0](ind_feat_set_upper, src_key_padding_mask=people_pad_mask_spatial)
            ind_feat_enc_upper_res = self.spatial_transformer_mlp[0](ind_feat_enc_upper)

            if self.cfg.feature_adapt_type == 'adaptor':
                ind_feat_enc_upper = ind_feat_enc_upper + ind_feat_enc_upper_res + self.spatial_transformer_encoder_adaptor[0](ind_feat_enc_upper_res)
            else:
                ind_feat_enc_upper = ind_feat_set_upper + ind_feat_enc_upper_res

            ind_feat_enc_upper = ind_feat_enc_upper.view(B, T, N, NFB)
            ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2).contiguous().view(B*N, T, NFB)
            ind_feat_enc_upper = self.temporal_transformer_encoder[0](ind_feat_enc_upper)                
            ind_feat_enc_upper = ind_feat_enc_upper.view(B, N, T, NFB)
            ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2)

            # bottom branch
            if self.use_same_enc_dual_path:
                ind_feat_enc_bottom = self.temporal_transformer_encoder[0](ind_feat_set_bottom)
            else:
                ind_feat_enc_bottom = self.temporal_transformer_encoder[1](ind_feat_set_bottom)
            ind_feat_enc_bottom_res = self.temporal_transformer_mlp[0](ind_feat_enc_bottom)

            if self.cfg.feature_adapt_type == 'adaptor':
                ind_feat_enc_bottom = ind_feat_enc_bottom + ind_feat_enc_bottom_res + self.temporal_transformer_encoder_adaptor[0](ind_feat_enc_bottom_res)
            else:
                ind_feat_enc_bottom = ind_feat_set_bottom + ind_feat_enc_bottom_res

            ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, N, T, NFB)                
            ind_feat_enc_bottom = torch.transpose(ind_feat_enc_bottom, 1, 2).contiguous().view(B*T, N, NFB)
            if self.use_same_enc_dual_path:
                ind_feat_enc_bottom = self.spatial_transformer_encoder[0](ind_feat_enc_bottom, src_key_padding_mask=people_pad_mask_spatial)
            else:
                ind_feat_enc_bottom = self.spatial_transformer_encoder[1](ind_feat_enc_bottom, src_key_padding_mask=people_pad_mask_spatial)

            ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, T, N, NFB)
        else:
            # upper branch
            ind_feat_enc_upper_res = self.spatial_transformer_mlp[0](ind_feat_set_upper)
            ind_feat_enc_upper = ind_feat_set_upper + ind_feat_enc_upper_res
            ind_feat_enc_upper = ind_feat_enc_upper.view(B, T, N, NFB)
            ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2).contiguous().view(B*N, T, NFB)
            ind_feat_enc_upper = ind_feat_enc_upper.view(B, N, T, NFB)
            ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2)

            # bottom branch
            ind_feat_enc_bottom_res = self.temporal_transformer_mlp[0](ind_feat_set_bottom)
            ind_feat_enc_bottom = ind_feat_set_bottom + ind_feat_enc_bottom_res
            ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, N, T, NFB)                
            ind_feat_enc_bottom = torch.transpose(ind_feat_enc_bottom, 1, 2).contiguous().view(B*T, N, NFB)
            ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, T, N, NFB)

        if self.use_res_connect:
            ind_feat_enc_bottom = ind_feat_enc_bottom + boxes_features_emb
            ind_feat_enc_upper = ind_feat_enc_upper + boxes_features_emb

        boxes_states = torch.cat([ind_feat_enc_upper, ind_feat_enc_bottom], dim=-1)
        NFS = NFB*2

        # Delete individual features of masked people
        if self.adapt_mask:
            boxes_states = boxes_states * (~(people_pad_mask.bool())).view(B, T, N, 1)
        if 'perturbation_mask' in batch_data.keys():
            boxes_states = boxes_states * (~(people_pad_mask.bool())).view(B, T, N, 1)
        if self.eval_only and (self.eval_mask_num != 0):
            boxes_states = boxes_states * (~(people_pad_mask.bool())).view(B, T, N, 1)

        # pooling temporal features
        if self.people_pool_type == 'max':
            individual_feat, _ = torch.max(boxes_states, dim=1)
        elif self.people_pool_type == 'mean':
            individual_feat = torch.mean(boxes_states, dim=1)
        individual_feat_dim = individual_feat.shape[-1]

        # pooling individual features
        if self.people_pool_type == 'max':
            frame_feat, _ = torch.max(boxes_states, dim=2)
        elif self.people_pool_type == 'mean':
            frame_feat = torch.mean(boxes_states, dim=2)

        if self.cfg.use_grp_feat == 'hbb':
            pass
        elif 'whole' in self.cfg.use_grp_feat:
            if self.cfg.use_grp_feat == 'hbb_whole':
                img_feat = features_multiscale.mean(dim=(2, 3))
                img_feat = img_feat.view(B, T, -1)
                frame_feat_img = self.whole_img_fc_emb(img_feat)
            elif self.cfg.use_grp_feat == 'hbb_whole_dino':
                images_in_dino = torch.reshape(images_in, (B * T, 3, H, W))
                images_in_dino = self.transform_dino(images_in_dino)
                img_feat = self.whole_img_dino(images_in_dino)
                img_feat = img_feat.view(B, T, -1)
                frame_feat_img = self.whole_img_dino_fc_emb(img_feat)

            # if self.cfg.people_pool_type == 'max':
            #     img_feat_emb_pool, _ = torch.max(img_feat_emb, dim=1)
            # elif self.cfg.people_pool_type == 'mean':
            #     img_feat_emb_pool = torch.mean(img_feat_emb, dim=1)
            # group_feat = group_feat + img_feat_emb_pool
            frame_feat = frame_feat + frame_feat_img
        else:
            assert False, 'unknown group feature type'

        # if self.people_pool_type == 'max':
            # group_feat, _ = torch.max(individual_feat, dim=1)
        # elif self.people_pool_type == 'mean':
            # group_feat = torch.mean(individual_feat, dim=1)

        if self.people_pool_type == 'max':
            group_feat, _ = torch.max(frame_feat, dim=1)
        elif self.people_pool_type == 'mean':
            group_feat = torch.mean(frame_feat, dim=1)

        if (self.training_stage == 4 or self.training_stage == 5):
            if self.cfg.feature_adapt_type in ['line', 'mlp']:
                group_feat_res = self.fc_adaptation_layer(group_feat)
                if self.cfg.feature_adapt_residual:
                    group_feat = group_feat + group_feat_res
                else:
                    group_feat = group_feat_res

        # get the feature dimension of group activity features
        group_feat_dim = group_feat.shape[-1]

        # add positional encoding to group features
        pos_enc_iar = self.pos_enc_iar
        iar_loc_feat = torch.transpose(pos_enc_iar[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        iar_loc_feat = iar_loc_feat.view(B, T, N, group_feat_dim)
        iar_inp_feat = group_feat.view(B, 1, 1, -1).expand(B, T, N, group_feat_dim)
        if self.use_pos_cond:
            iar_inp_feat = iar_inp_feat + iar_loc_feat

        # add temporal encoding to group features
        tem_enc_iar = self.tem_enc_iar
        people_tem = self.people_tem
        # people_tem = people_tem[:T]
        people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
        iar_tem_feat = tem_enc_iar[people_temp, :].view(B, T, N, group_feat_dim)
        if self.use_tmp_cond:
            iar_inp_feat = iar_inp_feat + iar_tem_feat

        # define the dictionary for the return values
        ret_dic = {}
        ret_dic['group_feat'] = group_feat
        ret_dic['individual_feat'] = individual_feat.view(B, N*individual_feat_dim)

        if self.cfg.training_stage == 4 or self.cfg.training_stage == 5:
            if self.cfg.use_act_loss:
                if self.cfg.feature_adapt_type in ['line', 'mlp']:
                    actions_scores_with_gr = self.fc_actions_with_gr_adapt(iar_inp_feat)
                else:
                    actions_scores_with_gr = self.fc_actions_with_gr(iar_inp_feat)
                actions_scores_with_gr = torch.mean(actions_scores_with_gr, dim=1).reshape(B * N, -1)

                ret_dic['actions'] = actions_scores_with_gr

            if self.cfg.use_key_recog_loss:
                if self.cfg.key_recog_feat_type == 'gaf':
                    recog_key_person = self.fc_key_recog_with_gr(iar_inp_feat)
                elif self.cfg.key_recog_feat_type == 'iaf':
                    individual_feat_expand = individual_feat.view(B, 1, N, individual_feat_dim).expand(B, T, N, individual_feat_dim)
                    recog_key_person = self.fc_key_recog_with_gr(individual_feat_expand)
                else:
                    assert False, 'key_recog_feat_type error'
                ret_dic['recog_key_person'] = recog_key_person
                gt_key_person = batch_data['user_queries_people']
                # print(recog_key_person[:3, 0, :, 1])
                # print(gt_key_person[:3, 0, :])
            
            if self.cfg.use_query_classfication_loss:
                recog_query = self.fc_query_classfication_with_gr(group_feat)
                ret_dic['recog_query'] = recog_query
            
            if self.cfg.use_ga_recog_loss:
                recognized_ga = self.fc_ga_recog_with_gr(group_feat)
                ret_dic['recognized_ga'] = recognized_ga
            
            if self.cfg.use_ia_recog_loss:
                individual_feat_expand = individual_feat.view(B, 1, N, individual_feat_dim).expand(B, T, N, individual_feat_dim)
                recognized_ia = self.fc_ia_recog_with_gr(individual_feat_expand)
                ret_dic['recognized_ia'] = recognized_ia

            if self.cfg.feature_adapt_type == 'disent':
                ret_dic['group_feat_key'] = key_feat
                ret_dic['group_feat_non_key'] = non_key_feat
        elif self.training_stage == 3:
            if self.use_jae_loss:
                estimated_ja = self.fc_jae(iar_inp_feat).mean(dim=2)
                ret_dic['estimated_ja'] = estimated_ja
            else:
                activities = self.fc_activities(group_feat)
                ret_dic['activities'] = activities
                actions = self.fc_actions(individual_feat).reshape(B * N, -1)
                ret_dic['actions'] = actions
        elif self.training_stage == 2:
            if self.cfg.use_activity_loss:
                individual_feat_upper, _ = torch.max(ind_feat_enc_upper, dim=1)
                individual_feat_bottom, _ = torch.max(ind_feat_enc_bottom, dim=1)
                group_feat_upper, _ = torch.max(individual_feat_upper, dim=1)
                group_feat_bottom, _ = torch.max(individual_feat_bottom, dim=1)
                activities_upper = self.fc_activities(group_feat_upper)
                activities_bottom = self.fc_activities(group_feat_bottom)
                activities = (activities_upper + activities_bottom) / 2
                ret_dic['activities'] = activities
        else:
            pass

        if self.cfg.use_act_loss:
            if self.cfg.feature_adapt_type in ['line', 'mlp']:
                actions_scores_with_gr = self.fc_actions_with_gr_adapt(iar_inp_feat)
            else:
                actions_scores_with_gr = self.fc_actions_with_gr(iar_inp_feat)
            actions_scores_with_gr = torch.mean(actions_scores_with_gr, dim=1).reshape(B * N, -1)
            ret_dic['actions'] = actions_scores_with_gr

        if self.cfg.use_recon_loss:
            if self.cfg.feature_adapt_type in ['line', 'mlp']:
                recon_features = self.fc_recon_with_gr_adapt(iar_inp_feat)
            else:
                recon_features = self.fc_recon_with_gr(iar_inp_feat)
            ret_dic['recon_features'] = recon_features
            ret_dic['original_features'] = boxes_features

        if self.use_act_loss:
            actions_scores_with_gr = self.fc_actions_with_gr(iar_inp_feat)
            actions_scores_with_gr = torch.mean(actions_scores_with_gr, dim=1).reshape(B * N, -1)
        
        if self.cfg.use_recon_pose_feat_loss:
            recon_pose_feat = self.fc_recon_pose_feat_with_gr(iar_inp_feat)
            pose_feat_mid = pose_feat_mid.view(B, T, N, -1)
            ret_dic['recon_pose_features'] = recon_pose_feat
            ret_dic['original_pose_features'] = pose_feat_mid
        
        if self.cfg.use_recon_pose_coord_loss:
            recon_pose_coords = self.fc_recon_pose_coord_with_gr(iar_inp_feat)
            pose_feat_coords = pose_feat_coords.view(B, T, N, -1)
            ret_dic['recon_pose_coords'] = recon_pose_coords
            ret_dic['original_pose_coords'] = pose_feat_coords

        if self.cfg.use_jae_loss:
            estimated_ja = self.fc_jae(frame_feat)
            ret_dic['estimated_ja'] = estimated_ja
        
        if self.cfg.use_traj_loss:
            estimated_flow = self.fc_traj(iar_inp_feat)
            ret_dic['estimated_flow'] = estimated_flow
        
        if self.cfg.use_person_num_loss:
            estimated_person_num = self.fc_person_num(frame_feat)
            ret_dic['estimated_person_num'] = estimated_person_num
    
        return ret_dic

class GroupRelation_HiGCIN_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(GroupRelation_HiGCIN_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        # NFB = self.cfg.num_features_boxes
        NFB = D
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        # addtional parameters
        self.eval_only = self.cfg.eval_only
        self.use_random_mask = self.cfg.use_random_mask
        if self.use_random_mask:
            self.random_mask_type = self.cfg.random_mask_type
        self.use_ind_feat = self.cfg.use_ind_feat
        self.use_trans = self.cfg.use_trans
        self.trans_head_num = self.cfg.trans_head_num
        self.trans_layer_num = self.cfg.trans_layer_num
        self.people_pool_type = self.cfg.people_pool_type
        self.use_pos_cond = self.cfg.use_pos_cond
        self.use_tmp_cond = self.cfg.use_tmp_cond
        self.final_head_mid_num = self.cfg.final_head_mid_num
        self.use_recon_loss = self.cfg.use_recon_loss
        self.use_recon_diff_loss = self.cfg.use_recon_diff_loss
        self.use_act_loss = self.cfg.use_act_loss
        self.use_pose_loss = self.cfg.use_pose_loss
        self.use_old_act_rec = self.cfg.old_act_rec
        self.use_res_connect = self.cfg.use_res_connect
        self.use_gen_iar = self.cfg.use_gen_iar
        if self.use_gen_iar:
            self.gen_iar_ratio = self.cfg.gen_iar_ratio
        self.training_stage = self.cfg.training_stage
        self.ind_ext_feat_dim = K * K * D

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.person_avg_pool = nn.AvgPool2d((K**2, 1), stride = 1)
        self.BIM = CrossInferBlock(in_dim = D, Temporal = T, Spatial = K**2)
        self.PIM = CrossInferBlock(in_dim = D, Temporal = T, Spatial = N)
        self.dropout = nn.Dropout()
        self.fc_activities = nn.Linear(D, cfg.num_activities, bias = False)
        self.fc_actions = nn.Linear(D, cfg.num_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        H, W = self.cfg.image_size
        self.pos_enc_iar = positionalencoding2d(NFB, H, W)
        self.final_head_mid = nn.Sequential()
        for i in range(self.final_head_mid_num):
            if i == 0:
                self.final_head_mid.add_module('fc%d' % i, nn.Linear(NFB, NFB))
            else:
                self.final_head_mid.add_module('fc%d' % i, nn.Linear(NFB, NFB))
            self.final_head_mid.add_module('relu%d' % i, nn.ReLU())

        if self.use_act_loss:
            self.fc_actions_with_gr = nn.Sequential(
                    self.final_head_mid,
                    nn.Linear(NFB, self.cfg.num_actions),
            )
        if self.use_recon_loss:
            self.fc_recon_with_gr = nn.Sequential(
                    self.final_head_mid,
                    nn.Linear(NFB, self.ind_ext_feat_dim),
            )

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        # self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        # images_in, boxes_in = batch_data
        images_in = batch_data['images_in']
        boxes_in = batch_data['boxes_in']
        # images_person_in = batch_data['images_person_in']

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.view(B, T, N, D, K*K)
        boxes_features_recon = boxes_features.view(B, T, N, K*K*D)
        boxes_features = boxes_features.permute(0, 2, 1, 4, 3).contiguous()
        boxes_features = boxes_features.view(B*N, T, K*K, D) # B*N, T, K*K, D

        # HiGCIN Inference
        boxes_features = self.BIM(boxes_features) # B*N, T, K*K, D
        boxes_features = self.person_avg_pool(boxes_features) # B*N, T, D
        boxes_features = boxes_features.view(B, N, T, D).contiguous().permute(0, 2, 1, 3) # B, T, N, D
        boxes_states = self.PIM(boxes_features) # B, T, N, D
        boxes_states = self.dropout(boxes_states)
        torch.cuda.empty_cache()
        NFS = D

        # pooling individual features
        individual_feat, _ = torch.max(boxes_states, dim=1)
        group_feat, _ = torch.max(individual_feat, dim=1)
        group_feat_expand = group_feat.view(B, 1, 1, -1).expand(B, T, N, NFS)
        iar_inp_feat = group_feat_expand

        # encode position infromation
        boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
        boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
        boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
        boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        pos_enc_iar = self.pos_enc_iar.to(device=boxes_in.device)
        iar_loc_feat = torch.transpose(pos_enc_iar[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        iar_loc_feat = iar_loc_feat.view(B, T, N, NFS)
        iar_inp_feat = iar_inp_feat + iar_loc_feat

        ret_dic = {}
        ret_dic['group_feat'] = group_feat
        ret_dic['individual_feat'] = individual_feat.view(B, N*NFS)

        if self.training_stage == 3:
            activities = self.fc_activities(group_feat)
            ret_dic['activities'] = activities
            actions = self.fc_actions(individual_feat).reshape(B * N, -1)
            ret_dic['actions'] = actions
        else:
            if self.use_recon_loss:
                recon_features = self.fc_recon_with_gr(iar_inp_feat)
                ret_dic['recon_features'] = recon_features
                ret_dic['original_features'] = boxes_features_recon
            if self.use_act_loss:
                actions_scores_with_gr = self.fc_actions_with_gr(iar_inp_feat)
                actions_scores_with_gr = torch.mean(actions_scores_with_gr, dim=1).reshape(B * N, -1)
                ret_dic['actions'] = actions_scores_with_gr

        return ret_dic

class GroupRelation_DIN_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, cfg):
        super(GroupRelation_DIN_volleyball, self).__init__()
        self.cfg=cfg
        
        T, N=self.cfg.num_frames, self.cfg.num_boxes
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG=self.cfg.num_graph

        # addtional parameters
        self.eval_only = self.cfg.eval_only
        self.use_random_mask = self.cfg.use_random_mask
        if self.use_random_mask:
            self.random_mask_type = self.cfg.random_mask_type
        self.use_ind_feat = self.cfg.use_ind_feat
        self.use_trans = self.cfg.use_trans
        self.trans_head_num = self.cfg.trans_head_num
        self.trans_layer_num = self.cfg.trans_layer_num
        self.people_pool_type = self.cfg.people_pool_type
        self.use_pos_cond = self.cfg.use_pos_cond
        self.use_tmp_cond = self.cfg.use_tmp_cond
        self.final_head_mid_num = self.cfg.final_head_mid_num
        self.use_recon_loss = self.cfg.use_recon_loss
        self.use_recon_diff_loss = self.cfg.use_recon_diff_loss
        self.use_act_loss = self.cfg.use_act_loss
        self.use_pose_loss = self.cfg.use_pose_loss
        self.use_old_act_rec = self.cfg.old_act_rec
        self.use_res_connect = self.cfg.use_res_connect
        self.use_gen_iar = self.cfg.use_gen_iar
        if self.use_gen_iar:
            self.gen_iar_ratio = self.cfg.gen_iar_ratio
        self.training_stage = self.cfg.training_stage
        self.ind_ext_feat_dim = K * K * D
        
        if cfg.backbone=='inv3':
            self.backbone=MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone=='vgg16':
            self.backbone=MyVGG16(pretrained = True)
        elif cfg.backbone=='vgg19':
            self.backbone=MyVGG19(pretrained = True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained = True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False
        
        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad=False
        
        self.roi_align = RoIAlign(*self.cfg.crop_size)
        # self.avgpool_person = nn.AdaptiveAvgPool2d((1,1))
        self.fc_emb_1 = nn.Linear(K*K*D,NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])
        
        
        #self.gcn_list = torch.nn.ModuleList([ GCN_Module(self.cfg)  for i in range(self.cfg.gcn_layers) ])
        if self.cfg.lite_dim:
            in_dim = self.cfg.lite_dim
            print_log(cfg.log_path, 'Activate lite model inference.')
        else:
            in_dim = NFB
            print_log(cfg.log_path, 'Deactivate lite model inference.')

        if not self.cfg.hierarchical_inference:
            # self.DPI = Dynamic_Person_Inference(
            #     in_dim = in_dim,
            #     person_mat_shape = (10, 12),
            #     stride = cfg.stride,
            #     kernel_size = cfg.ST_kernel_size,
            #     dynamic_sampling=cfg.dynamic_sampling,
            #     sampling_ratio = cfg.sampling_ratio, # [1,2,4]
            #     group = cfg.group,
            #     scale_factor = cfg.scale_factor,
            #     beta_factor = cfg.beta_factor,
            #     parallel_inference = cfg.parallel_inference,
            #     cfg = cfg)
            self.DPI = Multi_Dynamic_Inference(
                in_dim = in_dim,
                person_mat_shape = (10, 12),
                stride = cfg.stride,
                kernel_size = cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio = cfg.sampling_ratio, # [1,2,4]
                group = cfg.group,
                scale_factor = cfg.scale_factor,
                beta_factor = cfg.beta_factor,
                parallel_inference = cfg.parallel_inference,
                num_DIM = cfg.num_DIM,
                cfg = cfg)
            print_log(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        else:
            self.DPI = Hierarchical_Dynamic_Inference(
                in_dim = in_dim,
                person_mat_shape=(10, 12),
                stride=cfg.stride,
                kernel_size=cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio=cfg.sampling_ratio,  # [1,2,4]
                group=cfg.group,
                scale_factor=cfg.scale_factor,
                beta_factor=cfg.beta_factor,
                parallel_inference=cfg.parallel_inference,
                cfg = cfg,)
            print(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        self.dpi_nl = nn.LayerNorm([T, N, in_dim])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)


        # Lite Dynamic inference
        if self.cfg.lite_dim:
            self.point_conv = nn.Conv2d(NFB, in_dim, kernel_size = 1, stride = 1)
            self.point_ln = nn.LayerNorm([T, N, in_dim])
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)
            self.fc_actions = nn.Linear(in_dim, self.cfg.num_actions)
        else:
            self.fc_activities=nn.Linear(NFG, self.cfg.num_activities)
            self.fc_actions = nn.Linear(NFG, self.cfg.num_actions)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.cfg.lite_dim:
            NFB = in_dim
        else:
            NFB = NFG

        H, W = self.cfg.image_size
        self.pos_enc_iar = positionalencoding2d(NFB, H, W)
        self.final_head_mid = nn.Sequential()
        for i in range(self.final_head_mid_num):
            if i == 0:
                self.final_head_mid.add_module('fc%d' % i, nn.Linear(NFB, NFB))
            else:
                self.final_head_mid.add_module('fc%d' % i, nn.Linear(NFB, NFB))
            self.final_head_mid.add_module('relu%d' % i, nn.ReLU())

        if self.use_act_loss:
            self.fc_actions_with_gr = nn.Sequential(
                    self.final_head_mid,
                    nn.Linear(NFB, self.cfg.num_actions),
            )
        if self.use_recon_loss:
            self.fc_recon_with_gr = nn.Sequential(
                    self.final_head_mid,
                    nn.Linear(NFB, self.ind_ext_feat_dim),
            )

                    
    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k,v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num +=1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num)+' parameters loaded for '+prefix)


    def forward(self,batch_data):
        # images_in, boxes_in = batch_data
        images_in = batch_data['images_in']
        boxes_in = batch_data['boxes_in']
        # images_person_in = batch_data['images_person_in']

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        N=self.cfg.num_boxes

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]

        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W
        boxes_in_flat=torch.reshape(boxes_in,(B*T*N,4))  #B*T*N, 4

        boxes_idx=[i * torch.ones(N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*N,))  #B*T*N,
        
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4]==torch.Size([OH,OW])
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW
        
        
        # RoI Align
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        boxes_features=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*N, D, K, K,
        boxes_features=boxes_features.reshape(B,T,N,-1)  #B,T,N, D*K*K
        boxes_features_recon = boxes_features.clone()

        # Embedding 
        boxes_features=self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features=self.nl_emb_1(boxes_features)
        boxes_features=F.relu(boxes_features, inplace = True)

        if self.cfg.lite_dim:
            boxes_features = boxes_features.permute(0, 3, 1, 2)
            boxes_features = self.point_conv(boxes_features)
            boxes_features = boxes_features.permute(0, 2, 3, 1)
            boxes_features = self.point_ln(boxes_features)
            boxes_features = F.relu(boxes_features, inplace = True)
        else:
            None

        # Dynamic graph inference
        # graph_boxes_features = self.DPI(boxes_features)
        graph_boxes_features, ft_infer_MAD = self.DPI(boxes_features)
        torch.cuda.empty_cache()


        if self.cfg.backbone == 'res18':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            graph_boxes_features = self.dpi_nl(graph_boxes_features)
            graph_boxes_features = F.relu(graph_boxes_features, inplace=True)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dropout_global(boxes_states)
        elif self.cfg.backbone == 'vgg16':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dpi_nl(boxes_states)
            boxes_states = F.relu(boxes_states, inplace = True)
            boxes_states = self.dropout_global(boxes_states)
        elif self.cfg.backbone == 'inv3':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            graph_boxes_features = self.dpi_nl(graph_boxes_features)
            graph_boxes_features = F.relu(graph_boxes_features, inplace=True)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dropout_global(boxes_states)

        # pooling individual features
        NFS = self.cfg.lite_dim
        individual_feat, _ = torch.max(boxes_states, dim=1)
        group_feat, _ = torch.max(individual_feat, dim=1)
        group_feat_expand = group_feat.view(B, 1, 1, -1).expand(B, T, N, NFS)
        iar_inp_feat = group_feat_expand

        # encode position infromation
        boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
        boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
        boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
        boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        pos_enc_iar = self.pos_enc_iar.to(device=boxes_in.device)
        iar_loc_feat = torch.transpose(pos_enc_iar[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        iar_loc_feat = iar_loc_feat.view(B, T, N, NFS)
        iar_inp_feat = iar_inp_feat + iar_loc_feat

        ret_dic = {}
        ret_dic['group_feat'] = group_feat
        ret_dic['individual_feat'] = individual_feat.view(B, N*NFS)

        if self.training_stage == 3:
            activities = self.fc_activities(group_feat)
            ret_dic['activities'] = activities
            actions = self.fc_actions(individual_feat).reshape(B * N, -1)
            ret_dic['actions'] = actions
        else:
            if self.use_recon_loss:
                recon_features = self.fc_recon_with_gr(iar_inp_feat)
                ret_dic['recon_features'] = recon_features
                ret_dic['original_features'] = boxes_features_recon
            if self.use_act_loss:
                actions_scores_with_gr = self.fc_actions_with_gr(iar_inp_feat)
                actions_scores_with_gr = torch.mean(actions_scores_with_gr, dim=1).reshape(B * N, -1)
                ret_dic['actions'] = actions_scores_with_gr

        return ret_dic

class GroupActivity_volleyball(nn.Module):
    """
    main module of GA recognition for the volleyball dataset
    """

    def __init__(self, cfg):
        super(GroupActivity_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        H, W = self.cfg.image_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes

        backbone_pretrain = True
        # backbone_pretrain = False
        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=backbone_pretrain)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=backbone_pretrain)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=backbone_pretrain)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=backbone_pretrain)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=backbone_pretrain)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # transformer
        self.pos_enc_ind = positionalencoding2d(NFB, H, W)
        self.tem_enc_ind = positionalencoding1d(NFB, 100)

        self.temporal_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
        ])
        self.spatial_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1)
        ])

        self.temporal_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])
        self.spatial_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])

        # IAR with locations
        self.pos_enc_iar = positionalencoding2d(2*NFB, H, W)
        # self.fc_actions_with_gr = nn.Linear(NFB*2, self.cfg.num_actions)

        self.fc_actions = nn.Linear(NFB*2, self.cfg.num_actions)
        self.fc_activities = nn.Linear(NFB*2, self.cfg.num_activities)
        # self.fc_activities = nn.Sequential(
        #         nn.Linear(NFB*2, NFB),
        #         nn.ReLU(),
        #         nn.Linear(NFB, NFB),
        #         nn.ReLU(),
        #         nn.Linear(NFB, cfg.num_activities),
        #         )

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        # images_in, boxes_in, images_person_in = batch_data
        images_in = batch_data['images_in']
        boxes_in = batch_data['boxes_in']
        images_person_in = batch_data['images_person_in']

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,

        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # encode position infromation
        boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
        boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
        boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
        boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        pos_enc_ind = self.pos_enc_ind.to(device=boxes_in.device)
        ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        ind_loc_feat = ind_loc_feat.view(B, T, N, NFB)

        # encode temporal infromation
        tem_enc_ind = self.tem_enc_ind.to(device=boxes_in.device)
        people_tem = torch.arange(T).to(device=boxes_in.device).long()
        people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
        ind_tem_feat = tem_enc_ind[people_temp, :].view(B, T, N, NFB)

        # generate individual features
        ind_feat_set = boxes_features + ind_loc_feat + ind_tem_feat
        ind_feat_set_upper = ind_feat_set.view(B*T, N, NFB)
        ind_feat_set_bottom = torch.transpose(ind_feat_set, 1, 2).contiguous().view(B*N, T, NFB)

        # generate padding masks for transformer
        people_pad_mask = (torch.sum(boxes_in, dim=-1)==0).bool()
        people_pad_mask_spatial = people_pad_mask.view(B*T, N)
        people_pad_mask_temporal = torch.transpose(people_pad_mask, 1, 2).contiguous().view(B*N, T)

        # transformer encoder
        ind_feat_enc_upper = self.spatial_transformer_encoder[0](ind_feat_set_upper, src_key_padding_mask=people_pad_mask_spatial)
        ind_feat_enc_upper_res = self.spatial_transformer_mlp[0](ind_feat_enc_upper)
        ind_feat_enc_upper = ind_feat_set_upper + ind_feat_enc_upper_res
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, T, N, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2).contiguous().view(B*N, T, NFB)
        ind_feat_enc_upper = self.temporal_transformer_encoder[0](ind_feat_enc_upper)                
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, N, T, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2)

        # bottom branch
        ind_feat_enc_bottom = self.temporal_transformer_encoder[1](ind_feat_set_bottom)
        ind_feat_enc_bottom_res = self.temporal_transformer_mlp[0](ind_feat_enc_bottom)
        ind_feat_enc_bottom = ind_feat_set_bottom + ind_feat_enc_bottom_res
        ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, N, T, NFB)                
        ind_feat_enc_bottom = torch.transpose(ind_feat_enc_bottom, 1, 2).contiguous().view(B*T, N, NFB)
        ind_feat_enc_bottom = self.spatial_transformer_encoder[1](ind_feat_enc_bottom, src_key_padding_mask=people_pad_mask_spatial)
        ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, T, N, NFB)

        boxes_states = torch.cat([ind_feat_enc_upper, ind_feat_enc_bottom], dim=-1)
        NFS = NFB*2

        # ============================= general group activity recognition
        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFS)  # B*T*N, NFS
        actions_scores = self.fc_actions(boxes_states_flat)  # B*T*N, actn_num

        # Predict activities
        # boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        # boxes_states_pooled_flat = boxes_states_pooled.reshape(-1, NFS)
        # activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num

        # Temporal fusion
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = torch.mean(actions_scores, dim=1).reshape(B * N, -1)
        # activities_scores = activities_scores.reshape(B, T, -1)
        # activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)
        # ============================= general group activity recognition

        # pooling individual features
        individual_feat, _ = torch.max(boxes_states, dim=1)
        group_feat, _ = torch.max(individual_feat, dim=1)
        group_feat_expand = group_feat.view(B, 1, 1, -1).expand(B, T, N, NFS)

        ret_dic = {}
        ret_dic['actions'] = actions_scores
        # ret_dic['activities'] = activities_scores
        ret_dic['group_feat'] = group_feat
        ret_dic['individual_feat'] = individual_feat.view(B, N*NFS)

        return ret_dic

class DualAI_volleyball(nn.Module):
    """
    main module of GA recognition for the volleyball dataset
    """

    def __init__(self, cfg):
        super(DualAI_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        H, W = self.cfg.image_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # transformer
        self.pos_enc_ind = positionalencoding2d(NFB, H, W)
        self.tem_enc_ind = positionalencoding1d(NFB, 100)

        self.temporal_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
        ])
        self.spatial_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1)
        ])
        self.temporal_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])
        self.spatial_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])

        self.fc_actions = nn.Linear(NFB*2, self.cfg.num_actions)
        # self.fc_activities = nn.Linear(NFB*2, self.cfg.num_activities)
        self.fc_activities = nn.Sequential(
                nn.Linear(NFB*2, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, cfg.num_activities),
                )

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        if len(batch_data) == 2:
            images_in, boxes_in = batch_data
        else:
            images_in = batch_data['images_in']
            boxes_in = batch_data['boxes_in']

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,

        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # encode position infromation
        boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
        boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
        boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
        boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        pos_enc_ind = self.pos_enc_ind.to(device=boxes_in.device)
        ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        ind_loc_feat = ind_loc_feat.view(B, T, N, NFB)

        # encode temporal infromation
        tem_enc_ind = self.tem_enc_ind.to(device=boxes_in.device)
        people_tem = torch.arange(T).to(device=boxes_in.device).long()
        people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
        ind_tem_feat = tem_enc_ind[people_temp, :].view(B, T, N, NFB)

        # generate individual features
        ind_feat_set = boxes_features + ind_loc_feat + ind_tem_feat
        ind_feat_set_upper = ind_feat_set.view(B*T, N, NFB)
        ind_feat_set_bottom = torch.transpose(ind_feat_set, 1, 2).contiguous().view(B*N, T, NFB)

        # generate padding masks for transformer
        people_pad_mask = (torch.sum(boxes_in, dim=-1)==0).bool()
        people_pad_mask_spatial = people_pad_mask.view(B*T, N)
        people_pad_mask_temporal = torch.transpose(people_pad_mask, 1, 2).contiguous().view(B*N, T)

        # transformer encoder
        ind_feat_enc_upper = self.spatial_transformer_encoder[0](ind_feat_set_upper, src_key_padding_mask=people_pad_mask_spatial)
        ind_feat_enc_upper_res = self.spatial_transformer_mlp[0](ind_feat_enc_upper)
        ind_feat_enc_upper = ind_feat_set_upper + ind_feat_enc_upper_res
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, T, N, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2).contiguous().view(B*N, T, NFB)
        ind_feat_enc_upper = self.temporal_transformer_encoder[0](ind_feat_enc_upper)                
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, N, T, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2)

        # bottom branch
        ind_feat_enc_bottom = self.temporal_transformer_encoder[1](ind_feat_set_bottom)
        ind_feat_enc_bottom_res = self.temporal_transformer_mlp[0](ind_feat_enc_bottom)
        ind_feat_enc_bottom = ind_feat_set_bottom + ind_feat_enc_bottom_res
        ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, N, T, NFB)                
        ind_feat_enc_bottom = torch.transpose(ind_feat_enc_bottom, 1, 2).contiguous().view(B*T, N, NFB)
        ind_feat_enc_bottom = self.spatial_transformer_encoder[1](ind_feat_enc_bottom, src_key_padding_mask=people_pad_mask_spatial)
        ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, T, N, NFB)

        boxes_states = torch.cat([ind_feat_enc_upper, ind_feat_enc_bottom], dim=-1)
        NFS = NFB*2

        # ============================= general group activity recognition
        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFS)  # B*T*N, NFS
        actions_scores = self.fc_actions(boxes_states_flat)  # B*T*N, actn_num

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(-1, NFS)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num

        # Temporal fusion
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = torch.mean(actions_scores, dim=1).reshape(B * N, -1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)
        # ============================= general group activity recognition

        ret_dic = {}
        ret_dic['actions'] = actions_scores
        ret_dic['activities'] = activities_scores

        return ret_dic

class PersonAction_volleyball(nn.Module):

    def __init__(self, cfg):
        super(PersonAction_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        H, W = self.cfg.image_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # transformer
        self.pos_enc_ind = positionalencoding2d(NFB, H, W)
        self.tem_enc_ind = positionalencoding1d(NFB, 100)

        self.temporal_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
        ])
        self.spatial_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1)
        ])

        self.temporal_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])
        self.spatial_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])

        # IAR with locations
        self.pos_enc_iar = positionalencoding2d(2*NFB, H, W)
        # self.fc_actions_with_gr = nn.Linear(NFB*2, self.cfg.num_actions)

        self.fc_actions = nn.Linear(NFB*2, self.cfg.num_actions)
        # self.fc_activities = nn.Linear(NFB*2, self.cfg.num_activities)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        # images_in, boxes_in, images_person_in = batch_data
        images_in = batch_data['images_in']
        boxes_in = batch_data['boxes_in']
        images_person_in = batch_data['images_person_in']

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,

        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # encode position infromation
        boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
        boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
        boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
        boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        pos_enc_ind = self.pos_enc_ind.to(device=boxes_in.device)
        ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        ind_loc_feat = ind_loc_feat.view(B, T, N, NFB)

        # encode temporal infromation
        tem_enc_ind = self.tem_enc_ind.to(device=boxes_in.device)
        people_tem = torch.arange(T).to(device=boxes_in.device).long()
        people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
        ind_tem_feat = tem_enc_ind[people_temp, :].view(B, T, N, NFB)

        # generate individual features
        ind_feat_set = boxes_features + ind_loc_feat + ind_tem_feat
        ind_feat_set_upper = ind_feat_set.view(B*T, N, NFB)
        ind_feat_set_bottom = torch.transpose(ind_feat_set, 1, 2).contiguous().view(B*N, T, NFB)

        # generate padding masks for transformer
        people_pad_mask = (torch.sum(boxes_in, dim=-1)==0).bool()
        people_pad_mask_spatial = people_pad_mask.view(B*T, N)
        people_pad_mask_temporal = torch.transpose(people_pad_mask, 1, 2).contiguous().view(B*N, T)

        # transformer encoder
        ind_feat_enc_upper = self.spatial_transformer_encoder[0](ind_feat_set_upper, src_key_padding_mask=people_pad_mask_spatial)
        ind_feat_enc_upper_res = self.spatial_transformer_mlp[0](ind_feat_enc_upper)
        ind_feat_enc_upper = ind_feat_set_upper + ind_feat_enc_upper_res
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, T, N, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2).contiguous().view(B*N, T, NFB)
        ind_feat_enc_upper = self.temporal_transformer_encoder[0](ind_feat_enc_upper)                
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, N, T, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2)

        # bottom branch
        ind_feat_enc_bottom = self.temporal_transformer_encoder[1](ind_feat_set_bottom)
        ind_feat_enc_bottom_res = self.temporal_transformer_mlp[0](ind_feat_enc_bottom)
        ind_feat_enc_bottom = ind_feat_set_bottom + ind_feat_enc_bottom_res
        ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, N, T, NFB)                
        ind_feat_enc_bottom = torch.transpose(ind_feat_enc_bottom, 1, 2).contiguous().view(B*T, N, NFB)
        ind_feat_enc_bottom = self.spatial_transformer_encoder[1](ind_feat_enc_bottom, src_key_padding_mask=people_pad_mask_spatial)
        ind_feat_enc_bottom = ind_feat_enc_bottom.view(B, T, N, NFB)

        boxes_states = torch.cat([ind_feat_enc_upper, ind_feat_enc_bottom], dim=-1)
        NFS = NFB*2

        # ============================= general group activity recognition
        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFS)  # B*T*N, NFS
        actions_scores = self.fc_actions(boxes_states_flat)  # B*T*N, actn_num

        # Temporal fusion
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = torch.mean(actions_scores, dim=1).reshape(B * N, -1)
        # ============================= general group activity recognition

        # pooling individual features
        individual_feat, _ = torch.max(boxes_states, dim=1)
        group_feat, _ = torch.max(individual_feat, dim=1)
        group_feat_expand = group_feat.view(B, 1, 1, -1).expand(B, T, N, NFS)

        ret_dic = {}
        ret_dic['pseudo_scores'] = actions_scores
        ret_dic['person_features'] = individual_feat.view(B, N, NFS)

        return ret_dic

class PersonActionSigleBranch_volleyball(nn.Module):

    def __init__(self, cfg):
        super(PersonActionSigleBranch_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        H, W = self.cfg.image_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # transformer
        self.pos_enc_ind = positionalencoding2d(NFB, H, W)
        self.tem_enc_ind = positionalencoding1d(NFB, 100)

        self.temporal_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
        ])
        self.spatial_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
        ])

        self.temporal_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])
        self.spatial_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])

        # IAR with locations
        self.pos_enc_iar = positionalencoding2d(NFB, H, W)
        # self.fc_actions_with_gr = nn.Linear(NFB*2, self.cfg.num_actions)

        self.fc_actions = nn.Linear(NFB, self.cfg.num_actions)
        # self.fc_activities = nn.Linear(NFB*2, self.cfg.num_activities)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        # images_in, boxes_in, images_person_in = batch_data
        images_in = batch_data['images_in']
        boxes_in = batch_data['boxes_in']
        images_person_in = batch_data['images_person_in']

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,

        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # encode position infromation
        boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
        boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
        boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
        boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        pos_enc_ind = self.pos_enc_ind.to(device=boxes_in.device)
        ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        ind_loc_feat = ind_loc_feat.view(B, T, N, NFB)

        # encode temporal infromation
        tem_enc_ind = self.tem_enc_ind.to(device=boxes_in.device)
        people_tem = torch.arange(T).to(device=boxes_in.device).long()
        people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
        ind_tem_feat = tem_enc_ind[people_temp, :].view(B, T, N, NFB)

        # generate individual features
        ind_feat_set = boxes_features + ind_loc_feat + ind_tem_feat
        ind_feat_set_upper = ind_feat_set.view(B*T, N, NFB)
        ind_feat_set_bottom = torch.transpose(ind_feat_set, 1, 2).contiguous().view(B*N, T, NFB)

        # generate padding masks for transformer
        people_pad_mask = (torch.sum(boxes_in, dim=-1)==0).bool()
        people_pad_mask_spatial = people_pad_mask.view(B*T, N)
        people_pad_mask_temporal = torch.transpose(people_pad_mask, 1, 2).contiguous().view(B*N, T)

        # transformer encoder
        ind_feat_enc_upper = self.spatial_transformer_encoder[0](ind_feat_set_upper, src_key_padding_mask=people_pad_mask_spatial)
        ind_feat_enc_upper_res = self.spatial_transformer_mlp[0](ind_feat_enc_upper)
        ind_feat_enc_upper = ind_feat_set_upper + ind_feat_enc_upper_res
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, T, N, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2).contiguous().view(B*N, T, NFB)
        ind_feat_enc_upper = self.temporal_transformer_encoder[0](ind_feat_enc_upper)                
        ind_feat_enc_upper = ind_feat_enc_upper.view(B, N, T, NFB)
        ind_feat_enc_upper = torch.transpose(ind_feat_enc_upper, 1, 2)

        boxes_states = torch.cat([ind_feat_enc_upper], dim=-1)
        NFS = NFB

        # ============================= general group activity recognition
        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFS)  # B*T*N, NFS
        actions_scores = self.fc_actions(boxes_states_flat)  # B*T*N, actn_num

        # Temporal fusion
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = torch.mean(actions_scores, dim=1).reshape(B * N, -1)
        # ============================= general group activity recognition

        # pooling individual features
        individual_feat, _ = torch.max(boxes_states, dim=1)
        group_feat, _ = torch.max(individual_feat, dim=1)
        group_feat_expand = group_feat.view(B, 1, 1, -1).expand(B, T, N, NFS)

        ret_dic = {}
        ret_dic['pseudo_scores'] = actions_scores
        ret_dic['person_features'] = individual_feat.view(B, N, NFS)

        return ret_dic

class PersonActionSigleBranchTemporal_volleyball(nn.Module):

    def __init__(self, cfg):
        super(PersonActionSigleBranchTemporal_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        H, W = self.cfg.image_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # transformer
        self.pos_enc_ind = positionalencoding2d(NFB, H, W)
        self.tem_enc_ind = positionalencoding1d(NFB, 100)

        self.temporal_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=NFB, nhead=1, batch_first=True, dropout=0.0), num_layers=1),
        ])

        self.temporal_transformer_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NFB, NFB),
                nn.ReLU(),
                nn.Linear(NFB, NFB),
            )
        ])

        # IAR with locations
        self.pos_enc_iar = positionalencoding2d(NFB, H, W)
        # self.fc_actions_with_gr = nn.Linear(NFB*2, self.cfg.num_actions)

        self.fc_actions = nn.Linear(NFB, self.cfg.num_actions)
        # self.fc_activities = nn.Linear(NFB*2, self.cfg.num_activities)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        # images_in, boxes_in, images_person_in = batch_data
        images_in = batch_data['images_in']
        boxes_in = batch_data['boxes_in']
        images_person_in = batch_data['images_person_in']

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,

        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # encode position infromation
        boxes_in_x_center = (boxes_in[:, :, :, 0]+boxes_in[:, :, :, 2])/2
        boxes_in_y_center = (boxes_in[:, :, :, 1]+boxes_in[:, :, :, 3])/2
        boxes_in_x_center_view = boxes_in_x_center.view(B*T*N)
        boxes_in_y_center_view = boxes_in_y_center.view(B*T*N)
        pos_enc_ind = self.pos_enc_ind.to(device=boxes_in.device)
        ind_loc_feat = torch.transpose(pos_enc_ind[:, boxes_in_y_center_view.long(), boxes_in_x_center_view.long()], 0, 1)
        ind_loc_feat = ind_loc_feat.view(B, T, N, NFB)

        # encode temporal infromation
        tem_enc_ind = self.tem_enc_ind.to(device=boxes_in.device)
        people_tem = torch.arange(T).to(device=boxes_in.device).long()
        people_temp = people_tem.view(1, T, 1).expand(B, T, N).reshape(B*T*N)
        ind_tem_feat = tem_enc_ind[people_temp, :].view(B, T, N, NFB)

        # generate individual features
        ind_feat_set = boxes_features + ind_loc_feat + ind_tem_feat
        ind_feat_set = ind_feat_set.view(B, T, N, NFB)
        ind_feat_set = torch.transpose(ind_feat_set, 1, 2).contiguous().view(B*N, T, NFB)
        ind_feat_set = self.temporal_transformer_encoder[0](ind_feat_set)                
        ind_feat_set = ind_feat_set.view(B, N, T, NFB)
        ind_feat_set = torch.transpose(ind_feat_set, 1, 2)
        boxes_states = torch.cat([ind_feat_set], dim=-1)
        NFS = NFB

        # ============================= general group activity recognition
        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFS)  # B*T*N, NFS
        actions_scores = self.fc_actions(boxes_states_flat)  # B*T*N, actn_num

        # Temporal fusion
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = torch.mean(actions_scores, dim=1).reshape(B * N, -1)
        # ============================= general group activity recognition

        # pooling individual features
        individual_feat, _ = torch.max(boxes_states, dim=1)
        group_feat, _ = torch.max(individual_feat, dim=1)
        group_feat_expand = group_feat.view(B, 1, 1, -1).expand(B, T, N, NFS)

        ret_dic = {}
        ret_dic['pseudo_scores'] = actions_scores
        ret_dic['person_features'] = individual_feat.view(B, N, NFS)

        return ret_dic