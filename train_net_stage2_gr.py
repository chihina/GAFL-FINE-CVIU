'''
    Train the model with person action/appearance features.
'''

import torch
import torch.optim as optim

import time
import random
import os
import sys
import wandb
from tqdm import tqdm
from collections import OrderedDict

from config import *
from volleyball import *
from collective import *
from basketball import *
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

def train_net(cfg):
    """
    training gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list
    
    # Show config parameters
    cfg.init_config()
    show_config(cfg)

    # Save config parameters
    cfg.model_exp_name = os.path.basename(cfg.result_path)
    cfg_save_path = os.path.join(cfg.result_path, 'cfg.pickle')
    with open(cfg_save_path, "wb") as f:
        pickle.dump(cfg, f)

    # acceralate the training process
    torch.backends.cudnn.benchmark = True
    
    # Reading dataset
    training_set, validation_set, all_set = return_dataset(cfg)
    cfg.num_boxes = all_set.get_num_boxes_max()

    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 8, # 4,
        'pin_memory': True,
    }
    training_loader=data.DataLoader(training_set,**params)
    
    params['batch_size']=cfg.test_batch_size
    validation_loader=data.DataLoader(validation_set,**params)
    
    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    torch.cuda.manual_seed(cfg.train_random_seed)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    cfg.device = device
    
    # Build model and optimizer
    basenet_list={'volleyball':Basenet_volleyball, 'collective':Basenet_collective}
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
    
    # build main GAFL network
    if cfg.training_stage==1:
        Basenet = basenet_list[cfg.dataset_name]
        model = Basenet(cfg)
    elif cfg.training_stage==2:
        GCNnet = gcnnet_list[cfg.inference_module_name]
        model = GCNnet(cfg)
        # Load backbone
        if cfg.load_stage2model:
            state = torch.load(cfg.stage2model)
            new_state = OrderedDict()
            for k, v in state['state_dict'].items():
                name = k[7:] 
                new_state[name] = v
            model.load_state_dict(new_state)
            # model.load_state_dict(new_state, strict=False)
            print_log(cfg.log_path, 'Loading stage2 model: ' + cfg.stage2model)

            # model.load_state_dict(state['state_dict'])
            print_log(cfg.log_path, 'Loading stage2 model: ' + cfg.stage2model)
        elif cfg.load_backbone_stage2:
            model.loadmodel(cfg.stage1_model_path)
        else:
            print_log(cfg.log_path, 'Not loading stage1 or stage2 model.')
    else:
        assert(False)


    # move models to gpu
    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)

    model=model.to(device=device)

    # set mode of models    
    model.train()
    if cfg.set_bn_eval:
        model.apply(set_bn_eval)

    # set parameters to be optimized
    optimizer_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer=optim.Adam(optimizer_params, lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)

    models = {'model': model}

    # train_list={'volleyball':train_volleyball, 'collective':train_collective, 'basketball':train_volleyball}
    # train_list={'volleyball':train_collective, 'collective':train_collective, 'basketball':train_collective}
    # test_list={'volleyball':test_volleyball, 'collective':test_collective, 'basketball':test_volleyball}
    # test_list={'volleyball':test_collective, 'collective':test_collective, 'basketball':test_collective}
    # train=train_list[cfg.dataset_name]
    # test=test_list[cfg.dataset_name]
    train = train_collective
    test = test_collective
    
    if cfg.test_before_train:
        test_info=test(validation_loader, models, device, 0, cfg)
        print(test_info)

    # Training iteration
    best_result = {'epoch':0, 'activities_acc':0, 'actions_acc':0, 'loss':100000000000000}
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch+cfg.max_epoch):
        
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])
            
        # One epoch of forward and backward
        train_info=train(training_loader, models, device, optimizer, epoch, cfg)
        show_epoch_info('Train', cfg.log_path, train_info)
        for wandb_loss_name in cfg.wandb_loss_list:
            wandb.log({f"Train {wandb_loss_name}": train_info[wandb_loss_name]}, step=epoch)

        # Test
        if epoch % cfg.test_interval_epoch == 0:
            test_info=test(validation_loader, models, device, epoch, cfg)
            show_epoch_info('Test', cfg.log_path, test_info)
            for wandb_loss_name in cfg.wandb_loss_list:
                wandb.log({f"Test {wandb_loss_name}": test_info[wandb_loss_name]}, step=epoch)

            if test_info['loss']<best_result['loss']:
                best_result=test_info
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, cfg.result_path+'/best_model.pth')
                print('Best model saved.')
            print_log(cfg.log_path, 
                      'Best loss: %.2f%% at epoch #%d.'%(best_result['loss'], best_result['epoch']))

            # Save model
            if cfg.training_stage==2:
                pass
                # None
                # state = {
                #     'epoch': epoch,
                #     'state_dict': model.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                # }
                # filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['loss'])
                # torch.save(state, filepath)
                # print('model saved to:',filepath)
            elif cfg.training_stage==1:
                if test_info['loss'] == best_result['loss']:
                    for m in model.modules():
                        if isinstance(m, Basenet):
                            filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['loss'])
                            m.savemodel(filepath)
            else:
                assert False
    
    return cfg

def train_collective(data_loader, models, device, optimizer, epoch, cfg):
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    loss_act_meter=AverageMeter()
    loss_recon_meter=AverageMeter()
    loss_recon_pose_feat_meter=AverageMeter()
    loss_recon_pose_coord_meter=AverageMeter()
    loss_jae_meter=AverageMeter()
    loss_traj_meter=AverageMeter()
    loss_person_num_meter=AverageMeter()
    epoch_timer=Timer()
    activities_conf = ConfusionMeter(cfg.num_activities)
    actions_conf = ConfusionMeter(cfg.num_actions)

    model = models['model']

    for batch_idx, batch_data in enumerate(tqdm(data_loader)):
        if batch_idx % 100 == 0 and batch_idx > 0:
            print('Training in processing {}/{}, Loss: {:.4f}'.format(batch_idx, len(data_loader), loss_meter.avg))

        start_time = time.time()
        model.train()
        model.apply(set_bn_eval)
    
        # prepare batch data
        for key in batch_data.keys():
            if torch.is_tensor(batch_data[key]):
                batch_data[key] = batch_data[key].to(device=device)

        images_in = batch_data['images_in']
        batch_size, num_frames, _, _, _ = images_in.shape
        actions_in=batch_data['actions_in'].reshape((batch_size,num_frames,cfg.num_boxes))
        activities_in=batch_data['activities_in'].reshape((batch_size,num_frames))
        activities_in=activities_in[:,0].reshape((batch_size,))
        boxes_in=batch_data['boxes_in'].reshape((batch_size,num_frames,cfg.num_boxes,4))
        bboxes_num=batch_data['bboxes_num'].reshape(batch_size,num_frames)

        ret = model(batch_data)

        loss_list = []
        if 'activities' in list(ret.keys()):
            activities_scores = ret['activities']
            activities_loss = F.cross_entropy(activities_scores, activities_in)
            loss_list.append(activities_loss)
            activities_labels = torch.argmax(activities_scores, dim=1)
            activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
            activities_accuracy = activities_correct.item() / activities_scores.shape[0]
            activities_meter.update(activities_accuracy, activities_scores.shape[0])
            activities_conf.add(activities_labels, activities_in)

        if 'actions' in list(ret.keys()):
            actions_scores = ret['actions'].reshape(batch_size, cfg.num_boxes, -1)
            actions_in_nopad=[]
            actions_scores_nopad=[]
            for b in range(batch_size):
                N = bboxes_num[b][0]
                actions_in_nopad.append(actions_in[b][0][:N])
                actions_scores_nopad.append(actions_scores[b][:N])
            actions_scores=torch.cat(actions_scores_nopad,dim=0).reshape(-1, cfg.num_actions)
            actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)
            actions_loss=F.cross_entropy(actions_scores,actions_in) * 1e-2
            loss_list.append(actions_loss)
            loss_act_meter.update(actions_loss.item(), batch_size)
            actions_labels=torch.argmax(actions_scores,dim=1)
            actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
            actions_accuracy = actions_correct.item() / actions_scores.shape[0]
            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            actions_conf.add(actions_labels, actions_in)

        if 'recon_features' in list(ret.keys()):
            recon_features = ret['recon_features']
            original_features = ret['original_features']
            
            recon_features_nopad=[]
            original_features_nopad=[]
            for b in range(batch_size):
                N = bboxes_num[b][0]
                recon_features_nopad.append(recon_features[b][:, :N, :].reshape(-1,))
                original_features_nopad.append(original_features[b][:, :N, :].reshape(-1,))
            recon_features_nopad=torch.cat(recon_features_nopad,dim=0).reshape(-1,)
            original_features_nopad=torch.cat(original_features_nopad,dim=0).reshape(-1,)
            recon_loss = F.mse_loss(recon_features_nopad, original_features_nopad)

            # recon_loss_all = F.mse_loss(recon_features, original_features, reduction='none')
            # recon_loss_all = recon_loss_all.mean(dim=-1)
            # people_pad_mask = (torch.sum(boxes_in, dim=-1)==0).bool()
            # recon_loss_all = recon_loss_all * ~people_pad_mask
            # recon_loss_all = recon_loss_all.flatten().sum()
            # recon_loss = recon_loss_all / torch.sum(~people_pad_mask.flatten())

            loss_list.append(recon_loss)
            loss_recon_meter.update(recon_loss.item(), batch_size)
        
        if 'recon_pose_features' in list(ret.keys()):
            recon_pose_features = ret['recon_pose_features']
            original_pose_features = ret['original_pose_features']
            recon_pose_features_nopad=[]
            original_pose_features_nopad=[]
            for b in range(batch_size):
                N = bboxes_num[b][0]
                recon_pose_features_nopad.append(recon_pose_features[b][:, :N, :].reshape(-1,))
                original_pose_features_nopad.append(original_pose_features[b][:, :N, :].reshape(-1,))
            recon_pose_features_nopad=torch.cat(recon_pose_features_nopad,dim=0).reshape(-1,)
            original_pose_features_nopad=torch.cat(original_pose_features_nopad,dim=0).reshape(-1,)
            recon_pose_feat_loss = F.mse_loss(recon_pose_features_nopad, original_pose_features_nopad)
            loss_list.append(recon_pose_feat_loss)
            loss_recon_pose_feat_meter.update(recon_pose_feat_loss.item(), batch_size)
        
        if 'recon_pose_coords' in list(ret.keys()):
            recon_pose_coords = ret['recon_pose_coords']
            original_pose_coords = ret['original_pose_coords']
            recon_pose_coords_nopad=[]
            original_pose_coords_nopad=[]
            for b in range(batch_size):
                N = bboxes_num[b][0]
                recon_pose_coords_nopad.append(recon_pose_coords[b][:, :N, :].reshape(-1,))
                original_pose_coords_nopad.append(original_pose_coords[b][:, :N, :].reshape(-1,))
            recon_pose_coords_nopad=torch.cat(recon_pose_coords_nopad,dim=0).reshape(-1,)
            original_pose_coords_nopad=torch.cat(original_pose_coords_nopad,dim=0).reshape(-1,)
            recon_pose_coord_loss = F.mse_loss(recon_pose_coords_nopad, original_pose_coords_nopad)
            loss_list.append(recon_pose_coord_loss)
            loss_recon_pose_coord_meter.update(recon_pose_coord_loss.item(), batch_size)
        
        if 'estimated_ja' in list(ret.keys()):
            estimated_ja = ret['estimated_ja']
            gt_ja = batch_data['ja_points']

            jae_inf_mask = torch.isinf(gt_ja).sum(dim=-1) == 0
            if jae_inf_mask.sum() == 0:
                jae_loss = torch.tensor(0.0, device=device)
            else:
                estimated_ja = estimated_ja[jae_inf_mask]
                gt_ja = gt_ja[jae_inf_mask]
                jae_loss = F.mse_loss(estimated_ja, gt_ja)
                loss_list.append(jae_loss)
                loss_jae_meter.update(jae_loss.item(), batch_size)
        
        if 'estimated_person_num' in list(ret.keys()):
            estimated_person_num = ret['estimated_person_num'].reshape(batch_size, num_frames)
            person_num_loss = F.mse_loss(estimated_person_num, bboxes_num.float())
            loss_list.append(person_num_loss)
            loss_person_num_meter.update(person_num_loss.item(), batch_size)

        if 'estimated_flow' in list(ret.keys()):
            # generate ground truth flow
            boxes_in = batch_data['boxes_in']
            boxes_in_xmid = (boxes_in[:, :, :, 0] + boxes_in[:, :, :, 2]) / 2
            boxes_in_ymid = (boxes_in[:, :, :, 1] + boxes_in[:, :, :, 3]) / 2
            boxes_in_xmid_diff = boxes_in_xmid[:, 1:] - boxes_in_xmid[:, :-1]
            boxes_in_ymid_diff = boxes_in_ymid[:, 1:] - boxes_in_ymid[:, :-1]
            gt_flow = torch.stack([boxes_in_xmid_diff, boxes_in_ymid_diff], dim=-1)

            # generate estimated flow
            estimated_flow_w_init = ret['estimated_flow']
            estimated_flow = estimated_flow_w_init[:, 1:]

            gt_flow_nopad = []
            estimated_flow_nopad = []
            for b in range(batch_size):
                N = bboxes_num[b][0]
                gt_flow_nopad.append(gt_flow[b][:N])
                estimated_flow_nopad.append(estimated_flow[b][:N])
            gt_flow = torch.cat(gt_flow_nopad, dim=0)
            estimated_flow = torch.cat(estimated_flow_nopad, dim=0)

            # calculate flow loss
            flow_loss = F.mse_loss(estimated_flow, gt_flow)
            loss_list.append(flow_loss)
            loss_traj_meter.update(flow_loss.item(), batch_size)

        # if batch_idx > 5:
            # break

        # Total loss
        total_loss = sum(loss_list)
        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    train_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'loss_act':loss_act_meter.avg,
        'loss_recon':loss_recon_meter.avg,
        'loss_recon_pose_feat':loss_recon_pose_feat_meter.avg,
        'loss_recon_pose_coord':loss_recon_pose_coord_meter.avg,
        'loss_jae':loss_jae_meter.avg,
        'loss_traj':loss_traj_meter.avg,
        'loss_person_num':loss_person_num_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'activities_conf':activities_conf.value(),
        'activities_MPCA':MPCA(activities_conf.value()),
        'actions_acc':actions_meter.avg*100,
        'actions_conf':actions_conf.value(),
        'actions_MPCA':MPCA(actions_conf.value()),
    }  

    return train_info
        
def test_collective(data_loader, models, device, epoch, cfg):
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    loss_act_meter=AverageMeter()
    loss_recon_meter=AverageMeter()
    loss_recon_pose_feat_meter=AverageMeter()
    loss_recon_pose_coord_meter=AverageMeter()
    loss_jae_meter=AverageMeter()
    loss_traj_meter=AverageMeter()
    loss_person_num_meter=AverageMeter()
    epoch_timer=Timer()
    activities_conf = ConfusionMeter(cfg.num_activities)
    actions_conf = ConfusionMeter(cfg.num_actions)

    model = models['model']
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader)):
            for key in batch_data.keys():
                if torch.is_tensor(batch_data[key]):
                    batch_data[key] = batch_data[key].to(device=device)

            images_in = batch_data['images_in']
            batch_size, num_frames, _, _, _ = images_in.shape
            actions_in=batch_data['actions_in'].reshape((batch_size,num_frames,cfg.num_boxes))
            activities_in=batch_data['activities_in'].reshape((batch_size,num_frames))
            activities_in=activities_in[:,0].reshape((batch_size,))
            bboxes_num=batch_data['bboxes_num'].reshape(batch_size,num_frames)
            ret = model(batch_data)

            loss_list = []

            if 'activities' in list(ret.keys()):
                activities_scores = ret['activities']
                activities_loss = F.cross_entropy(activities_scores, activities_in)
                loss_list.append(activities_loss)
                activities_labels = torch.argmax(activities_scores, dim=1)
                activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
                activities_accuracy = activities_correct.item() / activities_scores.shape[0]
                activities_meter.update(activities_accuracy, activities_scores.shape[0])
                activities_conf.add(activities_labels, activities_in)

            if 'actions' in list(ret.keys()):
                actions_scores = ret['actions'].reshape(batch_size, cfg.num_boxes, -1)
                actions_in_nopad=[]
                actions_scores_nopad=[]
                for b in range(batch_size):
                    N = bboxes_num[b][0]
                    actions_in_nopad.append(actions_in[b][0][:N])
                    actions_scores_nopad.append(actions_scores[b][:N])
                actions_scores=torch.cat(actions_scores_nopad,dim=0).reshape(-1, cfg.num_actions)
                actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)
                actions_loss=F.cross_entropy(actions_scores,actions_in)
                loss_list.append(actions_loss)
                loss_act_meter.update(actions_loss.item(), batch_size)
                actions_labels=torch.argmax(actions_scores,dim=1)
                actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
                actions_accuracy = actions_correct.item() / actions_scores.shape[0]
                actions_meter.update(actions_accuracy, actions_scores.shape[0])
                actions_conf.add(actions_labels, actions_in)

            if 'recon_features' in list(ret.keys()):
                recon_features = ret['recon_features']
                original_features = ret['original_features']
                recon_features_nopad=[]
                original_features_nopad=[]
                for b in range(batch_size):
                    N = bboxes_num[b][0]
                    recon_features_nopad.append(recon_features[b][:, :N, :].reshape(-1,))
                    original_features_nopad.append(original_features[b][:, :N, :].reshape(-1,))
                recon_features_nopad=torch.cat(recon_features_nopad,dim=0).reshape(-1,)
                original_features_nopad=torch.cat(original_features_nopad,dim=0).reshape(-1,)
                recon_loss = F.mse_loss(recon_features_nopad, original_features_nopad)
                loss_list.append(recon_loss)
                loss_recon_meter.update(recon_loss.item(), batch_size)
            
            if 'recon_pose_features' in list(ret.keys()):
                recon_pose_features = ret['recon_pose_features']
                original_pose_features = ret['original_pose_features']
                recon_pose_features_nopad=[]
                original_pose_features_nopad=[]
                for b in range(batch_size):
                    N = bboxes_num[b][0]
                    recon_pose_features_nopad.append(recon_pose_features[b][:, :N, :].reshape(-1,))
                    original_pose_features_nopad.append(original_pose_features[b][:, :N, :].reshape(-1,))
                recon_pose_features_nopad=torch.cat(recon_pose_features_nopad,dim=0).reshape(-1,)
                original_pose_features_nopad=torch.cat(original_pose_features_nopad,dim=0).reshape(-1,)
                recon_pose_feat_loss = F.mse_loss(recon_pose_features_nopad, original_pose_features_nopad)
                loss_list.append(recon_pose_feat_loss)
                loss_recon_pose_feat_meter.update(recon_pose_feat_loss.item(), batch_size)
            
            if 'recon_pose_coords' in list(ret.keys()):
                recon_pose_coords = ret['recon_pose_coords']
                original_pose_coords = ret['original_pose_coords']
                recon_pose_coords_nopad=[]
                original_pose_coords_nopad=[]
                for b in range(batch_size):
                    N = bboxes_num[b][0]
                    recon_pose_coords_nopad.append(recon_pose_coords[b][:, :N, :].reshape(-1,))
                    original_pose_coords_nopad.append(original_pose_coords[b][:, :N, :].reshape(-1,))
                recon_pose_coords_nopad=torch.cat(recon_pose_coords_nopad,dim=0).reshape(-1,)
                original_pose_coords_nopad=torch.cat(original_pose_coords_nopad,dim=0).reshape(-1,)
                recon_pose_coord_loss = F.mse_loss(recon_pose_coords_nopad, original_pose_coords_nopad)
                loss_list.append(recon_pose_coord_loss)
                loss_recon_pose_coord_meter.update(recon_pose_coord_loss.item(), batch_size)
            
            if 'estimated_ja' in list(ret.keys()):
                estimated_ja = ret['estimated_ja']
                gt_ja = batch_data['ja_points']

                jae_inf_mask = torch.isinf(gt_ja).sum(dim=-1) == 0
                if jae_inf_mask.sum() == 0:
                    jae_loss = torch.tensor(0.0, device=device)
                else:
                    estimated_ja = estimated_ja[jae_inf_mask]
                    gt_ja = gt_ja[jae_inf_mask]
                    jae_loss = F.mse_loss(estimated_ja, gt_ja)
                    loss_list.append(jae_loss)
                    loss_jae_meter.update(jae_loss.item(), batch_size)
            
            if 'estimated_person_num' in list(ret.keys()):
                estimated_person_num = ret['estimated_person_num'].reshape(batch_size, num_frames)
                person_num_loss = F.mse_loss(estimated_person_num, bboxes_num.float())
                loss_list.append(person_num_loss)
                loss_person_num_meter.update(person_num_loss.item(), batch_size)
            
            if 'estimated_flow' in list(ret.keys()):
                # generate ground truth flow
                boxes_in = batch_data['boxes_in']
                boxes_in_xmid = (boxes_in[:, :, :, 0] + boxes_in[:, :, :, 2]) / 2
                boxes_in_ymid = (boxes_in[:, :, :, 1] + boxes_in[:, :, :, 3]) / 2
                boxes_in_xmid_diff = boxes_in_xmid[:, 1:] - boxes_in_xmid[:, :-1]
                boxes_in_ymid_diff = boxes_in_ymid[:, 1:] - boxes_in_ymid[:, :-1]
                gt_flow = torch.stack([boxes_in_xmid_diff, boxes_in_ymid_diff], dim=-1)

                # generate estimated flow
                estimated_flow_w_init = ret['estimated_flow']
                estimated_flow = estimated_flow_w_init[:, 1:]

                gt_flow_nopad = []
                estimated_flow_nopad = []
                for b in range(batch_size):
                    N = bboxes_num[b][0]
                    gt_flow_nopad.append(gt_flow[b][:N])
                    estimated_flow_nopad.append(estimated_flow[b][:N])
                gt_flow = torch.cat(gt_flow_nopad, dim=0)
                estimated_flow = torch.cat(estimated_flow_nopad, dim=0)

                # calculate flow loss
                flow_loss = F.mse_loss(estimated_flow, gt_flow)
                loss_list.append(flow_loss)
                loss_traj_meter.update(flow_loss.item(), batch_size)

            # if batch_idx > 5:
                # break

            # Total loss
            total_loss = sum(loss_list)
            loss_meter.update(total_loss.item(), batch_size)

    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'loss_act':loss_act_meter.avg,
        'loss_recon':loss_recon_meter.avg,
        'loss_recon_pose_feat':loss_recon_pose_feat_meter.avg,
        'loss_recon_pose_coord':loss_recon_pose_coord_meter.avg,
        'loss_jae':loss_jae_meter.avg,
        'loss_traj':loss_traj_meter.avg,
        'loss_person_num':loss_person_num_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
        'actions_acc':actions_meter.avg*100,
        'actions_conf':actions_conf.value(),
        'actions_MPCA':MPCA(actions_conf.value()),
    }

    return test_info