from volleyball import *
from collective import *
from basketball import *
from jrdb import *

import pickle
from glob import glob

def return_dataset(cfg):
    if cfg.dataset_name=='volleyball':
        all_tracks = pickle.load(open(cfg.data_path + '/tracks_normalized.pkl', 'rb'))

        train_anns = volley_read_dataset(cfg.data_path, cfg.train_seqs, cfg.jae_ann_dir,
                                            cfg.query_dir, cfg.query_type, all_tracks, cfg)
        train_frames = volley_all_frames(train_anns)

        test_anns = volley_read_dataset(cfg.data_path, cfg.test_seqs, cfg.jae_ann_dir, 
                                        cfg.query_dir, cfg.query_type, all_tracks, cfg)
        test_frames = volley_all_frames(test_anns)

        all_anns = {**train_anns, **test_anns}
        all_frames = train_frames + test_frames
        training_set=VolleyballDataset(all_anns,all_tracks,train_frames,
                                      cfg.data_path,cfg.image_size,cfg.person_size,cfg.out_size,cfg.inference_module_name, cfg, num_before=cfg.num_before,
                                       num_after=cfg.num_after,is_training=True,is_finetune=(cfg.training_stage==1),use_jae_loss=cfg.use_jae_loss)

        validation_set=VolleyballDataset(all_anns,all_tracks,test_frames,
                                      cfg.data_path,cfg.image_size,cfg.person_size,cfg.out_size,cfg.inference_module_name, cfg, num_before=cfg.num_before,
                                         num_after=cfg.num_after,is_training=False,is_finetune=(cfg.training_stage==1), use_jae_loss=cfg.use_jae_loss)
        
        all_set = VolleyballDataset(all_anns,all_tracks,all_frames,
                                        cfg.data_path,cfg.image_size,cfg.person_size,cfg.out_size,cfg.inference_module_name, cfg, num_before=cfg.num_before,
                                         num_after=cfg.num_after,is_training=False,is_finetune=(cfg.training_stage==1),use_jae_loss=cfg.use_jae_loss)
    
    elif cfg.dataset_name=='collective':
        train_anns=collective_read_dataset(cfg.data_path, cfg.train_seqs, cfg.query_dir, 
                                           cfg.query_type, cfg)
        train_frames=collective_all_frames(train_anns)

        test_anns=collective_read_dataset(cfg.data_path, cfg.test_seqs, cfg.query_dir, 
                                          cfg.query_type, cfg)
        test_frames=collective_all_frames(test_anns)

        all_anns = {**train_anns, **test_anns}
        all_frames = train_frames + test_frames

        training_set=CollectiveDataset(train_anns,train_frames,
                                      cfg.data_path,cfg.image_size,cfg.person_size,cfg.out_size, cfg,
                                      num_frames = cfg.num_frames, is_training=True,is_finetune=(cfg.training_stage==1))

        validation_set=CollectiveDataset(test_anns,test_frames,
                                      cfg.data_path,cfg.image_size,cfg.person_size,cfg.out_size, cfg,
                                      num_frames = cfg.num_frames, is_training=False,is_finetune=(cfg.training_stage==1))
        
        all_set = CollectiveDataset(all_anns,all_frames,
                                        cfg.data_path,cfg.image_size,cfg.person_size,cfg.out_size, cfg,
                                        num_frames = cfg.num_frames, is_training=False,is_finetune=(cfg.training_stage==1))
    elif cfg.dataset_name=='basketball':
        all_tracks = []

        train_anns=basketball_read_dataset(cfg.data_path, cfg.train_seqs, cfg.query_dir, cfg.query_type, all_tracks, cfg)
        train_frames=basketball_all_frames(train_anns)

        test_anns=basketball_read_dataset(cfg.data_path, cfg.test_seqs, cfg.query_dir, cfg.query_type, all_tracks, cfg)
        test_frames=basketball_all_frames(test_anns)

        all_anns = {**train_anns, **test_anns}
        all_frames = train_frames + test_frames
        
        training_set=BasketballDataset(all_anns,all_tracks,train_frames,
                                      cfg.data_path,cfg.image_size,cfg.person_size,cfg.out_size,cfg.inference_module_name, cfg, 
                                      num_boxes=cfg.num_boxes,num_before=cfg.num_before, num_after=cfg.num_after,
                                      is_training=True,is_finetune=(cfg.training_stage==1),use_jae_loss=cfg.use_jae_loss)
        validation_set=BasketballDataset(all_anns,all_tracks,test_frames,
                                        cfg.data_path,cfg.image_size,cfg.person_size,cfg.out_size,cfg.inference_module_name, cfg,
                                        num_boxes=cfg.num_boxes,num_before=cfg.num_before, num_after=cfg.num_after,
                                        is_training=False,is_finetune=(cfg.training_stage==1),use_jae_loss=cfg.use_jae_loss)
        all_set = BasketballDataset(all_anns,all_tracks,all_frames,
                                        cfg.data_path,cfg.image_size,cfg.person_size,cfg.out_size,cfg.inference_module_name, cfg,
                                        num_boxes=cfg.num_boxes,num_before=cfg.num_before, num_after=cfg.num_after,
                                        is_training=False,is_finetune=(cfg.training_stage==1),use_jae_loss=cfg.use_jae_loss)
    elif cfg.dataset_name == 'jrdb':
        image_path = os.path.join(cfg.data_path, 'videos')
        train_anns = jrdb_read_dataset_new(cfg.data_path, cfg.train_seqs, cfg.num_actions, cfg.num_activities, cfg.num_social_activities)
        train_frames = jrdb_all_frames(train_anns)

        test_anns = jrdb_read_dataset_new(cfg.data_path, cfg.test_seqs, cfg.num_actions, cfg.num_activities, cfg.num_social_activities)
        test_frames = jrdb_all_frames(test_anns)

        all_anns = {**train_anns, **test_anns}
        all_frames = train_frames + test_frames

        training_set = JRDB_Dataset(cfg.num_actions, cfg.num_activities,
                                    cfg.num_social_activities, train_anns, train_frames,
                                    image_path, cfg.image_size, cfg.out_size, cfg, num_boxes=cfg.num_boxes,
                                    num_frame=cfg.num_frames, is_training=True,
                                    is_finetune=False)
        
        training_set_for_val = JRDB_Dataset(cfg.num_actions, cfg.num_activities,
                                    cfg.num_social_activities, train_anns, train_frames,
                                    image_path, cfg.image_size, cfg.out_size, cfg, num_boxes=cfg.num_boxes,
                                    num_frame=cfg.num_frames, is_training=False,
                                    is_finetune=False)

        validation_set = JRDB_Dataset(cfg.num_actions, cfg.num_activities,
                                    cfg.num_social_activities, test_anns, test_frames,
                                    image_path, cfg.image_size, cfg.out_size, cfg, num_boxes=cfg.num_boxes,
                                    num_frame=cfg.num_frames, is_training=False,
                                    is_finetune=False)
        
        all_set = JRDB_Dataset(cfg.num_actions, cfg.num_activities,
                                    cfg.num_social_activities, all_anns, all_frames,
                                    image_path, cfg.image_size, cfg.out_size, cfg, num_boxes=cfg.num_boxes,
                                    num_frame=cfg.num_frames, is_training=False,
                                    is_finetune=False)
    else:
        assert False
    
    print('Reading dataset finished...')
    print('%d train samples'%len(train_frames))
    print('%d test samples'%len(test_frames))
    print('%d all samples'%len(all_frames))

    if cfg.dataset_name in ['basketball']:
        num_boxes_max = all_set.get_num_boxes_max()
        training_set.set_num_boxes(num_boxes_max)
        validation_set.set_num_boxes(num_boxes_max)

    return training_set, validation_set, all_set
    