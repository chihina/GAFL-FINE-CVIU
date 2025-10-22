import numpy as np
import skimage.io
import skimage.transform

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models

from PIL import Image
import random

import sys
import os
import cv2
import pandas as pd
from tqdm import tqdm
import time

from hr_net_func import box_to_center_scale, transform_image_hrnet
from utils import gen_ja_points

"""
Reference:
https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
"""

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']
NUM_ACTIVITIES = 8

ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
           'moving', 'setting', 'spiking', 'standing',
           'waiting']
NUM_ACTIONS = 9

def volley_get_actions():
    return ACTIONS

def volley_get_activities():
    return ACTIVITIES

def volley_read_annotations(path):
    """
    reading annotations for the given sequence
    """
    annotations = {}

    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    act_to_id = {name: i for i, name in enumerate(ACTIONS)}

    with open(path) as f:
        for l in f.readlines():
            values = l[:-1].split(' ')
            file_name = values[0]
            activity = gact_to_id[values[1]]

            values = values[2:]
            num_people = len(values) // 5

            action_names = values[4::5]
            actions = [act_to_id[name]
                       for name in action_names]

            def _read_bbox(xywh):
                x, y, w, h = map(int, xywh)
                return y, x, y+h, x+w
            bboxes = np.array([_read_bbox(values[i:i+4])
                               for i in range(0, 5*num_people, 5)])

            fid = int(file_name.split('.')[0])

            annotations[fid] = {
                'file_name': file_name,
                'group_activity': activity,
                'actions': actions,
                'bboxes': bboxes,
            }
    return annotations

# joint attention estimation
def read_ja_bbox_from_csv(csv_file_path: str) -> list:
    try:
        df_anno = pd.read_csv(csv_file_path,  header=None)
    except pd.errors.EmptyDataError:
        return False, False
    
    bbox_array = np.zeros((df_anno.shape[0], 4))
    for img_idx in range(df_anno.shape[0]):
        anno_row = df_anno.iloc[img_idx, :].values[0].split(" ")
        x_min_ball, y_min_ball, x_max_ball, y_max_ball = map(int, anno_row[1:5])
        # lost, occluded = map(int, anno_row[6:8])
        bbox_array[img_idx, :] = [x_min_ball, y_min_ball, x_max_ball, y_max_ball]

    return bbox_array, True

def volley_read_dataset_jae(jae_ann_dir, data, sid, cfg) -> dict:
    data_sid = data[sid]
    for fid, ann in data_sid.items():
        if cfg.use_jae_type == 'gt':
            ja_ann_path = os.path.join(jae_ann_dir, f'volleyball_{sid}_{fid}_ver3.csv')
            ja_bboxes, ja_flag = read_ja_bbox_from_csv(ja_ann_path)
            if ja_flag:
                data_sid[fid]['ja_bboxes'] = ja_bboxes
            else:
                data_sid[fid]['ja_bboxes'] = np.zeros((41, 4))
        elif cfg.use_jae_type == 'pred':
            ja_ann_path = os.path.join(cfg.jae_pred_dir, str(sid), f'{fid}.txt')
            df_point = pd.read_csv(ja_ann_path, header=None, sep=' ')
            ja_x_mid, ja_y_mid = df_point.iloc[:, 0].values, df_point.iloc[:, 1].values
            ja_box_width, ja_box_height = 20, 20
            ja_x_min = ja_x_mid - ja_box_width / 2
            ja_x_max = ja_x_mid + ja_box_width / 2
            ja_y_min = ja_y_mid - ja_box_height / 2
            ja_y_max = ja_y_mid + ja_box_height / 2
            ja_bboxes = np.zeros((ja_x_mid.shape[0], 4))
            ja_bboxes[:, 0] = ja_x_min
            ja_bboxes[:, 1] = ja_y_min
            ja_bboxes[:, 2] = ja_x_max
            ja_bboxes[:, 3] = ja_y_max
            data_sid[fid]['ja_bboxes'] = ja_bboxes

    return data_sid

# user queries
def volley_read_user_queries(query_dir, query_type, data, sid, cfg):
    """
    reading user queries for the given sequence
    """

    data_sid = data[sid]
    path = os.path.join(query_dir, f'{sid}', f'query_{query_type}_{cfg.use_individual_action_type}.txt')
    with open(path) as f:
        for l in f.readlines():
            seq_id, use_query_flag = l.split()
            data_sid[int(seq_id)]['use_query'] = int(use_query_flag)
    
    path_people = os.path.join(query_dir, f'{sid}', f'query_people_{query_type}_{cfg.use_individual_action_type}.txt')
    with open(path_people) as f:
        for l in f.readlines():
            seq_id, use_query_people_flag = l.split()[0], l.split()[1:]
            use_query_people_flag = list(map(int, use_query_people_flag))
            data_sid[int(seq_id)]['use_query_people'] = use_query_people_flag

    return data_sid

# define the max and min coordinates of the bounding box in each video
def volley_read_video_bbox_info(data, sid, all_tracks):
    '''
    read the max and min coordinates of the bounding box in each video
    '''

    # get the data of the given sequence
    data_sid = data[sid]
    data_sid_keys = sorted(list(data_sid.keys()))

    # get the max and min coordinates of the bounding box in each video
    x_min_vid, y_min_vid = 10000, 10000
    x_max_vid, y_max_vid = 0, 0
    for track_key, track in all_tracks.items():
        track_sid, track_fid = track_key
        if track_sid == sid:
            for fid, bboxes in track.items():
                y_min_vid = min(y_min_vid, bboxes[:, 0].min())
                x_min_vid = min(x_min_vid, bboxes[:, 1].min())
                y_max_vid = max(y_max_vid, bboxes[:, 2].max())
                x_max_vid = max(x_max_vid, bboxes[:, 3].max())

                y_min_img = bboxes[:, 0].min()
                x_min_img = bboxes[:, 1].min()
                y_max_img = bboxes[:, 2].max()
                x_max_img = bboxes[:, 3].max()
                x_mid_img_grav = (bboxes[:, 1].mean() + bboxes[:, 3].mean()) / 2
                y_mid_img_grav = (bboxes[:, 0].mean() + bboxes[:, 2].mean()) / 2
                if track_fid in data_sid_keys:
                    data_sid[track_fid]['x_min_img'] = x_min_img
                    data_sid[track_fid]['y_min_img'] = y_min_img
                    data_sid[track_fid]['x_max_img'] = x_max_img
                    data_sid[track_fid]['y_max_img'] = y_max_img
                    data_sid[track_fid]['x_mid_img_grav'] = x_mid_img_grav
                    data_sid[track_fid]['y_mid_img_grav'] = y_mid_img_grav

    # update the bounding box info in each video
    for fid in list(data_sid.keys()):
        data_sid[fid]['x_min_vid'] = x_min_vid
        data_sid[fid]['y_min_vid'] = y_min_vid
        data_sid[fid]['x_max_vid'] = x_max_vid
        data_sid[fid]['y_max_vid'] = y_max_vid
    
    return data_sid

# define the max and min coordinates of the bounding box in each video
def volley_read_net_bbox_info(data, sid, cfg):
    data_sid = data[sid]
    for fid in list(data_sid.keys()):
        net_det_path = os.path.join(cfg.net_det_dir, f'{sid}', f'{fid}', f'{fid}.csv')
        df_net = pd.read_csv(net_det_path, index_col=False)
        data_sid[fid]['x_min_net'] = df_net['xmin'].loc[0]
        data_sid[fid]['y_min_net'] = df_net['ymin'].loc[0]
        data_sid[fid]['x_max_net'] = df_net['xmax'].loc[0]
        data_sid[fid]['y_max_net'] = df_net['ymax'].loc[0]
    
    return data_sid

def volley_read_line_bbox_info(data, sid, cfg):
    data_sid = data[sid]
    for fid in list(data_sid.keys()):
        line_det_path = os.path.join(cfg.line_det_dir, f'{sid}', f'{fid}', f'{fid}.csv')
        df_line = pd.read_csv(line_det_path, index_col=False)
        x_min_net, x_max_net = data_sid[fid]['x_min_net'], data_sid[fid]['x_max_net']
        x_mid_net = (x_min_net + x_max_net) / 2

        df_line_left = df_line[df_line['xmax'] < x_mid_net]
        df_line_left_mean = df_line_left.mean()
        data_sid[fid]['x_min_line_left'] = df_line_left_mean['xmin']
        data_sid[fid]['y_min_line_left'] = df_line_left_mean['ymin']
        data_sid[fid]['x_max_line_left'] = df_line_left_mean['xmax']
        data_sid[fid]['y_max_line_left'] = df_line_left_mean['ymax']

        df_line_right = df_line[df_line['xmin'] > x_mid_net]
        df_line_right_mean = df_line_right.mean()
        data_sid[fid]['x_min_line_right'] = df_line_right_mean['xmin']
        data_sid[fid]['y_min_line_right'] = df_line_right_mean['ymin']
        data_sid[fid]['x_max_line_right'] = df_line_right_mean['xmax']
        data_sid[fid]['y_max_line_right'] = df_line_right_mean['ymax']

    return data_sid

def volley_read_dataset(path, seqs, jae_ann_dir, query_dir, query_type, all_tracks, cfg):
    data = {}
    
    for sid in seqs:
        # read annotations
        data[sid] = volley_read_annotations(path + '/%d/annotations.txt' % sid)

        # read user queries
        data[sid] = volley_read_user_queries(query_dir, query_type, data, sid, cfg)

        # read bbox info in each video
        data[sid] = volley_read_video_bbox_info(data, sid, all_tracks)

        # read ja bbox annotations
        data[sid] = volley_read_dataset_jae(jae_ann_dir, data, sid, cfg)

        # read net bbox info in each video
        data[sid] = volley_read_net_bbox_info(data, sid, cfg)

        # read line bbox info in each video
        data[sid] = volley_read_line_bbox_info(data, sid, cfg)

    return data

def volley_all_frames(data):
    frames = []
    for sid, anns in data.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))
    return frames

def volley_random_frames(data, num_frames):
    frames = []
    for sid in np.random.choice(list(data.keys()), num_frames):
        fid = int(np.random.choice(list(data[sid]), []))
        frames.append((sid, fid))
    return frames

def volley_frames_around(frame, num_before=5, num_after=4):
    sid, src_fid = frame
    return [(sid, src_fid, fid)
            for fid in range(src_fid-num_before, src_fid+num_after+1)]

# def load_samples_sequence(anns,tracks,images_path,frames,image_size,num_boxes=12,):
#     """
#     load samples of a bath
    
#     Returns:
#         pytorch tensors
#     """
#     images, boxes, boxes_idx = [], [], []
#     activities, actions = [], []
#     for i, (sid, src_fid, fid) in enumerate(frames):
#         #img=skimage.io.imread(images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
#         #img=skimage.transform.resize(img,(720, 1280),anti_aliasing=True)
        
#         img = Image.open(images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
        
#         img=transforms.functional.resize(img,image_size)
#         img=np.array(img)
        
#         # H,W,3 -> 3,H,W
#         img=img.transpose(2,0,1)
#         images.append(img)

#         boxes.append(tracks[(sid, src_fid)][fid])
#         actions.append(anns[sid][src_fid]['actions'])
#         if len(boxes[-1]) != num_boxes:
#           boxes[-1] = np.vstack([boxes[-1], boxes[-1][:num_boxes-len(boxes[-1])]])
#           actions[-1] = actions[-1] + actions[-1][:num_boxes-len(actions[-1])]
#         boxes_idx.append(i * np.ones(num_boxes, dtype=np.int32))
#         activities.append(anns[sid][src_fid]['group_activity'])


#     images = np.stack(images)
#     activities = np.array(activities, dtype=np.int32)
#     bboxes = np.vstack(boxes).reshape([-1, num_boxes, 4])
#     bboxes_idx = np.hstack(boxes_idx).reshape([-1, num_boxes])
#     actions = np.hstack(actions).reshape([-1, num_boxes])
    
#     #convert to pytorch tensor
#     images=torch.from_numpy(images).float()
#     bboxes=torch.from_numpy(bboxes).float()
#     bboxes_idx=torch.from_numpy(bboxes_idx).int()
#     actions=torch.from_numpy(actions).long()
#     activities=torch.from_numpy(activities).long()

#     return images, bboxes, bboxes_idx, actions, activities, joint_attention_bbox




class VolleyballDataset(data.Dataset):
    """
    Characterize volleyball dataset for pytorch
    """
    def __init__(self,anns,tracks,frames,images_path,image_size,person_size,feature_size,inference_module_name, cfg, 
                 num_boxes=12, num_before=4,num_after=4,is_training=True,is_finetune=False, use_jae_loss=False):
        self.anns=anns
        self.tracks=tracks
        self.frames=frames
        self.images_path=images_path
        self.image_size=image_size
        self.person_size=person_size
        self.feature_size=feature_size
        self.inference_module_name = inference_module_name
        
        self.num_boxes=num_boxes
        self.num_before=num_before
        self.num_after=num_after
        
        self.is_training=is_training
        self.is_finetune=is_finetune

        self.use_jae_loss = use_jae_loss
        self.cfg = cfg

        self.prev_keyword_list = ['autoencoder', 'HRN', 'VGG']
        self.prev_flag = any(keyword in cfg.model_exp_name for keyword in self.prev_keyword_list)
        self.pose_flag = 'PPF' in self.cfg.mode or 'PPC' in self.cfg.mode

        # self.frames_seq = np.empty((1337, 2), dtype = np.int)
        # self.flag = 0

    def __len__(self):
        """
        Return the total number of samples
        """

        return len(self.frames)
    
    def __getitem__(self,index):
        """
        Generate one sample of the dataset
        """

        select_frames = self.volley_frames_sample(self.frames[index])
        sample = self.load_samples_sequence(select_frames)
        
        return sample
    
    def get_bboxs(self, frame_index):
        """
        Get the bounding boxes of the given index
        """

        select_frames = self.volley_frames_sample(self.frames[frame_index])
        sample = self.load_bboxes_sequence(select_frames)

        return sample
    
    def get_activities_in_all_list(self):
        activities_list = []
        for frame in self.frames:
            sid, src_fid = frame
            activities_list.append(self.anns[sid][src_fid]['group_activity'])
        activities_list = np.array(activities_list)
        return activities_list
    
    def get_frame_id_list(self):
        frame_id_list = []
        for frame in self.frames:
            sid, src_fid = frame
            frame_id_list.append(f'{sid}_{src_fid}_{str(int(src_fid)-5)}')
        return frame_id_list
    
    def get_query_list(self):
        query_list = []
        for frame in self.frames:
            sid, src_fid = frame
            query_list.append(self.anns[sid][src_fid]['use_query'])
        query_list = np.array(query_list)
        return query_list
    
    def get_query(self, frame):
        select_frames = self.volley_frames_sample(frame)
        sid, src_fid, fid = select_frames[0]
        return self.anns[sid][src_fid]['use_query']
    
    def set_query(self, frame, query):
        sid, src_fid = frame
        self.anns[sid][src_fid]['use_query'] = query

    def get_frames_all(self):
        return self.frames

    def set_frames(self, new_frames):
        self.frames = new_frames

    def volley_frames_sample(self,frame):
        sid, src_fid = frame
        
        if self.is_finetune:
            if self.is_training:
                fid=random.randint(src_fid-self.num_before, src_fid+self.num_after)
                return [(sid, src_fid, fid)]
            else:
                return [(sid, src_fid, fid)
                        for fid in range(src_fid-self.num_before, src_fid+self.num_after+1)]
        else:
            # if self.is_training:
            #     sample_frames=random.sample(range(src_fid-self.num_before, src_fid+self.num_after+1), 3)
            #     return [(sid, src_fid, fid)
            #             for fid in sample_frames]
            # else:
            #     return [(sid, src_fid, fid)
            #             for fid in  [src_fid-3,src_fid,src_fid+3, src_fid-4,src_fid-1,src_fid+2, src_fid-2,src_fid+1,src_fid+4 ]]
            if self.inference_module_name == 'arg_volleyball':
                if self.is_training:
                    sample_frames=random.sample(range(src_fid-self.num_before, src_fid+self.num_after+1), 3)
                    return [(sid, src_fid, fid)
                            for fid in sample_frames]
                else:
                    return [(sid, src_fid, fid)
                            for fid in  [src_fid-3,src_fid,src_fid+3, src_fid-4,src_fid-1,src_fid+2, src_fid-2,src_fid+1,src_fid+4 ]]
            else:
                if self.is_training:
                    return [(sid, src_fid, fid) for fid in range(src_fid-self.num_before, src_fid+self.num_after+1)]
                else:
                    return [(sid, src_fid, fid) for fid in range(src_fid-self.num_before, src_fid+self.num_after+1)]

    def get_num_boxes_max(self):
        max_num_boxes = 0
        for sid, src_fid in self.frames:
            max_num_boxes = max(max_num_boxes, len(self.tracks[(sid, src_fid)][src_fid]))
        return max_num_boxes

    def load_samples_sequence(self, select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        
        OH, OW = self.feature_size
        PH, PW = self.person_size
        
        images, boxes = [], []
        boxes_wo_norm = []
        boxes_wo_norm_bev_foot = []
        activities, actions = [], []
        bboxes_num=[]
        images_person = []
        images_info = []
        video_id = f'{select_frames[0][0]}_{select_frames[0][1]}_{select_frames[0][2]}'
        # select_frame_mid = select_frames[self.num_before]
        # video_id = f'{select_frame_mid[0]}_{select_frame_mid[1]}_{select_frame_mid[2]}'
        ja_bboxes = []
        user_queries = []
        user_queries_people = []
        bbox_region_vid = []
        bbox_region_img = []
        bbox_region_net = []
        bbox_region_line_left = []
        bbox_region_line_right = []

        self.use_crop_person = self.prev_flag
        for frame_idx, (sid, src_fid, fid) in enumerate(select_frames):
            start_time = time.time()

            img = Image.open(self.images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
            img_people = Image.open(self.images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
            IW, IH = img.size

            img=transforms.functional.resize(img,self.image_size)
            img=np.array(img)

            # H,W,3 -> 3,H,W
            img=img.transpose(2,0,1)
            images.append(img)

            # add height and width of the image
            images_info.append([IH, IW])

            # add bbox region in each video
            x_min_vid, x_max_vid = map(lambda x: x * IW, [self.anns[sid][src_fid]['x_min_vid'], self.anns[sid][src_fid]['x_max_vid']])
            y_min_vid, y_max_vid = map(lambda x: x * IH, [self.anns[sid][src_fid]['y_min_vid'], self.anns[sid][src_fid]['y_max_vid']])
            bbox_vid = [x_min_vid, y_min_vid, x_max_vid, y_max_vid]
            bbox_region_vid.append(bbox_vid)

            # add bbox of net in each video
            x_min_net, x_max_net = map(lambda x: x * IW, [self.anns[sid][src_fid]['x_min_net'], self.anns[sid][src_fid]['x_max_net']])
            y_min_net, y_max_net = map(lambda x: x * IH, [self.anns[sid][src_fid]['y_min_net'], self.anns[sid][src_fid]['y_max_net']])
            bbox_region_net.append([x_min_net, y_min_net, x_max_net, y_max_net])

            # add bbox of left line in each video
            x_min_line_left, x_max_line_left = map(lambda x: x * IW, [self.anns[sid][src_fid]['x_min_line_left'], self.anns[sid][src_fid]['x_max_line_left']])
            y_min_line_left, y_max_line_left = map(lambda x: x * IH, [self.anns[sid][src_fid]['y_min_line_left'], self.anns[sid][src_fid]['y_max_line_left']])
            bbox_line_left = [x_min_line_left, y_min_line_left, x_max_line_left, y_max_line_left]
            bbox_region_line_left.append(bbox_line_left)

            # add bbox of right line in each video
            x_min_line_right, x_max_line_right = map(lambda x: x * IW, [self.anns[sid][src_fid]['x_min_line_right'], self.anns[sid][src_fid]['x_max_line_right']])
            y_min_line_right, y_max_line_right = map(lambda x: x * IH, [self.anns[sid][src_fid]['y_min_line_right'], self.anns[sid][src_fid]['y_max_line_right']])
            bbox_line_right = [x_min_line_right, y_min_line_right, x_max_line_right, y_max_line_right]
            bbox_region_line_right.append(bbox_line_right)

            # calculate the perspective transform matrix
            lu_pts = np.array([x_max_line_left, y_min_line_left])
            lb_pts = np.array([x_min_line_left, y_max_line_left])
            ru_pts = np.array([x_min_line_right, y_min_line_right])
            rb_pts = np.array([x_max_line_right, y_max_line_right])
            scale, pad = 7, (200, 175)
            pts1 = np.float32([lu_pts, ru_pts, lb_pts, rb_pts])
            pts2 = np.float32([[6*scale+pad[0],0*scale+pad[1]], [12*scale+pad[0],0*scale+pad[1]],
                            [6*scale+pad[0], 9*scale+pad[1]],[12*scale+pad[0], 9*scale+pad[1]]])
            H_m = cv2.getPerspectiveTransform(pts1, pts2)

            # add bbox region in each image
            x_mid_img_grav = self.anns[sid][src_fid]['x_mid_img_grav'] * IW
            y_mid_img_grav = self.anns[sid][src_fid]['y_mid_img_grav'] * IH
            x_min_img, x_max_img = map(lambda x: x * IW, [self.anns[sid][src_fid]['x_min_img'], self.anns[sid][src_fid]['x_max_img']])
            y_min_img, y_max_img = map(lambda x: x * IH, [self.anns[sid][src_fid]['y_min_img'], self.anns[sid][src_fid]['y_max_img']])
            bbox_img = [x_min_img, y_min_img, x_max_img, y_max_img, x_mid_img_grav, y_mid_img_grav]
            bbox_region_img.append(bbox_img)

            temp_boxes=np.ones_like(self.tracks[(sid, src_fid)][fid])
            temp_boxes_wo_norm = np.ones_like(self.tracks[(sid, src_fid)][fid])
            temp_boxes_wo_norm_bev = np.ones_like(self.tracks[(sid, src_fid)][fid])
            temp_images_person=np.zeros((self.tracks[(sid, src_fid)][fid].shape[0], 3, PH, PW))

            for person_idx,track in enumerate(self.tracks[(sid, src_fid)][fid]):
                y1,x1,y2,x2 = track
                w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                temp_boxes[person_idx]=np.array([w1,h1,w2,h2])
                temp_boxes_wo_norm[person_idx]=np.array([x1*IW,y1*IH,x2*IW,y2*IH])
                temp_boxes_wo_norm_bev[person_idx]=np.array([x1*IW,y1*IH,x2*IW,y2*IH])
                Pw1,Ph1,Pw2,Ph2 = x1*IW, y1*IH, x2*IW, y2*IH

                if self.pose_flag:
                    PH, PW = self.person_size
                    center, scale = box_to_center_scale([(Pw1, Ph1), (Pw2, Ph2)], PW, PH)
                    img_people_cv2 = cv2.cvtColor(np.array(img_people), cv2.COLOR_RGB2BGR)
                    img_person = transform_image_hrnet(img_people_cv2, center, scale, PH, PW)
                    img_person = img_person.numpy()
                else:
                    img_person = img_people.crop([Pw1,Ph1,Pw2,Ph2])
                    img_person = transforms.functional.resize(img_person,(PH,PW))
                    img_person = np.array(img_person)
                    img_person = img_person.transpose(2,0,1)
                temp_images_person[person_idx]=np.array(img_person)

            boxes.append(temp_boxes)
            boxes_wo_norm.append(temp_boxes_wo_norm)
            temp_boxes_wo_norm_bev_xmin = temp_boxes_wo_norm_bev[:, 0]
            temp_boxes_wo_norm_bev_xmax = temp_boxes_wo_norm_bev[:, 2]
            temp_boxes_wo_norm_bev_xmid = (temp_boxes_wo_norm_bev_xmin + temp_boxes_wo_norm_bev_xmax) / 2
            temp_boxes_wo_norm_bev_ymax = temp_boxes_wo_norm_bev[:, 3]
            temp_boxes_wo_norm_bev_foot = np.vstack([temp_boxes_wo_norm_bev_xmid, temp_boxes_wo_norm_bev_ymax]).T
            temp_boxes_wo_norm_bev_foot_exp = np.expand_dims(temp_boxes_wo_norm_bev_foot, axis=0)
            temp_boxes_wo_norm_bev_trans = cv2.perspectiveTransform(temp_boxes_wo_norm_bev_foot_exp, H_m)[0, :, :]
            boxes_wo_norm_bev_foot.append(temp_boxes_wo_norm_bev_trans)
            actions.append(self.anns[sid][src_fid]['actions'])
            use_query_people = self.anns[sid][src_fid]['use_query_people']
            if self.pose_flag or self.prev_flag:
                images_person.append(temp_images_person)

            if len(boxes[-1]) != self.num_boxes:
                boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes-len(boxes[-1])]])
                boxes_wo_norm[-1] = np.vstack([boxes_wo_norm[-1], boxes_wo_norm[-1][:self.num_boxes-len(boxes_wo_norm[-1])]])
                boxes_wo_norm_bev_foot[-1] = np.vstack([boxes_wo_norm_bev_foot[-1], boxes_wo_norm_bev_foot[-1][:self.num_boxes-len(boxes_wo_norm_bev_foot[-1])]])
                actions[-1] = actions[-1] + actions[-1][:self.num_boxes-len(actions[-1])]
                use_query_people = use_query_people + use_query_people[:self.num_boxes-len(use_query_people)]
                if self.pose_flag or self.prev_flag:
                    images_person[-1] = np.vstack([images_person[-1], images_person[-1][:self.num_boxes-len(images_person[-1])]])
                    # print('load time (stack-person):', time.time()-start_time)
                
            activities.append(self.anns[sid][src_fid]['group_activity'])

            # add bbox_num
            bboxes_num.append(len(actions[-1]))

            # add joint attention bbox
            ja_bbox = self.anns[sid][src_fid]['ja_bboxes'][fid-src_fid+20]
            ja_bboxes_norm = np.ones_like(ja_bbox)
            ja_bboxes_norm[::2] = ja_bbox[::2] / IW
            ja_bboxes_norm[1::2] = ja_bbox[1::2] / IH
            ja_bboxes.append(ja_bboxes_norm)

            # add user query flag
            user_queries.append(self.anns[sid][src_fid]['use_query'])
            user_queries_people.append(use_query_people)

        activities = np.array(activities, dtype=np.int32)
        bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4])
        bboxes_wo_norm = np.vstack(boxes_wo_norm).reshape([-1, self.num_boxes, 4])
        bboxes_wo_norm_bev_foot = np.vstack(boxes_wo_norm_bev_foot).reshape([-1, self.num_boxes, 2])
        actions = np.hstack(actions).reshape([-1, self.num_boxes])
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        # images = np.stack(images)
        # images_person = np.stack(images_person)

        #convert to pytorch tensor
        bboxes=torch.from_numpy(bboxes).float()
        bboxes_wo_norm=torch.from_numpy(bboxes_wo_norm).float()
        boxes_wo_norm_bev_foot = np.stack(boxes_wo_norm_bev_foot)
        actions=torch.from_numpy(actions).long()
        activities=torch.from_numpy(activities).long()
        bboxes_num=torch.from_numpy(bboxes_num).int()
        ja_bboxes = np.stack(ja_bboxes)
        ja_bboxes = torch.from_numpy(ja_bboxes).float()
        user_queries = torch.tensor(user_queries).float()
        user_queries_people = torch.tensor(user_queries_people).float()
        bbox_region_vid = torch.tensor(bbox_region_vid).float()
        bbox_region_img = torch.tensor(bbox_region_img).float()
        bbox_region_net = torch.tensor(bbox_region_net).float()
        bbox_region_line_left = torch.tensor(bbox_region_line_left).float()
        bbox_region_line_right = torch.tensor(bbox_region_line_right).float()
        images_info = torch.tensor(images_info).float()
        # images=torch.from_numpy(images).float()
        # images_person=torch.from_numpy(images_person).float()
        images = torch.tensor(images).float()
        frame = (select_frames[0][0], select_frames[0][1])

        data = {}
        data['images_in'] = images
        if self.pose_flag or self.prev_flag:
            images_person = torch.tensor(images_person).float()
            data['images_person_in'] = images_person
        data['boxes_in'] = bboxes
        data['boxes_wo_norm_in'] = bboxes_wo_norm
        data['boxes_wo_norm_bev_foot'] = boxes_wo_norm_bev_foot
        data['actions_in'] = actions
        data['activities_in'] = activities
        data['video_id'] = video_id
        data['ja_bboxes'] = ja_bboxes
        data['ja_points'] = gen_ja_points(ja_bboxes)
        data['user_queries'] = user_queries
        data['user_queries_people'] = user_queries_people
        data['bbox_region_vid'] = bbox_region_vid
        data['bbox_region_img'] = bbox_region_img
        data['bbox_region_net'] = bbox_region_net
        data['bbox_region_line_left'] = bbox_region_line_left
        data['bbox_region_line_right'] = bbox_region_line_right
        data['images_info'] = images_info
        data['bboxes_num'] = bboxes_num
        data['frame'] = frame

        return data

    def load_bboxes_sequence(self, select_frames):
        OH, OW = self.feature_size
        boxes, boxes_wo_norm = [], []
        boxes_wo_norm_bev_foot = []
        for frame_idx, (sid, src_fid, fid) in enumerate(select_frames):
            img = Image.open(self.images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
            IW, IH = img.size
            temp_boxes=np.ones_like(self.tracks[(sid, src_fid)][fid])
            temp_boxes_wo_norm = np.ones_like(self.tracks[(sid, src_fid)][fid])
            temp_boxes_wo_norm_bev = np.ones_like(self.tracks[(sid, src_fid)][fid])
            for person_idx,track in enumerate(self.tracks[(sid, src_fid)][fid]):
                y1,x1,y2,x2 = track
                w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                temp_boxes[person_idx]=np.array([w1,h1,w2,h2])
                temp_boxes_wo_norm[person_idx]=np.array([x1*IW,y1*IH,x2*IW,y2*IH])
                temp_boxes_wo_norm_bev[person_idx]=np.array([x1*IW,y1*IH,x2*IW,y2*IH])

            boxes.append(temp_boxes)
            boxes_wo_norm.append(temp_boxes_wo_norm)

            # add bbox of left line in each video
            x_min_line_left, x_max_line_left = map(lambda x: x * IW, [self.anns[sid][src_fid]['x_min_line_left'], self.anns[sid][src_fid]['x_max_line_left']])
            y_min_line_left, y_max_line_left = map(lambda x: x * IH, [self.anns[sid][src_fid]['y_min_line_left'], self.anns[sid][src_fid]['y_max_line_left']])
            x_min_line_right, x_max_line_right = map(lambda x: x * IW, [self.anns[sid][src_fid]['x_min_line_right'], self.anns[sid][src_fid]['x_max_line_right']])
            y_min_line_right, y_max_line_right = map(lambda x: x * IH, [self.anns[sid][src_fid]['y_min_line_right'], self.anns[sid][src_fid]['y_max_line_right']])

            # calculate the perspective transform matrix
            lu_pts = np.array([x_max_line_left, y_min_line_left])
            lb_pts = np.array([x_min_line_left, y_max_line_left])
            ru_pts = np.array([x_min_line_right, y_min_line_right])
            rb_pts = np.array([x_max_line_right, y_max_line_right])
            scale, pad = 7, (200, 175)
            pts1 = np.float32([lu_pts, ru_pts, lb_pts, rb_pts])
            pts2 = np.float32([[6*scale+pad[0],0*scale+pad[1]], [12*scale+pad[0],0*scale+pad[1]],
                            [6*scale+pad[0], 9*scale+pad[1]],[12*scale+pad[0], 9*scale+pad[1]]])
            H_m = cv2.getPerspectiveTransform(pts1, pts2)

            bev_xmin = temp_boxes_wo_norm_bev[:, 0]
            bev_xmax = temp_boxes_wo_norm_bev[:, 2]
            bev_xmid = (bev_xmin + bev_xmax) / 2
            bev_ymax = temp_boxes_wo_norm_bev[:, 3]
            bev_foot = np.vstack([bev_xmid, bev_ymax]).T
            bev_foot_exp = np.expand_dims(bev_foot, axis=0)
            bev_trans = cv2.perspectiveTransform(bev_foot_exp, H_m)[0, :, :]
            boxes_wo_norm_bev_foot.append(bev_trans)

            if len(boxes[-1]) != self.num_boxes:
                boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes-len(boxes[-1])]])
                boxes_wo_norm[-1] = np.vstack([boxes_wo_norm[-1], boxes_wo_norm[-1][:self.num_boxes-len(boxes_wo_norm[-1])]])
                boxes_wo_norm_bev_foot[-1] = np.vstack([boxes_wo_norm_bev_foot[-1], boxes_wo_norm_bev_foot[-1][:self.num_boxes-len(boxes_wo_norm_bev_foot[-1])]])

        bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4])
        bboxes_wo_norm = np.vstack(boxes_wo_norm).reshape([-1, self.num_boxes, 4])
        boxes_wo_norm_bev_foot = np.vstack(boxes_wo_norm_bev_foot).reshape([-1, self.num_boxes, 2])
        bboxes=torch.from_numpy(bboxes).float()
        bboxes_wo_norm=torch.from_numpy(bboxes_wo_norm).float()

        data = {}
        data['boxes_in'] = bboxes
        data['boxes_wo_norm_in'] = bboxes_wo_norm

        return data