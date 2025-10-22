import numpy as np
import skimage.io
import skimage.transform

import torch
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

from utils import gen_ja_points
from hr_net_func import box_to_center_scale, transform_image_hrnet


"""
Reference:
https://github.com/dk-kim/DFWSGAR
"""

ACTIVITIES = ['2p-succ.', '2p-fail.-def.', '2p-fail.-off.',
              '2p-layup-succ.', '2p-layup-fail.-def.', '2p-layup-fail.-off.',
              '3p-succ.', '3p-fail.-def.', '3p-fail.-off.']
NUM_ACTIVITIES = 9

# ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
        #    'moving', 'setting', 'spiking', 'standing',
        #    'waiting']
# NUM_ACTIONS = 9
ACTIONS = ['non_action']
NUM_ACTIONS = 1

def basketball_get_actions():
    return ACTIONS

def basketball_get_activities():
    return ACTIVITIES

def basketball_read_annotations(path):
    """
    reading annotations for the given sequence
    """
    annotations = {}

    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    act_to_id = {name: i for i, name in enumerate(ACTIONS)}

    with open(path) as f:
        for l in f.readlines():
            fid = int(l.split()[0])
            activity = gact_to_id[l.split()[1]]
            annotations[fid] = {
                'group_activity': activity,
            }
    return annotations

# user queries
def basketball_read_user_queries(query_dir, query_type, data, sid, cfg):
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

def basketball_read_tracks(data_sid, sid, cfg):
    """
    reading tracks for the given sequence
    """

    col_names = ['frame_id', 'player_id', 'x1', 'y1', 'w', 'h', 'conf', 'n1', 'n2', 'n3']
    for fid, ann in data_sid.items():
        track_path = os.path.join(cfg.nba_track_dir, f'{sid}', f'{fid}', f'{sid}_{fid}.txt')
        df_track = pd.read_csv(track_path, header=None, names=col_names, sep=',')

        # filter people who most appear in the video
        use_player_id_list = df_track['player_id'].value_counts().head(cfg.num_boxes).index.tolist()
        df_track = df_track[df_track['player_id'].isin(use_player_id_list)]
        df_track['player_id'] = df_track['player_id'].astype('category')
        df_track['player_id'] = df_track['player_id'].cat.codes

        if df_track.empty:
            print(f'Warning: empty track for {sid} {fid}')
            track = np.zeros((72, cfg.num_boxes, 4))
        else:
            df_track_frame_id = df_track['frame_id'].values
            df_track_player_id = df_track['player_id'].values
            player_indices = df_track_player_id - 1
            player_num = np.unique(player_indices).size
            track = np.zeros((72, player_num, 4))
            frame_indices = df_track_frame_id - df_track_frame_id.min()
            track[frame_indices, player_indices] = df_track[['x1', 'y1', 'w', 'h']].values
        
        data_sid[fid].update(dict(zip(range(72), track)))

    return data_sid

def basketball_read_jae_tracks(data_sid, sid, cfg):
    """
    reading jae tracks for the given sequence
    """

    bbox_width = 20
    bbox_height = 20
    for fid, ann in data_sid.items():
        track_path = os.path.join(cfg.nba_ball_track_dir, f'{sid}', f'{fid}.txt')
        df_track = pd.read_csv(track_path, header=None, sep=' ')
        ja_bboxes = np.zeros((df_track.shape[0], 4))
        xmid, ymid = df_track[0], df_track[1]
        ja_bboxes[:, 0] = xmid - bbox_width / 2
        ja_bboxes[:, 1] = ymid - bbox_height / 2
        ja_bboxes[:, 2] = xmid + bbox_width / 2
        ja_bboxes[:, 3] = ymid + bbox_height / 2

        # sample ja_bboxes based nba_track_step
        # ja_bboxes_sample = np.zeros((df_track.shape[0] // cfg.nba_track_step, 4))
        # for i in range(ja_bboxes_sample.shape[0]):
            # ja_bboxes_sample[i] = ja_bboxes[i * cfg.nba_track_step]
        # data_sid[fid]['ja_bboxes'] = ja_bboxes_sample

        data_sid[fid]['ja_bboxes'] = ja_bboxes

    return data_sid

def basketball_read_dataset(path, seqs, query_dir, query_type, all_tracks, cfg):
    data = {}
    
    for sid in tqdm(seqs):
        data[sid] = basketball_read_annotations(os.path.join(path, f'{sid}', 'annotations.txt'))

        data[sid] = basketball_read_user_queries(query_dir, query_type, data, sid, cfg)

        if cfg.use_jae_loss:
            data[sid] = basketball_read_jae_tracks(data[sid], sid, cfg)

        data[sid] = basketball_read_tracks(data[sid], sid, cfg)

    return data

def basketball_all_frames(data):
    frames = []
    for sid, anns in data.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))
    return frames

def basketball_random_frames(data, num_frames):
    frames = []
    for sid in np.random.choice(list(data.keys()), num_frames):
        fid = int(np.random.choice(list(data[sid]), []))
        frames.append((sid, fid))
    return frames

def basketball_frames_around(frame, num_before=0, num_after=0):
    sid, src_fid = frame
    return [(sid, src_fid, fid)
            for fid in range(src_fid-num_before, src_fid+num_after+1)]


# class VolleyballDataset(data.Dataset):
class BasketballDataset(data.Dataset):
    """
    Characterize basketball dataset for pytorch
    
    """
    def __init__(self,anns,tracks,frames,images_path,image_size,person_size,feature_size,inference_module_name, cfg, 
                 num_boxes=16, num_before=4,num_after=4,is_training=True,is_finetune=False, use_jae_loss=False):
        self.anns=anns
        self.tracks=tracks
        self.frames=frames
        self.images_path=images_path
        self.image_size=image_size
        self.person_size=person_size
        self.feature_size=feature_size
        self.inference_module_name = inference_module_name
        
        # self.num_boxes=num_boxes
        self.num_boxes = cfg.num_boxes
        # print(f'num_boxes: {self.num_boxes}')
        self.num_boxes = self.get_num_boxes_max()
        print(f'The number of people: {self.num_boxes}')

        self.num_before=num_before
        self.num_after=num_after
        self.all_num_frames = 72
        
        self.is_training=is_training
        self.is_finetune=is_finetune

        self.use_jae_loss = use_jae_loss
        self.cfg = cfg

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

        select_frames = self.basketball_frames_sample(self.frames[index])
        sample = self.load_samples_sequence(select_frames)

        return sample   
    
    def get_bboxs(self, frame_index):
        """
        Get the bounding boxes of the given index
        """

        select_frames = self.basketball_frames_sample(self.frames[frame_index])
        sample = self.load_bboxes_sequence(select_frames)

        return sample

    def get_num_boxes_max(self):
        max_num_boxes = 0
        for sid, src_fid in self.frames:
            for fid in self.anns[sid][src_fid]:
                # if fid in ['group_activity', 'use_query', 'use_query_people', 'ja_bboxes']:
                    # continue
                # else:
                    # max_num_boxes = max(max_num_boxes, self.anns[sid][src_fid][fid].shape[0])
                if type(fid) == int:
                    max_num_boxes = max(max_num_boxes, self.anns[sid][src_fid][fid].shape[0])

        return max_num_boxes
    
    def set_num_boxes(self, new_num_boxes):
        self.num_boxes = new_num_boxes

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
            frame_id_list.append(f'{sid}_{src_fid}_{src_fid}')
        return frame_id_list
    
    def get_query_list(self):
        query_list = []
        for frame in self.frames:
            sid, src_fid = frame
            query_list.append(self.anns[sid][src_fid]['use_query'])
        query_list = np.array(query_list)
        return query_list
    
    def get_query(self, frame):
        select_frames = self.basketball_frames_sample(frame)
        sid, src_fid, fid = select_frames[0]
        return self.anns[sid][src_fid]['use_query']
    
    def set_query(self, frame, query):
        sid, src_fid = frame
        self.anns[sid][src_fid]['use_query'] = query

    def get_frames_all(self):
        return self.frames

    def set_frames(self, new_frames):
        self.frames = new_frames

    def basketball_frames_sample(self,frame):
        sid, src_fid = frame

        if self.is_training:
            if self.cfg.random_sampling:
                sample_frames = random.sample(range(self.all_num_frames), self.cfg.num_frames)
                sample_frames.sort()
            else:
                segment_duration = self.all_num_frames // self.cfg.num_frames
                sample_frames = np.multiply(list(range(self.cfg.num_frames)), segment_duration) + np.random.randint(
                    segment_duration, size=self.cfg.num_frames)
        else:
            if self.cfg.num_frames == 3:
                # [12, 36, 60]
                sample_frames = list(range(12, 72, 24))
            elif self.cfg.num_frames == 6:
                # [6, 18, 30, 42, 54, 66]
                sample_frames = list(range(6, 72, 12))
            elif self.cfg.num_frames == 12:
                # [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70]
                sample_frames = list(range(4, 72, 6))
            elif self.cfg.num_frames == 18:
                # [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70]
                sample_frames = list(range(2, 72, 4))
            else:
                segment_duration = self.all_num_frames // self.cfg.num_frames
                sample_frames = np.multiply(list(range(self.cfg.num_frames)), segment_duration) + segment_duration // 2
        
        return [(sid, src_fid, fid) for fid in sample_frames]

        # return [(sid, src_fid, fid) for fid in range(0, self.all_num_frames, self.cfg.nba_track_step)]

        # if self.is_finetune:
        #     if self.is_training:
        #         fid=random.randint(src_fid-self.num_before, src_fid+self.num_after)
        #         return [(sid, src_fid, fid)]
        #     else:
        #         return [(sid, src_fid, fid)
        #                 for fid in range(src_fid-self.num_before, src_fid+self.num_after+1)]
        # else:
        #     if self.inference_module_name == 'arg_volleyball':
        #         if self.is_training:
        #             sample_frames=random.sample(range(src_fid-self.num_before, src_fid+self.num_after+1), 3)
        #             return [(sid, src_fid, fid)
        #                     for fid in sample_frames]
        #         else:
        #             return [(sid, src_fid, fid)
        #                     for fid in  [src_fid-3,src_fid,src_fid+3, src_fid-4,src_fid-1,src_fid+2, src_fid-2,src_fid+1,src_fid+4 ]]
        #     else:
        #         return [(sid, src_fid, fid) for fid in range(src_fid-self.num_before, src_fid+self.num_after+1)]


    def load_samples_sequence(self,select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        
        OH, OW = self.feature_size
        PH, PW = self.person_size
        
        images, boxes = [], []
        boxes_wo_norm = []
        activities, actions = [], []
        bboxes_num=[]
        images_person = []
        user_queries = []
        user_queries_people = []
        images_info = []
        # video_id = f'{select_frames[0][0]}_{select_frames[0][1]}_{select_frames[0][2]}'
        video_id = f'{select_frames[0][0]}_{select_frames[0][1]}_{select_frames[0][1]}'
        ja_bboxes = []

        self.use_crop_person = ('PPF' in self.cfg.mode) or ('PPC' in self.cfg.mode) or ('group_relation' in self.cfg.inference_module_name)

        start_time = time.time()
        for frame_idx, (sid, src_fid, fid) in enumerate(select_frames):

            time_start = time.time()
            img_path = os.path.join(self.images_path, f'{sid}', f'{src_fid}', f'{str(fid).zfill(6)}.jpg')
            
            # img = Image.open(img_path)
            # IW, IH = img.size
            # img=transforms.functional.resize(img, self.image_size)
            # img=np.array(img)
            # img=img.transpose(2,0,1)
            # images.append(img)
            # img_people = Image.open(img_path)
            # img_people = np.array(img_people)
            # print(f'0: {time.time()-time_start:.4f}s')

            # Load image once using OpenCV
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
            IH, IW = img.shape[:2]
            img_resized = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
            images.append(img_resized.transpose(2, 0, 1))
            # print(f'1: {time.time()-time_start:.4f}s')

            # add height and width of the image
            images_info.append([IH, IW])
            temp_boxes=np.ones((self.num_boxes, 4))
            temp_boxes_wo_norm = np.ones((self.num_boxes, 4))
            temp_images_person = np.zeros((self.num_boxes, 3, PH, PW))

            if fid not in self.anns[sid][src_fid]:
                print(f'Warning: no tarcking for {sid} {src_fid} {fid}')
            else:
                track_img = self.anns[sid][src_fid][fid]
                for person_idx in range(track_img.shape[0]):
                    x1, y1, w, h = track_img[person_idx]
                    if x1 == 0 and y1 == 0 and w == 0 and h == 0:
                        continue
                    x2, y2 = x1 + w, y1 + h
                    x1, x2 = x1 / IW, x2 / IW
                    y1, y2 = y1 / IH, y2 / IH
                    w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH
                    temp_boxes[person_idx]=np.array([w1,h1,w2,h2])
                    temp_boxes_wo_norm[person_idx]=np.array([x1*IW,y1*IH,x2*IW,y2*IH])
                    Pw1,Ph1,Pw2,Ph2 = x1*IW, y1*IH, x2*IW, y2*IH  

                    if ('PPF' in self.cfg.mode or 'PPC' in self.cfg.mode):
                        PH, PW = self.person_size
                        center, scale = box_to_center_scale([(Pw1, Ph1), (Pw2, Ph2)], PW, PH)
                        img_person = transform_image_hrnet(img, center, scale, PH, PW)
                        img_person = img_person.numpy()
                    else:
                        # img_person = img_people.crop([Pw1,Ph1,Pw2,Ph2])
                        # img_person = img_people[int(Ph1):int(Ph2), int(Pw1):int(Pw2)]
                        # img_person = Image.fromarray(img_person)
                        # img_person = img_person.resize((PW, PH))
                        # img_person = np.array(img_person)
                        # img_person = img_person.transpose(2,0,1)

                        img_person = img[int(Ph1):int(Ph2), int(Pw1):int(Pw2)]
                        img_person = cv2.resize(img_person, (PW, PH), interpolation=cv2.INTER_LINEAR)
                        img_person = img_person.transpose(2, 0, 1)

                    temp_images_person[person_idx]=np.array(img_person)

            boxes.append(temp_boxes)
            boxes_wo_norm.append(temp_boxes_wo_norm)
            bboxes_num.append(len(temp_boxes))
            actions.append([0]*self.num_boxes)
            use_query_people = [0]*self.num_boxes
            if self.use_crop_person:
                images_person.append(temp_images_person)

            if len(boxes[-1]) != self.num_boxes:
                boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes-len(boxes[-1])]])
                boxes_wo_norm[-1] = np.vstack([boxes_wo_norm[-1], boxes_wo_norm[-1][:self.num_boxes-len(boxes_wo_norm[-1])]])
                actions[-1] = actions[-1] + actions[-1][:self.num_boxes-len(actions[-1])]
                use_query_people = use_query_people + use_query_people[:self.num_boxes-len(use_query_people)]
                if 'PPF' in self.cfg.mode or 'PPC' in self.cfg.mode:
                    images_person[-1] = np.vstack([images_person[-1], images_person[-1][:self.num_boxes-len(images_person[-1])]])
                    print('load time (stack-person):', time.time()-start_time)

            activities.append(self.anns[sid][src_fid]['group_activity'])
            user_queries.append(self.anns[sid][src_fid]['use_query'])
            user_queries_people.append(use_query_people)
            
            # add joint attention bbox
            if self.use_jae_loss:
                ja_bbox = self.anns[sid][src_fid]['ja_bboxes'][frame_idx]
                ja_bboxes_norm = np.ones_like(ja_bbox)
                ja_bboxes_norm[::2] = ja_bbox[::2] / IW
                ja_bboxes_norm[1::2] = ja_bbox[1::2] / IH
                ja_bboxes.append(ja_bboxes_norm)

        # print(f'load_samples_sequence-00: {time.time()-start_time:.4f}s')
        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4])
        # print(f'load_samples_sequence-01: {time.time()-start_time:.4f}s')
        bboxes_wo_norm = np.vstack(boxes_wo_norm).reshape([-1, self.num_boxes, 4])
        # print(f'load_samples_sequence-02: {time.time()-start_time:.4f}s')
        actions = np.hstack(actions).reshape([-1, self.num_boxes])
        # images_person = np.stack(images_person)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)

        # print(f'load_samples_sequence-1: {time.time()-start_time:.4f}s')

        #convert to pytorch tensor
        images=torch.from_numpy(images).float()
        bboxes=torch.from_numpy(bboxes).float()
        bboxes_wo_norm=torch.from_numpy(bboxes_wo_norm).float()
        actions=torch.from_numpy(actions).long()
        activities=torch.from_numpy(activities).long()
        # images_person=torch.from_numpy(images_person).float()
        bboxes_num=torch.from_numpy(bboxes_num).int()
        user_queries = torch.tensor(user_queries).float()
        user_queries_people = torch.tensor(user_queries_people).float()
        images_info = torch.tensor(images_info).float()
        frame = (select_frames[0][0], select_frames[0][1])

        # print(f'load_samples_sequence-2: {time.time()-start_time:.4f}s')

        data = {}
        data['images_in'] = images
        data['boxes_in'] = bboxes
        data['boxes_wo_norm_in'] = bboxes_wo_norm
        # data['boxes_wo_norm_bev_foot'] = boxes_wo_norm_bev_foot
        data['actions_in'] = actions
        data['activities_in'] = activities
        if self.use_crop_person:
            images_person = torch.tensor(images_person).float()
            data['images_person_in'] = images_person
        # data['images_person_in'] = images_person
        data['video_id'] = video_id
        if self.use_jae_loss:
            ja_bboxes = np.stack(ja_bboxes)
            ja_bboxes = torch.from_numpy(ja_bboxes).float()
            data['ja_bboxes'] = ja_bboxes
            data['ja_points'] = gen_ja_points(ja_bboxes)
        data['user_queries'] = user_queries
        data['user_queries_people'] = user_queries_people
        data['bboxes_num'] = bboxes_num
        # data['bbox_region_vid'] = bbox_region_vid
        # data['bbox_region_img'] = bbox_region_img
        # data['bbox_region_net'] = bbox_region_net
        # data['bbox_region_line_left'] = bbox_region_line_left
        # data['bbox_region_line_right'] = bbox_region_line_right
        data['images_info'] = images_info
        data['frame'] = frame

        # print(f'load_samples_sequence-3: {time.time()-start_time:.4f}s')

        return data

    def load_bboxes_sequence(self,select_frames):
            OH, OW = self.feature_size
            PH, PW = self.person_size
            
            images, boxes = [], []
            boxes_wo_norm = []
            for frame_idx, (sid, src_fid, fid) in enumerate(select_frames):
                img_path = os.path.join(self.images_path, f'{sid}', f'{src_fid}', f'{str(fid).zfill(6)}.jpg')
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
                IH, IW = img.shape[:2]

                temp_boxes=np.ones((self.num_boxes, 4))
                temp_boxes_wo_norm = np.ones((self.num_boxes, 4))
                if fid not in self.anns[sid][src_fid]:
                    print(f'Warning: no tarcking for {sid} {src_fid} {fid}')
                else:
                    track_img = self.anns[sid][src_fid][fid]
                    for person_idx in range(track_img.shape[0]):
                        x1, y1, w, h = track_img[person_idx]
                        if x1 == 0 and y1 == 0 and w == 0 and h == 0:
                            continue
                        x2, y2 = x1 + w, y1 + h
                        x1, x2 = x1 / IW, x2 / IW
                        y1, y2 = y1 / IH, y2 / IH
                        w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH
                        temp_boxes[person_idx]=np.array([w1,h1,w2,h2])
                        temp_boxes_wo_norm[person_idx]=np.array([x1*IW,y1*IH,x2*IW,y2*IH])
                        Pw1,Ph1,Pw2,Ph2 = x1*IW, y1*IH, x2*IW, y2*IH  

                boxes.append(temp_boxes)
                boxes_wo_norm.append(temp_boxes_wo_norm)
                if len(boxes[-1]) != self.num_boxes:
                    boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes-len(boxes[-1])]])
                    boxes_wo_norm[-1] = np.vstack([boxes_wo_norm[-1], boxes_wo_norm[-1][:self.num_boxes-len(boxes_wo_norm[-1])]])

            bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4])
            bboxes_wo_norm = np.vstack(boxes_wo_norm).reshape([-1, self.num_boxes, 4])
            bboxes=torch.from_numpy(bboxes).float()
            bboxes_wo_norm=torch.from_numpy(bboxes_wo_norm).float()

            data = {}
            data['boxes_in'] = bboxes
            data['boxes_wo_norm_in'] = bboxes_wo_norm

            return data