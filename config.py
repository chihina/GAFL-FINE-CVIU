import time
import os


class Config(object):
    """
    class to save config parameter
    """

    def __init__(self, dataset_name):
        # Global
        self.image_size = 720, 1280  #input image size
        self.person_size = 224, 224  #input person image size
        self.batch_size =  32  #train batch size 
        self.test_batch_size = 8  #test batch size
        
        # Gpu
        self.use_gpu=True
        self.use_multi_gpu=True   
        self.device_list="0,1,2,3"  #id list of gpus used for training 
        
        # Dataset
        assert(dataset_name in ['volleyball', 'collective', 'basketball', 'jrdb'])
        self.dataset_name=dataset_name 
        
        if dataset_name=='volleyball':
            self.data_path = 'data/volleyball/videos' #data path for the volleyball dataset
            self.train_seqs = [ 1,3,6,7,10,13,15,16,18,22,23,31,32,36,38,39,40,41,42,48,50,52,53,54,
                                0,2,8,12,17,19,24,26,27,28,30,33,46,49,51]  #video id list of train set 
            self.test_seqs = [4,5,9,11,14,20,21,25,29,34,35,37,43,44,45,47]  #video id list of test set
            self.num_boxes = 12  #max number of bounding boxes in each frame
            # self.train_seqs = self.train_seqs[:3]
            # self.test_seqs = self.test_seqs[:3]
        elif dataset_name=='collective':
            self.data_path='data/collective'  #data path for the collective dataset
            self.test_seqs=[5,6,7,8,9,10,11,15,16,25,28,29]
            self.train_seqs=[s for s in range(1,45) if s not in self.test_seqs]
            self.num_boxes = 12  #max number of bounding boxes in each frame
        elif dataset_name == 'basketball':
            self.data_path = 'data/basketball/videos'  # data path for the basketball dataset
            train_seq_path = os.path.join(self.data_path, 'train_video_ids')
            with open(train_seq_path, 'r') as f:
                self.train_seqs = list(map(int, f.readlines()[0].strip().split(',')[:-1]))
            test_seq_path = os.path.join(self.data_path, 'test_video_ids')
            with open(test_seq_path, 'r') as f:
                self.test_seqs = list(map(int, f.readlines()[0].strip().split(',')[:-1]))
            # self.train_seqs = self.train_seqs[:10]
            # self.test_seqs = self.test_seqs[:10]
            self.num_boxes = 15
        elif dataset_name == 'jrdb':
            self.data_path = 'data/jrdb_par'
            self.test_seqs = [2, 7, 11, 16, 17, 25, 26]
            self.train_seqs = [s for s in range(0, 26) if s not in self.test_seqs]
        else:
            raise ValueError('dataset_name not supported')

        # Backbone 
        self.backbone='res18'
        self.crop_size = 5, 5  #crop size of roi align
        self.train_backbone = False  #if freeze the feature extraction part of network, True for stage 1, False for stage 2
        self.out_size = 87, 157  #output feature map size of backbone 
        self.emb_features=1056   #output feature map channel of backbone

        
        # Activity Action
        self.num_actions = 9  #number of action categories
        self.num_activities = 8  #number of activity categories
        self.actions_loss_weight = 1.0  #weight used to balance action loss and activity loss
        self.actions_weights = None

        # Sample
        # self.num_frames = 3 
        self.num_frames = 10
        self.num_before = 5
        self.num_after = 4

        # ARG params
        self.num_features_boxes = 1024
        self.num_features_relation=256
        self.num_graph=16  #number of graphs
        self.num_features_gcn=self.num_features_boxes
        self.gcn_layers=1  #number of GCN layers
        self.tau_sqrt=False
        self.pos_threshold=0.2  #distance mask threshold in position relation

        # Training Parameters
        self.train_random_seed = 0
        self.train_learning_rate = 1e-4  #initial learning rate
        self.lr_plan = {11:3e-5, 21:1e-5}  #change learning rate in these epochs
        self.train_dropout_prob = 0.3  #dropout probability
        self.weight_decay = 0  #l2 weight decay
    
        self.max_epoch = 30  #max training epoch
        self.test_interval_epoch = 1
        
        # Exp
        self.training_stage=1  #specify stage1 or stage2
        self.stage1_model_path=''  #path of the base model, need to be set in stage2
        self.test_before_train=False
        self.exp_note='Group-Activity-Recognition'
        self.exp_name=None
        self.set_bn_eval = False
        self.inference_module_name = 'dynamic_volleyball'

        # Dynamic Inference
        self.stride = 1
        self.ST_kernel_size = 3
        self.dynamic_sampling = True
        self.sampling_ratio = [1, 3]  # [1,2,4]
        self.group = 1
        self.scale_factor = True
        self.beta_factor = True
        self.load_backbone_stage2 = False
        self.parallel_inference = False
        self.hierarchical_inference = False
        self.lite_dim = None
        self.num_DIM = 1
        self.load_stage2model = False
        self.stage2model = None

        # Actor Transformer
        self.temporal_pooled_first = False

        # SACRF + BiUTE
        self.halting_penalty = 0.0001
        
    def init_config(self, need_new_folder=True):
        if self.eval_only:
            self.exp_name=self.model_exp_name
            self.result_path='result/%s'%self.exp_name
            self.log_path='result/%s/log.txt'%self.exp_name
            return

        if self.exp_name is None:
            time_str=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name='[%s_stage%d]<%s>'%(self.exp_note,self.training_stage,time_str)
            
        self.result_path='result/%s'%self.exp_name
        self.log_path='result/%s/log.txt'%self.exp_name
            
        # if need_new_folder:
            # os.mkdir(self.result_path)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
