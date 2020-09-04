# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:59:02 2020

@author: Warmachine
"""



import os
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py
import numpy as np
import time
import pdb
from core.ProceLDataset import ProceLDataset
from global_setting import raw_data_dir,data_path_tr_CrossTask,data_path_tst_CrossTask,NFS_path

dict_n_keystep = {'23521':6,'59684':5,'71781':8,'113766':11,
                  '105222':6,'94276':6,'53193':6,'105253':11,
                  '44047':8,'76400':10,'16815':3,'95603':7,
                  '109972':5,'44789':8,'40567':11,'77721':5,'87706':9,'91515':8}

annot_dir = NFS_path+"data/CrossTask/crosstask_release/annotations/"


class FeatureVGGDataset_CrossTask(Dataset):
    """Feature VGG Dataset."""
    
    '''
        !!!!!!!!!! Need to perform sanity check of correct video category !!!!!!
    '''
    
    def __init__(self, root_dir, verbose = False, is_visualize = False,target_cat = None, is_all = False):
        self.root_dir = root_dir
        self.target_fps = 2
        self.verbose = verbose
        self.cat_video_tuples = []
        self.cat_video_ll = []
        self.cat2idx = {}
        
        self.idx2cat = []
        self.is_visualize = is_visualize
        self.is_all = is_all
        input_size = 224
        
        ### for visualization ###
        self.raw_data_dir = raw_data_dir#'/mnt/raptor/datasets/ProceL_Dat/'
        
        self.transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()
        ])
        ### for visualization ###    
        
        list_dir = os.listdir(root_dir)
        list_dir.sort()
        for idx_cat,category in enumerate(list_dir):
            cat_path = os.path.join(root_dir, category)
            self.cat2idx[category] = idx_cat
            self.idx2cat.append(category)
            self.cat_video_ll.append([])
            for video_files in os.listdir(cat_path):
               self.cat_video_ll[idx_cat].append(video_files)
            
        self.n_cat = len(self.cat_video_ll)
        
        if target_cat is None:
            print('Alternate category loader')
            counter = 0
            is_cont = True
            while is_cont:
                is_cont = False
                for idx_cat in range(self.n_cat):
                    if counter < len(self.cat_video_ll[idx_cat]):           #as long as there is video in some cats then is_cont = True
                        is_cont = True
                        self.cat_video_tuples.append((self.idx2cat[idx_cat], self.cat_video_ll[idx_cat][counter]))   # creates a tuple list of category and its videos
                counter += 1
            self.n_video = sum([len(cat) for cat in self.cat_video_ll])
        else:
            if self.is_all:
                print("!!!! Load all videos from both training and testing !!!!")
                assert root_dir == data_path_tr_CrossTask
                ### Augment test video in evaluation
                print("Augment training video with testing video")
                cat_path_aug = os.path.join(data_path_tst_CrossTask, target_cat)
                target_cat_idx = self.cat2idx[target_cat]
                for video_files in os.listdir(cat_path_aug):
                   self.cat_video_ll[target_cat_idx].append(video_files)
                ### Augment test video in evaluation
            
            print('Target Cat {}'.format(target_cat))
            target_cat_idx = self.cat2idx[target_cat]
            for cat_video in self.cat_video_ll[target_cat_idx]:
                self.cat_video_tuples.append((self.idx2cat[target_cat_idx],cat_video))
            self.n_video = len(self.cat_video_tuples)
        
        self.dict_n_keystep = dict_n_keystep
    
    def __len__(self):
        return self.n_video
    
    def check_match_annotation(self,category,video):
        csv_path = os.path.join(annot_dir, category+'_'+video+'.csv')
        return os.path.isfile(csv_path)
            
    def __getitem__(self, idx):
        ###
        tic = time.clock()
        
        category = self.cat_video_tuples[idx][0]
        video = self.cat_video_tuples[idx][1]
        
        file_path = os.path.join(self.root_dir, category, video)
        
        if self.is_all:
            ### Augment training video with testing video (minimal edit)
            if not os.path.isfile(file_path):
                file_path = os.path.join(data_path_tst_CrossTask, category, video)
            ### Augment training video with testing video (minimal edit)
        
        file = h5py.File(file_path, 'r')
        
        feature = file['features'].value
        feature_idx = torch.tensor(list(range(1, len(feature)+1)))
        if self.verbose:
            print('load video {}'.format(time.clock()-tic))
        ###
        
        ### load raw frames for visualization ###
#        if self.is_visualize:
#            cat_path = os.path.join(self.raw_data_dir, category)
#            frame_path = os.path.join(cat_path, 'frames')
#            frame_path = os.path.join(frame_path, video.split('.')[0])
#            frames = self.load_frames(frame_path)
        ###
        
#        ###
#        tic = time.clock()
#        mat_data = self.annotate(self.mat_path, category)
#        if self.verbose:
#            print('load annotation {}'.format(time.clock()-tic))
#        ###
        
        is_match = self.check_match_annotation(category,video)
        
        ###
        tic = time.clock()
        fps = video.split('_')[-1].split('f')[0]#self.original_fps(category, video, self.mat_data)
        if self.verbose:
            print('time load fps {}'.format(time.clock()-tic))
        ###
        if self.verbose:
            print('fps {}'.format(fps))
        
        ###
        tic = time.clock()
        subsampled_feature = feature
        subsampled_feature_idx = feature_idx
        subsampled_segment_list = feature_idx
        
        del feature
        
        if self.is_visualize:
            pass
        else:
            subsampled_frames = torch.zeros(0)
            
        if self.verbose:
            print('time subsample {}'.format(time.clock()-tic))
        ###
        
        ###
        tic = time.clock()
        key_step_list = file['gt'].value
        n_keysteps = dict_n_keystep[category]
        if self.verbose:
            print('time load keystep {}'.format(time.clock()-tic))
        ###
        
        
        file.close()
        out_package = {'cat_labels':self.cat2idx[category], 'cat_names':category, 'video':video[:-17], 'subsampled_feature':subsampled_feature,
                   'subsampled_segment_list':subsampled_segment_list, 'key_step_list':key_step_list, 'n_og_keysteps':n_keysteps,
                   'subsampled_frames':subsampled_frames,'is_match':is_match}
        return out_package