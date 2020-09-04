# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:59:19 2020

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
from global_setting import raw_data_dir,data_path_tr_fast,data_path_tst_fast,data_path_tr,data_path_tst

class FeatureVGGDataset_Fast(Dataset):
    """Feature VGG Dataset."""

    def __init__(self, root_dir, mat_path, target_fps,verbose = False, is_visualize = False,target_cat = None, is_all = False):
        """
        Args:
            root_dir (string): Directory with all the feature
            hdf5 files.
            mat_path (string): Directory with all the annotation
            matlab files
        """

        self.root_dir = root_dir
        self.mat_path = mat_path
        self.target_fps = target_fps
        self.verbose = verbose
        self.cat_video_tuples = []
        self.cat_video_ll = []
        self.cat2idx = {}
        self.mat_data={}
        
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
               
            cat_path = os.path.join(self.mat_path, category + '_data.mat')
            self.mat_data[category] = sio.loadmat(cat_path)
            
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
                assert root_dir == data_path_tr_fast
                ### Augment test video in evaluation
                print("Augment training video with testing video")
                cat_path_aug = os.path.join(data_path_tst_fast, target_cat)
                target_cat_idx = self.cat2idx[target_cat]
                for video_files in os.listdir(cat_path_aug):
                   self.cat_video_ll[target_cat_idx].append(video_files)
                ### Augment test video in evaluation
            
            print('Target Cat {}'.format(target_cat))
            target_cat_idx = self.cat2idx[target_cat]
            for cat_video in self.cat_video_ll[target_cat_idx]:
                self.cat_video_tuples.append((self.idx2cat[target_cat_idx],cat_video))
            self.n_video = len(self.cat_video_tuples)
        
    
    def __len__(self):
        return self.n_video

    def load_frames(self, category, video, mat_data,batch_size = 200):
        old_dir = ''
        if self.root_dir == data_path_tr_fast:
            old_dir = data_path_tr
        else:
            old_dir = data_path_tst
            
        file_path = os.path.join(old_dir, category, video)
        
        file = h5py.File(file_path, 'r')
        
        key = list(file.keys())[0]
        file.close()
        
        video = video[:-17]      #removing "_feature_vgg.hdf5" from the video name
        video_no = int(video[-2:])    #getting the last two video number
        video_no-=1
        n_segments = len(mat_data[category]['superframe_time'][video_no][0])
        n_secs = mat_data[category]['superframe_time'][video_no][0][n_segments-1][1]
        n_frames = mat_data[category]['superframe_frame'][video_no][0][n_segments-1][1]

        fps = int(n_frames/n_secs)
    
        factor = max(int(fps/self.target_fps),1)
        
        idx_frames = list(range(1,n_frames+1,factor))           #this model starts from one !!!!
        
        cat_path = os.path.join(self.raw_data_dir, category)
        frame_path = os.path.join(cat_path, 'frames')
        frame_path = os.path.join(frame_path, key)
        
        
        proceL_dataset = ProceLDataset(frame_path , 
                                       transform = self.transforms,idx_frames=idx_frames)
        dataset_loader = DataLoader(proceL_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=10)      #this has to be zero if you use multi-thread in VGG dataset. You can not create thread in thread
        
        all_frames = []
        for i_batch, frames in enumerate(dataset_loader):
            all_frames.append(frames.cpu())
        all_frames = torch.cat(all_frames,0)
        all_frames = all_frames.permute(0,2,3,1)               #n,244,244,3 <== n,3,244,244 
        return all_frames
    
    def __getitem__(self, idx):
        
        category = self.cat_video_tuples[idx][0]
        video = self.cat_video_tuples[idx][1]
        
        file_path = os.path.join(self.root_dir, category, video)
        
        if self.is_all:
            ### Augment training video with testing video (minimal edit)
            if not os.path.isfile(file_path):
                file_path = os.path.join(data_path_tst_fast, category, video)
            ### Augment training video with testing video (minimal edit)
        
        file = h5py.File(file_path, 'r')
        
        
        subsampled_feature = file["subsampled_feature"].value[0]
        subsampled_segment_list = file["subsampled_segment_list"].value[0]
        key_step_list = file["key_step_list"].value[0]
        n_og_keysteps = file["n_og_keysteps"].value[0]
        
        ### load raw frames for visualization ###
        if self.is_visualize:
            frames = self.load_frames(category, video, self.mat_data) 
        else:
            frames = -1
        ###
        
        file.close()
        out_package = {'cat_labels':self.cat2idx[category], 'cat_names':category, 'video':video[:-17], 'subsampled_feature':subsampled_feature,
                   'subsampled_segment_list':subsampled_segment_list, 'key_step_list':key_step_list, 'n_og_keysteps':n_og_keysteps,
                   'subsampled_frames':frames,'is_match':-1,'full_video_name':video}
        return out_package
