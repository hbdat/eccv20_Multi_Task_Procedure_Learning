# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:43:02 2020

@author: Warmachine
"""

from __future__ import print_function, division
import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd)
#%%
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py
import numpy as np
import time
import pdb
from core.ProceLDataset import ProceLDataset
from core.FeatureVGGDataset import FeatureVGGDataset
from global_setting import raw_data_dir,data_path_tr,data_path_tst,docker_path,mat_path

#%%
target_fps = 2
verbose = True
batch_size = 1
num_worker = 4
#%%
feature_dataset_tr = FeatureVGGDataset(data_path_tr, mat_path, target_fps,verbose = verbose,is_visualize=False,target_cat=None)
dataset_loader_tr = DataLoader(feature_dataset_tr,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = num_worker)


save_dir = docker_path+'datasets/ProceL/features/training_data/'

for data_package in dataset_loader_tr:
    tic = time.clock()
    print("training set")
    
    cat_labels, cat_names, full_video_name, subsampled_feature, subsampled_segment_list, key_step_list, n_og_keysteps \
                    = data_package['cat_labels'],data_package['cat_names'],data_package['full_video_name'],data_package['subsampled_feature'],data_package['subsampled_segment_list'],data_package['key_step_list'],data_package['n_og_keysteps']
    
    save_cat_dir = save_dir + cat_names[0]
    if not os.path.exists(save_cat_dir):
        os.makedirs(save_cat_dir)
    
    save_path = save_cat_dir +'/'+full_video_name[0]
    
    f = h5py.File(save_path, "w")
    _ = f.create_dataset("subsampled_feature", data = subsampled_feature,compression ='gzip')
    _ = f.create_dataset("subsampled_segment_list", data = subsampled_segment_list)
    _ = f.create_dataset("key_step_list", data = key_step_list)
    _ = f.create_dataset("n_og_keysteps", data = n_og_keysteps)
    f.close()
        
    print("Shape: ", subsampled_feature.shape)
    print("Done with", full_video_name, time.clock()-tic)
#%%
feature_dataset_tst = FeatureVGGDataset(data_path_tst, mat_path, target_fps,verbose = verbose,is_visualize=False,target_cat=None)
dataset_loader_tst = DataLoader(feature_dataset_tst,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = num_worker)


save_dir = docker_path+'datasets/ProceL/features/testing_data/'

for data_package in dataset_loader_tst:
    tic = time.clock()
    print("testing set")
    
    cat_labels, cat_names, full_video_name, subsampled_feature, subsampled_segment_list, key_step_list, n_og_keysteps \
                    = data_package['cat_labels'],data_package['cat_names'],data_package['full_video_name'],data_package['subsampled_feature'],data_package['subsampled_segment_list'],data_package['key_step_list'],data_package['n_og_keysteps']
    
    save_cat_dir = save_dir + cat_names[0]
    if not os.path.exists(save_cat_dir):
        os.makedirs(save_cat_dir)
    
    save_path = save_cat_dir +'/'+full_video_name[0]
    
    f = h5py.File(save_path, "w")
    _ = f.create_dataset("subsampled_feature", data = subsampled_feature,compression ='gzip')
    _ = f.create_dataset("subsampled_segment_list", data = subsampled_segment_list)
    _ = f.create_dataset("key_step_list", data = key_step_list)
    _ = f.create_dataset("n_og_keysteps", data = n_og_keysteps)
    f.close()
        
    print("Shape: ", subsampled_feature.shape)
    print("Done with", full_video_name, time.clock()-tic)
    


    