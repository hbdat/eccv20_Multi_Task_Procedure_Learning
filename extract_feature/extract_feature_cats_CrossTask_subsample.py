# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:55:38 2020

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
import torch
from torch import nn
import pandas as pd
from PIL import Image
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils, models
from core.CrossTaskDataset import CrossTaskDataset
import h5py
import time
import pdb
import threading
from global_setting import docker_path
#%%
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

idx_GPU = 0
is_save = True
batch_size = 32
subsample_rate = 2

frame_dir = docker_path+"/datasets/CrossTask/frames/"
video_dir = docker_path+"/datasets/CrossTask/videos/"
save_dir = docker_path+"/datasets/CrossTask/features/"
annot_dir = docker_path+"/data/CrossTask/crosstask_release/annotations/"

cats = os.listdir(frame_dir)
cats.sort()
GPUs = [2,3,4,5,6,7]

def thread_function(idx_cat):
    idx_GPU = GPUs[idx_cat%len(GPUs)]
    
    device = torch.device("cuda:{}".format(idx_GPU) 
                          if torch.cuda.is_available() else "cpu")
    
    model_ref = models.vgg19(pretrained=True)
    model_ref.eval()
    
    ##### Updates from PyTorch make the VGG model has 2 components instead of 3 components like before ##### 
    model_f = nn.Sequential(*list(model_ref.children())[:1])
    model_f.to(device)
    #model_f = nn.DataParallel(model_f)
    model_f.eval()
    
    for param in model_f.parameters():
        param.requires_grad = False
        
    input_size = 224
    data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])
    
    category = str(cats[idx_cat])
    cat_path = os.path.join(frame_dir, category)
    
    videos = [f for f in os.listdir(cat_path)]
    videos.sort()
    
    cat_dir = save_dir + category+'/'
    if is_save and not os.path.exists(cat_dir):
            os.makedirs(cat_dir)

    for idx_v,video in enumerate(videos):
        print(video)
        new_name_video = video[:-2]
        file_path = cat_dir + new_name_video + '_feature_vgg_{}fps.hdf5'.format(subsample_rate)
        
        if os.path.isfile(file_path):
            print('file exists {}'.format(file_path))
            continue
        
        tic = time.clock()
        
        img_dir = os.path.join(cat_path, video)
        csv_path = os.path.join(annot_dir, category+'_'+video+'.csv')
        video_path = video_dir+category+'/'+video+'.mp4'
        if not os.path.isfile(video_path):
            video_path = video_dir+category+'/'+video+'.webm'
            
        assert os.path.isfile(video_path)
            
        
        crossTask_dataset = CrossTaskDataset(img_dir , video_path, csv_path, subsample_rate = subsample_rate,
                                       transform = data_transforms)
        dataset_loader = DataLoader(crossTask_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=0)

        all_features = []
        count = 0
        for i_batch, imgs in enumerate(dataset_loader):
            imgs=imgs.to(device)
            count+=1
            features = model_f(imgs)
            all_features.append(features.cpu().numpy())
        all_features = np.concatenate(all_features,axis=0)
        #feat_dict['video'] = all_features
        
        if is_save:
            f = h5py.File(file_path, "w")
            _ = f.create_dataset("features", data = all_features,compression ='gzip')
            _ = f.create_dataset("gt", data = crossTask_dataset.gt)
            f.close()
            
        print("Shape: ", all_features.shape)
        print("Done with", video, time.clock()-tic)

#%% testing
#%%
threads = list()
for cat in range(len(cats)):
    x = threading.Thread(target=thread_function, args=(cat,))
    threads.append(x)
    x.start()
