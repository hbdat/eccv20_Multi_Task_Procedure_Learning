# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:01:44 2019

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
from core.ProceLDataset import ProceLDataset
import h5py
import time
import pdb

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


idx_GPU = 0
is_save = True
batch_size = 32

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

device = torch.device("cuda:{}".format(idx_GPU) 
                      if torch.cuda.is_available() else "cpu")

model_ref = models.vgg19(pretrained=True)
model_ref.eval()

model_f = nn.Sequential(*list(model_ref.children())[:-2])
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
    
categories = ['chromecast', 'clarinet',
              'pbj', 'phone_battery',
              'salmon', 'tie', 'toilet']

save_dir = './datasets/ProceL/features_full/{}/{}/'

train_partition = pd.read_csv('./data_partition/ProceL_train_partition.csv')['video'].values

'''
The code saves all videos from the same task into a single file. Please seperate each video into different file
'''
    
cat_feat_list = []
for category in categories:
    cat_path = './datasets/ProceL/'
    cat_path = os.path.join(cat_path, category)
    videos_path = os.path.join(cat_path, 'videos')
    frames_path = os.path.join(cat_path, 'frames')
    
    videos = [f for f in os.listdir(videos_path) if os.path.isfile(os.path.join(videos_path, f))]
    videos.sort()
    videos = [v[:-4] for v in videos]
    num_videos = len(videos)
    feat_dict = {}

    for idx_v,video in enumerate(videos):
        
        if video in train_partition:
            cats_dir = save_dir.format('training_data',category)
        else:
            cats_dir = save_dir.format('testing_data',category)
        
        if is_save and not os.path.exists(cats_dir):
                os.makedirs(cats_dir)
        
        new_name_video = video[:-2]+"{:02d}".format(idx_v+1)            #this is because the names in ProceL are not continuous
        file_path = cats_dir + new_name_video + '_feature_vgg.hdf5'
        
        if os.path.isfile(file_path):
            print('file exists {}'.format(file_path))
            continue
        
        tic = time.clock()
        img_dir = os.path.join(frames_path, video)
        
        proceL_dataset = ProceLDataset(img_dir , 
                                       transform = data_transforms)
        dataset_loader = DataLoader(proceL_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4)

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
            hfd5_cat_video = f.create_dataset(video, data = all_features,
                                                    compression ='gzip')
            f.close()
            
        print("Number of frames:", all_features.shape[0])
        print("Done with", video, time.clock()-tic)
    #cat_feat_list.append(feat_dict)
    
    #print("Done with: ", category, time.clock()-tic)

