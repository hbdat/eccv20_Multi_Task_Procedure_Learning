# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:08:04 2020

@author: Warmachine
"""


import os
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import time
import pandas as pd
from PIL import Image
import cv2
import pdb

def construct_gt(df,n_frame,duration):
    gt = np.zeros(n_frame)
    for i, row in df.iterrows(): 
        ks = row[0]
        s = row[1]
        e = row[2]
        
        idx_s = int(s*n_frame/duration)
        idx_e = int(e*n_frame/duration)
        
        assert idx_s <= idx_e
        
        gt[idx_s:idx_e] = ks
    
    return gt

class CrossTaskDataset(Dataset):
    """Procedure Learning dataset."""

    def __init__(self, frame_dir, video_path, csv_path, subsample_rate = 2, transform=None):
        """
        Args:
            frame_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to 
            be applied on a sample.
        """
        vidcap = cv2.VideoCapture(video_path)
        
        extension = '.jpg'
        num_jpg = len([f for f in os.listdir(frame_dir)
                         if os.path.isfile(os.path.join(frame_dir, f)) and f.endswith(extension)])
        
        
        vidcap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
        
        self.duration = vidcap.get(cv2.CAP_PROP_POS_MSEC)/1000
        self.frame_rate = int(round(num_jpg/self.duration))        #this function gets the wrong framerate for webm - vidcap.get(cv2.CAP_PROP_FPS)
#        if abs(self.frame_rate-vidcap.get(cv2.CAP_PROP_FPS)) <= 1:
#            self.frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
        
        print("video {} framerate {}".format(video_path.split('/')[-1],self.frame_rate))
        df = pd.read_csv(csv_path,header=None)
        
        
        self.subsample_rate = subsample_rate
        self.sampling_ratio = min(self.frame_rate,self.frame_rate//subsample_rate)
        
        self.img_names = pd.Series([str('{0:0>7}'.format(x)) 
                          for x in range(0,num_jpg) if x%self.sampling_ratio == 0])
        
        self.img_names = self.img_names.astype(str) + extension
        self.frame_dir = frame_dir
        self.video_path = video_path
        self.transform = transform
        self.num_files = len(self.img_names)
        
        self.gt = construct_gt(df,self.num_files,self.duration)
#        if self.num_files <=50:
#            pdb.set_trace()
        
    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        img_name = os.path.join(self.frame_dir,
                                self.img_names.iloc[idx])
        image = Image.open(img_name)
        #landmarks = self.landmarks_frame.iloc[idx, 1:]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            image = self.transform(image)

        return image