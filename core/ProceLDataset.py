# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:08:31 2019

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
#import pdb

class ProceLDataset(Dataset):
    """Procedure Learning dataset."""

    def __init__(self, root_dir, csv_file=None, transform=None, idx_frames = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to 
            be applied on a sample.
        """
        extension = '.jpg'
        self.num_files = len([f for f in os.listdir(root_dir)
                         if os.path.isfile(os.path.join(root_dir, f)) and f.endswith(extension)])
        #print(num_files)
        
        if idx_frames is None:
            idx_frames = range(1,self.num_files+1)
        
        self.img_names = pd.Series([str('{0:0>6}'.format(x)) 
                          for x in idx_frames])
    
        self.idx_frames = idx_frames
        self.img_names = self.img_names.astype(str) + extension
        #print(self.img_names)
#         self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.idx_frames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.img_names.iloc[idx])
        image = Image.open(img_name)
        #landmarks = self.landmarks_frame.iloc[idx, 1:]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            image = self.transform(image)

        return image