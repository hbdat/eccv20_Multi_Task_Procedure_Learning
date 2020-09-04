# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 06:21:17 2019

@author: Warmachine
"""


import torch
import torch.backends.cudnn as cudnn
import numpy as np


seed = 214
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
cudnn.benchmark = True

docker_path = './'

NFS_path = docker_path

## ProceL ##
data_path_tr = docker_path+'datasets/ProceL/features_full/training_data/'
data_path_tst = docker_path+'datasets/ProceL/features_full/testing_data/'

data_path_tr_fast = docker_path+'datasets/ProceL/features/training_data/'
data_path_tst_fast = docker_path+'datasets/ProceL/features/features/testing_data/'

mat_path = docker_path+'datasets/ProceL/anno/'

## Cross-Task
data_path_tr_CrossTask = docker_path+'/datasets/CrossTask/features/training_data/'
data_path_tst_CrossTask = docker_path+'/datasets/CrossTask/features/testing_data/'

