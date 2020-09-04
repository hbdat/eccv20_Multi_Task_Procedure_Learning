# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:17:08 2020

@author: badat
"""

import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd)
#%%
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%

import pandas as pd
import numpy as np
from global_setting import NFS_path,mat_path
import shutil
#%%
map_file_name = {'setup_chromecast':'chromecast_data.mat','assemble_clarinet':'clarinet_data.mat','make_pbj_sandwich':'pbj_data.mat',
                 'change_iphone_battery':'phone_battery_data.mat','make_smoke_salmon_sandwich':'salmon_data.mat','tie_tie':'tie_data.mat',
                 'change_toilet_seat':'toilet_data.mat','change_tire':'changing_tire_data.mat','make_coffee':'coffee_data.mat',
                 'perform_cpr':'cpr_data.mat','jump_car':'jump_car_data.mat','repot_plant':'repot_data.mat'}
data_path = './data/ProceL_dataset/'

for cat in os.listdir(data_path):
    if cat in map_file_name:
        old_name = data_path + cat +'/data.mat'
        new_name = mat_path+map_file_name[cat]
        shutil.copy(old_name, new_name)