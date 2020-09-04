# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 22:12:50 2020

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
import pytube
import pandas as pd
import os
from global_setting import NFS_path,docker_path

'''
fix 
pip3 install git+https://github.com/giacaglia/pytube.git
https://github.com/nficano/pytube/issues/467
'''
#%%
meta_video_path = NFS_path+'/data/CrossTask/ProceL/ProceL_dataset/'
download_dir = docker_path+'/datasets/ProceL/videos/{}/'
#%%
df_error = pd.DataFrame()
for cat in os.listdir(meta_video_path):
    task_path = download_dir.format(cat)
    if not os.path.isdir(task_path):
        os.mkdir(task_path)
    
    with open(meta_video_path+cat+'/readme.txt','r') as f:
        while True: 
            line = f.readline() 
            if not line: 
                break
            
            items = line.split('\t')
            
            url = items[0]
            video_name = items[1]
            
            try:
                youtube = pytube.YouTube(url)
                
                #video = youtube.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').first()
                video = youtube.streams.first()
                video.download(task_path,filename = video_name+'.mp4') # path, where to video download.
            except Exception as e:
                print(e)
                df_error = df_error.append([video_name,url])
                df_error.to_csv('./download_youtube/error_video.csv')

    
        