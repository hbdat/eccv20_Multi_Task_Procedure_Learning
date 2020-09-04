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
primary_task = [23521,59684,71781,113766,105222,94276,53193,105253,44047,76400,16815,95603,109972,44789,40567,77721,87706,91515]
meta_video_path = NFS_path+'/data/CrossTask/crosstask_release/videos.csv'
download_dir = docker_path+'/datasets/CrossTask/videos/{}/'
#%%
for id in primary_task:
    task_path = download_dir.format(id)
    if not os.path.isdir(task_path):
        os.mkdir(task_path)
#%%
df_video = pd.read_csv(meta_video_path,header=None)
#%%
df_error = pd.DataFrame()

for i, row in df_video.iterrows(): 
    task_id = row[0]
    video_name = row[1]
    url = row[2]
    task_path = download_dir.format(task_id)
    
    if task_id not in primary_task:
        continue
    print(task_id,video_name,url)
    print('error {}'.format(len(df_error)))
    if os.path.isfile(task_path+video_name+'.mp4'):
        print('skip')
        continue
    try:
        youtube = pytube.YouTube(url)
        
        #video = youtube.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').first()
        video = youtube.streams.first()
        video.download(task_path,filename = video_name) # path, where to video download.
    except Exception as e:
        print(e)
        df_error = df_error.append(row)
        df_error.to_csv('./download_youtube/error_video.csv')
        