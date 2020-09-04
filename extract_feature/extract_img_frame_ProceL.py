# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:44:27 2020

@author: Warmachine
"""



import cv2
import os
import threading

import pdb

dataset_dir = "./datasets/ProceL/videos/"
save_dir = "./datasets/ProceL/frames/"

leading_zero = 7

def thread_function(cat):
    cat_video_dir = dataset_dir+cat+'/'
    
    cat_video_dir_save = save_dir+cat+'/'
    
    if not os.path.isdir(cat_video_dir_save):
        os.mkdir(cat_video_dir_save)
    
    for video in os.listdir(cat_video_dir):
        print(video)
        video_path_save = cat_video_dir_save+video.split('.')[0]+'/'
        
        if not os.path.isdir(video_path_save):
            os.mkdir(video_path_save)
        
        video_path = cat_video_dir+video
        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(video_path_save+('{0:0>'+str(leading_zero)+'}.jpg').format(count), image)     # save frame as JPEG file
            success,image = vidcap.read()
#            print('Read a new frame: ', success)
            
            count += 1
    
    
        print('Done cat {} video {} frames {}'.format(cat,video,count))

threads = list()

#extension = 'mp4' #could be mp4 or webm
for cat in os.listdir(dataset_dir):
    x = threading.Thread(target=thread_function, args=(cat,))
    threads.append(x)
    x.start()
    