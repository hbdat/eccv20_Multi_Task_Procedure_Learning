
import os
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py
import numpy as np
import time
import pdb
from core.ProceLDataset import ProceLDataset
from global_setting import raw_data_dir,data_path_tr,data_path_tst

class FeatureVGGDataset(Dataset):
    """Feature VGG Dataset."""

    def __init__(self, root_dir, mat_path, target_fps,verbose = False, is_visualize = False,target_cat = None, is_all = False):
        """
        Args:
            root_dir (string): Directory with all the feature
            hdf5 files.
            mat_path (string): Directory with all the annotation
            matlab files
        """

        self.root_dir = root_dir
        self.mat_path = mat_path
        self.target_fps = target_fps
        self.verbose = verbose
        self.cat_video_tuples = []
        self.cat_video_ll = []
        self.cat2idx = {}
        self.mat_data={}
        
        self.idx2cat = []
        self.is_visualize = is_visualize
        self.is_all = is_all
        input_size = 224
        
        ### for visualization ###
        self.raw_data_dir = raw_data_dir#'/mnt/raptor/datasets/ProceL_Dat/'
        
        self.transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()
        ])
        ### for visualization ###    
        
        list_dir = os.listdir(root_dir)
        list_dir.sort()
        for idx_cat,category in enumerate(list_dir):
            cat_path = os.path.join(root_dir, category)
            self.cat2idx[category] = idx_cat
            self.idx2cat.append(category)
            self.cat_video_ll.append([])
            for video_files in os.listdir(cat_path):
               self.cat_video_ll[idx_cat].append(video_files)
               
            cat_path = os.path.join(self.mat_path, category + '_data.mat')
            self.mat_data[category] = sio.loadmat(cat_path)
            
        self.n_cat = len(self.cat_video_ll)
        
        if target_cat is None:
            print('Alternate category loader')
            counter = 0
            is_cont = True
            while is_cont:
                is_cont = False
                for idx_cat in range(self.n_cat):
                    if counter < len(self.cat_video_ll[idx_cat]):           #as long as there is video in some cats then is_cont = True
                        is_cont = True
                        self.cat_video_tuples.append((self.idx2cat[idx_cat], self.cat_video_ll[idx_cat][counter]))   # creates a tuple list of category and its videos
                counter += 1
            self.n_video = sum([len(cat) for cat in self.cat_video_ll])
        else:
            if self.is_all:
                print("!!!! Load all videos from both training and testing !!!!")
                assert root_dir == data_path_tr
                ### Augment test video in evaluation
                print("Augment training video with testing video")
                cat_path_aug = os.path.join(data_path_tst, target_cat)
                target_cat_idx = self.cat2idx[target_cat]
                for video_files in os.listdir(cat_path_aug):
                   self.cat_video_ll[target_cat_idx].append(video_files)
                ### Augment test video in evaluation
            
            print('Target Cat {}'.format(target_cat))
            target_cat_idx = self.cat2idx[target_cat]
            for cat_video in self.cat_video_ll[target_cat_idx]:
                self.cat_video_tuples.append((self.idx2cat[target_cat_idx],cat_video))
            self.n_video = len(self.cat_video_tuples)
        
    def load_frames(self,frame_path,batch_size = 200):
        
        proceL_dataset = ProceLDataset(frame_path , 
                                       transform = self.transforms)
        dataset_loader = DataLoader(proceL_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=10)      #this has to be zero if you use multi-thread in VGG dataset. You can not create thread in thread
        
        all_frames = []
        for i_batch, frames in enumerate(dataset_loader):
            all_frames.append(frames.cpu())
        all_frames = torch.cat(all_frames,0)
        all_frames = all_frames.permute(0,2,3,1)               #n,244,244,3 <== n,3,244,244 
        return all_frames
    
    def __len__(self):
        return self.n_video

#    def annotate(self, path, category):
#
#        """
#        This function loads the data from the annotation matlab files and
#        brings it for use in the python environment
#        """
#        mat_data={}
#        cat_path = os.path.join(path, category + '_data.mat')
#        mat_data[category] = sio.loadmat(cat_path)
#        return mat_data

    def create_segment_list(self, category, video, mat_data):

        """
        Function to fetch data as to which segment does each frame of a video
        belong to

        Parameters
        ----------
        category: Current video's category
        video: File name of the current video
        mat_data: Annotation data of the video

        Returns
        -------
        segment_video_list: A list of size equal to number of frames denoting
        the segment number of the frame
        """
        video = video[:-20]      #removing "_ft_feature_vgg.hdf5" from the video name
        video_no = int(video[-2:])    #getting the last two video number
        video_no-=1
        n_segments = len(mat_data[category]['superframe_frame'][video_no][0])

        j = 1
        segment_video_list = []
        for s in range(n_segments):
            first_frame = mat_data[category]['superframe_frame'][video_no][0][s][0]
            last_frame = mat_data[category]['superframe_frame'][video_no][0][s][1]
            for f in range(first_frame, last_frame+1):
                segment_video_list.append(j)
            j+=1

        return segment_video_list

    def original_fps(self, category, video, mat_data):

        """
        Function to calculate the frames per second rate of the video

        Parameters
        ----------
        category: Current video's category
        video: File name of the current video
        mat_data: Annotation data of the video

        Returns
        -------
        fps: Frames per second rate

        """
        video = video[:-20]      #removing "_ft_feature_vgg.hdf5" from the video name
        video_no = int(video[-2:])    #getting the last two video number
        video_no-=1
        n_segments = len(mat_data[category]['superframe_time'][video_no][0])
        n_secs = mat_data[category]['superframe_time'][video_no][0][n_segments-1][1]
        n_frames = mat_data[category]['superframe_frame'][video_no][0][n_segments-1][1]

        return int(n_frames/n_secs)

    def check_match_annotation(self,category,video,mat_data,feature):
        video = video[:-20]      #removing "_ft_feature_vgg.hdf5" from the video name
        video_no = int(video[-2:])    #getting the last two video number
        video_no-=1
        n_segments = len(mat_data[category]['superframe_time'][video_no][0])
        n_frames_a = mat_data[category]['superframe_frame'][video_no][0][n_segments-1][1]
        n_frames_f = feature.shape[0]
        is_match = n_frames_a == n_frames_f
        if self.verbose:
            print(video)
            if is_match:
                print('match annotations {} {}'.format(n_frames_a,n_frames_f))
            else:
                print('miss-match annotations {} {}'.format(n_frames_a,n_frames_f))
        return int(is_match)

    def create_mask_seg_list(self, category, video, feature, mat_data, target_fps = 2, original_fps = 30):

        """
        Function to create a mask and the segment list which is used to
        subsample the features

        Parameters
        ----------
        category: Current video's category
        video: File name of the current video
        feature: vgg features of a video
        mat_data: Annotation data of the video
        factor: subsampling factor
        original_fps: original frame rate

        Returns
        -------
        segment_list: A list of size equal to number of frames denoting
        the segment number of the frame
        mask_video_list: A list of Trues and Falses where True indicates which
        frame to keep and False indicates which frames to drop.

        """
        n_frames = len(feature)
        segment_list = self.create_segment_list(category, video, mat_data)
        mask_video_list = []
        i = 0
#        assert original_fps >= target_fps
        self.factor = max(int(original_fps/target_fps),1)
        for f in range(n_frames):                                             #Creating mask for subsampling
            if(i%self.factor==0):
                mask_video_list.append(True)
            else:
                mask_video_list.append(False)
            i+=1
        return mask_video_list, torch.tensor(segment_list)

    def create_key_step_list(self, category, video, feature, frame_idx, mat_data):

        """
        Function to keep track as to which key step does each frame in the
        subsampled feature matrix belong to.

        Parameters
        ----------
        category: Current video's category
        video: File name of the current video
        feature: subsampled vgg features of a video
        frame_idx: indices of frames starting from 1
        mat_data: Annotation data of the video

        Returns
        -------
        key_step_list: A tensor of size equal to number of frames denoting
        the key step number each frame belongs to. If a key step does not
        belong to any key step, it stores 0 at that index.

        If a frame belongs to multiple key steps, it stores the key step
        number which is smaller. This might be because of error
        while annotating videos

        """
        video = video[:-20]
        video_no = int(video[-2:])
        video_no-=1

        n_frames = len(feature)

        n_keysteps = 0
        steps_tuple = []
        for keystep in mat_data[category]['key_steps_frame'][video_no][0]:      # each annotation has shape [1,2] or [0,0] (empty)
            n_keysteps += 1
            for k in range(len(keystep)):
                for kx in keystep[k]:
                    steps_tuple.append((n_keysteps, kx))                      #Creates tuples of key step number and its intervals
        

        def getKey(item):                                             #Sorting tuple based on start frame of each interval
            return item[1][0]

        steps_tuple = sorted(steps_tuple, key = getKey)
        for i in range(len(steps_tuple)):                                #To detect overlapping of intervals
            if(i>0):
                if steps_tuple[i][1][0]<=steps_tuple[i-1][1][1]:
                    overlap_start = steps_tuple[i][1][0]
                    overlap_end = steps_tuple[i-1][1][1]
                    overlap_mean = int((overlap_start + overlap_end)/2)
                    steps_tuple[i-1][1][1] = overlap_mean
                    steps_tuple[i][1][0] = overlap_mean+1
                    #print("Overlap!!!")
                else:
                    continue
                    #print("No Overlap!!!")

        ## Bad implementation
#        for f in range(n_frames):
#            flag = False
#            for idx_k, key_step in steps_tuple:
#
#                if frame_idx[f] in range(key_step[0], key_step[1]+1):
#                    key_step_list.append(idx_k)
#                    flag = True
#                    #break
#            if(flag == False):
#                key_step_list.append(0)
        
        ## Bad implementation
        keysteps = torch.zeros((n_frames))
        for step in steps_tuple:
            start,end = step[1]
            idx_step = step[0]
            
            ## these always compute lower bound
            idx_start = (start-1)//self.factor
            if idx_start*self.factor+1 < start:
                idx_start += 1
            
            idx_end = (end-1)//self.factor
            
            ## these always compute lower bound
            
            keysteps[idx_start:idx_end+1]=idx_step
        
        return keysteps, n_keysteps

    def __getitem__(self, idx):
        ###
        tic = time.clock()
        
        category = self.cat_video_tuples[idx][0]
        video = self.cat_video_tuples[idx][1]
        
        file_path = os.path.join(self.root_dir, category, video)
        
        if self.is_all:
            ### Augment training video with testing video (minimal edit)
            if not os.path.isfile(file_path):
                file_path = os.path.join(data_path_tst, category, video)
            ### Augment training video with testing video (minimal edit)
        
        file = h5py.File(file_path, 'r')
        
        key = list(file.keys())[0]
        feature = file[key].value
        feature_idx = torch.tensor(list(range(1, len(feature)+1)))
        if self.verbose:
            print('load video {}'.format(time.clock()-tic))
        ###
        
        ### load raw frames for visualization ###
        if self.is_visualize:
            cat_path = os.path.join(self.raw_data_dir, category)
            frame_path = os.path.join(cat_path, 'frames')
            frame_path = os.path.join(frame_path, key)
            frames = self.load_frames(frame_path)
        ###
        
#        ###
#        tic = time.clock()
#        mat_data = self.annotate(self.mat_path, category)
#        if self.verbose:
#            print('load annotation {}'.format(time.clock()-tic))
#        ###
        
        is_match = self.check_match_annotation(category,video,self.mat_data,feature)
        
        ###
        tic = time.clock()
        fps = self.original_fps(category, video, self.mat_data)
        if self.verbose:
            print('time load fps {}'.format(time.clock()-tic))
        ###
        if self.verbose:
            print('fps {}'.format(fps))
        ###
        tic = time.clock()
        mask_list, seg_list = self.create_mask_seg_list(category, video, feature,  self.mat_data, self.target_fps, fps)
        if self.verbose:
            print('time load segment {}'.format(time.clock()-tic))
        ###
        
        ###
        tic = time.clock()
        subsampled_feature = feature[mask_list]
        subsampled_feature_idx = feature_idx[mask_list]
        subsampled_segment_list = seg_list[mask_list]
        
        del feature
        
        if self.is_visualize:
            subsampled_frames = frames[mask_list]
        else:
            subsampled_frames = torch.zeros(0)
            
        if self.verbose:
            print('time subsample {}'.format(time.clock()-tic))
        ###
        
        ###
        tic = time.clock()
        key_step_list, n_keysteps = self.create_key_step_list(category, video, subsampled_feature, subsampled_feature_idx, self.mat_data)
        if self.verbose:
            print('time load keystep {}'.format(time.clock()-tic))
        ###
        
        
        file.close()
        out_package = {'cat_labels':self.cat2idx[category], 'cat_names':category, 'video':video[:-17], 'subsampled_feature':subsampled_feature,
                   'subsampled_segment_list':subsampled_segment_list, 'key_step_list':key_step_list, 'n_og_keysteps':n_keysteps,
                   'subsampled_frames':subsampled_frames,'is_match':is_match,'full_video_name':video}
        return out_package
