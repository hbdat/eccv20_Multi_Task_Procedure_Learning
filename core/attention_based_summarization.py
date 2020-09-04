# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 00:25:34 2019

@author: Warmachine
"""

import scipy.io as sio
import os
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
from PIL import Image
from skimage import io, transform
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim 
from torchvision import transforms, utils, models
import h5py
import time
import pdb

from core.attention_model import AttentionModel
from core.helper import get_parameters,fcsn_preprocess_fbar,convert_keystep_2_keyframe,get_list_param_norm
from core.fcsn import FCSN


from collections import OrderedDict 

class AttentionSummarization(nn.Module):
    def __init__(self,n_class,n_category,lambda_1,dim_input = 512,verbose=True,temporal_att = True,spatial_att = True,keystep_att = True,is_balance=True,is_hof = False):
        super(AttentionSummarization, self).__init__()
        
        self.verbose = verbose
        self.temporal_att = temporal_att
        self.spatial_att = spatial_att
        self.keystep_att = keystep_att
        self.is_hof = is_hof
        
        dim_hof = 2000
        
        if is_hof:
            dim_input_extend = dim_hof+dim_input
            print("!!!!!!!!! USE HOF !!!!!!!!!")
        else:
            dim_input_extend = dim_input
        
        ## Attention
        self.model_att = AttentionModel(dim_input, 1, bias=True,is_att=spatial_att)
        
        self.model_att_2 = AttentionModel(dim_input_extend, 1, bias=True,is_att=keystep_att,normalize_F=False)
#        model_att.to(device)
        print('-'*30)
        print('spatial att')
        self.att_params = get_parameters(self.model_att,verbose=self.verbose)
        
        print('-'*30)
        print('keystep att')
        self.att_params_2 = get_parameters(self.model_att_2,verbose=self.verbose)
        
        ## FCSN
        self.n_class = n_class
        self.n_category = n_category
        self.lambda_1 = lambda_1
        self.model_fcsn = FCSN(n_class,n_category,lambda_1,dim_input = dim_input_extend,verbose=self.verbose,temporal_att= temporal_att)
#        model_fcsn.to(device)
        print('fcsn_params')
        self.fcsn_params = get_parameters(self.model_fcsn,verbose=self.verbose)
        
        print("n_classes {}".format(n_class))
        
        ##Loss
        self.func_loss_cat = nn.CrossEntropyLoss()
        print('pos-weight in loss keystep')
        print('lambda is for loss_cat: focus loss in loss_keystep')
        
        if self.spatial_att:
            print('Use spatial attention')
        else:
            print('No spatial attention')
        
        if self.temporal_att:
            print('Use temporal attention')
        else:
            print('No temporal attention')
            
        if self.keystep_att:
            print('Use keystep attention')
        else:
            print('No keystep attention')
            
        print('-'*30)
        print("n_class: {}".format(n_class))
        self.is_balance = is_balance
        
        if self.is_balance:
            print('is_balance !!!')
        
        dim_classifier = dim_input_extend
        
        
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(dim_classifier, n_category))
            ]))
    
        
#        self.classifier = nn.Sequential(OrderedDict([
#            ('fc1', nn.Linear(dim_classifier, dim_classifier//2)),
#            ('relu1', nn.ReLU(inplace=True)),
#            ('drop1', nn.Dropout()),
#            ('fc2', nn.Linear(dim_classifier//2, n_category)),
#            ]))
    
        self.classifier_params = get_parameters(self.classifier,verbose=self.verbose)
    
    def set_att(self, temporal_att=None,spatial_att=None,keystep_att=None):
        assert temporal_att==keystep_att
        if temporal_att is not None:
            self.temporal_att = temporal_att
            self.keystep_att = keystep_att
            self.model_att_2.is_att = self.keystep_att
            
        if spatial_att is not None:
            self.spatial_att = spatial_att
            self.model_att.is_att = self.spatial_att
        
    
    def aggregate_features(self,feature, seg_list,feature_hof):

        """
        Function to aggregate all the frames belonging to a segment into one
        tensor by taking mean across them.
    
        Parameters
        ----------
        feature: subsampled vgg features of a video
        seg_list: subsampled list of segments
    
        Returns
        -------
        aggregated_feature: Returns the features such that each tensor
        represent a segment found in the original sub sampled features where
        the data of all the tensors of one segment is aggregated into one
        tensor
    
        """
#        pdb.set_trace()
        assert feature.shape[0] == 1
        aggregated_features = []
        for b in range(feature.shape[0]):
            segments = np.unique(seg_list[b])
            for s in segments:
                indicator = seg_list[b] == s
                aggregated_seg_feature = torch.mean(feature[:,indicator,:], 1, True)
                if feature_hof is not None and self.is_hof:
                    concat_feature = feature_hof[:,s-1,:]
                    concat_feature = concat_feature[:,None,:].float()
                    aggregated_seg_feature = torch.cat([aggregated_seg_feature,concat_feature],dim=2)
                aggregated_features.append(aggregated_seg_feature)
            aggregate_features = torch.cat(aggregated_features,dim=1)
        return aggregate_features
    
    def forward_middle(self,feature,subsampled_segment_list,feature_hof=None):           #features: [1,T,49,512]
        fbar,alphas = self.model_att(feature)
        
        fbar_seg = self.aggregate_features(fbar, subsampled_segment_list,feature_hof)
        
        fbar_seg = torch.transpose(fbar_seg, dim0 = 2, dim1 = 1)
        
        fbar_seg = fcsn_preprocess_fbar(fbar_seg,self.verbose)
        
        return fbar_seg
    
    def forward(self,feature,subsampled_segment_list,feature_hof=None):      #features: [1,T,49,512]
        fbar,alphas_sp = self.model_att(feature)               #[1,T,512] <= [1,T,49,512]
        
        fbar_seg = self.aggregate_features(fbar, subsampled_segment_list,feature_hof)       #[1,S,512]   
        
        fbar_seg = torch.transpose(fbar_seg, dim0 = 2, dim1 = 1)                #[1,512,S] <== #[1,S,512]
        
        fbar_seg = fcsn_preprocess_fbar(fbar_seg,self.verbose)                  ##[1,512,S]
        
        keysteps = self.model_fcsn(fbar_seg)                                    #bmt: [1,M,S] <== Warning this score haven't been normalized yet
        
        if self.temporal_att:
            att_seg_of_ks = F.softmax(keysteps,dim=1)
        else:
            att_seg_of_ks = F.softmax(keysteps.new_full(keysteps.size(),1),dim=1)
        
#        print("attention seg: {}".format(att_seg_of_ks))
#        print("att_seg_norm_parmas {}".format(np.sum(get_list_param_norm(self.fcsn_params))))
        # attention over segment to get the keystep feature
        f_keysteps = torch.einsum('bds,bms->bmd',fbar_seg,att_seg_of_ks)         #[1,M,512] <== [1,512,S],[1,M,S]
        
        f_keysteps = f_keysteps[:,None,:,:]                 #[1,1,M,512] <== [1,M,512]
                
        assert f_keysteps.size(1) == 1  
        
        f_videos,alphas_ks = self.model_att_2(f_keysteps)            #[1,1,512] <== [1,1,M,512], take in a rank 4 tensor
        
        
#        print("attention ks: {}".format(alphas_ks))
#        print("att_ks_norm_parmas {}".format(np.sum(get_list_param_norm(self.att_params_2))))
        
#        print("classifier_norm_parmas {}".format(np.sum(get_list_param_norm(self.classifier_params))))
        
        assert f_videos.size(1) == 1  
        
        f_videos = f_videos[:,0]                            #[1,512] <== [1,1,512]
        
        cats = self.classifier(f_videos)
        
        return keysteps,cats,alphas_sp,alphas_ks
    
    def compute_loss_keystep_cat(self,keysteps,cats,keystep_labels,cat_labels):
        device = keysteps.device
        keystep_labels=keystep_labels.to(device)
        cat_labels = cat_labels.to(device)
        
        keysteps = torch.transpose(keysteps,dim0=2,dim1=1)
        
        keysframe_labels=convert_keystep_2_keyframe(keystep_labels)
        keystep_frame_1_hot = F.one_hot(keysframe_labels,2).float()
        
        assert keystep_labels.size(0) == 1
        
        n_pos =  torch.max(torch.sum(keysframe_labels,dim=1).float(),torch.ones(1).to(device))
        pos_weight = (keystep_labels.size(1)-n_pos)/n_pos
        
        loss_keystep = F.binary_cross_entropy_with_logits(keysteps,keystep_frame_1_hot,pos_weight=pos_weight )
#        loss_keystep = F.binary_cross_entropy_with_logits(keysteps,keystep_frame_1_hot)
        loss_cat = self.func_loss_cat(cats,cat_labels)
        loss = loss_keystep+self.lambda_1*loss_cat
        package = {'loss':loss,'loss_keystep':loss_keystep,'loss_cat':loss_cat,'pos_weight':-1}
        return package
    
    def compute_loss_rank_keystep_cat(self,keysteps,cats,keystep_labels,cat_labels):
        device = keysteps.device
        cat_labels = cat_labels.to(device)
        
        loss_cat = self.func_loss_cat(cats,cat_labels)#torch.ones(1).to(device)*-1#
        
        if keystep_labels is not None:
            
            keystep_labels=keystep_labels.to(device)
            keysteps = torch.transpose(keysteps,dim0=2,dim1=1)          #[1,T,M] <== [1,M,T]
            
            assert keysteps.size(-1) == self.n_class
            
            class_weights = keystep_labels.new_zeros(self.n_class).float()
            frame_weights = keystep_labels.new_zeros(keystep_labels.size()).float()
            
            n_frames = keystep_labels.size(1)*1.0
            ## compute weights for every frames
            for c in range(self.n_class):
                assert frame_weights.size() == keystep_labels.size()
                mask_c = keystep_labels == c            #correct query as long as keystep_labels.size() == frame_weights.size()
                if self.is_balance:
                    class_weights[c] = n_frames/(torch.sum(mask_c).float()+1.0)
                else:
                    class_weights[c] = 1.0#
                frame_weights[mask_c] = class_weights[c]
                
            #s_max = torch.max(keysteps,dim = 2)     #[1,T] <== [1,T,M]
            s = keysteps
            s_gt = torch.gather(keysteps,2,keystep_labels.view(1,-1,1).long())
            
            margin = 1-(s_gt-s)                                         #[1,T,M] <== [1,T,1] - [1,T,M]
            loss_keystep = torch.max(margin,torch.zeros_like(margin))  
            loss_keystep = torch.mean(loss_keystep,dim = -1)#torch.max(loss_keystep,dim = -1)#
            assert loss_keystep.size() == frame_weights.size()
            loss_keystep = torch.mean(loss_keystep*frame_weights)  
            loss = loss_keystep+self.lambda_1*loss_cat
        else:
            class_weights = torch.ones(1)*-1
            loss_keystep = torch.ones(1)*-1
            loss = loss_cat
        package = {'loss':loss,'loss_keystep':loss_keystep,'loss_cat':loss_cat,'class_weights':class_weights}
        return package
    
    def compute_loss_cat(self,cats,cat_labels):
        device = cats.device
        cat_labels = cat_labels.to(device)
        
        assert cat_labels.size(0) == 1
        
        loss_cat = self.func_loss_cat(cats,cat_labels)
        loss = loss_cat
        package = {'loss':loss,'loss_cat':loss_cat}
        return package
    
    def compute_loss_cat_dummy_keystep(self,keysteps,cats,keystep_labels,cat_labels):
        device = keysteps.device
        keystep_labels=keystep_labels.to(device)
        cat_labels = cat_labels.to(device)
        
        keysteps = torch.transpose(keysteps,dim0=2,dim1=1)
        
        keysframe_labels=convert_keystep_2_keyframe(keystep_labels)
        
        n_pos =  torch.max(torch.sum(keysframe_labels,dim=1).float(),torch.ones(1).to(device))
        
        assert keystep_labels.size(1) > n_pos
        pos_weight = (keystep_labels.size(1)-n_pos)/n_pos
        
        keysframe_labels = keysframe_labels*2-1
        keysframe_labels[keysframe_labels>0] = pos_weight
        
        
        assert keystep_labels.size(0) == 1
        
        s_1 = keysteps[:,:,1]
        s_0 = keysteps[:,:,0]
        
        
        
        margin = 1-(s_1-s_0)
        margin = margin*keysframe_labels.float()
        
        loss_keystep = torch.max(margin,torch.zeros_like(margin))  
        loss_keystep = torch.mean(loss_keystep)
        
        loss_cat = self.func_loss_cat(cats,cat_labels)
        loss = 0*loss_keystep+loss_cat
        package = {'loss':loss,'loss_keystep':loss_keystep,'loss_cat':loss_cat,'pos_weight':pos_weight}
        return package
    
    def compute_loss_keystep(self,keysteps,keystep_labels):
        device = keysteps.device
        keystep_labels=keystep_labels.to(device)
        
        keysteps = torch.transpose(keysteps,dim0=2,dim1=1)
        
        keysframe_labels=convert_keystep_2_keyframe(keystep_labels)
        keystep_frame_1_hot = F.one_hot(keysframe_labels,2).float()
        
        assert keystep_labels.size(0) == 1
        
#        n_pos =  torch.max(torch.sum(keystep_labels,dim=1).float(),torch.ones(1).to(device))
#        pos_weight = (keystep_labels.size(1)-n_pos)/n_pos
        
#        loss_keystep = F.binary_cross_entropy_with_logits(keysteps,keystep_frame_1_hot,pos_weight=pos_weight )
        loss_keystep = F.binary_cross_entropy_with_logits(keysteps,keystep_frame_1_hot)
        loss = loss_keystep
        package = {'loss':loss,'loss_keystep':loss_keystep,'pos_weight':-1}
        return package