# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:29:36 2019

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
import pickle

from core.FeatureVGGDataset_Fast import FeatureVGGDataset_Fast
from core.attention_based_summarization import AttentionSummarization

from core.self_supervision_summarization_cat_batch import SelfSupervisionSummarization

from core.helper import aggregated_keysteps,fcsn_preprocess_keystep,\
                        get_parameters,get_weight_decay,evaluation_align,\
                        visualize_attention,Logger
                        
from core.alignment import compute_align_F1
from global_setting import NFS_path,data_path_tr_fast,data_path_tst_fast,mat_path
# Ignore warnings
#import warnings
#warnings.filterwarnings("ignore")
#%%
folder = sys.argv[1]#"all_cat_ss_same_cat_batch_SS"#sys.argv[1]
print('Folder {}'.format(folder))
#%%
plt.ion()   # interactive mode
#%% use GPU
M = 50
repNum = int(sys.argv[3])#15
#%%
idx_GPU = sys.argv[2]
device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
#%% hyper-params

batch_size = 1
target_fps = 2

verbose = False
n_video_iters = 1
n_class = M
num_worker = 5
#number_eval = 5
is_save = True
n_epoches = 4
is_balance = True
switch_period = 12*10           # n_task * n_video_per_task: 12*10
is_ss = True
lambda_1 = 0.5

per_keystep = False
#%%
if is_save:
    print("Save")
    print("!"*30)
#%%
list_cat = os.listdir(data_path_tr_fast)
#%%
feature_dataset_tr = FeatureVGGDataset_Fast(data_path_tr_fast, mat_path, target_fps,verbose = verbose,is_visualize=False,target_cat=None)
dataset_loader_tr = DataLoader(feature_dataset_tr,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = num_worker)

feature_dataset_tr_2 = FeatureVGGDataset_Fast(data_path_tr_fast, mat_path, target_fps,verbose = verbose,is_visualize=False,target_cat=None)
dataset_loader_tr_2 = DataLoader(feature_dataset_tr_2,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = 0)

n_category = len(feature_dataset_tr.cat2idx)
n_train = feature_dataset_tr.n_video
print('Training set size: {}'.format(n_train))

#n_test = 0
#list_dataloader_tst = []
#for cat_name in list_cat:
#    print("Test dataloader: {}".format(cat_name))
#    feature_dataset_tst = FeatureVGGDataset(data_path_tst, mat_path, target_fps,verbose = verbose,target_cat=cat_name)
#    dataset_loader_tst = DataLoader(feature_dataset_tst,
#                                batch_size = batch_size,
#                                shuffle = False,
#                                num_workers = num_worker)
#    list_dataloader_tst.append([cat_name,dataset_loader_tst])
#    n_test += feature_dataset_tst.n_video
#    
#print('Testing set size: {}'.format(n_test))
#%%
#n_keysteps = feature_dataset_tr.mat_data[cat_name]['grammar'].shape[0]
#n_class = n_keysteps+1
#%%
experiment_dir = NFS_path+'results/'+folder+'/rank_key_all_cat_ss_K_lambda_{}_{}_GPU_{}_time_{}/'.format(lambda_1,repNum,idx_GPU,str(time.time()).replace('.','d'))
if is_save:
    os.makedirs(experiment_dir)
    orig_stdout = sys.stdout
    f = open(experiment_dir+'specs.txt', 'w')
    sys.stdout = f
    
assert M >= n_class    
#if is_save:
#    with open(experiment_dir+'log.txt', 'a') as file:
#        file.write("n_keystep: {}\n".format(n_keysteps))
#%%
model = AttentionSummarization(M,n_category,lambda_1,dim_input = 512,verbose=verbose,temporal_att=True,is_balance=is_balance)

assert model.lambda_1 > 0

model.to(device)
print('fcsn_params')
att_params = model.att_params
fcsn_params = model.fcsn_params

#%%
if is_ss:
    ss_model = SelfSupervisionSummarization(M = M,repNum=repNum)
else:
    ss_model = None
#%%
lr = 0.001#0.001
weight_decay = 0.000
momentum = 0.0
params = [{'params':att_params,'lr':lr,'weight_decay':weight_decay},
          {'params':fcsn_params,'lr':lr,'weight_decay':weight_decay}]
#optimizer = optim.Adam(params)
optimizer  = optim.RMSprop( model.parameters() ,lr=lr,weight_decay=weight_decay, momentum=momentum)
#%%
print('-'*30)
print('rank loss for keystep')
print('pos_weight')
print('-'*30)
print('GPU {}'.format(idx_GPU))
print('lambda_1 {}'.format(lambda_1))
print('lr {} weight_decay {} momentum {}'.format(lr,weight_decay,momentum))
print('target_fps {}'.format(target_fps))
print('n_video_iters {}'.format(n_video_iters))
print('num_worker {}'.format(num_worker))
print('n_epoches {}'.format(n_epoches))
print('repNum {}'.format(repNum))
print('is_balance {}'.format(is_balance))
print("Switch period {}".format(switch_period))
#input('confirm?')
#%%
if is_save:
    train_logger=Logger(experiment_dir+'train.csv',['loss','loss_cat','loss_key','pos_weight'])
    test_logger=Logger(experiment_dir+'test.csv',['R_pred','P_pred','all_acc'])
#%%
if is_save:
    sys.stdout = orig_stdout
    f.close()
#%%
def measurement(is_test=True):
    eps = 1e-8
    
    list_P_pred = []
    list_R_pred = []
    
    list_P_pseudo = []
    list_R_pseudo = []
    
    list_n_video = []
    
    list_acc = []
    
    
    if is_test:
        prefix = 'tst_'
    else:
        prefix = 'tr_'
    
    for cat_name in list_cat:
        if is_test:
            feature_dataset_tst = FeatureVGGDataset_Fast(data_path_tst_fast, mat_path, target_fps,verbose = verbose,target_cat=cat_name)
        else:
            feature_dataset_tst = FeatureVGGDataset_Fast(data_path_tr_fast, mat_path, target_fps,verbose = verbose,target_cat=cat_name)
        dataset_loader_tst = DataLoader(feature_dataset_tst,
                                    batch_size = batch_size,
                                    shuffle = False,
                                    num_workers = num_worker)
        
        print(cat_name)
        
        out_package = evaluation_align(model,ss_model,dataset_loader_tst,device)
    
        R_pred, P_pred = out_package['R_pred'],out_package['P_pred']
        R_pseudo, P_pseudo = out_package['R_pseudo'],out_package['P_pseudo']
        
        acc = out_package['per_class_acc']
        
        
        list_P_pred.append(P_pred)
        list_R_pred.append(R_pred)
        
        list_P_pseudo.append(P_pseudo)
        list_R_pseudo.append(R_pseudo)
        
        list_n_video.append(feature_dataset_tst.n_video)
        
        list_acc.append(acc)
        
        if is_save:
            with open(experiment_dir+prefix+'log_{}.txt'.format(cat_name), 'a') as file:
                
                file.write("R_pred {} P_pred {}\n".format(R_pred,P_pred))
                file.write("R_pseudo {} P_pseudo {}\n".format(R_pseudo,P_pseudo))
                
                file.write("-"*30)
                file.write("\n")

                file.write("classification acc {}\n".format(acc))
       
                file.write("-"*30)
                file.write("\n")
        
                
        print("-"*30)
        del feature_dataset_tst
        del dataset_loader_tst
        
    if is_save: 
        
        test_logger.add([np.mean(list_R_pred),np.mean(list_P_pred),np.mean(list_acc)])
        test_logger.save()
#%%
def inf_train_gen(dataloader):
    while True:
        for output in dataloader:
            yield output

### two dataloaders need to be keep in sync ###
gen_tr = inf_train_gen(dataset_loader_tr)
gen_tr_2 = inf_train_gen(dataset_loader_tr_2)
#%%
for i_epoch in range(n_epoches):
    counter = 0
    
    while (counter < n_train):
        
        #1st pass
        
        list_F1_pseudo = []
        list_F1_pred = []
        
        if is_ss:
            ss_model.flush()
            for idx_v in range(switch_period):
                with torch.no_grad():
                    model.eval()
                    data_package = next(gen_tr)
                    cat_labels, cat_names, video, subsampled_feature, subsampled_segment_list, key_step_list, n_og_keysteps \
                    = data_package['cat_labels'],data_package['cat_names'],data_package['video'],data_package['subsampled_feature'],data_package['subsampled_segment_list'],data_package['key_step_list'],data_package['n_og_keysteps']
                    
                    # flatten the feature vector: [512,7,7] -> [512,49]
                    flatten_feature = subsampled_feature.view(batch_size,-1,512,7*7).to(device)
        #            print("Flatten tensor shape:", flatten_feature.shape)
                
                    #Transposing the flattened features
                    flatten_feature = torch.transpose(flatten_feature, dim0 = 2, dim1 = 3)
        #            print("Transposed Flatten tensor shape:", flatten_feature.shape)
                    print(idx_v,cat_names, video)
                    
                    
                    keystep_labels = aggregated_keysteps(subsampled_segment_list, key_step_list)
                    keystep_labels = fcsn_preprocess_keystep(keystep_labels,verbose = verbose)
                    
                    fbar_seg = model.forward_middle(flatten_feature,subsampled_segment_list)        #[1,512,T]
                    
                    ss_model.add_video(fbar_seg,video,cat_labels.item())                                              #[T,512]
            
            print('-'*30)
            print('subset selection')
            ss_model.foward()
            print('-'*30)
            print('unique assignment {} number of represent {} number cluster {}'.format(np.unique(ss_model.reps).shape,np.unique(ss_model.assignments).shape,ss_model.kmeans.cluster_centers_.shape))
            if is_save:
                with open(experiment_dir+'log.txt', 'a') as file:
                    file.write('unique assignment {} number of represent {} number cluster {}\n'.format(np.unique(ss_model.reps).shape,np.unique(ss_model.assignments).shape,ss_model.kmeans.cluster_centers_.shape))
        
        measurement()
#        if test_logger.get_len()-1 >= 8:        #this change across dataset
#            measurement(is_test=False)
        
        torch.save(model.state_dict(), experiment_dir+'model_ES_pred_or_{}'.format(test_logger.get_len()-1))
        pickle.dump(ss_model,open(experiment_dir+'SS_model_ES_pred_or_{}'.format(test_logger.get_len()-1),'wb'))
        
        for idx_v in range(switch_period):
            counter += 1
            data_package = next(gen_tr_2)
            
#            for _ in range(10):
                
            model.train()
            optimizer.zero_grad()
            
            cat_labels, cat_names, video, subsampled_feature, subsampled_segment_list, key_step_list, n_og_keysteps \
            = data_package['cat_labels'],data_package['cat_names'],data_package['video'],data_package['subsampled_feature'],data_package['subsampled_segment_list'],data_package['key_step_list'],data_package['n_og_keysteps']
            
            
            # flatten the feature vector: [1,T,512,7,7] -> [1,T,512,49]
            flatten_feature = subsampled_feature.view(batch_size,-1,512,7*7).to(device)
    #        print("Flatten tensor shape:", flatten_feature.shape)
        
            #Transposing the flattened features 
            flatten_feature = torch.transpose(flatten_feature, dim0 = 2, dim1 = 3)          #[1,T,49,512] <== [1,T,512,49]
            
    #        print("Transposed Flatten tensor shape:", flatten_feature.shape)
            print(idx_v,cat_names, video)
            
            
            keystep_labels = aggregated_keysteps(subsampled_segment_list, key_step_list)
            keystep_labels = fcsn_preprocess_keystep(keystep_labels,verbose = verbose)
        
            
            keysteps,cats,_,_ = model(flatten_feature,subsampled_segment_list)
            
            if is_ss:
                keystep_pseudo_labels = ss_model.get_key_step_label(video,cat_labels.item())
            else:
                keystep_pseudo_labels = None
            
            
    #        package = model.compute_loss_rank_keystep_cat(keysteps,cats,keystep_labels,cat_labels)
            
            package = model.compute_loss_rank_keystep_cat(keysteps,cats,keystep_pseudo_labels,cat_labels)
            
            loss,loss_cat,loss_key,class_weights = package['loss'],package['loss_cat'],package['loss_keystep'],package['class_weights']
            
            train_stats = [loss.item(),loss_cat.item(),loss_key.item(),class_weights.cpu().numpy()]
            print('loss {} loss_cat {} loss_key {} pos_weight {}'.format(*train_stats))
            print('weight_decay {}'.format(get_weight_decay(optimizer)))
            if is_save:
                train_logger.add(train_stats)
                train_logger.save()
            loss.backward()
            optimizer.step()
                
            n_keystep_background = n_og_keysteps.item()+1
            
        
        if is_save:
            with open(experiment_dir+'log.txt', 'a') as file:
                file.write("Pseudo {} Pred {}\n".format(np.mean(list_F1_pseudo),np.mean(list_F1_pred)))
        
        
#        if is_save and test_logger.is_max('all_F1_pred_or'):
#            torch.save(model.state_dict(), experiment_dir+'model_ES_pred_or')
#            pickle.dump(ss_model,open(experiment_dir+'SS_model_ES_pred_or','wb'))
#        
#        if is_save and test_logger.is_max('all_F1_pseudo_ks'):
#            torch.save(model.state_dict(), experiment_dir+'model_ES_pseudo_ks')
#            pickle.dump(ss_model,open(experiment_dir+'SS_model_ES_pred_ks','wb'))
#%%
measurement()
#measurement(is_test=False)
#%%
torch.save(model.state_dict(), experiment_dir+'model_final')
pickle.dump(ss_model,open(experiment_dir+'SS_model_final','wb'))