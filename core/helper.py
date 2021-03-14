import numpy as np
import torch
from torch.nn import functional as F
from collections import Counter
from sklearn.metrics import f1_score,precision_score,recall_score
import skimage.transform
import matplotlib.pyplot as plt
import pandas as pd
from core.frame_base_measurement import compute_align_MoF_UoI,compute_align_MoF_UoI_bg, compute_align_MoF_UoI_no_align, compute_align_MoF_UoI_bg_no_align 
import pdb
import os
from multiprocessing import Process,Queue


def get_list_param_norm(params):
    list_norms = []
    with torch.no_grad():
        for p in params:
            list_norms.append(torch.norm(p).cpu().item())
    return list_norms

def find_folder_with_pattern(pattern,path_dir):
    for path_folder in os.listdir(path_dir):
        if pattern in path_folder:
            return path_dir+path_folder+'/'
    return None
    

def aggregated_keysteps(subsampled_segment_list, key_step_list):

    """
    Function which aggregate the subsampled keysteps assigning the
    keystep which is found in majority for each segment

    Parameters
    ----------
    subsampled_segment_list: subsampled list of segments
    key_step_list: a list denoting the key step number
    each frame belongs to.

    Returns
    -------
    batch_aggregated_key_list: list denoting which segment in the aggregated
    feature belongs to which key step
    """
    #assert aggregated_features.shape[0] == 1
    batch_aggregated_key_list = []
    for b in range(subsampled_segment_list.shape[0]):
        segments = np.unique(subsampled_segment_list[b])
        aggregated_key_list = []

        for s in segments:
#            indicator = []
#            for idx in subsampled_segment_list[b]:
#                if s == idx:
#                    indicator.append(True)
#                else:
#                    indicator.append(False)
            indicator = subsampled_segment_list[b] == s
            segment_keysteps = key_step_list[b][indicator]
            unique_keys, key_freq = torch.unique(segment_keysteps, return_counts = True)

            max_v, max_k= 0,0
            for i in range(len(unique_keys)):
                if key_freq[i] > max_v:
                    max_v = key_freq[i]
                    max_k = unique_keys[i]

            aggregated_key_list.append(max_k)
        batch_aggregated_key_list.append(aggregated_key_list)

    return torch.tensor(batch_aggregated_key_list)      #[T]

def fcsn_preprocess_fbar(fbar_seg,verbose = False):     #fbar_seg [bx512xT]
     ## padding to be multiple of 32
    if verbose:
        print('before padding fbar {}'.format(fbar_seg.size()))
    n_frames = fbar_seg.size(2)
    n_pad = 0
    if n_frames % 32 != 0:      #this guarantees number of frames is at least 32
        n_pad = 32-n_frames%32      
    if n_frames+n_pad < 64:     # if number of frames if 32, then make it 64 for batchnorm
        n_pad += 32
    fbar_seg = F.pad(fbar_seg,(0,n_pad))
    if verbose:
        print('after padding fbar {}'.format(fbar_seg.size()))
        
    return fbar_seg

def fcsn_preprocess_keystep(keystep_labels,verbose = False):        #keysteps [bxT]
     ## padding to be multiple of 32
    if verbose:
        print('before padding keystep {}'.format(keystep_labels.size()))
    n_frames = keystep_labels.size(1)
    n_pad = 0
    if n_frames % 32 != 0:      #this guarantees number of frames is at least 32
        n_pad = 32-n_frames%32      
    if n_frames+n_pad < 64:     # if number of frames if 32, then make it 64 for batchnorm
        n_pad += 32
    keystep_labels = F.pad(keystep_labels,(0,n_pad))
    if verbose:
        print('after padding keystep {}'.format(keystep_labels.size()))
        
    return keystep_labels

def get_parameters(model,verbose = True):
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            if verbose:
                print("\t",name)
    return params_to_update

def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return lr

def get_weight_decay(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['weight_decay'])
    return lr

def convert_keystep_2_keyframe(keystep_labels):
    keysframe_labels = torch.clamp(keystep_labels,0,1).long()
    return keysframe_labels

def top_k_acc(cat_labels,cat_preds,k):
    cat_labels = cat_labels.cpu()
    cat_preds = cat_preds.cpu()
    idx_sort = torch.argsort(cat_preds,dim = 1,descending=True)
    top_k = idx_sort[:,:k]
    top_k = top_k.cpu().numpy()
    assert top_k.shape[0] == len(cat_labels)
    avg_acc = 0
    cat_labels = cat_labels.numpy()
    for i,cat_label in enumerate(cat_labels):
        if cat_label in top_k[i]:
            avg_acc+=1
    avg_acc /= len(cat_labels)
    return avg_acc

def compute_per_class_acc(test_label, predicted_label):
    
    test_label = test_label.cpu().numpy()
    
    predicted_label = predicted_label.cpu().numpy()
    predicted_label = np.argmax(predicted_label,-1)
    
    target_classes = np.unique(test_label)
    per_class_accuracies = np.zeros(target_classes.shape[0])
    
    for i in range(target_classes.shape[0]):
        is_class = test_label == target_classes[i]

        per_class_accuracies[i] = np.sum(predicted_label[is_class]==test_label[is_class])/np.sum(is_class)

    return np.mean(per_class_accuracies)

def evaluation_align(model,ss_model,dataset_loader_tst,device):
    k=1
    batch_size = 1
    
    list_ks_pred = []
    list_ks_pseudo = []
    list_ks_label = []
    
    list_top_k_acc = []
    all_cat_labels = []
    all_cat_preds = []
    segment_per_video = []
    
    counter = 0
    print('EVALUATION')
    with torch.no_grad():
        for data_package in dataset_loader_tst:
            model.eval()
            counter += 1
            
            cat_labels, cat_names, video, subsampled_feature, subsampled_segment_list, key_step_list, n_og_keysteps \
            = data_package['cat_labels'],data_package['cat_names'],data_package['video'],data_package['subsampled_feature'],data_package['subsampled_segment_list'],data_package['key_step_list'],data_package['n_og_keysteps']
#            print(video)
            
            if 'feature_hof' in data_package:
                feature_hof=data_package['feature_hof'].to(device)
            else:
                feature_hof = None
            
            flatten_feature = subsampled_feature.view(batch_size,-1,512,7*7).to(device)
        
            #Transposing the flattened features
            flatten_feature = torch.transpose(flatten_feature, dim0 = 2, dim1 = 3)
            
            keystep_labels = aggregated_keysteps(subsampled_segment_list, key_step_list)
            
            keystep_labels = fcsn_preprocess_keystep(keystep_labels)
            
            keysteps,cats,_,_ = model(flatten_feature,subsampled_segment_list,feature_hof)
        
            n_keystep_background = n_og_keysteps.item()+1
            
            ### evaluation for each video ###
            
            if ss_model is not None:
                pred_cat = np.argmax(cats.cpu().numpy(),-1)
                fbar_seg = model.forward_middle(flatten_feature,subsampled_segment_list)
                _,keystep_pseudo_labels = ss_model.predict(fbar_seg,pred_cat.item())
                M = ss_model.M
                print("pseudo: N_gt {} M {}".format(n_keystep_background,M))
                
                list_ks_pseudo.append(keystep_pseudo_labels)
#                P_pseudo,R_pseudo,F1_pseudo = [-1.0,-1.0,-1.0]
                pass
            else:
                pass
            
            keysteps_pred = torch.argmax(keysteps,dim = 1)
            M = keysteps.size(1)
            
            list_ks_pred.append(keysteps_pred)
            list_ks_label.append(keystep_labels)
            
            list_top_k_acc.append(top_k_acc(cat_labels,cats,k))
            all_cat_labels.append(cat_labels)
            all_cat_preds.append(cats)
            
            segment_per_video.append(keysteps_pred.size(1))
            
    out_package = {}
    
    arr_ks_pred = torch.cat(list_ks_pred,dim=1)
    arr_ks_label = torch.cat(list_ks_label,dim=1)
    if ss_model is not None:
        arr_ks_pseudo = torch.cat(list_ks_pseudo,dim=1)
    
    
    ######## evaluate ########
    
    MoF_pred, IoU_pred, P_pred = compute_align_MoF_UoI(keystep_pred=arr_ks_pred,keystep_gt=arr_ks_label,
                                                     n_keystep=n_keystep_background,M=M)
    
    MoF_pred_bg, IoU_pred_bg = compute_align_MoF_UoI_bg(keystep_pred=arr_ks_pred,keystep_gt=arr_ks_label,
                                                     n_keystep=n_keystep_background,M=M)
    
    if ss_model is not None:
        assert ss_model.M == M
        MoF_pseudo, IoU_pseudo, P_pseudo = compute_align_MoF_UoI(keystep_pred=arr_ks_pseudo,keystep_gt=arr_ks_label,
                                                         n_keystep=n_keystep_background,M=ss_model.M)
        
        MoF_pseudo_bg, IoU_pseudo_bg = compute_align_MoF_UoI_bg(keystep_pred=arr_ks_pseudo,keystep_gt=arr_ks_label,
                                                         n_keystep=n_keystep_background,M=ss_model.M)
    else:
        MoF_pseudo, IoU_pseudo, P_pseudo,MoF_pseudo_bg, IoU_pseudo_bg = [-1,-1,-1,-1,-1]
    
    all_cat_labels = torch.cat(all_cat_labels,dim = 0)
    all_cat_preds = torch.cat(all_cat_preds,dim = 0)
    per_class_acc = compute_per_class_acc(all_cat_labels,all_cat_preds)
    
    ######## evaluate ########
    
    out_package['list_top_k_acc'] = list_top_k_acc
    out_package['per_class_acc'] = per_class_acc
    
    out_package['R_pred'] = IoU_pred
    out_package['P_pred'] = P_pred
    
    out_package['R_pseudo'] = IoU_pseudo
    out_package['P_pseudo'] = P_pseudo
    
    return out_package

class Logger:
    def __init__(self,filename,cols,is_save=True):
        self.df = pd.DataFrame()
        self.cols = cols
        self.filename=filename
        self.is_save=is_save
        
    def add(self,values):
        self.df=self.df.append(pd.DataFrame([values],columns=self.cols),ignore_index=True)
        
    def get_len(self):
        return len(self.df)
        
    def save(self):
        if self.is_save:
            self.df.to_csv(self.filename)
    def get_max(self,col):
        return np.max(self.df[col])
    
    def is_max(self,col):
        return self.df[col].iloc[-1] >= np.max(self.df[col])