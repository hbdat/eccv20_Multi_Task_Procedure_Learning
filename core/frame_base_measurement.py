# -*- coding: utf-8 -*- 
""" 
Created on Mon Mar  2 13:21:49 2020 
 
@author: Warmachine 
""" 
 
from scipy.optimize import linear_sum_assignment 
import torch.nn.functional as F 
import torch 
import numpy as np 
#%% 
import pdb 
 
eps = 1e-8 
def compute_align_MoF_UoI(keystep_pred,keystep_gt,n_keystep,M,per_keystep=False):         #[1,T] 
#    assert random==False 
    assert keystep_pred.size(0)==keystep_gt.size(0)==1 
     
    keystep_pred = keystep_pred[0].long()                      #[T]<==[1,T]     
    keystep_gt = keystep_gt[0].long()                          #[T]<==[1,T] 
     
    
#    print("single class assignment !!!!!\n"*10) 
#    keystep_pred = torch.ones(size=keystep_gt.size()).long()
    
#    if random: 
#        print("Random assignment !!!!!\n"*10) 
#        keystep_pred = torch.randint(low=torch.min(keystep_gt), high=torch.max(keystep_gt)+1, size=keystep_gt.size()) 
#     
    Z_pred = F.one_hot(keystep_pred,M).float().cpu().numpy()                  #[T,M] <== [T] 
    Z_gt = F.one_hot(keystep_gt,n_keystep).float().cpu().numpy()              #[T,K] <== [T] 
     
    Z_gt = Z_gt[:,1:]                                           #discard background info 
     
    ## pure numpy 
    assert Z_pred.shape[0] == Z_gt.shape[0] 
    T = Z_gt.shape[0]*1.0 
     
    Dis = 1.0 - np.matmul(np.transpose(Z_gt),Z_pred)/T            #[K,M] <== [T,K]^T matmul [T,M] 
     
    perm_gt, perm_pred = linear_sum_assignment(Dis) 
    print('perm_gt {} perm_pred {}'.format(perm_gt,perm_pred)) 
     
    Z_pred_perm = Z_pred[:,perm_pred] 
    Z_gt_perm = Z_gt[:,perm_gt] 
     
    print('alignment cost {}'.format(np.sum(Dis[perm_gt,perm_pred]))) 
    
    if per_keystep: 
        list_MoF = [] 
        list_IoU = [] 
        for idx_k in range(Z_gt_perm.shape[1]):             #[T,K] 
            pred_k =  Z_pred_perm[:,idx_k] 
            gt_k = Z_gt_perm[:,idx_k] 
             
            intersect = np.multiply(pred_k,gt_k)     #since these are binary matrix, the elementwise product is only 1 if and only if the prediction and ground-truth values are both 1 
            union = np.clip((pred_k+gt_k).astype(np.float),0,1)
            
            n_intersect = np.sum(intersect) 
            n_union = np.sum(union) 
             
            n_gt = np.sum(gt_k==1) 
             
            if n_gt != 0: 
                MoF_k = n_intersect/n_gt 
                IoU_k = n_intersect/n_union 
            else: 
                MoF_k,IoU_k = [-1,-1] 
            list_MoF.append(MoF_k) 
            list_IoU.append(IoU_k) 
         
        arr_MoF = np.array(list_MoF) 
        arr_IoU = np.array(list_IoU) 
         
        mask = arr_MoF!=-1 
        MoF = np.mean(arr_MoF[mask]) 
        IoU = np.mean(arr_IoU[mask]) 
    else: 
        print('overall') 
        intersect = np.multiply(Z_pred_perm,Z_gt_perm)     #since these are binary matrix, the elementwise product is only 1 if and only if the prediction and ground-truth values are both 1 
        union = np.clip((Z_pred_perm+Z_gt_perm).astype(np.float),0,1) 
         
        n_intersect = np.sum(intersect) 
        n_union = np.sum(union) 
        n_predict = np.sum(Z_pred_perm)
         
        n_gt = np.sum(Z_gt_perm) 
         
        MoF = n_intersect/n_gt 
        IoU = n_intersect/n_union 
        Precision = n_intersect/n_predict
    return MoF,IoU,Precision

def compute_align_MoF_UoI_bg(keystep_pred,keystep_gt,n_keystep,M,per_keystep=False):         #[1,T] 
#    assert random==False 
    assert keystep_pred.size(0)==keystep_gt.size(0)==1 
     
    keystep_pred = keystep_pred[0].long()                      #[T]<==[1,T]     
    keystep_gt = keystep_gt[0].long()                          #[T]<==[1,T] 
    
    Z_pred = F.one_hot(keystep_pred,M).float().cpu().numpy()                  #[T,M] <== [T] 
    Z_gt = F.one_hot(keystep_gt,n_keystep).float().cpu().numpy()              #[T,K] <== [T] 
     
    Z_gt = Z_gt[:,1:]                                           #discard background info 
     
    ## pure numpy 
    assert Z_pred.shape[0] == Z_gt.shape[0] 
    T = Z_gt.shape[0]*1.0 
     
    Dis = 1.0 - np.matmul(np.transpose(Z_gt),Z_pred)/T            #[K,M] <== [T,K]^T matmul [T,M] 
     
    perm_gt, perm_pred = linear_sum_assignment(Dis) 
    print('perm_gt {} perm_pred {}'.format(perm_gt,perm_pred)) 
     
    Z_pred_perm = Z_pred[:,perm_pred]           #[T,K]
    Z_gt_perm = Z_gt[:,perm_gt]                 #[T,K]
     
    ####### background augmentation #######
    
    pred_bg = 1-np.sum(Z_pred_perm,1)           #[T]
    gt_bg = 1-np.sum(Z_gt_perm,1)               #[T]
    
    Z_pred_perm = np.concatenate([Z_pred_perm,pred_bg[:,np.newaxis]],axis=1)     #[T,K+1] <== [T,K],[T,1]
    Z_gt_perm = np.concatenate([Z_gt_perm,gt_bg[:,np.newaxis]],axis=1)     #[T,K+1] <== [T,K],[T,1]
    
    assert np.min(Z_pred_perm) == 0 and np.max(Z_pred_perm) == 1
    assert np.min(Z_gt_perm) == 0 and np.max(Z_gt_perm) == 1
    
    ####### background augmentation #######
    
    print('alignment cost {}'.format(np.sum(Dis[perm_gt,perm_pred]))) 
    
    if per_keystep: 
        list_MoF = [] 
        list_IoU = [] 
        for idx_k in range(Z_gt_perm.shape[1]):             #[T,K] 
            pred_k =  Z_pred_perm[:,idx_k] 
            gt_k = Z_gt_perm[:,idx_k] 
             
            intersect = np.multiply(pred_k,gt_k)     #since these are binary matrix, the elementwise product is only 1 if and only if the prediction and ground-truth values are both 1 
            union = np.clip((pred_k+gt_k).astype(np.float),0,1)
            
            n_intersect = np.sum(intersect) 
            n_union = np.sum(union) 
             
            n_gt = np.sum(gt_k==1) 
             
            if n_gt != 0: 
                MoF_k = n_intersect/n_gt 
                IoU_k = n_intersect/n_union 
            else: 
                MoF_k,IoU_k = [-1,-1] 
            list_MoF.append(MoF_k) 
            list_IoU.append(IoU_k) 
         
        arr_MoF = np.array(list_MoF) 
        arr_IoU = np.array(list_IoU) 
         
        mask = arr_MoF!=-1 
        MoF = np.mean(arr_MoF[mask]) 
        IoU = np.mean(arr_IoU[mask]) 
    else: 
        print('overall') 
        intersect = np.multiply(Z_pred_perm,Z_gt_perm)     #since these are binary matrix, the elementwise product is only 1 if and only if the prediction and ground-truth values are both 1 
        union = np.clip((Z_pred_perm+Z_gt_perm).astype(np.float),0,1) 
         
        n_intersect = np.sum(intersect) 
        n_union = np.sum(union) 
         
        n_gt = np.sum(Z_gt_perm) 
         
        MoF = n_intersect/n_gt 
        IoU = n_intersect/n_union 
    return MoF,IoU

#%%
def compute_align_MoF_UoI_no_align(keystep_pred,keystep_gt,n_keystep,M,per_keystep=False):         #[1,T] 
#    assert random==False 
    assert keystep_pred.size(0)==keystep_gt.size(0)==1 
    assert n_keystep == M
     
    keystep_pred = keystep_pred[0].long()                      #[T]<==[1,T]     
    keystep_gt = keystep_gt[0].long()                          #[T]<==[1,T] 
    
    
#    print("single class assignment !!!!!\n"*10) 
#    keystep_pred = torch.ones(size=keystep_gt.size()).long()
    
#    if random: 
#        print("Random assignment !!!!!\n"*10) 
#        keystep_pred = torch.randint(low=torch.min(keystep_gt), high=torch.max(keystep_gt)+1, size=keystep_gt.size()) 
#     
    Z_pred = F.one_hot(keystep_pred,M).float().cpu().numpy()                  #[T,M] <== [T] 
    Z_gt = F.one_hot(keystep_gt,n_keystep).float().cpu().numpy()              #[T,K] <== [T] 
    
#   
    Z_pred = Z_pred[:,1:]                                           #discard background info 
    Z_gt = Z_gt[:,1:]                                           #discard background info 
#     
#    ## pure numpy 
#    assert Z_pred.shape[0] == Z_gt.shape[0] 
#    T = Z_gt.shape[0]*1.0 
#     
#    Dis = 1.0 - np.matmul(np.transpose(Z_gt),Z_pred)/T            #[K,M] <== [T,K]^T matmul [T,M] 
#     
#    perm_gt, perm_pred = linear_sum_assignment(Dis) 
#    print('perm_gt {} perm_pred {}'.format(perm_gt,perm_pred)) 
#     
    Z_pred_perm = Z_pred
    Z_gt_perm = Z_gt
#     
#    print('alignment cost {}'.format(np.sum(Dis[perm_gt,perm_pred]))) 
    
    if per_keystep: 
        list_MoF = [] 
        list_IoU = [] 
        for idx_k in range(Z_gt_perm.shape[1]):             #[T,K] 
            pred_k =  Z_pred_perm[:,idx_k] 
            gt_k = Z_gt_perm[:,idx_k] 
             
            intersect = np.multiply(pred_k,gt_k)     #since these are binary matrix, the elementwise product is only 1 if and only if the prediction and ground-truth values are both 1 
            union = np.clip((pred_k+gt_k).astype(np.float),0,1)
            
            n_intersect = np.sum(intersect) 
            n_union = np.sum(union) 
             
            n_gt = np.sum(gt_k==1) 
             
            if n_gt != 0: 
                MoF_k = n_intersect/n_gt 
                IoU_k = n_intersect/n_union 
            else: 
                MoF_k,IoU_k = [-1,-1] 
            list_MoF.append(MoF_k) 
            list_IoU.append(IoU_k) 
         
        arr_MoF = np.array(list_MoF) 
        arr_IoU = np.array(list_IoU) 
         
        mask = arr_MoF!=-1 
        MoF = np.mean(arr_MoF[mask]) 
        IoU = np.mean(arr_IoU[mask]) 
    else: 
        print('overall') 
        intersect = np.multiply(Z_pred_perm,Z_gt_perm)     #since these are binary matrix, the elementwise product is only 1 if and only if the prediction and ground-truth values are both 1 
        union = np.clip((Z_pred_perm+Z_gt_perm).astype(np.float),0,1) 
         
        n_intersect = np.sum(intersect) 
        n_union = np.sum(union) 
        n_predict = np.sum(Z_pred_perm)
         
        n_gt = np.sum(Z_gt_perm) 
         
        MoF = n_intersect/n_gt 
        IoU = n_intersect/n_union 
        Precision = n_intersect/n_predict
    return MoF,IoU,Precision

def compute_align_MoF_UoI_bg_no_align(keystep_pred,keystep_gt,n_keystep,M,per_keystep=False):         #[1,T] 
#    assert random==False 
    assert keystep_pred.size(0)==keystep_gt.size(0)==1 
    assert n_keystep == M
     
    keystep_pred = keystep_pred[0].long()                      #[T]<==[1,T]     
    keystep_gt = keystep_gt[0].long()                          #[T]<==[1,T] 
    
    Z_pred = F.one_hot(keystep_pred,M).float().cpu().numpy()                  #[T,M] <== [T] 
    Z_gt = F.one_hot(keystep_gt,n_keystep).float().cpu().numpy()              #[T,K] <== [T] 
#     
#    Z_pred = Z_pred[:,1:]                                           #discard background info 
#    Z_gt = Z_gt[:,1:] 
#     
#    ## pure numpy 
#    assert Z_pred.shape[0] == Z_gt.shape[0] 
#    T = Z_gt.shape[0]*1.0 
#     
#    Dis = 1.0 - np.matmul(np.transpose(Z_gt),Z_pred)/T            #[K,M] <== [T,K]^T matmul [T,M] 
#     
#    perm_gt, perm_pred = linear_sum_assignment(Dis) 
#    print('perm_gt {} perm_pred {}'.format(perm_gt,perm_pred)) 
     
    Z_pred_perm = Z_pred          #[T,K]
    Z_gt_perm = Z_gt                #[T,K]
     
    ####### background augmentation #######
    
    pred_bg = 1-np.sum(Z_pred_perm,1)           #[T]
    gt_bg = 1-np.sum(Z_gt_perm,1)               #[T]
    
    Z_pred_perm = np.concatenate([Z_pred_perm,pred_bg[:,np.newaxis]],axis=1)     #[T,K+1] <== [T,K],[T,1]
    Z_gt_perm = np.concatenate([Z_gt_perm,gt_bg[:,np.newaxis]],axis=1)     #[T,K+1] <== [T,K],[T,1]
    
    assert np.min(Z_pred_perm) == 0 and np.max(Z_pred_perm) == 1
    assert np.min(Z_gt_perm) == 0 and np.max(Z_gt_perm) == 1
    
    ####### background augmentation #######
    
#    print('alignment cost {}'.format(np.sum(Dis[perm_gt,perm_pred]))) 
    
    if per_keystep: 
        list_MoF = [] 
        list_IoU = [] 
        for idx_k in range(Z_gt_perm.shape[1]):             #[T,K] 
            pred_k =  Z_pred_perm[:,idx_k] 
            gt_k = Z_gt_perm[:,idx_k] 
             
            intersect = np.multiply(pred_k,gt_k)     #since these are binary matrix, the elementwise product is only 1 if and only if the prediction and ground-truth values are both 1 
            union = np.clip((pred_k+gt_k).astype(np.float),0,1)
            
            n_intersect = np.sum(intersect) 
            n_union = np.sum(union) 
             
            n_gt = np.sum(gt_k==1) 
             
            if n_gt != 0: 
                MoF_k = n_intersect/n_gt 
                IoU_k = n_intersect/n_union 
            else: 
                MoF_k,IoU_k = [-1,-1] 
            list_MoF.append(MoF_k) 
            list_IoU.append(IoU_k) 
         
        arr_MoF = np.array(list_MoF) 
        arr_IoU = np.array(list_IoU) 
         
        mask = arr_MoF!=-1 
        MoF = np.mean(arr_MoF[mask]) 
        IoU = np.mean(arr_IoU[mask]) 
    else: 
        print('overall') 
        intersect = np.multiply(Z_pred_perm,Z_gt_perm)     #since these are binary matrix, the elementwise product is only 1 if and only if the prediction and ground-truth values are both 1 
        union = np.clip((Z_pred_perm+Z_gt_perm).astype(np.float),0,1) 
         
        n_intersect = np.sum(intersect) 
        n_union = np.sum(union) 
         
        n_gt = np.sum(Z_gt_perm) 
         
        MoF = n_intersect/n_gt 
        IoU = n_intersect/n_union 
    return MoF,IoU