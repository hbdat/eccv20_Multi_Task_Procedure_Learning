#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:23:25 2019

@author: war-machince
"""

'''
https://github.com/weirme/Video_Summary_using_FCSN
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from collections import OrderedDict 


class FCSN(nn.Module):
    def __init__(self, n_class,n_category,lambda_1,dim_input = 512,verbose = False,temporal_att=True):
        super(FCSN, self).__init__()
        
        self.n_class = n_class
        self.n_category = n_category
        self.verbose = verbose
        self.temporal_att = temporal_att
        
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv1d(dim_input, dim_input, 3, padding=1)),
            ('bn1_1', nn.BatchNorm1d(dim_input)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv1d(dim_input, dim_input, 3, padding=1)),
            ('bn1_2', nn.BatchNorm1d(dim_input)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool1d(2, stride=2, ceil_mode=True))
            ])) # 1/2

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv2_1', nn.Conv1d(dim_input, dim_input, 3, padding=1)),
            ('bn2_1', nn.BatchNorm1d(dim_input)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv1d(dim_input, dim_input, 3, padding=1)),
            ('bn2_2', nn.BatchNorm1d(dim_input)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool1d(2, stride=2, ceil_mode=True))
            ])) # 1/4

        self.conv3 = nn.Sequential(OrderedDict([
            ('conv3_1', nn.Conv1d(dim_input, dim_input, 3, padding=1)),
            ('bn3_1', nn.BatchNorm1d(dim_input)),
            ('relu3_1', nn.ReLU(inplace=True)),
            ('conv3_2', nn.Conv1d(dim_input, dim_input, 3, padding=1)),
            ('bn3_2', nn.BatchNorm1d(dim_input)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('conv3_3', nn.Conv1d(dim_input, dim_input, 3, padding=1)),
            ('bn3_3', nn.BatchNorm1d(dim_input)),
            ('relu3_3', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool1d(2, stride=2, ceil_mode=True))
            ])) # 1/8

        self.conv4 = nn.Sequential(OrderedDict([
            ('conv4_1', nn.Conv1d(dim_input, dim_input*2, 3, padding=1)),
            ('bn4_1', nn.BatchNorm1d(dim_input*2)),
            ('relu4_1', nn.ReLU(inplace=True)),
            ('conv4_2', nn.Conv1d(dim_input*2, dim_input*2, 3, padding=1)),
            ('bn4_2', nn.BatchNorm1d(dim_input*2)),
            ('relu4_2', nn.ReLU(inplace=True)),
            ('conv4_3', nn.Conv1d(dim_input*2, dim_input*2, 3, padding=1)),
            ('bn4_3', nn.BatchNorm1d(dim_input*2)),
            ('relu4_3', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool1d(2, stride=2, ceil_mode=True))
            ])) # 1/16

        self.conv5 = nn.Sequential(OrderedDict([
            ('conv5_1', nn.Conv1d(dim_input*2, dim_input*2, 3, padding=1)),
            ('bn5_1', nn.BatchNorm1d(dim_input*2)),
            ('relu5_1', nn.ReLU(inplace=True)),
            ('conv5_2', nn.Conv1d(dim_input*2, dim_input*2, 3, padding=1)),
            ('bn5_2', nn.BatchNorm1d(dim_input*2)),
            ('relu5_2', nn.ReLU(inplace=True)),
            ('conv5_3', nn.Conv1d(dim_input*2, dim_input*2, 3, padding=1)),
            ('bn5_3', nn.BatchNorm1d(dim_input*2)),
            ('relu5_3', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool1d(2, stride=2, ceil_mode=True))
            ])) # 1/32

        self.conv6 = nn.Sequential(OrderedDict([
            ('fc6', nn.Conv1d(dim_input*2, dim_input*4, 1)),
            ('bn6', nn.BatchNorm1d(dim_input*4)),
            ('relu6', nn.ReLU(inplace=True)),
            ('drop6', nn.Dropout())
            ]))
   
        self.conv7 = nn.Sequential(OrderedDict([
            ('fc7', nn.Conv1d(dim_input*4, dim_input*4, 1)),
            ('bn7', nn.BatchNorm1d(dim_input*4)),
            ('relu7', nn.ReLU(inplace=True)),
            ('drop7', nn.Dropout())
            ]))

        self.conv8 = nn.Sequential(OrderedDict([
            ('fc8', nn.Conv1d(dim_input*4, n_class, 1)),
            ('bn8', nn.BatchNorm1d(n_class)),
            ('relu8', nn.ReLU(inplace=True)),
            ]))
        
        self.conv_pool4 = nn.Conv1d(dim_input*2, n_class, 1)
        self.bn_pool4 = nn.BatchNorm1d(n_class)

        self.deconv1 = nn.ConvTranspose1d(n_class, n_class, 4, padding=1, stride=2, bias=False)
        self.deconv2 = nn.ConvTranspose1d(n_class, n_class, 16, stride=16, bias=False)
        
        #self.func_loss_keystep = nn.BCEWithLogitsLoss()
        
        self.lambda_1 = lambda_1
        
        print('.'*30)
        print('config FCSN')
        print('Linear classifier')
        if self.temporal_att:
            print('Temporal Attention')
        else:
            print('No temporal attention')
        print('.'*30)
        
    def forward(self, x):

        h = x
        h = self.conv1(h)               #bd(t/2)
#        f_conv1 = h
        
#        agg_h = torch.mean(h,dim=2)
#        cats = self.classifier(agg_h)
        
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        pool4 = h

        h = self.conv5(h)
        h = self.conv6(h)
        h = self.conv7(h)
        
        h = self.conv8(h)
        
        
        h = self.deconv1(h)
        upscore2 = h

        h = self.conv_pool4(pool4)
        h = self.bn_pool4(h)
        score_pool4 = h

        h = upscore2 + score_pool4

        keysteps = self.deconv2(h)
        
#        dim_C = 1
        
#        assert keysteps.size(dim_C) == 2
#        if self.temporal_att:
#            keysteps_atten = F.softmax(keysteps,dim = dim_C)[:,1,:] # bt
#        else:
#            keysteps_atten = x.new_ones((x.size(0),x.size(-1)))
#        
#        feature_clss = torch.einsum('bdt,bt->bd',x,keysteps_atten)
#        cats = self.classifier(feature_clss)
        
        return keysteps #bmt,bc            m: is n_keystep and c: is n_catgory 


if __name__ == '__main__':
    net = FCSN()
    data = torch.randn((1, 1024, 320))
    keysteps, cats = net(data)
    print(keysteps.shape,cats.shape)
