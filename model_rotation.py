#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: bixuelei
@Contact: xueleibi@gmail.com
@File: model.py
@Time: 2022/1/15 17:11 PM
"""
import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from attention_util import *
from util import *
from pointnet_util import PointNetSetAbstractionMsg, query_ball_point, index_points,find_goals,find_goals_rectify
from torch.autograd import Variable
# from display import *


def knn(x, k):
    """
    Input:
        points: input points data, [B, N, C]
    Return:
        idx: sample index data, [B, N, K]
    """
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]            # (batch_size, num_points, k)
    return idx



def index_points_neighbors(x, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    batch_size = x.size(0)
    num_points = x.size(1)
    num_dims= x.size(2)

    device=idx.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx=idx+idx_base
    neighbors = x.view(batch_size*num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, -1, num_dims)

    return neighbors



def get_neighbors(x,k=20):
    """
    Input:
        points: input points data, [B, C, N]
    Return:
        feature_points:, indexed points data, [B, 2*C, N, K]
    """
    batch_size = x.size(0)
    num_dims= x.size(1)
    num_points = x.size(2)
    idx = knn(x, k)                                         # batch_size x num_points x 20
    x = x.transpose(2, 1).contiguous()
    neighbors = index_points_neighbors(x, idx)  
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) 
    feature = torch.cat((neighbors-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature



def visialize_cluster(input,indices):
        input=input.permute(0, 2, 1).float()
        bs_,n_point,_=input.shape
        to_display=input
        man_made_label=torch.zeros((bs_,n_point,1)).to(input.device)
        to_display=torch.cat((to_display,man_made_label),dim=-1)        #[bs,n_points,C+1]
        bs,n_superpoints,num_topk=indices.shape
        indices_=indices.view(bs,-1)
        sample_point=index_points(input,indices_)                       #from [bs,n_points,3] to[bs,n_superpoint*num_topk,3]
        sample_point=sample_point.view(bs,n_superpoints,num_topk,3)     #[bs,n_superpoint*num_topk,3]->[bs,n_superpoint,num_topk,3]
        man_made_points=torch.zeros((bs,n_superpoints,num_topk,4)).to(input.device)
        label_n_superpoints=torch.zeros((bs,n_superpoints,num_topk,1)).to(input.device)
        for i in range(n_superpoints):
            label_n_superpoints[:,i,:,:]=i+1
            man_made_points[:,i,:,0:3]=sample_point[:,i,:,0:3]
            man_made_points[:,i,:,3]=label_n_superpoints[:,i,:,0]
        man_made_points=man_made_points.view(bs,-1,4)                     
        for i in range(bs):
            sampled=man_made_points[i,:,:].squeeze(0)
            original=to_display[i,:,:].squeeze(0)
            Visuell_superpoint(sampled,original)            



class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x))) #bs features 2048
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x



class STN3d_Rotate(nn.Module):
    def __init__(self, channel):
        super(STN3d_Rotate, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x))) #bs features 2048
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = F.relu(self.bn6(self.fc3(x)))
        x = self.fc4(x)
        x=x*3.1415927410125732
        rotation_matrix=torch.zeros((batchsize,3,3),dtype=torch.float32).cuda()
        for k in range(x.shape[0]):
            Rx = torch.tensor([[1,0,0],
                        [0,torch.cos(x[k][0]),-torch.sin(x[k][0])],
                        [0,torch.sin(x[k][0]),torch.cos(x[k][0])]])
            Ry = torch.tensor([[torch.cos(x[k][1]),0,torch.sin(x[k][1])],
                        [0,1,0],
                        [-torch.sin(x[k][1]),0,torch.cos(x[k][1])]])
            Rz = torch.tensor([[torch.cos(x[k][2]),-torch.sin(x[k][2]),0],
                        [torch.sin(x[k][2]),torch.cos(x[k][2]),0],
                        [0,0,1]])
            R = torch.matmul(Rz, torch.matmul(Ry,Rx))
            R=R.float().cuda()
            rotation_matrix[k,:,:] = R
        return rotation_matrix



class DGCNN_semseg(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg, self).__init__()
        self.args = args
        self.k = args.k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)                                                             
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn8,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 6, kernel_size=1, bias=False)   #256*6=1536
        #dgcnn_con      1244800

        

    def forward(self, x,input_for_alignment_all_structure):
        batch_size = x.size(0)
        num_points = x.size(2)
        x=x.float() 

        # x=x.permute(0,2,1)                      #(batch_size, 3, num_points)->(batch_size,  num_points,3)
        # Visuell_PointCloud_per_batch(x,target)
        # x=x.permute(0,2,1)

        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        
        return x,None,None,None



class DGCNN_semseg_conv(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg_conv, self).__init__()
        self.args = args
        self.k = args.k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bnmax11 = nn.BatchNorm2d(64)
        self.bnmax12 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bnmax21 = nn.BatchNorm2d(64)
        self.bnmax22 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bnmax31 = nn.BatchNorm2d(64)
        self.bnmax32 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)                                                             
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv_max1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(1,20), bias=False),       #64*64*20=40960*2
                                   self.bnmax11,        #256
                                   nn.LeakyReLU(negative_slope=0.2),        #0
                                   )
        self.conv_max_together1 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),       #64*64*10=40960
                                   self.bnmax12,        #256
                                   nn.LeakyReLU(negative_slope=0.2),        #0
                                   )

        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv_max2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(1,20), bias=False),       #64*64*10=40960
                                   self.bnmax21,        #256
                                   nn.LeakyReLU(negative_slope=0.2),        #0
                                   )
        self.conv_max_together2 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),       #64*64*10=40960
                                   self.bnmax22,        #256
                                   nn.LeakyReLU(negative_slope=0.2),        #0
                                   )
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv_max3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(1,20), bias=False),       #64*64*10=40960
                                   self.bnmax31,        #256
                                   nn.LeakyReLU(negative_slope=0.2),        #0
                                   )
        self.conv_max_together3 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),       #64*64*10=40960
                                   self.bnmax32,        #256
                                   nn.LeakyReLU(negative_slope=0.2),        #0
                                   )
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn8,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 6, kernel_size=1, bias=False)   #256*6=1536
        #dgcnn_con      1244800

        

    def forward(self, x,input_for_alignment_all_structure):
        batch_size = x.size(0)
        num_points = x.size(2)
        x=x.float() 

 
        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x11 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x12=self.conv_max1(x).squeeze(-1)        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)\
        x1 = torch.cat((x11, x12), dim=1)        # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points)
        x1 = self.conv_max_together1(x1)          # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)

        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x21 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x22 = self.conv_max2(x).squeeze(-1)                  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2 = torch.cat((x21, x22), dim=1)        # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points)
        x2 = self.conv_max_together2(x2)          # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)


        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x31 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x32 = self.conv_max3(x).squeeze(-1)       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x3 = torch.cat((x31, x32), dim=1)        # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points)
        x3 = self.conv_max_together3(x3)          # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)


        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        
        return x,None,None,None



class DGCNN_semseg_attention(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg_attention, self).__init__()
        self.args = args
        self.k = args.k
        self.s3n=STN3d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.my_self_attn_inter1=MultiHeadAttention(args.num_heads,64,64,16,16)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.my_self_attn_inter2=MultiHeadAttention(args.num_heads,64,64,16,16)
        self.bn5 = nn.BatchNorm2d(64)
        self.my_self_attn_inter3=MultiHeadAttention(args.num_heads,64,64,16,16)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)                                                             
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn8,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 6, kernel_size=1, bias=False)   #256*6=1536
        #dgcnn_con      1244800

        

    def forward(self, x,input_for_alignment_all_structure):
        batch_size = x.size(0)
        num_points = x.size(2)
        x=x.float()

        trans=self.s3n(x)
        x=x.permute(0,2,1)                      #(batch_size, 3, num_points)->(batch_size,  num_points,3)
        x = torch.bmm(x, trans)
        #Visuell_PointCloud_per_batch(x,target)
        x=x.permute(0,2,1)
 
        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1=x1.permute(0,2,1).contiguous()       #(batch_size, 64, num_points)->(batch_size,num_points,64)
        x1=self.my_self_attn_inter1(x1,x1,x1)
        x1=x1.permute(0,2,1).contiguous()


        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # x2=x2.permute(0,2,1).contiguous()
        # x2=self.my_self_attn_inter2(x2,x2,x2)
        # x2=x2.permute(0,2,1).contiguous()


        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # x3=x3.permute(0,2,1).contiguous()
        # x3=self.my_self_attn_inter3(x3,x3,x3)
        # x3=x3.permute(0,2,1).contiguous()


        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        
        return x,trans,None,None



class DGCNN_semseg_conv_attention(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg_conv_attention, self).__init__()
        self.args = args
        self.k = args.k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.my_self_attn=MultiHeadAttention(args.num_heads,3,3,16,16)
        self.my_self_attn_inter1=MultiHeadAttention(args.num_heads,64,64,16,16)
        self.bnmax11 = nn.BatchNorm2d(64)
        self.bnmax12 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.my_self_attn_inter2=MultiHeadAttention(args.num_heads,64,64,16,16)
        self.bnmax21 = nn.BatchNorm2d(64)
        self.bnmax22 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bnmax31 = nn.BatchNorm2d(64)
        self.bnmax32 = nn.BatchNorm1d(64)
        self.my_self_attn_inter3=MultiHeadAttention(args.num_heads,64,64,16,16)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)                                                             
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv_max1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(1,20), bias=False),       #64*64*10=40960
                                   self.bnmax11,        #256
                                   nn.LeakyReLU(negative_slope=0.2),        #0
                                   )
        self.conv_max_together1 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),       #64*64*10=40960
                                   self.bnmax12,        #256
                                   nn.LeakyReLU(negative_slope=0.2),        #0
                                   )

        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv_max2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(1,20), bias=False),       #64*64*10=40960
                                   self.bnmax21,        #256
                                   nn.LeakyReLU(negative_slope=0.2),        #0
                                   )
        self.conv_max_together2 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),       #64*64*10=40960
                                   self.bnmax22,        #256
                                   nn.LeakyReLU(negative_slope=0.2),        #0
                                   )
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv_max3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(1,20), bias=False),       #64*64*10=40960
                                   self.bnmax31,        #256
                                   nn.LeakyReLU(negative_slope=0.2),        #0
                                   )
        self.conv_max_together3 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),       #64*64*10=40960
                                   self.bnmax32,        #256
                                   nn.LeakyReLU(negative_slope=0.2),        #0
                                   )
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn8,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 6, kernel_size=1, bias=False)   #256*6=1536
        #dgcnn_con      1244800

        

    def forward(self, x,input_for_alignment_all_structure):
        batch_size = x.size(0)
        num_points = x.size(2)
        x=x.float() 

 
        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x11 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x12=self.conv_max1(x).squeeze(-1)        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)\
        x1 = torch.cat((x11, x12), dim=1)        # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points)
        x1 = self.conv_max_together1(x1)          # (batch_size, 64*2, num_points) -> (batch_size, 64, num_points)
        x1=x1.permute(0,2,1).contiguous()
        x1=self.my_self_attn_inter1(x1,x1,x1)
        x1=x1.permute(0,2,1).contiguous()



        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x21 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x22 = self.conv_max2(x).squeeze(-1)                  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2 = torch.cat((x21, x22), dim=1)        # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points)
        x2 = self.conv_max_together2(x2)          # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)



        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x31 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x32 = self.conv_max3(x).squeeze(-1)       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x3 = torch.cat((x31, x32), dim=1)        # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points)
        x3 = self.conv_max_together3(x3)          # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)


        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)


        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        
        return x,None,None,None



class PCT_semseg(nn.Module):
    def __init__(self, args):
        super(PCT_semseg, self).__init__()
        self.args = args
        self.k = args.k
        self.s3n = STN3d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        self.sa1=SA_Layer_Single_Head(128)
        self.sa2=SA_Layer_Single_Head(128)
        self.sa3=SA_Layer_Single_Head(128)
        self.sa4=SA_Layer_Single_Head(128)
        self.bnmax11 = nn.BatchNorm1d(64)
        self.bnmax12 = nn.BatchNorm1d(64)

                                                            
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64*2, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
                                   
        self.conv5 = nn.Conv1d(1024 * 3, 512, 1)
        self.dp5 = nn.Dropout(0.5)
        self.conv6 = nn.Conv1d(512, 256, 1)
        self.conv7 = nn.Conv1d(256, 6, 1)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        

    def forward(self, x,input_for_alignment_all_structure):
        batch_size = x.size(0)
        num_points = x.size(2)
        x=x.float()  

        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x = self.conv3(x)                      # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points)

        x=x.permute(0,2,1)
        x1 = self.sa1(x)                       #(batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)  50MB
        x2 = self.sa2(x1)                      #(batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)
        x3 = self.sa3(x2)                      #(batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)
        x4 = self.sa4(x3)                      #(batch_size, 64*2, num_points)->(batch_size, 64*2, num_points)
        x = torch.cat((x1, x2, x3, x4), dim=-1)      #(batch_size, 64*2, num_points)*4->(batch_size, 512, num_points)
        x=x.permute(0,2,1)
        x = self.conv4(x)                           # (batch_size, 512, num_points)->(batch_size, 1024, num_points) 
        x11 = x.max(dim=-1, keepdim=False)[0]       # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x11=x11.unsqueeze(-1).repeat(1,1,num_points)# (batch_size, 1024)->(batch_size, 1024, num_points)
        x12=torch.mean(x,dim=2,keepdim=False)       # (batch_size, 1024, num_points) -> (batch_size,1024)
        x12=x12.unsqueeze(-1).repeat(1,1,num_points)# (batch_size, 1024)->(batch_size, 1024, num_points)
        x_global = torch.cat((x11, x12), dim=1)     # (batch_size,1024,num_points)+(batch_size, 1024,num_points)-> (batch_size, 2048,num_points)
        x=torch.cat((x,x_global),dim=1)             # (batch_size,2048,num_points)+(batch_size, 1024,num_points) ->(batch_size, 3036,num_points)
        x=self.relu(self.bn5(self.conv5(x)))        # (batch_size, 3036,num_points)-> (batch_size, 512,num_points)
        x=self.dp5(x)                      
        x=self.relu(self.bn6(self.conv6(x)))        # (batch_size, 512,num_points) ->(batch_size,256,num_points)
        x=self.conv7(x)                             # # (batch_size, 256,num_points) ->(batch_size,6,num_points)
        
        return x,None,None,None



class SortNet(nn.Module):
    def __init__(self, num_feat, input_dims, actv_fn=F.relu, feat_dims=256, top_k=16):
        """This function returns a sorted Tensor of Points. The Points are sorted
        according to a 1D score which is generated using rFF's (row-wise Feed Forward).

        Arguments:
            args {args.args} -- args class holding network parameters
            num_feat {int} -- number of features (dims) per point
            device {torch.device} -- Device to run (CPU or GPU)

        Keyword Arguments:
            mode {str} -- Mode to create score (default: {"max"})
        """
        super(SortNet, self).__init__()

        self.num_feat = num_feat
        self.actv_fn = actv_fn
        self.input_dims = input_dims

        self.top_k = top_k
        self.d_model = 512
        self.radius = 0.3
        self.max_radius_points = 64

        self.input_selfattention_layer = nn.TransformerEncoderLayer(
            self.num_feat, nhead=8
        )
        self.input_selfattention = nn.TransformerEncoder(
            self.input_selfattention_layer, num_layers=2
        )

        self.feat_channels = [num_feat,64, 16, 1]
        self.feat_generator = create_conv1d_serials(self.feat_channels)
        self.feat_generator.apply(init_weights)
        self.feat_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.feat_channels[i+1])
                for i in range(len(self.feat_channels)-1)
            ]
        )

        self.radius_ch = [128, 256, self.d_model]
        self.radius_cnn = create_conv3d_serials(self.radius_ch, self.max_radius_points + 1, 3)
        self.radius_cnn.apply(init_weights)
        self.radius_bn = nn.ModuleList(
            [
                nn.BatchNorm3d(num_features=self.radius_ch[i])
                for i in range(len(self.radius_ch))
            ]
        )

        dim_flatten = self.d_model * self.top_k
        self.flatten_linear_ch = [dim_flatten, 1024, self.d_model]
        self.flatten_linear = nn.ModuleList(
            [
                nn.Linear(
                    in_features=self.flatten_linear_ch[i],
                    out_features=self.flatten_linear_ch[i + 1],
                )
                for i in range(len(self.flatten_linear_ch) - 1)
            ]
        )
        self.flatten_linear.apply(init_weights)
        self.flatten_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.flatten_linear_ch[i + 1])
                for i in range(len(self.flatten_linear_ch) - 1)
            ]
        )

    def forward(self, sortvec, input):                                  #[bs,1,features,n_points]   [bs,C,n_points]

        top_k = self.top_k                                              #16
        sortvec_feat = sortvec                                          #[bs,1,features,n_points]
        feat_dim = input.shape[1]

        sortvec_att = sortvec                                           #[bs,features,n_points]
        sortvec_att = sortvec_att.permute(2, 0, 1)                      #[bs,features,n_points]->[n_points,bs,features]
        sortvec_att = self.input_selfattention(sortvec_att)             #[n_points,bs,features]->[n_points,bs,features]
        sortvec_att = sortvec_att.permute(1, 2, 0)                      #[n_points,bs,features]->[bs,features,n_points]

        for i, conv in enumerate(self.feat_generator):                  #[bs,features,n_points]->[bs,1,n_points]
            bn = self.feat_bn[i]
            sortvec = self.actv_fn(bn(conv(sortvec)))

        topk = torch.topk(sortvec, k=top_k, dim=-1)                     #[bs,1,n_points]->[bs,1,cluster]
        indices = topk.indices.squeeze(1)                                #[bs,1,cluster]->[bs,cluster]
        sorted_input = torch.zeros((sortvec_feat.shape[0], feat_dim, top_k)).to(    
            input.device                                                #[bs,C,cluster]
        )

        if top_k == 1:
            indices = indices.unsqueeze(dim=-1)

        sorted_input = index_points(input.permute(0, 2, 1).float(), indices).permute(0, 2, 1)       #[bs,cluster]->[bs,C,cluster]

        all_points = input.permute(0, 2, 1).float()                             #[bs,C,n_points]->[bs,n_points,C]
        query_points = sorted_input.permute(0, 2, 1)                    #[bs,C,cluster]->[bs,cluster,C]

        radius_indices = query_ball_point(                              #idx=[bs,cluster,n_sample]
            self.radius,
            self.max_radius_points,
            all_points[:, :, :3],
            query_points[:, :, :3],
        )

        radius_points = index_points(all_points, radius_indices)        #[bs,cluster,n_sample,C]

        radius_centroids = query_points.unsqueeze(dim=-2)               #[bs,cluster,1,C]
        radius_grouped = torch.cat([radius_centroids, radius_points], dim=-2).unsqueeze(
            dim=1                                                       #[bs,cluster,n_sample,C]+[bs,cluster,1,C]->[bs,cluster,n_sample+1,C]
        )

        for i, radius_conv in enumerate(self.radius_cnn):               #[bs,cluster,n_sample+1,C]->[bs,512,cluster,1,1]
            bn = self.radius_bn[i]
            radius_grouped = self.actv_fn(bn(radius_conv(radius_grouped)))

        radius_grouped = radius_grouped.squeeze(dim=-1).squeeze(dim=-1) #[bs,512,cluster,1,1]->[bs,512,cluster]
        sorted_idx = indices                                            #[bs,cluster]
        sorted_input = radius_grouped                                   #[bs,512,cluster]

        sorted_input = torch.flatten(sorted_input, start_dim=1)         ##[bs,512,cluster]->[bs,512*cluster]

        for i, linear in enumerate(self.flatten_linear):                #[bs,512*cluster]->[bs,512]
            bn = self.flatten_bn[i]
            #sorted_input = self.actv_fn(linear(sorted_input))
            sorted_input = self.actv_fn(bn(linear(sorted_input)))

        sorted_input = sorted_input.unsqueeze(dim=-1)                   #[bs,512]->[bs,512,1]

        return sorted_input, sorted_idx



class PCT_Nico(nn.Module):
    '''
    this is net from nico,which need huge storage of gpu(10 times of SortNet).we need change n_points to 1024,ohterwise
    there is no enough storage for us to straining
    and its results of Sortnet are  also not exclusive
    '''
    def __init__(self, args):
        super(PCT_Nico, self).__init__()

        # Parameters
        self.actv_fn = Mish()

        self.p_dropout = args.dropout
        self.input_dim = 3
        self.num_sort_nets = 10
        self.top_k = 16
        self.d_model = args.hidden_size

        ## Create rFF to project input points to latent feature space
        ## Local Feature Generation --> rFF
        self.sort_ch = [self.input_dim,64, 128, 256]
        self.sort_cnn = create_conv1d_serials(self.sort_ch)
        self.sort_cnn.apply(init_weights)
        self.sort_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.sort_ch[i+1])
                for i in range(len(self.sort_ch)-1)
            ]
        )

        self.sortnets = nn.ModuleList(
            [
                SortNet(
                    self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k
                )
                for _ in range(self.num_sort_nets)
               
            ]
        )

        ## Create rFF to project input points to latent feature space
        ## Global Feature Generation --> rFF
        self.global_ch = [self.input_dim,64, 128, 256]
        self.global_cnn = create_conv1d_serials(self.global_ch)
        self.global_cnn.apply(init_weights)
        self.global_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.global_ch[i+1])
                for i in range(len(self.global_ch)-1)
            ]
        )
        self.global_selfattention = nn.TransformerEncoderLayer(
            self.global_ch[-1], nhead=8
        )

        ## Create set abstraction (MSG)
        ## Global Feature Generation --> Set Abstraction (MSG)
        in_channel = self.global_ch[-1]

        self.sa1 = PointNetSetAbstractionMsg(
            256,
            [0.1, 0.2, 0.4],
            [16, 32, 64],
            in_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            64,
            [0.2, 0.4, 0.6],
            [32, 64, 128],
            320,
            [[32, 64, 128], [64, 64, 128], [64, 128, 253]],
        )

        self.sa1.apply(init_weights)
        self.sa2.apply(init_weights)

        ## Create Local-Global Attention
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=8)
        self.last_layer = PTransformerDecoderLayer(self.d_model, nhead=8, last_dim=self.global_ch[-1])
        self.custom_decoder = PTransformerDecoder(self.decoder_layer, 8, self.last_layer)
        self.transformer_model = nn.Transformer(d_model=self.d_model,nhead=8,num_encoder_layers=4,num_decoder_layers=4,custom_decoder=self.custom_decoder,)
        self.transformer_model.apply(init_weights)

        ## Create Part Segmentation Head
        self.interp_decoder_layer = nn.TransformerDecoderLayer(
            self.global_ch[-1], nhead=8
        )
        self.interp_last_layer = PTransformerDecoderLayer(
            self.global_ch[-1], nhead=8, last_dim=128       ##########################
        )
        self.interp_decoder = PTransformerDecoder(
            self.interp_decoder_layer, 1, self.interp_last_layer
        )

        # Per point classification
        num_classes = 6                                           #########################
        self.conv1 = nn.Conv1d(128, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(self.p_dropout)
        self.conv2 = nn.Conv1d(64, num_classes, 1)

        self.dropout1 = nn.Dropout(p=self.p_dropout)
        self.dropout2 = nn.Dropout(p=self.p_dropout)
        self.dropout3 = nn.Dropout(p=self.p_dropout)

    #def forward(self, input, cls_label):
    def forward(self, input,input_for_alignment_all_structure):

        #############################################
        ## Global Features
        #############################################
        xyz = input.float()
        B, _, _ = xyz.shape
        x_global = input.float()                                            #[bs,C,n_points]
        for i, global_conv in enumerate(self.global_cnn):                   #[bs,C,n_points]->[bs,256,n_points]
            bn = self.global_bn[i]
            x_global = self.actv_fn(bn(global_conv(x_global)))   

        x_global = x_global.squeeze(2).permute(2, 0, 1)                     #[bs,256,1,n_points]->[n_points,bs,256]       
        x_global = self.global_selfattention(x_global)                      #[n_points,bs,256]->[n_points,bs,256]
        x_global = x_global.permute(1, 2, 0)                                #[n_points,bs,256]->[bs,256,n_points]

        l1_xyz, l1_points = self.sa1(xyz, x_global)                         #pointnet++ [bs,3,n_points/4] [2,320,n_point/4]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)                     #pointnet++ [bs,3,n_points/16]  [bs,509,n_points/16]
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        out = torch.cat([l2_xyz, l2_points], dim=1)                         #[bs,512,n_points/16]




        #############################################
        ## Local Features
        #############################################

        x_local = input.float()                                             #[bs,1,Cor,n_points]
        for i, sort_conv in enumerate(self.sort_cnn):                       #[bs,1,Cor,n_points]->[bs,256,n_points]
            bn = self.sort_bn[i]
            x_local = self.actv_fn(bn(sort_conv(x_local)))

        x_local_sorted = torch.cat(
            [sortnet(x_local, input.float())[0] for sortnet in self.sortnets], dim=-1   
                                                                            #input  [bs,1,256,n_points] [bs,6,n_points]    output [bs,512,1*10]
        )



        #############################################
        ## Point Transformer
        #############################################

        source = out.permute(2, 0, 1)                                       #[bs,512,n_points/16]->[64,bs,512]
        target = x_local_sorted.permute(2, 0, 1)                            #[bs,512,10]->[10,bs,512]
        embedding = self.transformer_model(source, target)                  #[64,bs,512]+[10,bs,512]->[10,bs,512]->[10,bs,256]


        #############################################
        ##  Segmentation
        #############################################
        x_interp = x_global.permute(2, 0, 1)                                #bs,256,n_points]->[n_points,bs,256]
        input_feat = self.interp_decoder(x_interp, embedding)               #[n_points,bs,256]+[10,bs,256] -> [n_points,bs,256]->[n_points,bs,128]
        input_feat = input_feat.permute(1, 2, 0)                            #[n_points,bs,128]->[bs,128,n_points]


        # FC layers
        x = self.actv_fn(self.bn1(self.conv1(input_feat)))                  #[bs,128,n_points]->[bs,128,n_points]
        x = self.drop1(x)                                                   
        x = self.conv2(x)                                                   #[bs,128,n_points]->[bs,6,n_points]
        x = F.log_softmax(x, dim=1)
        output = x.permute(0, 2, 1)                                         #[bs,6,n_points]->[bs,n_points,num_class]

        return output,None,None,None



class Superpoint_sample(nn.Module):                                              #ball top1 query sample 
    def __init__(self,args, num_feats, input_dims, actv_fn=F.relu, top_k=16):
        """This function returns a sorted Tensor of Points. The Points are sorted

        """
        super(Superpoint_sample, self).__init__()
        self.num_heads=args.num_heads
        self.num_layers=args.num_layers
        self.num_latent_feats_inencoder=args.self_encoder_latent_features
        self.num_feats = num_feats
        self.actv_fn = actv_fn
        self.input_dims = input_dims

        self.top_k = top_k
        self.d_model = args.emb_dims
        self.radius = 0.3
        self.max_radius_points = 32

        self.self_atn_layer =SA_Layer_Multi_Head(args,256)
        self.selfatn_layers=SA_Layers(self.num_layers,self.self_atn_layer)
 
        self.feat_channels_1d = [self.num_feats,64, 32, 16]
        self.feat_generator = create_conv1d_serials(self.feat_channels_1d)
        self.feat_generator.apply(init_weights)
        self.feat_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.feat_channels_1d[i+1])
                for i in range(len(self.feat_channels_1d)-1)
            ]
        )


        self.feat_channels_3d = [128, 256, self.d_model]
        self.radius_cnn = create_conv3d_serials(self.feat_channels_3d, self.max_radius_points + 1, 3)
        self.radius_cnn.apply(init_weights)
        self.radius_bn = nn.ModuleList(
            [
                nn.BatchNorm3d(num_features=self.feat_channels_3d[i])
                for i in range(len(self.feat_channels_3d))
            ]
        )

    def forward(self, hoch_features, input):                                          #[bs,1,features,n_points]   [bs,C,n_points]

        top_k = self.top_k                                                             #16
        origial_hoch_features = hoch_features                                          #[bs,1,features,n_points]
        feat_dim = input.shape[1]

        hoch_features_att = hoch_features  
        ############################################################
        # official version
        # ######################################################### 
        # hoch_features_att = hoch_features_att.permute(2, 0, 1)                      #[bs,features,n_points]->[n_points,bs,features]
        # hoch_features_att = self.self_atn_official(hoch_features_att)             #[n_points,bs,features]->[n_points,bs,features]
        # hoch_features_att = hoch_features_att.permute(1, 2, 0)                      #[n_points,bs,features]->[bs,features,n_points]
  
  
        #############################################################
        #implemented by myself
        #############################################################
        hoch_features_att=hoch_features_att.permute(0,2,1)
        hoch_features_att=self.selfatn_layers(hoch_features_att)
        hoch_features_att=hoch_features_att.permute(0,2,1)

        #####################
        #repeate 5 times
        #####################
        # for i in range(5):
        #     high_inter=hoch_features_att
        #     for j, conv in enumerate(self.feat_generator):                      #[bs,features,n_points]->[bs,10,n_points]
        #         bn = self.feat_bn[j]
        #         high_inter = self.actv_fn(bn(conv(high_inter)))
        #     topk = torch.topk(high_inter, k=top_k, dim=-1)                      #[bs,10,n_points]->[bs,10,n_superpoint]
        #     if i==0:
        #         input=input.permute(0, 2, 1).float()
        #         indices = topk.indices                                          #[bs,n_superpoints,top_k]->[bs,n_superpoints,top_k]
        # #         ###########################
        # #         #visial the superpoint cluster
        # #         ##########################
        # #         bs_,n_point,_=input.shape
        # #         to_display=input
        # #         man_made_label=torch.zeros((bs_,n_point,1)).to(input.device)
        # #         to_display=torch.cat((to_display,man_made_label),dim=-1)        #[bs,n_points,C+1]
        # #         bs,n_superpoints,num_topk=indices.shape
        # #         indices_=indices.view(bs,-1)
        # #         sample_point=index_points(input,indices_)                       #from [bs,n_points,3] to[bs,n_superpoint*num_topk,3]
        # #         sample_point=sample_point.view(bs,n_superpoints,num_topk,3)     #[bs,n_superpoint*num_topk,3]->[bs,n_superpoint,num_topk,3]
        # #         man_made_points=torch.zeros((bs,n_superpoints,num_topk,4)).to(input.device)
        # #         label_n_superpoints=torch.zeros((bs,n_superpoints,num_topk,1)).to(input.device)
        # #         for i in range(n_superpoints):
        # #             label_n_superpoints[:,i,:,:]=i+1
        # #             man_made_points[:,i,:,0:3]=sample_point[:,i,:,0:3]
        # #             man_made_points[:,i,:,3]=label_n_superpoints[:,i,:,0]
        # #         man_made_points=man_made_points.view(bs,-1,4)                     
        # #         for i in range(bs):
        # #             sampled=man_made_points[i,:,:].squeeze(0)
        # #             original=to_display[i,:,:].squeeze(0)
        # #             Visuell_superpoint(sampled,original)            
        #     else:
        #         indices = torch.cat((indices,topk.indices),dim=-1)
        # indices=indices.cpu()
        # indices=torch.mode(indices,dim=-1)[0]
        # indices=indices.cuda()


        ##########################
        #
        ##########################
        high_inter=hoch_features_att
        for j, conv in enumerate(self.feat_generator):                      #[bs,features,n_points]->[bs,10,n_points]
            bn = self.feat_bn[j]
            high_inter = self.actv_fn(bn(conv(high_inter)))
        topk = torch.topk(high_inter, k=top_k, dim=-1)                      #[bs,10,n_points]->[bs,10,n_superpoint]
        indices = topk.indices                                          #[bs,n_superpoints,top_k]->[bs,n_superpoints,top_k]
        
        #visialize_cluster(input,indices)
        indices=indices[:,:,0]



        sorted_input = torch.zeros((origial_hoch_features.shape[0], feat_dim, top_k)).to(    
            input.device                                                        #[bs,C,n_superpoint]
        )

        if top_k == 1:
            indices = indices.unsqueeze(dim=-1)

        sorted_input = index_points(input.permute(0, 2, 1).float(), indices).permute(0, 2, 1)       #[bs,n_superpoint]->[bs,C,n_superpoint]

        all_points = input.permute(0, 2, 1).float()                              #[bs,C,n_points]->[bs,n_points,C]
        query_points = sorted_input.permute(0, 2, 1)                             #[bs,C,n_superpoint]->[bs,n_superpoint,C]

        radius_indices = query_ball_point(                                       #idx=[bs,n_superpoint,n_sample]
            self.radius,
            self.max_radius_points,
            all_points[:, :, :3],
            query_points[:, :, :3],
        )

        radius_points = index_points(all_points, radius_indices)                    #[bs,n_superpoint,n_sample,C]

        radius_centroids = query_points.unsqueeze(dim=-2)                           #[bs,n_superpoint,1,C]
        radius_grouped = torch.cat([radius_centroids, radius_points], dim=-2).unsqueeze(
            dim=1                                                                   #[bs,n_superpoint,n_sample,C]+[bs,n_superpoint,1,C]->[bs,n_superpoint,n_sample+1,C]
        )

        for i, radius_conv in enumerate(self.radius_cnn):                           #[bs,n_superpoint,n_sample+1,C]->[bs,512,n_superpoint,1,1]
            bn = self.radius_bn[i]
            radius_grouped = self.actv_fn(bn(radius_conv(radius_grouped)))

        radius_grouped = radius_grouped.squeeze(dim=-1).squeeze(dim=-1)             #[bs,512,n_superpoint,1,1]->[bs,512,n_superpoint]                                          #[bs,n_superpoint]
        sorted_input = radius_grouped                                               #[bs,512,n_superpoint]

        return sorted_input,None



class My_Network1(nn.Module):                                                    #ball top1 query sample and transform
    def __init__(self, args):
        super(My_Network1, self).__init__()
        self.actv_fn = nn.LeakyReLU(negative_slope=0.2)
        self.p_dropout = args.dropout
        self.input_dim = 3
        self.top_k = 64
        self.d_model = args.hidden_size
        self.num_classes= 6  

        #########################################################################
        #dynamic graph based network
        ########################################################################
        self.args = args
        self.k = 20
        self.s3n = STN3d(3)
        self.my_self_attn_inter1=MultiHeadAttention(args.num_heads,64,64,16,16)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)                                                             
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, 512, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))        
     




        #############################################################################
        #self desigened superpoint sample net and its net of generation local features 
        #############################################################################
        self.sort_ch = [self.input_dim,64, 128, 256]
        self.sort_cnn = create_conv1d_serials(self.sort_ch)   
        self.sort_cnn.apply(init_weights)
        self.sort_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.sort_ch[i+1])
                for i in range(len(self.sort_ch)-1)
            ]
        )
        self.superpointnet =Superpoint_sample(args,self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k)


        ############################################################## 
        #sortnet proposed by nico in point transformer   
        ##############################################################
        #self.sortnets = nn.ModuleList(
        #     [
        #         SortNet(
        #             self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k
        #         )
        #         for _ in range(10)
               
        #     ]
        # )


        #############################
        #
        #############################
        ## Create Local-Global Attention
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=4)
        self.last_layer = PTransformerDecoderLayer(self.d_model, nhead=4, last_dim=256)
        self.custom_decoder = PTransformerDecoder(self.decoder_layer, 4, self.last_layer)
        self.transformer_model = nn.Transformer(d_model=self.d_model,nhead=4,num_encoder_layers=2,num_decoder_layers=2,custom_decoder=self.custom_decoder,)
        self.transformer_model.apply(init_weights)

        ##########################################################
        #final segmentation layer
        ###################################################### 
        self.bn7 = nn.BatchNorm1d(1024)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),         
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn8 = nn.BatchNorm1d(1024)
        self.conv8 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn8,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn8 = nn.BatchNorm1d(512)
        self.conv8 = nn.Sequential(nn.Conv1d(1728, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn8,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn9 = nn.BatchNorm1d(256)
        self.conv9 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn9,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Conv1d(256, self.num_classes, kernel_size=1, bias=False)   #256*6=1536

    def forward(self, x,input_for_alignment_all_structure):
        x=x.float() 

        trans=self.s3n(x)
        x=x.permute(0,2,1)                      #(batch_size, 3, num_points)->(batch_size,  num_points,3)
        x = torch.bmm(x, trans)
        #Visuell_PointCloud_per_batch(x,target)
        x=x.permute(0,2,1)

        x_local=x
        input=x
        #############################################
        ## Global Features
        #############################################
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1=x1.permute(0,2,1).contiguous()       #(batch_size, 64, num_points)->(batch_size,num_points,64)
        x1=self.my_self_attn_inter1(x1,x1,x1)
        x1=x1.permute(0,2,1).contiguous()


        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_remain=x
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, 64)          # (batch_size, 1024, num_points)
        #x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        #############################################
        ## Local Features
        #############################################

        x_local = x_local.float()                                             #[bs,1,Cor,n_points]
        for i, sort_conv in enumerate(self.sort_cnn):                       #[bs,1,Cor,n_points]->[bs,256,n_points]
            bn = self.sort_bn[i]
            x_local = self.actv_fn(bn(sort_conv(x_local)))

        x_local_sorted=self.superpointnet(x_local,input)[0]              #[bs,256,n_points]->[bs,512,32]
        # x_local_sorted = torch.cat(
        #     [sortnet(x_local, input.float())[0] for sortnet in self.sortnets], dim=-1   
        #                                                                      #[bs,256,n_points]->[bs,512,10]
        # )


        #############################################
        ## Point Transformer
        #############################################
        source = x.permute(2, 0, 1)                                        # [bs,1024,64]->[64,bs,1024]
        target = x_local_sorted.permute(2, 0, 1)                           # [bs,1024,10]->[10,bs,1024]
        embedding = self.transformer_model(source, target)                 # [64,bs,1024]+[32,bs,1024]->[32,bs,1024]



        ################################################
        ##segmentation
        ################################################
        embedding=embedding.permute(1,2,0)                                  # [32,bs,512]->[bs,512,32]
        x=self.conv7(embedding)                                             # [bs,512,32]->[bs,1024,32]
        x = x.max(dim=-1, keepdim=True)[0]                                  # (batch_size, emb_dims, 10) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)                                      # (batch_size, 1024, n_points)
        x = torch.cat((x,x_remain,x1,x2,x3), dim=1)                         # (batch_size, 1024*2, num_points)
        x = self.conv8(x)                                                   # (batch_size, 1024*2, num_points) -> (batch_size, 512, num_points)
        x = self.conv9(x)                                                   # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv10(x)                                                   # (batch_size, 256, num_points) -> (batch_size, 6, num_points)
        return x,trans,None,None



class My_Network1_r(nn.Module):                                                    #ball top1 query sample and transform
    def __init__(self, args):
        super(My_Network1_r, self).__init__()
        self.actv_fn = nn.LeakyReLU(negative_slope=0.2)
        self.p_dropout = args.dropout
        self.input_dim = 3
        self.top_k = 64
        self.d_model = args.emb_dims
        self.num_classes= 6  

        #########################################################################
        #dynamic graph based network
        ########################################################################
        self.args = args
        self.k = 20
        self.s3n = STN3d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.my_self_attn_inter1=MultiHeadAttention(args.num_heads,64,64,16,16)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bnsample = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)                                                             
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))    
        self.conv_sample = nn.Sequential(nn.Conv1d(192, 256, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bnsample,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))    
     




        #############################################################################
        #self desigened superpoint sample net and its net of generation local features 
        #############################################################################
        self.sort_ch = [self.input_dim,64, 128, 256]
        self.sort_cnn = create_conv1d_serials(self.sort_ch)   
        self.sort_cnn.apply(init_weights)
        self.sort_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.sort_ch[i+1])
                for i in range(len(self.sort_ch)-1)
            ]
        )
        self.superpointnet =Superpoint_sample(args,self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k)


        ############################################################## 
        #sortnet proposed by nico in point transformer   
        ##############################################################
        #self.sortnets = nn.ModuleList(
        #     [
        #         SortNet(
        #             self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k
        #         )
        #         for _ in range(10)
               
        #     ]
        # )


        #############################
        #
        #############################
        ## Create Local-Global Attention
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=4)
        self.last_layer = PTransformerDecoderLayer(self.d_model, nhead=4, last_dim=256)
        self.custom_decoder = PTransformerDecoder(self.decoder_layer, 4, self.last_layer)
        self.transformer_model = nn.Transformer(d_model=self.d_model,nhead=4,num_encoder_layers=2,num_decoder_layers=2,custom_decoder=self.custom_decoder,)
        self.transformer_model.apply(init_weights)

        ##########################################################
        #final segmentation layer
        ###################################################### 
        self.bn7 = nn.BatchNorm1d(1024)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),         
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn8 = nn.BatchNorm1d(512)
        self.conv8 = nn.Sequential(nn.Conv1d(2240, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn8,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn9 = nn.BatchNorm1d(256)
        self.conv9 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn9,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Conv1d(256, self.num_classes, kernel_size=1, bias=False)   #256*6=1536

    def forward(self, x,input_for_alignment_all_structure):
        x=x.float()
        trans=self.s3n(x)
        x=x.permute(0,2,1)                      #(batch_size, 3, num_points)->(batch_size,  num_points,3)
        x = torch.bmm(x, trans)
        #Visuell_PointCloud_per_batch(x,target)
        x=x.permute(0,2,1)
        input=x
        #############################################
        ## Global Features
        #############################################
        num_points = x.size(2) 
        

        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1=x1.permute(0,2,1).contiguous()       #(batch_size, 64, num_points)->(batch_size,num_points,64)
        x1=self.my_self_attn_inter1(x1,x1,x1)
        x1=x1.permute(0,2,1).contiguous()

        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x_local=self.conv_sample(x)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_ = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x_remain=x_.repeat(1, 1, self.args.npoints)
        x_source=torch.topk(x, k=32, dim=-1)[0]
        #x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)


        x_local_sorted=self.superpointnet(x_local,input)[0]              #[bs,256,n_points]->[bs,512,32]
        # x_local_sorted = torch.cat(
        #     [sortnet(x_local, input.float())[0] for sortnet in self.sortnets], dim=-1   
        #                                                                      #[bs,256,n_points]->[bs,512,10]
        # )


        #############################################
        ## Point Transformer
        #############################################
        source = x_source.permute(2, 0, 1)                                        # [bs,1024,64]->[64,bs,1024]
        target = x_local_sorted.permute(2, 0, 1)                           # [bs,1024,10]->[10,bs,1024]
        embedding = self.transformer_model(source, target)                 # [64,bs,1024]+[32,bs,1024]->[32,bs,1024]



        ################################################
        ##segmentation
        ################################################
        embedding=embedding.permute(1,2,0)                                  # [32,bs,512]->[bs,512,32]
        x=self.conv7(embedding)                                             # [bs,512,32]->[bs,1024,32]
        x = x.max(dim=-1, keepdim=True)[0]                                  # (batch_size, emb_dims, 10) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)                                      # (batch_size, 1024, n_points)
        x = torch.cat((x,x_remain,x1,x2,x3), dim=1)                         # (batch_size, 1024*2, num_points)
        x = self.conv8(x)                                                   # (batch_size, 1024*2, num_points) -> (batch_size, 512, num_points)
        x = self.conv9(x)                                                   # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv10(x)                                                   # (batch_size, 256, num_points) -> (batch_size, 6, num_points)
        return x,trans,None,None



class My_Network1_r_1728(nn.Module):                                                    #ball top1 query sample and transform
    def __init__(self, args):
        super(My_Network1_r_1728, self).__init__()
        self.actv_fn = nn.LeakyReLU(negative_slope=0.2)
        self.p_dropout = args.dropout
        self.input_dim = 3
        self.top_k = 64
        self.d_model = args.emb_dims
        self.num_classes= 6  

        #########################################################################
        #dynamic graph based network
        ########################################################################
        self.args = args
        self.k = 20
        self.s3n = STN3d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.my_self_attn_inter1=MultiHeadAttention(args.num_heads,64,64,16,16)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bnsample = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)                                                             
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))    
        self.conv_sample = nn.Sequential(nn.Conv1d(192, 256, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bnsample,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))    
     




        #############################################################################
        #self desigened superpoint sample net and its net of generation local features 
        #############################################################################
        self.sort_ch = [self.input_dim,64, 128, 256]
        self.sort_cnn = create_conv1d_serials(self.sort_ch)   
        self.sort_cnn.apply(init_weights)
        self.sort_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.sort_ch[i+1])
                for i in range(len(self.sort_ch)-1)
            ]
        )
        self.superpointnet =Superpoint_sample(args,self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k)


        ############################################################## 
        #sortnet proposed by nico in point transformer   
        ##############################################################
        #self.sortnets = nn.ModuleList(
        #     [
        #         SortNet(
        #             self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k
        #         )
        #         for _ in range(10)
               
        #     ]
        # )


        #############################
        #
        #############################
        ## Create Local-Global Attention
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=4)
        self.last_layer = PTransformerDecoderLayer(self.d_model, nhead=4, last_dim=256)
        self.custom_decoder = PTransformerDecoder(self.decoder_layer, 4, self.last_layer)
        self.transformer_model = nn.Transformer(d_model=self.d_model,nhead=4,num_encoder_layers=2,num_decoder_layers=2,custom_decoder=self.custom_decoder,)
        self.transformer_model.apply(init_weights)

        ##########################################################
        #final segmentation layer
        ###################################################### 
        self.bn7 = nn.BatchNorm1d(256)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),         
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn8 = nn.BatchNorm1d(512)
        self.conv8 = nn.Sequential(nn.Conv1d(1472, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn8,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn9 = nn.BatchNorm1d(256)
        self.conv9 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn9,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Conv1d(256, self.num_classes, kernel_size=1, bias=False)   #256*6=1536

    def forward(self, x,input_for_alignment_all_structure):
        x=x.float()
        trans=self.s3n(x)
        x=x.permute(0,2,1)                      #(batch_size, 3, num_points)->(batch_size,  num_points,3)
        x = torch.bmm(x, trans)
        #Visuell_PointCloud_per_batch(x,target)
        x=x.permute(0,2,1)
        input=x
        #############################################
        ## Global Features
        #############################################
        batch_size = x.size(0)
        num_points = x.size(2)
        x=x.float() 
        

        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1=x1.permute(0,2,1).contiguous()       #(batch_size, 64, num_points)->(batch_size,num_points,64)
        x1=self.my_self_attn_inter1(x1,x1,x1)
        x1=x1.permute(0,2,1).contiguous()

        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x_local=self.conv_sample(x)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_ = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x_remain=x_.repeat(1, 1, self.args.npoints)
        x_source=torch.topk(x, k=32, dim=-1)[0]
        #x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)


        x_local_sorted=self.superpointnet(x_local,input)[0]              #[bs,256,n_points]->[bs,512,32]
        # x_local_sorted = torch.cat(
        #     [sortnet(x_local, input.float())[0] for sortnet in self.sortnets], dim=-1   
        #                                                                      #[bs,256,n_points]->[bs,512,10]
        # )


        #############################################
        ## Point Transformer
        #############################################
        source = x_source.permute(2, 0, 1)                                        # [bs,1024,64]->[64,bs,1024]
        target = x_local_sorted.permute(2, 0, 1)                           # [bs,1024,10]->[10,bs,1024]
        embedding = self.transformer_model(source, target)                 # [64,bs,1024]+[32,bs,1024]->[32,bs,1024]



        ################################################
        ##segmentation
        ################################################
        embedding=embedding.permute(1,2,0)                                  # [32,bs,512]->[bs,512,32]
        x=self.conv7(embedding)                                             # [bs,512,32]->[bs,1024,32]
        x = x.max(dim=-1, keepdim=True)[0]                                  # (batch_size, emb_dims, 10) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)                                      # (batch_size, 1024, n_points)
        x = torch.cat((x,x_remain,x1,x2,x3), dim=1)                         # (batch_size, 1024*2, num_points)
        x = self.conv8(x)                                                   # (batch_size, 1024*2, num_points) -> (batch_size, 512, num_points)
        x = self.conv9(x)                                                   # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv10(x)                                                   # (batch_size, 256, num_points) -> (batch_size, 6, num_points)
        return x,trans,None,None



class Superpoint_cluster(nn.Module):                                             #cluster 
    def __init__(self,args, num_feats, input_dims, actv_fn=F.relu, top_k=16):
        """This function returns a sorted Tensor of Points. The Points are sorted

        """
        super(Superpoint_cluster, self).__init__()
        self.num_heads=args.num_heads
        self.num_layers=args.num_layers
        self.num_latent_feats_inencoder=args.self_encoder_latent_features
        self.num_feats = num_feats
        self.actv_fn = actv_fn
        self.input_dims = input_dims

        self.top_k = top_k
        self.d_model = 512

        self.self_atn_layer =SA_Layer_Multi_Head(args,256)
        self.selfatn_layers=SA_Layers(self.num_layers,self.self_atn_layer)
 
        self.feat_channels_1d = [self.num_feats,64, 32, 16]
        self.feat_generator = create_conv1d_serials(self.feat_channels_1d)
        self.feat_generator.apply(init_weights)
        self.feat_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.feat_channels_1d[i+1])
                for i in range(len(self.feat_channels_1d)-1)
            ]
        )


        self.feat_channels_3d = [128, 256, self.d_model]
        self.radius_cnn = create_conv3d_serials(self.feat_channels_3d, 32, 3)
        self.radius_cnn.apply(init_weights)
        self.radius_bn = nn.ModuleList(
            [
                nn.BatchNorm3d(num_features=self.feat_channels_3d[i])
                for i in range(len(self.feat_channels_3d))
            ]
        )

    def forward(self, high_features, input):                                          #[bs,1,features,n_points]   [bs,C,n_points]

        top_k = self.top_k                                                             #16  
        #############################################################
        #implemented by myself
        #############################################################
        high_features=high_features.permute(0,2,1)
        high_features=self.selfatn_layers(high_features)
        high_features=high_features.permute(0,2,1)

        ##########################
        #
        ##########################
        for j, conv in enumerate(self.feat_generator):                      #[bs,features,n_points]->[bs,16,n_points]
            bn = self.feat_bn[j]
            high_features = self.actv_fn(bn(conv(high_features)))
        topk = torch.topk(high_features, k=top_k, dim=-1)                      #[bs,10,n_points]->[bs,10,n_superpoint]
        indices = topk.indices                                          #[bs,n_superpoints,top_k]->[bs,n_superpoints,top_k]
        #####################
        #visial the superpoints cluster
        ######################
        #visialize_cluster(input,indices)

        sorted_input = index_points(input.permute(0, 2, 1).float(), indices).unsqueeze(dim=1)       #[bs,n_superpoint]->[bs,C,n_superpoint]


        for i, radius_conv in enumerate(self.radius_cnn):                           #[bs,n_superpoint,n_sample+1,C]->[bs,512,n_superpoint,1,1]
            bn = self.radius_bn[i]
            sorted_input = self.actv_fn(bn(radius_conv(sorted_input)))

        sorted_input = sorted_input.squeeze(dim=-1).squeeze(dim=-1)             #[bs,512,n_superpoint,1,1]->[bs,512,n_superpoint]                                          #[bs,n_superpoint]

        return sorted_input,None



class My_Network2(nn.Module):                                                   #topk  and transform
    def __init__(self, args):
        super(My_Network2, self).__init__()
        self.actv_fn = nn.LeakyReLU(negative_slope=0.2)
        self.p_dropout = args.dropout
        self.input_dim = 3
        self.top_k = 32
        self.d_model = args.hidden_size
        self.num_classes= 6  

        #########################################################################
        #dynamic graph based network
        ########################################################################
        self.args = args
        self.k = 20
        self.s3n = STN3d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)                                                             
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, 512, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))        
     




        #############################################################################
        #self desigened superpoint sample net and its net of generation local features 
        #############################################################################
        self.sort_ch = [self.input_dim,64, 128, 256]
        self.sort_cnn = create_conv1d_serials(self.sort_ch)   
        self.sort_cnn.apply(init_weights)
        self.sort_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.sort_ch[i+1])
                for i in range(len(self.sort_ch)-1)
            ]
        )
        self.superpointnet =Superpoint_cluster(args,self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k)



        #############################
        #
        #############################
        ## Create Local-Global Attention
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=4)
        self.last_layer = PTransformerDecoderLayer(self.d_model, nhead=4, last_dim=256)
        self.custom_decoder = PTransformerDecoder(self.decoder_layer, 4, self.last_layer)
        self.transformer_model = nn.Transformer(d_model=self.d_model,nhead=4,num_encoder_layers=2,num_decoder_layers=2,custom_decoder=self.custom_decoder,)
        self.transformer_model.apply(init_weights)

        ##########################################################
        #final segmentation layer
        ###################################################### 
        self.bn7 = nn.BatchNorm1d(1024)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),         
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn8 = nn.BatchNorm1d(512)
        self.conv8 = nn.Sequential(nn.Conv1d(1728, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn8,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn9 = nn.BatchNorm1d(256)
        self.conv9 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn9,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Conv1d(256, self.num_classes, kernel_size=1, bias=False)   #256*6=1536

    def forward(self, x,input_for_alignment_all_structure):
        x=x.float()
        x_local=x
        input=x
        #############################################
        ## Global Features
        #############################################
        batch_size = x.size(0)
        num_points = x.size(2)
 
        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_remain=x
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, 64)          # (batch_size, 1024, num_points)

        #############################################
        ## Local Features
        #############################################

        x_local = x_local.float()                                             #[bs,1,Cor,n_points]
        for i, sort_conv in enumerate(self.sort_cnn):                       #[bs,1,Cor,n_points]->[bs,256,n_points]
            bn = self.sort_bn[i]
            x_local = self.actv_fn(bn(sort_conv(x_local)))

        x_local_sorted=self.superpointnet(x_local,input)[0]                   #[bs,256,n_points]->[bs,512,16]


        #############################################
        ## Point Transformer
        #############################################
        source = x.permute(2, 0, 1)                                        # [bs,1024,64]->[64,bs,1024]
        target = x_local_sorted.permute(2, 0, 1)                           # [bs,1024,10]->[10,bs,1024]
        embedding = self.transformer_model(source, target)                 # [64,bs,1024]+[32,bs,1024]->[32,bs,1024]



        ################################################
        ##segmentation
        ################################################
        embedding=embedding.permute(1,2,0)                                  # [32,bs,512]->[bs,512,32]
        x=self.conv7(embedding)                                             # [bs,512,32]->[bs,1024,32]
        x = x.max(dim=-1, keepdim=True)[0]                                  # (batch_size, emb_dims, 10) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)                                      # (batch_size, 1024, n_points)
        x = torch.cat((x,x_remain,x1,x2,x3), dim=1)                         # (batch_size, 1024*2, num_points)
        x = self.conv8(x)                                                   # (batch_size, 1024*2, num_points) -> (batch_size, 512, num_points)
        x = self.conv9(x)                                                   # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv10(x)                                                   # (batch_size, 256, num_points) -> (batch_size, 6, num_points)
        return x,None,None,None



class My_Network3(nn.Module):                                                   # topk and contatenate
    def __init__(self, args):
        super(My_Network3, self).__init__()
        self.actv_fn = nn.LeakyReLU(negative_slope=0.2)
        self.p_dropout = args.dropout
        self.input_dim = 3
        self.top_k = 32
        self.d_model = args.hidden_size
        self.num_classes= 6  

        #########################################################################
        #dynamic graph based network
        ########################################################################
        self.args = args
        self.k = 20
        self.s3n = STN3d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)                                                             
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))        
     




        #############################################################################
        #self desigened superpoint sample net and its net of generation local features 
        #############################################################################
        self.sort_ch = [self.input_dim,64, 128, 256]
        self.sort_cnn = create_conv1d_serials(self.sort_ch)   
        self.sort_cnn.apply(init_weights)
        self.sort_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.sort_ch[i+1])
                for i in range(len(self.sort_ch)-1)
            ]
        )
        self.superpointnet =Superpoint_cluster(args,self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k)


        ############################################################## 
        #sortnet proposed by nico in point transformer   
        ##############################################################
        #self.sortnets = nn.ModuleList(
        #     [
        #         SortNet(
        #             self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k
        #         )
        #         for _ in range(10)
               
        #     ]
        # )


        ##########################################################
        #final segmentation layer
        ###################################################### 
        self.bn7 = nn.BatchNorm1d(512)
        self.conv7 = nn.Sequential(nn.Conv1d(1728, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn8 = nn.BatchNorm1d(256)
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn8,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, self.num_classes, kernel_size=1, bias=False)   #256*6=1536

    def forward(self, x,input_for_alignment_all_structure):
        x_local=x
        input=x.float()
        #############################################
        ## Global Features
        #############################################
        batch_size = x.size(0)
        num_points = x.size(2)
        x=x.float() 
        
        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_remain=x
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, 64)          # (batch_size, 1024, num_points)
        #x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        #############################################
        ## Local Features
        #############################################

        x_local = x_local.float()                                             #[bs,1,Cor,n_points]
        for i, sort_conv in enumerate(self.sort_cnn):                       #[bs,1,Cor,n_points]->[bs,256,n_points]
            bn = self.sort_bn[i]
            x_local = self.actv_fn(bn(sort_conv(x_local)))

        x_local_sorted=self.superpointnet(x_local,input)[0]              #[bs,256,n_points]->[bs,512,16]




        ################################################
        ##segmentation
        ################################################
        x = x_local_sorted.max(dim=-1, keepdim=True)[0]                                  # (batch_size, emb_dims, 10) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)                                      # (batch_size, 1024, n_points)
        x = torch.cat((x,x_remain,x1,x2,x3), dim=1)                         # (batch_size, 1024*2, num_points)
        x = self.conv7(x)                                                   # (batch_size, 1024*2, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                                                   # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                                                   # (batch_size, 256, num_points) -> (batch_size, 6, num_points)
        return x,None,None,None



class My_Network3_r(nn.Module):                                                   # topk and contatenate
    def __init__(self, args):
        super(My_Network3_r, self).__init__()
        self.actv_fn = nn.LeakyReLU(negative_slope=0.2)
        self.p_dropout = args.dropout
        self.input_dim = 3
        self.top_k = 32
        self.d_model = args.hidden_size
        self.num_classes= 6  

        #########################################################################
        #dynamic graph based network
        ########################################################################
        self.args = args
        self.k = 20
        self.s3n = STN3d(3)
        self.my_self_attn_inter1=MultiHeadAttention(args.num_heads,64,64,16,16)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bnsample = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)                                                             
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv_sample = nn.Sequential(nn.Conv1d(192, 256, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bnsample,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))    
          




        #############################################################################
        #self desigened superpoint sample net and its net of generation local features 
        #############################################################################
        self.superpointnet =Superpoint_cluster(args,256, self.input_dim, self.actv_fn, top_k=self.top_k)



        ##########################################################
        #final segmentation layer
        ###################################################### 
        self.bn7 = nn.BatchNorm1d(512)
        self.conv7 = nn.Sequential(nn.Conv1d(1728, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn8 = nn.BatchNorm1d(256)
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn8,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, self.num_classes, kernel_size=1, bias=False)   #256*6=1536

    def forward(self, x,input_for_alignment_all_structure):
        x=x.float()
        trans=self.s3n(x)
        x=x.permute(0,2,1)                      #(batch_size, 3, num_points)->(batch_size,  num_points,3)
        x = torch.bmm(x, trans)
        #Visuell_PointCloud_per_batch(x,target)
        x=x.permute(0,2,1)
        input=x
        #############################################
        ## Global Features
        #############################################
        num_points = x.size(2) 
        
        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1=x1.permute(0,2,1).contiguous()       #(batch_size, 64, num_points)->(batch_size,num_points,64)
        x1=self.my_self_attn_inter1(x1,x1,x1)
        x1=x1.permute(0,2,1).contiguous()

        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x_local=self.conv_sample(x)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x_remain=x.repeat(1, 1, num_points)

        #x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        #############################################
        ## Local Features
        #############################################
        x_local_sorted=self.superpointnet(x_local,input)[0]              #[bs,256,n_points]->[bs,512,16]




        ################################################
        ##segmentation
        ################################################
        x = x_local_sorted.max(dim=-1, keepdim=True)[0]                                  # (batch_size, emb_dims, 10) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)                                      # (batch_size, 1024, n_points)
        x = torch.cat((x,x_remain,x1,x2,x3), dim=1)                         # (batch_size, 1024*2, num_points)
        x = self.conv7(x)                                                   # (batch_size, 1024*2, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                                                   # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                                                   # (batch_size, 256, num_points) -> (batch_size, 6, num_points)
        return x,trans,None,None



class PCT_BI(nn.Module):                                                    
    '''
    the same structure with My_Network3, but here i replace the dgcnn(for global features extraction)
    with point net++ structure, but its performance is too low,
    we should give up point net++ method anyway
    so the part of My_Network above are all based on the structure of dgcnn
    '''
    def __init__(self, args):
        super(PCT_BI, self).__init__()

        # Parameters
        self.actv_fn = Mish()

        self.p_dropout = args.dropout
        self.input_dim = 3
        self.num_sort_nets = 10
        self.top_k = 16
        self.d_model = args.hidden_size

        ## Create rFF to project input points to latent feature space
        ## Local Feature Generation --> rFF
        self.sort_ch = [self.input_dim,64, 128, 256]
        self.sort_cnn = create_conv1d_serials(self.sort_ch)
        self.sort_cnn.apply(init_weights)
        self.sort_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.sort_ch[i+1])
                for i in range(len(self.sort_ch)-1)
            ]
        )

        self.superpointnet =Superpoint_sample(args,self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k)
        self.sortnets = nn.ModuleList(
            [
                SortNet(
                    self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k
                )
                for _ in range(self.num_sort_nets)
               
            ]
        )
        ## Global Feature Generation --> rFF
        self.global_ch = [self.input_dim,64, 128, 256]
        self.global_cnn = create_conv1d_serials(self.global_ch)
        self.global_cnn.apply(init_weights)
        self.global_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.global_ch[i+1])
                for i in range(len(self.global_ch)-1)
            ]
        )
        self.global_selfattention = nn.TransformerEncoderLayer(
            self.global_ch[-1], nhead=8
        )

        ## Create set abstraction (MSG)
        ## Global Feature Generation --> Set Abstraction (MSG)
        in_channel = self.global_ch[-1]

        self.sa1 = PointNetSetAbstractionMsg(
            256,
            [0.1, 0.2, 0.4],
            [16, 32, 64],
            in_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            64,
            [0.2, 0.4, 0.6],
            [32, 64, 128],
            320,
            [[32, 64, 128], [64, 64, 128], [64, 128, 253]],
        )

        self.sa1.apply(init_weights)
        self.sa2.apply(init_weights)

        ## Create Local-Global Attention
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=4)
        self.last_layer = PTransformerDecoderLayer(self.d_model, nhead=4, last_dim=self.global_ch[-1])
        self.custom_decoder = PTransformerDecoder(self.decoder_layer, 4, self.last_layer)
        self.transformer_model = nn.Transformer(d_model=self.d_model,nhead=4,num_encoder_layers=2,num_decoder_layers=2,custom_decoder=self.custom_decoder,)
        self.transformer_model.apply(init_weights)

        # Per point classification
        num_classes = 6                                           #########################
        self.bn1 = nn.BatchNorm1d(1024)
        self.conv1 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),         
                                   self.bn1,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn2 = nn.BatchNorm1d(1024)
        self.conv2 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn2,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn3 = nn.BatchNorm1d(512)
        self.conv3 = nn.Sequential(nn.Conv1d(2048, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn3,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn4 = nn.BatchNorm1d(256)
        self.conv4 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn4,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv5 = nn.Conv1d(256, num_classes, kernel_size=1, bias=False)   #256*6=1536
        #dgcnn_con      1244800


    def forward(self, input,for_the_same_with_all_input_frame):

        #############################################
        ## Global Features
        #############################################
        xyz = input.float()
        B, _, N_POINTS = xyz.shape
        x_global = input.float()                                            #[bs,C,n_points]
        for i, global_conv in enumerate(self.global_cnn):                   #[bs,C,n_points]->[bs,256,n_points]
            bn = self.global_bn[i]
            x_global = self.actv_fn(bn(global_conv(x_global)))   

        x_global = x_global.squeeze(2).permute(2, 0, 1)                     #[bs,256,1,n_points]->[n_points,bs,256]       
        x_global = self.global_selfattention(x_global)                      #[n_points,bs,256]->[n_points,bs,256]
        x_global = x_global.permute(1, 2, 0)                                #[n_points,bs,256]->[bs,256,n_points]

        l1_xyz, l1_points = self.sa1(xyz, x_global)                         #pointnet++ [bs,3,n_points/4] [2,320,n_point/4]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)                     #pointnet++ [bs,3,n_points/16]  [bs,509,n_points/16]
        out = torch.cat([l2_xyz, l2_points], dim=1)                         #[bs,512,n_points/16]




        #############################################
        ## Local Features
        #############################################

        x_local = input.float()                                             #[bs,1,Cor,n_points]
        for i, sort_conv in enumerate(self.sort_cnn):                       #[bs,1,Cor,n_points]->[bs,256,n_points]
            bn = self.sort_bn[i]
            x_local = self.actv_fn(bn(sort_conv(x_local)))


        x_local_sorted=self.superpointnet(x_local,input.float())[0]              #[bs,256,n_points]->[bs,512,16]
        # x_local_sorted = torch.cat(
        #     [sortnet(x_local, input.float())[0] for sortnet in self.sortnets], dim=-1   
        #                                                                      #[bs,256,n_points]->[bs,512,16]
        # )


        #############################################
        ## Point Transformer
        #############################################

        source = out.permute(2, 0, 1)                                       #[bs,512,n_points/16]->[64,bs,512]
        target = x_local_sorted.permute(2, 0, 1)                            #[bs,512,16]->[16,bs,512]
        embedding = self.transformer_model(source, target)                  #[64,bs,512]+[16,bs,512]->[16,bs,512]->[16,bs,256]



        ################################################
        ##segmentation
        ################################################
        embedding=embedding.permute(1,2,0)                                  #[10,bs,512]->[bs,512,10]
        x=self.conv1(embedding)                                             #[bs,512,10]->[bs,1024,10]
        x = x.max(dim=-1, keepdim=True)[0]                                  # (batch_size, emb_dims, 10) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, N_POINTS)                                        # (batch_size, 1024, n_points)
        x_global=self.conv2(x_global)                                       #[bs,512,n_points]->[bs,1024,n_points]
        x = torch.cat((x, x_global), dim=1)                                 # (batch_size, 1024*2, num_points)
        x = self.conv3(x)                                                   # (batch_size, 1024*2, num_points) -> (batch_size, 512, num_points)
        x = self.conv4(x)                                                   # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv5(x)                                                   # (batch_size, 256, num_points) -> (batch_size, 6, num_points)
        return x,None,None,None



class Superpoint_cluster_with_goal(nn.Module):
    def __init__(self,args, num_feats, input_dims, actv_fn=F.relu, top_k=16):
        """This function returns a sorted Tensor of Points. The Points are sorted

        """
        super(Superpoint_cluster_with_goal, self).__init__()
        self.args=args
        self.num_heads=args.num_heads
        self.num_layers=args.num_layers
        self.num_latent_feats_inencoder=args.self_encoder_latent_features
        self.num_feats = num_feats
        self.actv_fn = actv_fn
        self.input_dims = input_dims
        self.d_model=512
        self.top_k = top_k
        self.self_atn_layer =SA_Layer_Multi_Head(args,256)
        self.selfatn_layers=SA_Layers(self.num_layers,self.self_atn_layer)
 
        self.feat_channels_1d = [self.num_feats,64, 32, 16]
        self.feat_generator = create_conv1d_serials(self.feat_channels_1d)
        self.feat_generator.apply(init_weights)
        self.feat_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.feat_channels_1d[i+1])
                for i in range(len(self.feat_channels_1d)-1)
            ]
        )

        self.feat_channels_3d = [128, 256, self.d_model]
        self.radius_cnn = create_conv3d_serials(self.feat_channels_3d, 32, 3)
        self.radius_cnn.apply(init_weights)
        self.radius_bn = nn.ModuleList(
            [
                nn.BatchNorm3d(num_features=self.feat_channels_3d[i])
                for i in range(len(self.feat_channels_3d))
            ]
        )


        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0

    def forward(self, high_features,input,target):                                          #[bs,1,features,n_points]   [bs,C,n_points]
                                                            

        #self attention worked on 
        high_features=high_features.permute(0,2,1)
        high_features=self.selfatn_layers(high_features)
        high_features=high_features.permute(0,2,1)


        for j, conv in enumerate(self.feat_generator):                      #[bs,features,n_points]->[bs,16,n_points]
            bn = self.feat_bn[j]
            high_features = self.actv_fn(bn(conv(high_features)))
        topk = torch.topk(high_features, k=self.top_k, dim=-1)                      #[bs,16,n_points]->[bs,16,n_cluster]
        indices = topk.indices                                          #[bs,n_cluster,top_k]->[bs,n_cluster,top_k]

        # make the top1 get close to the set goals
        result_net=0
        goal=0
        if not self.args.eval and self.args.training:
            top1 = torch.topk(high_features, k=1, dim=-1)                      #[bs,10,n_points]->[bs,10,n_superpoint]
            index = top1.indices                                          #[bs,n_cluster,top_k]->[bs,n_cluster,top_k]
            result_net = index_points(input.permute(0, 2, 1).float(), index)
            result_net=result_net.squeeze(2)
            goal=find_goals(input.permute(0, 2, 1),target)
        #visialize_cluster(input,indices)


        sorted_input = index_points(input.permute(0, 2, 1).float(), indices).unsqueeze(dim=1)       #[bs,n_superpoint]->[bs,C,n_superpoint]


        for i, radius_conv in enumerate(self.radius_cnn):                           #[bs,n_superpoint,n_sample+1,C]->[bs,512,n_superpoint,1,1]
            bn = self.radius_bn[i]
            sorted_input = self.actv_fn(bn(radius_conv(sorted_input)))
        sorted_input = sorted_input.squeeze(dim=-1).squeeze(dim=-1)             #[bs,512,n_superpoint,1,1]->[bs,512,n_superpoint]                                         #[bs,n_superpoint]


        # sorted_input = sorted_input.squeeze(1).permute(0,3,1,2)
        # sorted_input=self.conv1(sorted_input)
        # sorted_input=self.conv2(sorted_input)
        # sorted_input=self.conv3(sorted_input)
        # sorted_input = sorted_input.max(dim=-1, keepdim=False)[0]

        return sorted_input,result_net,goal



class My_Network4(nn.Module):                                               #top1 to approach and transform
    def __init__(self, args):
        super(My_Network4, self).__init__()
        self.actv_fn = nn.LeakyReLU(negative_slope=0.2)
        self.p_dropout = args.dropout
        self.input_dim = 3
        self.top_k = 32
        self.d_model = args.hidden_size
        self.num_classes= 6  

        #########################################################################
        #dynamic graph based network
        ########################################################################
        self.args = args
        self.k = 20
        self.s3n = STN3d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)                                                             
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, 512, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))        
     




        #############################################################################
        #self desigened superpoint sample net and its net of generation local features 
        #############################################################################
        self.sort_ch = [self.input_dim,64, 128, 256]
        self.sort_cnn = create_conv1d_serials(self.sort_ch)   
        self.sort_cnn.apply(init_weights)
        self.sort_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.sort_ch[i+1])
                for i in range(len(self.sort_ch)-1)
            ]
        )
        self.superpointnet =Superpoint_cluster_with_goal(args,self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k)



        #############################
        #
        #############################
        ## Create Local-Global Attention
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=4)
        self.last_layer = PTransformerDecoderLayer(self.d_model, nhead=4, last_dim=256)
        self.custom_decoder = PTransformerDecoder(self.decoder_layer, 4, self.last_layer)
        self.transformer_model = nn.Transformer(d_model=self.d_model,nhead=4,num_encoder_layers=2,num_decoder_layers=2,custom_decoder=self.custom_decoder,)
        self.transformer_model.apply(init_weights)

        ##########################################################
        #final segmentation layer
        ###################################################### 
        self.bn7 = nn.BatchNorm1d(1024)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),         
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn8 = nn.BatchNorm1d(512)
        self.conv8 = nn.Sequential(nn.Conv1d(1728, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn8,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn9 = nn.BatchNorm1d(256)
        self.conv9 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn9,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Conv1d(256, self.num_classes, kernel_size=1, bias=False)   #256*6=1536

    def forward(self, x, target):
        x=x.float()
        x_local=x
        input=x
        #############################################
        ## Global Features
        #############################################
        batch_size = x.size(0)
        num_points = x.size(2)
 
        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_remain=x
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, 64)          # (batch_size, 1024, num_points)

        #############################################
        ## Local Features
        #############################################

        x_local = x_local.float()                                             #[bs,1,Cor,n_points]
        for i, sort_conv in enumerate(self.sort_cnn):                       #[bs,1,Cor,n_points]->[bs,256,n_points]
            bn = self.sort_bn[i]
            x_local = self.actv_fn(bn(sort_conv(x_local)))

        x_local_sorted,result,goal=self.superpointnet(x_local,input,target)              #[bs,256,n_points]->[bs,512,16]


        #############################################
        ## Point Transformer  global to patch
        #############################################
        source = x.permute(2, 0, 1)                                        # [bs,1024,64]->[64,bs,1024]
        target = x_local_sorted.permute(2, 0, 1)                           # [bs,1024,10]->[10,bs,1024]
        embedding = self.transformer_model(source, target)                 # [64,bs,1024]+[16,bs,1024]->[16,bs,1024]


        # #############################################
        # ## Point Transformer  patch to global
        # #############################################
        # target = x.permute(2, 0, 1)                                        # [bs,1024,64]->[64,bs,1024]
        # source = x_local_sorted.permute(2, 0, 1)                           # [bs,1024,10]->[10,bs,1024]
        # embedding = self.transformer_model(source, target)                 # [16,bs,1024]+[64,bs,1024]->[64,bs,1024]



        ################################################
        ##segmentation
        ################################################
        embedding=embedding.permute(1,2,0)                                  # [32,bs,512]->[bs,512,32]
        x=self.conv7(embedding)                                             # [bs,512,32]->[bs,1024,32]
        x = x.max(dim=-1, keepdim=True)[0]                                  # (batch_size, emb_dims, 10) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)                                      # (batch_size, 1024, n_points)
        x = torch.cat((x,x_remain,x1,x2,x3), dim=1)                         # (batch_size, 1024*2, num_points)
        x = self.conv8(x)                                                   # (batch_size, 1024*2, num_points) -> (batch_size, 512, num_points)
        x = self.conv9(x)                                                   # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv10(x)                                                   # (batch_size, 256, num_points) -> (batch_size, 6, num_points)

        return x,None,result,goal



class Superpoint_cluster_with_goal_mine(nn.Module):                        # dgcnn-like ending structure
    def __init__(self,args, num_feats, input_dims, actv_fn=F.relu, top_k=16):
        """This function returns a sorted Tensor of Points. The Points are sorted

        """
        super(Superpoint_cluster_with_goal_mine, self).__init__()
        self.args=args
        self.num_heads=args.num_heads
        self.num_layers=args.num_layers
        self.num_latent_feats_inencoder=args.self_encoder_latent_features
        self.num_feats = num_feats
        self.actv_fn = actv_fn
        self.input_dims = input_dims
        self.d_model=512
        self.top_k = top_k
        self.self_atn_layer =SA_Layer_Multi_Head(args,256)
        self.selfatn_layers=SA_Layers(self.num_layers,self.self_atn_layer)
 
        self.feat_channels_1d = [self.num_feats,64, 32, 16]
        self.feat_generator = create_conv1d_serials(self.feat_channels_1d)
        self.feat_generator.apply(init_weights)
        self.feat_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.feat_channels_1d[i+1])
                for i in range(len(self.feat_channels_1d)-1)
            ]
        )


        self.loss_function=nn.MSELoss()


        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0

    def forward(self, high_features,input,target):                                          #[bs,1,features,n_points]   [bs,C,n_points]
                                                            

        #self attention worked on 
        high_features=high_features.permute(0,2,1)
        high_features=self.selfatn_layers(high_features)
        high_features=high_features.permute(0,2,1)


        for j, conv in enumerate(self.feat_generator):                      #[bs,features,n_points]->[bs,16,n_points]
            bn = self.feat_bn[j]
            high_features = self.actv_fn(bn(conv(high_features)))
        topk = torch.topk(high_features, k=self.top_k, dim=-1)                      #[bs,16,n_points]->[bs,16,n_cluster]
        indices = topk.indices                                          #[bs,n_cluster,top_k]->[bs,n_cluster,top_k]

        # make the top1 get close to the set goals
        result_net=0
        goals=0
        if not self.args.eval and self.args.training:
            top1 = torch.topk(high_features, k=1, dim=-1)                      #[bs,10,n_points]->[bs,10,n_superpoint]
            index = top1.indices                                          #[bs,n_cluster,top_k]->[bs,n_cluster,top_k]
            result_net = index_points(input.permute(0, 2, 1).float(), index)
            result_net=result_net.squeeze(2)
            goals=find_goals(input.permute(0, 2, 1),target)


        #visialize_cluster(input,indices)


        sorted_input = index_points(input.permute(0, 2, 1).float(), indices).unsqueeze(dim=1)       #[bs,n_superpoint]->[bs,C,n_superpoint]


        sorted_input = sorted_input.squeeze(1).permute(0,3,1,2)
        sorted_input=self.conv1(sorted_input)
        sorted_input=self.conv2(sorted_input)
        sorted_input=self.conv3(sorted_input)
        sorted_input = sorted_input.max(dim=-1, keepdim=False)[0]

        return sorted_input, result_net,goals



class My_Network5(nn.Module):                                              #supercluster with my dgcnn-like ending structure
    def __init__(self, args):
        super(My_Network5, self).__init__()
        self.actv_fn = nn.LeakyReLU(negative_slope=0.2)
        self.p_dropout = args.dropout
        self.input_dim = 3
        self.top_k = 32
        self.d_model = args.hidden_size
        self.num_classes= 6  

        #########################################################################
        #dynamic graph based network
        ########################################################################
        self.args = args
        self.k = 20
        self.s3n = STN3d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)                                                             
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, 512, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))        
     




        #############################################################################
        #self desigened superpoint sample net and its net of generation local features 
        #############################################################################
        self.sort_ch = [self.input_dim,64, 128, 256]
        self.sort_cnn = create_conv1d_serials(self.sort_ch)   
        self.sort_cnn.apply(init_weights)
        self.sort_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.sort_ch[i+1])
                for i in range(len(self.sort_ch)-1)
            ]
        )
        self.superpointnet =Superpoint_cluster_with_goal_mine(args,self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k)



        #############################
        #
        #############################
        ## Create Local-Global Attention
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=4)
        self.last_layer = PTransformerDecoderLayer(self.d_model, nhead=4, last_dim=256)
        self.custom_decoder = PTransformerDecoder(self.decoder_layer, 4, self.last_layer)
        self.transformer_model = nn.Transformer(d_model=self.d_model,nhead=4,num_encoder_layers=2,num_decoder_layers=2,custom_decoder=self.custom_decoder,)
        self.transformer_model.apply(init_weights)

        ##########################################################
        #final segmentation layer
        ###################################################### 
        self.bn7 = nn.BatchNorm1d(1024)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),         
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn8 = nn.BatchNorm1d(512)
        self.conv8 = nn.Sequential(nn.Conv1d(1728, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn8,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn9 = nn.BatchNorm1d(256)
        self.conv9 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn9,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Conv1d(256, self.num_classes, kernel_size=1, bias=False)   #256*6=1536

    def forward(self, x, target):
        x=x.float()
        x_local=x
        input=x
        #############################################
        ## Global Features
        #############################################
        batch_size = x.size(0)
        num_points = x.size(2)
 
        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_remain=x
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, 64)          # (batch_size, 1024, num_points)

        #############################################
        ## Local Features
        #############################################

        x_local = x_local.float()                                             #[bs,1,Cor,n_points]
        for i, sort_conv in enumerate(self.sort_cnn):                       #[bs,1,Cor,n_points]->[bs,256,n_points]
            bn = self.sort_bn[i]
            x_local = self.actv_fn(bn(sort_conv(x_local)))

        x_local_sorted,result,goal=self.superpointnet(x_local,input,target)              #[bs,256,n_points]->[bs,512,16]


        #############################################
        ## Point Transformer  global to local
        #############################################
        source = x.permute(2, 0, 1)                                        # [bs,1024,64]->[64,bs,1024]
        target = x_local_sorted.permute(2, 0, 1)                           # [bs,1024,10]->[10,bs,1024]
        embedding = self.transformer_model(source, target)                 # [64,bs,1024]+[16,bs,1024]->[16,bs,1024]


        # #############################################
        # ## Point Transformer  local to global
        # #############################################
        # target = x.permute(2, 0, 1)                                        # [bs,1024,64]->[64,bs,1024]
        # source = x_local_sorted.permute(2, 0, 1)                           # [bs,1024,10]->[10,bs,1024]
        # embedding = self.transformer_model(source, target)                 # [16,bs,1024]+[64,bs,1024]->[64,bs,1024]



        ################################################
        ##segmentation
        ################################################
        embedding=embedding.permute(1,2,0)                                  # [32,bs,512]->[bs,512,32]
        x=self.conv7(embedding)                                             # [bs,512,32]->[bs,1024,32]
        x = x.max(dim=-1, keepdim=True)[0]                                  # (batch_size, emb_dims, 10) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)                                      # (batch_size, 1024, n_points)
        x = torch.cat((x,x_remain,x1,x2,x3), dim=1)                         # (batch_size, 1024*2, num_points)
        x = self.conv8(x)                                                   # (batch_size, 1024*2, num_points) -> (batch_size, 512, num_points)
        x = self.conv9(x)                                                   # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv10(x)                                                   # (batch_size, 256, num_points) -> (batch_size, 6, num_points)

        return x,None,result,goal



class Superpoint_cluster_with_goal_rectify(nn.Module):                     #top1(topk) to approach and change the original goals
    def __init__(self,args, num_feats, input_dims, actv_fn=F.relu, top_k=16):
        """This function returns a sorted Tensor of Points. The Points are sorted

        """
        super(Superpoint_cluster_with_goal_rectify, self).__init__()
        self.args=args
        self.num_heads=args.num_heads
        self.num_layers=args.num_layers
        self.num_latent_feats_inencoder=args.self_encoder_latent_features
        self.num_feats = num_feats
        self.actv_fn = actv_fn
        self.input_dims = input_dims
        self.d_model=512
        self.top_k = top_k
        self.self_atn_layer =SA_Layer_Multi_Head(args,256)
        self.selfatn_layers=SA_Layers(self.num_layers,self.self_atn_layer)
 
        self.feat_channels_1d = [self.num_feats,64, 32, 16]
        self.feat_generator = create_conv1d_serials(self.feat_channels_1d)
        self.feat_generator.apply(init_weights)
        self.feat_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.feat_channels_1d[i+1])
                for i in range(len(self.feat_channels_1d)-1)
            ]
        )


        self.loss_function=nn.MSELoss()

        self.feat_channels_3d = [128, 256, self.d_model]
        self.radius_cnn = create_conv3d_serials(self.feat_channels_3d, 32, 3)
        self.radius_cnn.apply(init_weights)
        self.radius_bn = nn.ModuleList(
            [
                nn.BatchNorm3d(num_features=self.feat_channels_3d[i])
                for i in range(len(self.feat_channels_3d))
            ]
        )


        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0

    def forward(self, high_features,input,target):                                          #[bs,1,features,n_points]   [bs,C,n_points]
                                                            

        #self attention worked on 
        high_features=high_features.permute(0,2,1)
        high_features=self.selfatn_layers(high_features)
        high_features=high_features.permute(0,2,1)


        for j, conv in enumerate(self.feat_generator):                      #[bs,features,n_points]->[bs,16,n_points]
            bn = self.feat_bn[j]
            high_features = self.actv_fn(bn(conv(high_features)))
        topk = torch.topk(high_features, k=self.top_k, dim=-1)                      #[bs,16,n_points]->[bs,16,n_cluster]
        indices = topk.indices                                          #[bs,n_cluster,top_k]->[bs,n_cluster,top_k]

        # make the top1 get close to the set goals
        result_net=0
        goal=0
        if not self.args.eval and self.args.training:
            top1 = torch.topk(high_features, k=1, dim=-1)                      #[bs,10,n_points]->[bs,10,n_superpoint]
            index = top1.indices                                          #[bs,n_cluster,top_k]->[bs,n_cluster,top_k]
            result_net = index_points(input.permute(0, 2, 1).float(), index)
            result_net=result_net.squeeze(2)
            goal=find_goals_rectify(input.permute(0, 2, 1),target)



        visialize_cluster(input,indices)


        sorted_input = index_points(input.permute(0, 2, 1).float(), indices).unsqueeze(dim=1)       #[bs,n_superpoint]->[bs,C,n_superpoint]


        for i, radius_conv in enumerate(self.radius_cnn):                           #[bs,n_superpoint,n_sample+1,C]->[bs,512,n_superpoint,1,1]
            bn = self.radius_bn[i]
            sorted_input = self.actv_fn(bn(radius_conv(sorted_input)))
        sorted_input = sorted_input.squeeze(dim=-1).squeeze(dim=-1)             #[bs,512,n_superpoint,1,1]->[bs,512,n_superpoint]                                         #[bs,n_superpoint]


        # sorted_input = sorted_input.squeeze(1).permute(0,3,1,2)
        # sorted_input=self.conv1(sorted_input)
        # sorted_input=self.conv2(sorted_input)
        # sorted_input=self.conv3(sorted_input)
        # sorted_input = sorted_input.max(dim=-1, keepdim=False)[0]

        return sorted_input, result_net,goal



class My_Network6(nn.Module):                                               #top1(topk) to approach(change the goals of cluster and omit gearcontainer and chager) and transform
    def __init__(self, args):
        super(My_Network6, self).__init__()
        self.actv_fn = nn.LeakyReLU(negative_slope=0.2)
        self.p_dropout = args.dropout
        self.input_dim = 3
        self.top_k = 32
        self.d_model = args.hidden_size
        self.num_classes= 6  

        #########################################################################
        #dynamic graph based network
        ########################################################################
        self.args = args
        self.k = 20
        self.s3n = STN3d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)                                                             
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, 512, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))        
     




        #############################################################################
        #self desigened superpoint sample net and its net of generation local features 
        #############################################################################
        self.sort_ch = [self.input_dim,64, 128, 256]
        self.sort_cnn = create_conv1d_serials(self.sort_ch)   
        self.sort_cnn.apply(init_weights)
        self.sort_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.sort_ch[i+1])
                for i in range(len(self.sort_ch)-1)
            ]
        )
        self.superpointnet =Superpoint_cluster_with_goal_rectify(args,self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k)



        #############################
        #
        #############################
        ## Create Local-Global Attention
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=4)
        self.last_layer = PTransformerDecoderLayer(self.d_model, nhead=4, last_dim=256)
        self.custom_decoder = PTransformerDecoder(self.decoder_layer, 4, self.last_layer)
        self.transformer_model = nn.Transformer(d_model=self.d_model,nhead=4,num_encoder_layers=2,num_decoder_layers=2,custom_decoder=self.custom_decoder,)
        self.transformer_model.apply(init_weights)

        ##########################################################
        #final segmentation layer
        ###################################################### 
        self.bn7 = nn.BatchNorm1d(1024)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),         
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn8 = nn.BatchNorm1d(512)
        self.conv8 = nn.Sequential(nn.Conv1d(1728, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn8,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn9 = nn.BatchNorm1d(256)
        self.conv9 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn9,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Conv1d(256, self.num_classes, kernel_size=1, bias=False)   #256*6=1536

    def forward(self, x, target):
        x=x.float()
        x_local=x
        input=x
        #############################################
        ## Global Features
        #############################################
        batch_size = x.size(0)
        num_points = x.size(2)
 
        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_remain=x
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, 64)          # (batch_size, 1024, num_points)

        #############################################
        ## Local Features
        #############################################

        x_local = x_local.float()                                             #[bs,1,Cor,n_points]
        for i, sort_conv in enumerate(self.sort_cnn):                       #[bs,1,Cor,n_points]->[bs,256,n_points]
            bn = self.sort_bn[i]
            x_local = self.actv_fn(bn(sort_conv(x_local)))

        x_local_sorted,result,goal=self.superpointnet(x_local,input,target)              #[bs,256,n_points]->[bs,512,16]


        #############################################
        ## Point Transformer  global to local
        #############################################
        source = x.permute(2, 0, 1)                                        # [bs,1024,64]->[64,bs,1024]
        target = x_local_sorted.permute(2, 0, 1)                           # [bs,1024,10]->[10,bs,1024]
        embedding = self.transformer_model(source, target)                 # [64,bs,1024]+[16,bs,1024]->[16,bs,1024]


        # #############################################
        # ## Point Transformer  local to global
        # #############################################
        # target = x.permute(2, 0, 1)                                        # [bs,1024,64]->[64,bs,1024]
        # source = x_local_sorted.permute(2, 0, 1)                           # [bs,1024,10]->[10,bs,1024]
        # embedding = self.transformer_model(source, target)                 # [16,bs,1024]+[64,bs,1024]->[64,bs,1024]



        ################################################
        ##segmentation
        ################################################
        embedding=embedding.permute(1,2,0)                                  # [32,bs,512]->[bs,512,32]
        x=self.conv7(embedding)                                             # [bs,512,32]->[bs,1024,32]
        x = x.max(dim=-1, keepdim=True)[0]                                  # (batch_size, emb_dims, 10) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)                                      # (batch_size, 1024, n_points)
        x = torch.cat((x,x_remain,x1,x2,x3), dim=1)                         # (batch_size, 1024*2, num_points)
        x = self.conv8(x)                                                   # (batch_size, 1024*2, num_points) -> (batch_size, 512, num_points)
        x = self.conv9(x)                                                   # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv10(x)                                                   # (batch_size, 256, num_points) -> (batch_size, 6, num_points)

        return x,None,result,goal



class ball_query_sample_with_goal(nn.Module):                                #top1(ball) to approach 
    def __init__(self,args, num_feats, input_dims, actv_fn=F.relu, top_k=16):
        """This function returns a sorted Tensor of Points. The Points are sorted

        """
        super(ball_query_sample_with_goal, self).__init__()
        self.args=args
        self.num_heads=args.num_heads
        self.num_layers=args.num_layers
        self.num_latent_feats_inencoder=args.self_encoder_latent_features
        self.num_feats = num_feats
        self.actv_fn = actv_fn
        self.input_dims = input_dims

        self.top_k = 32
        self.d_model = 512
        self.radius = 0.3
        self.max_radius_points = 32

        self.self_atn_layer =SA_Layer_Multi_Head(args,256)
        self.selfatn_layers=SA_Layers(self.num_layers,self.self_atn_layer)

        self.loss_function=nn.MSELoss()
 
        self.feat_channels_1d = [self.num_feats,64, 32, 16]
        self.feat_generator = create_conv1d_serials(self.feat_channels_1d)
        self.feat_generator.apply(init_weights)
        self.feat_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.feat_channels_1d[i+1])
                for i in range(len(self.feat_channels_1d)-1)
            ]
        )


        self.feat_channels_3d = [128, 256, self.d_model]
        self.radius_cnn = create_conv3d_serials(self.feat_channels_3d, self.max_radius_points + 1, 3)
        self.radius_cnn.apply(init_weights)
        self.radius_bn = nn.ModuleList(
            [
                nn.BatchNorm3d(num_features=self.feat_channels_3d[i])
                for i in range(len(self.feat_channels_3d))
            ]
        )

    def forward(self, hoch_features, input,target):                                          #[bs,1,features,n_points]   [bs,C,n_points]

        top_k = self.top_k                                                             #16
        origial_hoch_features = hoch_features                                          #[bs,1,features,n_points]
        feat_dim = input.shape[1]

        hoch_features_att = hoch_features  
        ############################################################
        # official version
        # ######################################################### 
        # hoch_features_att = hoch_features_att.permute(2, 0, 1)                      #[bs,features,n_points]->[n_points,bs,features]
        # hoch_features_att = self.self_atn_official(hoch_features_att)             #[n_points,bs,features]->[n_points,bs,features]
        # hoch_features_att = hoch_features_att.permute(1, 2, 0)                      #[n_points,bs,features]->[bs,features,n_points]
  
  
        #############################################################
        #implemented by myself
        #############################################################
        hoch_features_att=hoch_features_att.permute(0,2,1)
        hoch_features_att=self.selfatn_layers(hoch_features_att)
        hoch_features_att=hoch_features_att.permute(0,2,1)

        #####################
        #repeate 5 times
        #####################
        # for i in range(5):
        #     high_inter=hoch_features_att
        #     for j, conv in enumerate(self.feat_generator):                      #[bs,features,n_points]->[bs,10,n_points]
        #         bn = self.feat_bn[j]
        #         high_inter = self.actv_fn(bn(conv(high_inter)))
        #     topk = torch.topk(high_inter, k=top_k, dim=-1)                      #[bs,10,n_points]->[bs,10,n_superpoint]
        #     if i==0:
        #         input=input.permute(0, 2, 1).float()
        #         indices = topk.indices                                          #[bs,n_superpoints,top_k]->[bs,n_superpoints,top_k]
        # #         ###########################
        # #         #visial the superpoint cluster
        # #         ##########################
        # #         bs_,n_point,_=input.shape
        # #         to_display=input
        # #         man_made_label=torch.zeros((bs_,n_point,1)).to(input.device)
        # #         to_display=torch.cat((to_display,man_made_label),dim=-1)        #[bs,n_points,C+1]
        # #         bs,n_superpoints,num_topk=indices.shape
        # #         indices_=indices.view(bs,-1)
        # #         sample_point=index_points(input,indices_)                       #from [bs,n_points,3] to[bs,n_superpoint*num_topk,3]
        # #         sample_point=sample_point.view(bs,n_superpoints,num_topk,3)     #[bs,n_superpoint*num_topk,3]->[bs,n_superpoint,num_topk,3]
        # #         man_made_points=torch.zeros((bs,n_superpoints,num_topk,4)).to(input.device)
        # #         label_n_superpoints=torch.zeros((bs,n_superpoints,num_topk,1)).to(input.device)
        # #         for i in range(n_superpoints):
        # #             label_n_superpoints[:,i,:,:]=i+1
        # #             man_made_points[:,i,:,0:3]=sample_point[:,i,:,0:3]
        # #             man_made_points[:,i,:,3]=label_n_superpoints[:,i,:,0]
        # #         man_made_points=man_made_points.view(bs,-1,4)                     
        # #         for i in range(bs):
        # #             sampled=man_made_points[i,:,:].squeeze(0)
        # #             original=to_display[i,:,:].squeeze(0)
        # #             Visuell_superpoint(sampled,original)            
        #     else:
        #         indices = torch.cat((indices,topk.indices),dim=-1)
        # indices=indices.cpu()
        # indices=torch.mode(indices,dim=-1)[0]
        # indices=indices.cuda()


        ##########################
        #
        ##########################
        high_inter=hoch_features_att
        for j, conv in enumerate(self.feat_generator):                      #[bs,features,n_points]->[bs,10,n_points]
            bn = self.feat_bn[j]
            high_inter = self.actv_fn(bn(conv(high_inter)))
        topk = torch.topk(high_inter, k=top_k, dim=-1)                      #[bs,10,n_points]->[bs,10,n_superpoint]
        indices = topk.indices                                          #[bs,n_superpoints,top_k]->[bs,n_superpoints,top_k]
        
        #visialize_cluster(input,indices)
        indices=indices[:,:,0]
        # make the top1 get close to the set goals
        result_net =0
        goal=0
        if not self.args.eval and self.args.training:
            index = indices                                         #[bs,n_cluster,top_k]->[bs,n_cluster,top_k]
            result_net = index_points(input.permute(0, 2, 1).float(), index)
            result_net=result_net.squeeze(2)
            goal=find_goals(input.permute(0, 2, 1),target)



        sorted_input = torch.zeros((origial_hoch_features.shape[0], feat_dim, top_k)).to(    
            input.device                                                        #[bs,C,n_superpoint]
        )

        if top_k == 1:
            indices = indices.unsqueeze(dim=-1)

        sorted_input = index_points(input.permute(0, 2, 1).float(), indices).permute(0, 2, 1)       #[bs,n_superpoint]->[bs,C,n_superpoint]

        all_points = input.permute(0, 2, 1).float()                              #[bs,C,n_points]->[bs,n_points,C]
        query_points = sorted_input.permute(0, 2, 1)                             #[bs,C,n_superpoint]->[bs,n_superpoint,C]

        radius_indices = query_ball_point(                                       #idx=[bs,n_superpoint,n_sample]
            self.radius,
            self.max_radius_points,
            all_points[:, :, :3],
            query_points[:, :, :3],
        )

        radius_points = index_points(all_points, radius_indices)                    #[bs,n_superpoint,n_sample,C]

        radius_centroids = query_points.unsqueeze(dim=-2)                           #[bs,n_superpoint,1,C]
        radius_grouped = torch.cat([radius_centroids, radius_points], dim=-2).unsqueeze(
            dim=1                                                                   #[bs,n_superpoint,n_sample,C]+[bs,n_superpoint,1,C]->[bs,n_superpoint,n_sample+1,C]
        )

        for i, radius_conv in enumerate(self.radius_cnn):                           #[bs,n_superpoint,n_sample+1,C]->[bs,512,n_superpoint,1,1]
            bn = self.radius_bn[i]
            radius_grouped = self.actv_fn(bn(radius_conv(radius_grouped)))

        radius_grouped = radius_grouped.squeeze(dim=-1).squeeze(dim=-1)             #[bs,512,n_superpoint,1,1]->[bs,512,n_superpoint]                                          #[bs,n_superpoint]
        sorted_input = radius_grouped                                               #[bs,512,n_superpoint]

        return sorted_input,result_net,goal



class My_Network7(nn.Module):                                                #top1(ball) to approach  and transform
    def __init__(self, args):
        super(My_Network7, self).__init__()
        self.actv_fn = nn.LeakyReLU(negative_slope=0.2)
        self.p_dropout = args.dropout
        self.input_dim = 3
        self.top_k = 64
        self.d_model = args.hidden_size
        self.num_classes= 6  

        #########################################################################
        #dynamic graph based network
        ########################################################################
        self.args = args
        self.k = 20
        self.s3n = STN3d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)                                                             
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #3*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, 512, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))        
     




        #############################################################################
        #self desigened superpoint sample net and its net of generation local features 
        #############################################################################
        self.sort_ch = [self.input_dim,64, 128, 256]
        self.sort_cnn = create_conv1d_serials(self.sort_ch)   
        self.sort_cnn.apply(init_weights)
        self.sort_bn = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=self.sort_ch[i+1])
                for i in range(len(self.sort_ch)-1)
            ]
        )
        self.superpointnet =ball_query_sample_with_goal(args,self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k)


        ############################################################## 
        #sortnet proposed by nico in point transformer   
        ##############################################################
        #self.sortnets = nn.ModuleList(
        #     [
        #         SortNet(
        #             self.sort_ch[-1], self.input_dim, self.actv_fn, top_k=self.top_k
        #         )
        #         for _ in range(10)
               
        #     ]
        # )


        #############################
        #
        #############################
        ## Create Local-Global Attention
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=4)
        self.last_layer = PTransformerDecoderLayer(self.d_model, nhead=4, last_dim=256)
        self.custom_decoder = PTransformerDecoder(self.decoder_layer, 4, self.last_layer)
        self.transformer_model = nn.Transformer(d_model=self.d_model,nhead=4,num_encoder_layers=2,num_decoder_layers=2,custom_decoder=self.custom_decoder,)
        self.transformer_model.apply(init_weights)

        ##########################################################
        #final segmentation layer
        ###################################################### 
        self.bn7 = nn.BatchNorm1d(1024)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),         
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn8 = nn.BatchNorm1d(1024)
        self.conv8 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn8,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn8 = nn.BatchNorm1d(512)
        self.conv8 = nn.Sequential(nn.Conv1d(1728, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn8,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn9 = nn.BatchNorm1d(256)
        self.conv9 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn9,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Conv1d(256, self.num_classes, kernel_size=1, bias=False)   #256*6=1536

    def forward(self, x,target):
        x_local=x
        input=x.float()
        #############################################
        ## Global Features
        #############################################
        batch_size = x.size(0)
        num_points = x.size(2)
        x=x.float() 
        

        x = get_neighbors(x, k=self.k)       # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_remain=x
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, 64)          # (batch_size, 1024, num_points)
        #x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        #############################################
        ## Local Features
        #############################################

        x_local = x_local.float()                                             #[bs,1,Cor,n_points]
        for i, sort_conv in enumerate(self.sort_cnn):                       #[bs,1,Cor,n_points]->[bs,256,n_points]
            bn = self.sort_bn[i]
            x_local = self.actv_fn(bn(sort_conv(x_local)))

        x_local_sorted,result,goal=self.superpointnet(x_local,input,target)              #[bs,256,n_points]->[bs,512,32]
        # x_local_sorted = torch.cat(
        #     [sortnet(x_local, input.float())[0] for sortnet in self.sortnets], dim=-1   
        #                                                                      #[bs,256,n_points]->[bs,512,10]
        # )


        #############################################
        ## Point Transformer
        #############################################
        source = x.permute(2, 0, 1)                                        # [bs,1024,64]->[64,bs,1024]
        target = x_local_sorted.permute(2, 0, 1)                           # [bs,1024,10]->[10,bs,1024]
        embedding = self.transformer_model(source, target)                 # [64,bs,1024]+[32,bs,1024]->[32,bs,1024]



        ################################################
        ##segmentation
        ################################################
        embedding=embedding.permute(1,2,0)                                  # [32,bs,512]->[bs,512,32]
        x=self.conv7(embedding)                                             # [bs,512,32]->[bs,1024,32]
        x = x.max(dim=-1, keepdim=True)[0]                                  # (batch_size, emb_dims, 10) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)                                      # (batch_size, 1024, n_points)
        x = torch.cat((x,x_remain,x1,x2,x3), dim=1)                         # (batch_size, 1024*2, num_points)
        x = self.conv8(x)                                                   # (batch_size, 1024*2, num_points) -> (batch_size, 512, num_points)
        x = self.conv9(x)                                                   # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv10(x)                                                   # (batch_size, 256, num_points) -> (batch_size, 6, num_points)
        return x,None,result,goal