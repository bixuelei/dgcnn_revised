#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: bixuelei
@Contact: xueleibi@gmail.com
@File: model.py
@Time: 2022/1/15 17:11 PM
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from attention_util import *
from util import *
from pointnet_util import PointNetSetAbstractionMsg, query_ball_point, index_points
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



class STN3d(nn.Module):
    '''
    training with a 3X3 rotation matrix and limit it with R * transpose(R) to I
    
    '''
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
    '''
    training with euler angles,but the result is bad
    '''
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



class DGCNN_semseg_rotate(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg_rotate, self).__init__()
        self.args = args
        self.k = args.k
        self.s3n = STN3d(3)
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
        
        trans=self.s3n(x)
        x=x.permute(0,2,1)                      #(batch_size, 3, num_points)->(batch_size,  num_points,3)
        x = torch.bmm(x, trans)
        #Visuell_PointCloud_per_batch(x,target)
        x=x.permute(0,2,1)
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

        return x ,trans,None,None



class DGCNN_semseg_rotate_conv(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg_rotate_conv, self).__init__()
        self.args = args
        self.k = args.k
        self.s3n = STN3d(3)
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

        trans=self.s3n(x)
        x=x.permute(0,2,1)                      #(batch_size, 3, num_points)->(batch_size,  num_points,3)
        x = torch.bmm(x, trans)
        #Visuell_PointCloud_per_batch(x,target)
        x=x.permute(0,2,1)

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
        
        return x,trans,None,None



class DGCNN_semseg_rotate_attention(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg_rotate_attention, self).__init__()
        self.args = args
        self.k = args.k
        self.s3n = STN3d(3)
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

        

    def forward(self, x,target):
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



class DGCNN_semseg_rotate_conv_attention(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg_rotate_conv_attention, self).__init__()
        self.args = args
        self.k = args.k
        self.s3n = STN3d(3)
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

        

    def forward(self, x,for_alignment):
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
        
        return x,trans,None,None



class PCT_semseg_rotate(nn.Module):
    def __init__(self, args):
        super(PCT_semseg_rotate, self).__init__()
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

        

    def forward(self, x,target):
        batch_size = x.size(0)
        num_points = x.size(2)
        x=x.float()  

        trans=self.s3n(x)
        x=x.permute(0,2,1)
        x = torch.bmm(x, trans)
        x=x.permute(0,2,1)
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
        
        return x,trans,None,None



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

        self.top_k = 32
        self.d_model = 512
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

        sorted_input = radius_grouped.squeeze(1).permute(0,3,1,2)
        sorted_input=self.conv1(sorted_input)
        sorted_input=self.conv2(sorted_input)
        sorted_input=self.conv3(sorted_input)
        sorted_input = sorted_input.max(dim=-1, keepdim=False)[0]

        # for i, radius_conv in enumerate(self.radius_cnn):                           #[bs,n_superpoint,n_sample+1,C]->[bs,512,n_superpoint,1,1]
        #     bn = self.radius_bn[i]
        #     radius_grouped = self.actv_fn(bn(radius_conv(radius_grouped)))

        # radius_grouped = radius_grouped.squeeze(dim=-1).squeeze(dim=-1)             #[bs,512,n_superpoint,1,1]->[bs,512,n_superpoint]                                          #[bs,n_superpoint]
        # sorted_input = radius_grouped                                               #[bs,512,n_superpoint]

        return sorted_input,None



class Final(nn.Module):
    def __init__(self, args):
        super(Final, self).__init__()
        self.actv_fn = nn.LeakyReLU(negative_slope=0.2)
        self.p_dropout = args.dropout
        self.input_dim = 3
        self.top_k = 64
        self.d_model = args.hidden_size
        self.num_classes= 6  
        self.args = args
        self.k = args.k
        self.s3n = STN3d(3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.my_self_attn_inter1=MultiHeadAttention(args.num_heads,64,64,16,16)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.my_self_attn_inter2=MultiHeadAttention(args.num_heads,64,64,16,16)
        self.bn5 = nn.BatchNorm2d(64)
        self.my_self_attn_inter3=MultiHeadAttention(args.num_heads,64,64,16,16)
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
        x=x.float()
        x_local=x
        input=x.float()
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
