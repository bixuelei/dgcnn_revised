#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: bixuelei
@Contact: xueleibi@gmail.com
@File: model.py
@Time: 2022/1/15 17:11 PM
"""

import torch
import torch.nn as nn
from attention_util import *
from util import *
from pointnet_util import index_points
from torch.autograd import Variable



def knn(x, k):
    """
    Input:
        points: input points data, [B, C, N]
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



class DGCNN_semseg(nn.Module):                                                  #original dgcnn
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



class DGCNN_semseg_conv(nn.Module):                                             #original dgcnn with additional conv1d as aggregation operator.
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



class DGCNN_semseg_attention(nn.Module):                                       #original dgcnn with additional self-attention layer
    def __init__(self, args):
        super(DGCNN_semseg_attention, self).__init__()
        self.args = args
        self.k = args.k
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
        
        return x,None,None,None



class DGCNN_semseg_conv_attention(nn.Module):                                 #original dgcnn with conv1d as additional aggregation operator and self-attention
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



class PCT_semseg(nn.Module):                                                  # replace edge with self-attention(PCT Net)
    def __init__(self, args):
        super(PCT_semseg, self).__init__()
        self.args = args
        self.k = args.k
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