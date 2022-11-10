import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import time
from .patching import same_padding,extract_image_patches

"""
Graph model
"""
class CE(nn.Module):
    def __init__(self, ksize=7, stride_1=4, stride_2=1, softmax_scale=10,shape=64 ,p_len=64,in_channels=64
                 , inter_channels=16,use_multiple_size=False,use_topk=False,add_SE=False,num_edge = 50,search_cfg=None):
        super(CE, self).__init__()
        self.ksize = ksize
        self.shape=shape
        self.p_len=p_len
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size=use_multiple_size
        self.use_topk=use_topk
        self.add_SE=add_SE
        self.num_edge = num_edge
        self.g = nn.Conv2d(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=3, stride=1,
                           padding=1)
        self.W = nn.Conv2d(in_channels=self.inter_channels,
                           out_channels=self.in_channels,
                           kernel_size=1, stride=1,
                           padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                               kernel_size=1, stride=1,
                               padding=0)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=ksize**2*inter_channels,
                      out_features=(ksize**2*inter_channels)//4),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=ksize**2*inter_channels,
                      out_features=(ksize**2*inter_channels)//4),
            nn.ReLU()
        )

        self.thr_conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=1,
                                  kernel_size=ksize,
                                  stride=stride_1,padding=0)
        self.bias_conv = nn.Conv2d(in_channels=in_channels,
                                   out_channels=1,
                                   kernel_size=ksize,
                                   stride=stride_1,padding=0)

    def forward(self, b, inds_prev=None):
        b1 = self.g(b)
        b2 = self.theta(b)
        b3 = b1

        raw_int_bs = list(b1.size())  # b*c*h*w
        b4, _ = same_padding(b,[self.ksize,self.ksize],[self.stride_1,self.stride_1],[1,1])
        soft_thr = self.thr_conv(b4).view(raw_int_bs[0],-1)
        soft_bias = self.bias_conv(b4).view(raw_int_bs[0],-1)
        patch_28, paddings_28 = extract_image_patches(b1, ksizes=[self.ksize, self.ksize],
                                                      strides=[self.stride_1, self.stride_1],
                                                      rates=[1, 1],
                                                      padding='same')
        patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_28 = patch_28.permute(0, 4, 1, 2, 3)
        patch_28_group = torch.split(patch_28, 1, dim=0)

        patch_112, paddings_112 = extract_image_patches(b2, ksizes=[self.ksize, self.ksize],
                                                        strides=[self.stride_2, self.stride_2],
                                                        rates=[1, 1],
                                                        padding='same')

        patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_112 = patch_112.permute(0, 4, 1, 2, 3)
        patch_112_group = torch.split(patch_112, 1, dim=0)

        patch_112_2, paddings_112_2 = extract_image_patches(b3, ksizes=[self.ksize, self.ksize],
                                                        strides=[self.stride_2, self.stride_2],
                                                        rates=[1, 1],
                                                        padding='same')

        patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)
        patch_112_group_2 = torch.split(patch_112_2, 1, dim=0)
        y = []
        w, h = raw_int_bs[2], raw_int_bs[3]
        _, paddings = same_padding(b3[0,0].unsqueeze(0).unsqueeze(0), [self.ksize, self.ksize], [self.stride_2, self.stride_2], [1, 1])

        for xi, wi,pi,thr,bias in zip(patch_112_group_2, patch_28_group, patch_112_group,soft_thr,soft_bias):
            c_s = pi.shape[2]
            k_s = wi[0].shape[2]
            # print("[a] wi.shape: ",wi.shape,wi.view(wi.shape[1],-1).shape)
            wi = self.fc1(wi.view(wi.shape[1],-1))
            # print("[b] wi.shape: ",wi.shape)
            # print("[a] xi.shape: ",xi.shape,xi.view(xi.shape[1],-1).shape)
            xi = self.fc2(xi.view(xi.shape[1],-1)).permute(1,0)
            # print("[b] xi.shape: ",xi.shape)
            score_map = torch.matmul(wi,xi)
            score_map = score_map.view(1, score_map.shape[0], math.ceil(w / self.stride_2),
                                       math.ceil(h / self.stride_2))
            b_s, l_s, h_s, w_s = score_map.shape
            yi = score_map.view(l_s, -1)

            mask = F.relu(yi-yi.mean(dim=1,keepdim=True)*thr.unsqueeze(1)+bias.unsqueeze(1))
            mask_b = (mask!=0.).float()

            yi = yi * mask
            yi = F.softmax(yi * self.softmax_scale, dim=1)
            yi = yi * mask_b

            print("[a] pi.shape: ",pi.shape)
            pi = pi.view(h_s * w_s, -1)
            print("[b] pi.shape: ",pi.shape)
            yi = torch.mm(yi, pi)
            yi = yi.view(b_s, l_s, c_s, k_s, k_s)[0]
            zi = yi.view(1, l_s, -1).permute(0, 2, 1)

            zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize), padding=paddings[0], stride=self.stride_1)
            inp = torch.ones_like(zi)

            inp_unf = torch.nn.functional.unfold(inp, (self.ksize, self.ksize), padding=paddings[0], stride=self.stride_1)
            out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize), padding=paddings[0], stride=self.stride_1)
            out_mask += (out_mask==0.).float()

            zi = zi / out_mask
            y.append(zi)
        y = torch.cat(y, dim=0)
        print("y.shape: ",y.shape)
        return y,None

    def GSmap(self,a,b):
        return torch.matmul(a,b)
