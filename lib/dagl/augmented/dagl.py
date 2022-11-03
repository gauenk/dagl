import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import time
from .blocks import default_conv,ResBlock,MeanShift
from .ce_layer import CE

def make_model(args, parent=False):
    return RR(args)

class RR(nn.Module):
    def __init__(self, args, search_cfg,
                 conv=default_conv):
        super(RR, self).__init__()
        # define basic setting
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        msa = CES(in_channels=n_feats,search_cfg=search_cfg)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, nn.PReLU(), res_scale=args.res_scale
            ) for _ in range(n_resblocks // 2)
        ]
        m_body.append(msa)
        for i in range(n_resblocks // 2):
            m_body.append(ResBlock(conv, n_feats, kernel_size,
                                          nn.PReLU(), res_scale=args.res_scale))

        m_body.append(conv(n_feats, n_feats, kernel_size))
        m_tail = [
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        res = self.head(x)

        res = self.body(res)

        res = self.tail(res)

        return x+res

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
class CES(nn.Module):
    def __init__(self,in_channels,search_cfg,num=4):
        super(CES,self).__init__()
        RBS1 = [
            ResBlock(default_conv, n_feats=in_channels,
                kernel_size=3, act=nn.PReLU(), res_scale=1
            ) for _ in range(num)
        ]
        self.RBS1 = nn.Sequential(
            *RBS1
        )
        RBS2 = [
            ResBlock(default_conv, n_feats=in_channels,
                kernel_size=3, act=nn.PReLU(), res_scale=1
            ) for _ in range(num)
        ]
        self.RBS2 = nn.Sequential(
            *RBS2
        )
        # stage 1 (4 head)
        search_cfg_l = search_config_list(search_cfg,4*3)
        self.c1_1 = CE(in_channels=in_channels,search_cfg=search_cfg_l[0])
        self.c1_2 = CE(in_channels=in_channels,search_cfg=search_cfg_l[1])
        self.c1_3 = CE(in_channels=in_channels,search_cfg=search_cfg_l[2])
        self.c1_4 = CE(in_channels=in_channels,search_cfg=search_cfg_l[3])
        self.c1_c = nn.Conv2d(in_channels,in_channels,1,1,0)
        # stage 2 (4 head)
        self.c2_1 = CE(in_channels=in_channels,search_cfg=search_cfg_l[4])
        self.c2_2 = CE(in_channels=in_channels,search_cfg=search_cfg_l[5])
        self.c2_3 = CE(in_channels=in_channels,search_cfg=search_cfg_l[6])
        self.c2_4 = CE(in_channels=in_channels,search_cfg=search_cfg_l[7])
        self.c2_c = nn.Conv2d(in_channels,in_channels,1,1,0)
        # stage 3 (4 head)
        self.c3_1 = CE(in_channels=in_channels,search_cfg=search_cfg_l[8])
        self.c3_2 = CE(in_channels=in_channels,search_cfg=search_cfg_l[9])
        self.c3_3 = CE(in_channels=in_channels,search_cfg=search_cfg_l[10])
        self.c3_4 = CE(in_channels=in_channels,search_cfg=search_cfg_l[11])
        self.c3_c = nn.Conv2d(in_channels,in_channels,1,1,0)

    def forward(self, x):
        # 4head-3stages
        nstages = 3
        nheads = 4
        stage_out = x
        for stage in range(nstages):
            input_x = stage_out
            stage_out,stage_inds = [],[]
            inds_prev = None
            for head in range(nheads):
                cstr = "c%d_%d" % (stage+1,head+1)
                layer = getattr(self,cstr)
                out_,inds_ = layer(input_x,inds_prev)
                inds_prev = inds_
                stage_out.append(out_)
            conv = getattr(self,"c%d_c" % (stage+1))
            stage_out = conv(torch.cat(stage_out,dim=1))+input_x
            if stage < (nstages-1):
                RBS = getattr(self,"RBS%d" % (stage+1))
                stage_out = RBS(stage_out)
        return stage_out

        # out = self.c1_c(torch.cat((self.c1_1(x),self.c1_2(x),self.c1_3(x),self.c1_4(x)),dim=1))+x
        # out = self.RBS1(out)
        # out = self.c2_c(torch.cat((self.c2_1(out),self.c2_2(out),self.c2_3(out),self.c2_4(out)),dim=1))+out
        # out  = self.RBS2(out)
        # out = self.c3_c(torch.cat((self.c3_1(out),self.c3_2(out),self.c3_3(out),self.c3_4(out)),dim=1))+out
        # return out

def search_config_list(cfg,nblocks=12):
    cfgs = []
    for _ in range(nblocks):
        cfgs.append(None)
    return cfgs

