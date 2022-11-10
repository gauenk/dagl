import torch as th
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import time


# -- clean code --
from dagl.utils import clean_code
from dagl.utils.config_blocks import config_to_list
from . import shared_mods
from . import inds_buffer

# -- blocks --
from .blocks import default_conv,ResBlock,MeanShift
# from .ce_layer import CE
from .ce_aug import CE

def make_model(args, parent=False):
    return RR(args)

@clean_code.add_methods_from(shared_mods)
@clean_code.add_methods_from(inds_buffer)
class RR(nn.Module):
    def __init__(self, args, search_cfg,
                 conv=default_conv):
        super(RR, self).__init__()
        # define basic setting
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        self.n_resblocks = n_resblocks
        self.inds_buffer = []
        self.return_inds = args.return_inds

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        msa = CES(in_channels=n_feats,return_inds=args.return_inds,
                  search_cfg=search_cfg)
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

        # -- inds buffer --
        self.use_inds_buffer = self.return_inds
        self.inds_buffer = []

    def forward(self, x, flows=None):
        res = self.head(x)
        inds = None
        mid = self.n_resblocks//2
        for _name,layer in self.body.named_children():
            if int(_name) == mid:
                res,inds = layer(res,flows)
                self.update_inds_buffer(inds)
            else: res = layer(res)
        res = self.tail(res)
        return x+res

@clean_code.add_methods_from(inds_buffer)
class CES(nn.Module):
    def __init__(self,in_channels,return_inds,search_cfg,num=4):
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
        self.return_inds = return_inds
        search_cfg_l = config_to_list(search_cfg,4*3)
        print(search_cfg_l[0])
        print(search_cfg_l[11])
        self.c1_1 = CE(in_channels=in_channels,**search_cfg_l[0])
        self.c1_2 = CE(in_channels=in_channels,**search_cfg_l[1])
        self.c1_3 = CE(in_channels=in_channels,**search_cfg_l[2])
        self.c1_4 = CE(in_channels=in_channels,**search_cfg_l[3])
        self.c1_c = nn.Conv2d(in_channels,in_channels,1,1,0)
        # stage 2 (4 head)
        self.c2_1 = CE(in_channels=in_channels,**search_cfg_l[4])
        self.c2_2 = CE(in_channels=in_channels,**search_cfg_l[5])
        self.c2_3 = CE(in_channels=in_channels,**search_cfg_l[6])
        self.c2_4 = CE(in_channels=in_channels,**search_cfg_l[7])
        self.c2_c = nn.Conv2d(in_channels,in_channels,1,1,0)
        # stage 3 (4 head)
        self.c3_1 = CE(in_channels=in_channels,**search_cfg_l[8])
        self.c3_2 = CE(in_channels=in_channels,**search_cfg_l[9])
        self.c3_3 = CE(in_channels=in_channels,**search_cfg_l[10])
        self.c3_4 = CE(in_channels=in_channels,**search_cfg_l[11])
        self.c3_c = nn.Conv2d(in_channels,in_channels,1,1,0)

        # -- inds buffer --
        self.use_inds_buffer = self.return_inds
        self.inds_buffer = []


    def forward(self, x, flows=None):
        # 4head-3stages
        nstages = 3
        nheads = 4
        stage_out = x
        inds_prev = None
        for stage in range(nstages):
            input_x = stage_out
            stage_out,stage_inds = [],[]
            for head in range(nheads):
                cstr = "c%d_%d" % (stage+1,head+1)
                layer = getattr(self,cstr)
                # th.cuda.synchronize()
                # print("layer: ",cstr)
                out_,inds_ = layer(input_x,flows,inds_prev)
                # th.cuda.synchronize()
                # print("inds.shape: ",inds_.shape)
                self.update_inds_buffer(inds_)
                inds_prev = inds_
                stage_out.append(out_)
            conv = getattr(self,"c%d_c" % (stage+1))
            # th.cuda.synchronize()
            stage_out = conv(torch.cat(stage_out,dim=1))+input_x
            # th.cuda.synchronize()
            if stage < (nstages-1):
                RBS = getattr(self,"RBS%d" % (stage+1))
                stage_out = RBS(stage_out)

        # -- grab and clear --
        # inds = self.get_inds_buffer()
        inds = self.inds_buffer
        self.clear_inds_buffer()

        return stage_out,inds

