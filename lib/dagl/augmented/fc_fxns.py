

import dnls
import torch as th
from einops import rearrange
from dagl.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)
from torch.nn.functional import unfold,pad,relu
from dagl.utils.misc import assert_nonan

@register_method
def extract_fc(self,vid1,vid3):
    if self.use_pfc:
        return self.extract_fc_pfc(vid1,vid3)
    else:
        return self.extract_fc_standard(vid1,vid3)

@register_method
def extract_fc_pfc(self,vid1,vid3):
    # print("vid1.shape:" ,vid1.shape)
    # vid1 = th.zeros_like(vid1)[...,:4,:,:].contiguous()
    # vid3 = th.zeros_like(vid3)[...,:4,:,:].contiguous()
    vid1 = self.pfc0(vid1[None,:])[0]
    vid3 = self.pfc1(vid3[None,:])[0]
    # assert_nonan(vid1)
    # assert_nonan(vid3)
    return vid1,vid3

@register_method
def extract_fc_standard(self,vid1,vid3):
    patches1 = self.iunfold(vid1,self.stride0)
    patches3 = self.iunfold(vid3,self.stride1)
    T,C,H,W = vid1.shape
    ifold1 = self.init_ifold_fc((1,T,4,H,W),vid1.device,self.stride0)
    ifold3 = self.init_ifold_fc((1,T,4,H,W),vid3.device,self.stride1)
    ps = self.ps
    shape_str = 't q (c h w) -> 1 (t q) 1 1 c h w'
    qstart = 0
    for p1,p3 in zip(patches1[None,:],patches3[None,:]):
        p1 = self.fc2(p1)
        p3 = self.fc1(p3)
        p1 = rearrange(p1,shape_str,h=ps,w=ps)
        p3 = rearrange(p3,shape_str,h=ps,w=ps)
        ifold1(p1,qstart)
        ifold3(p3,qstart)
        qstart += p3.shape[1]
    vid1 = ifold1.vid/ifold1.zvid
    vid3 = ifold3.vid/ifold3.zvid
    assert_nonan(vid1)
    assert_nonan(vid3)
    # print("vid1.shape:" ,vid1.shape)
    # print("vid3.shape:" ,vid3.shape)
    return vid1[0],vid3[0]

@register_method
def init_ifold_fc(self,vshape,device,stride):
    rbounds = True
    dil = self.search.dilation
    ifold = dnls.iFoldz(vshape,None,stride=stride,dilation=dil,
                        adj=0,only_full=False,use_reflect=rbounds,device=device)
    return ifold

@register_method
def iunfold(self,vid,stride):
    pads = [(self.ps)//2,]*4
    vid_pad = pad(vid,pads,mode="reflect")
    patches = unfold(vid_pad,(self.ps,self.ps),stride=stride)
    patches = patches.transpose(2,1)
    return patches
