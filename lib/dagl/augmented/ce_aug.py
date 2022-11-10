
# -- misc --
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -- import --
from dagl.utils.timer import ExpTimer
from dagl.utils.misc import assert_nonan
from .patching import same_padding,extract_image_patches

# -- inds buffer --
from dagl.utils import clean_code
from . import fc_fxns
from . import attn_fxns
from . import inds_buffer

# -- profiling --
import torch.autograd.profiler as profiler
from ..utils.timer import ExpTimer,TimeIt


"""
Graph model
"""
@clean_code.add_methods_from(fc_fxns)
@clean_code.add_methods_from(attn_fxns)
@clean_code.add_methods_from(inds_buffer)
class CE(nn.Module):
    def __init__(self, in_channels=64, inter_channels=16,
                 use_multiple_size=False,add_SE=False,
                 softmax_scale=10,shape=64,p_len=64,
                 attn_mode="dnls_k",k_s=100,k_a=100,ps=7,pt=1,
                 ws=27,ws_r=3,wt=0,stride0=4,stride1=1,dilation=1,bs=-1,
                 rbwd=True,nbwd=1,exact=False,reflect_bounds=False,
                 refine_inds=False,return_inds=False,use_pfc=False):
        # print(shape,in_channels,inter_channels,ps)
        # ksize=7, stride_1=4, stride_2=1,
        #          softmax_scale=10, shape=64 ,p_len=64,in_channels=64,
        #          inter_channels=16,use_multiple_size=False,
        #          add_SE=False,num_edge = 50,search_cfg=None):
        super(CE, self).__init__()
        self.ps = ps
        self.stride0 = stride0
        self.stride1 = stride1
        self.refine_inds = refine_inds
        self.use_inds_buffer = return_inds
        self.use_pfc = use_pfc
        self.bs = bs
        self.k_s = k_s
        self.k_a = k_a
        print("use_pfc: ",use_pfc)

        self.shape=shape
        self.p_len=p_len
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size=use_multiple_size
        self.add_SE=add_SE
        self.g = nn.Conv2d(in_channels=self.in_channels,
                           out_channels=self.inter_channels, kernel_size=3, stride=1,
                           padding=1)
        self.W = nn.Conv2d(in_channels=self.inter_channels,
                           out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                               kernel_size=1, stride=1,
                               padding=0)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=ps**2*inter_channels,
                      out_features=(ps**2*inter_channels)//4),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=ps**2*inter_channels,
                      out_features=(ps**2*inter_channels)//4),
            nn.ReLU()
        )

        # pad = (ps - stride0)//2
        # self.fc1 = nn.Sequential(
        #     nn.Conv2d(in_features=inter_channels,out_channels=inter_channels//4,
        #               stride=stride0,padding=pad,bias=True)
        #     nn.ReLU()
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Conv2d(in_features=inter_channels,out_channels=inter_channels//4,
        #               stride=stride0,padding=pad,bias=True)
        #     nn.ReLU()
        # )
        # print("k_s,k_a,ws,ws_r,wt,stride0,stride1: ",
        #        k_s,k_a,ws,ws_r,wt,stride0,stride1)

        self.thr_conv = nn.Conv2d(in_channels=in_channels,out_channels=1,
                                  kernel_size=ps,stride=stride0,padding=0)
        self.bias_conv = nn.Conv2d(in_channels=in_channels,out_channels=1,
                                   kernel_size=ps,stride=stride0,padding=0)
        self.search = self.init_search(attn_mode=attn_mode,k=k_s,ps=ps,pt=pt,
                                       ws=ws,ws_r=ws_r,wt=wt,
                                       stride0=stride0,stride1=stride1,
                                       dilation=dilation,rbwd=rbwd,nbwd=nbwd,
                                       exact=exact,reflect_bounds=reflect_bounds,
                                       refine_inds=refine_inds)
        self.wpsum = self.init_wpsum(ps=ps,pt=pt,dilation=dilation,
                                     reflect_bounds=reflect_bounds,exact=exact)

        c_in = inter_channels
        c_out = inter_channels//4
        self.pfc0 = self.init_pfc(c_in,c_out,ps,stride0)
        self.pfc1 = self.init_pfc(c_in,c_out,ps,stride1)

    def extract_features(self,vid):

        # -- basic features --
        B = vid.shape[0]
        vid1 = self.g(vid)
        vid2 = self.theta(vid)
        vid3 = vid1
        # vid = vid[None,:]
        vid4, _ = same_padding(vid,[self.ps,self.ps],
                             [self.stride0,self.stride0],[1,1])

        # -- apply the graph convolution to patches --
        # vid3 = self.fc1(vid3.view(wi.shape[1],-1))
        # vid1 = self.fc2(vid1.view(xi.shape[1],-1)).permute(1,0)
        vid1,vid3 = self.extract_fc(vid1,vid3)

        # -- thresholding --
        soft_thr = self.thr_conv(vid4)#.view(B,-1)
        soft_bias = self.bias_conv(vid4)#.view(B,-1)

        return vid1,vid2,vid3,vid4,soft_thr,soft_bias

    def forward(self, vid, flows=None, inds_pred=None):

        # -- init pfc --
        # print(dir(self.fc1))
        # print(dir(dict(self.fc1.named_modules())['0']))
        # print(dict(self.fc1.named_modules())['0'].parameters())
        self.pfc0.set_from_fc(dict(self.fc1.named_modules())['0'])
        self.pfc1.set_from_fc(dict(self.fc2.named_modules())['0'])

        # -- init buffs --
        self.clear_inds_buffer()

        # -- init timer --
        use_timer = True
        timer = ExpTimer(use_timer)

        # -- get features --
        with TimeIt(timer,"extract"):
            ftrs = self.extract_features(vid)
            vid1,vid2,vid3,vid4,soft_thr,soft_bias = ftrs

        # -- add batch dim --
        vid1 = vid1[None,:]
        vid2 = vid2[None,:]
        vid3 = vid3[None,:]
        # vid4
        # soft_thr
        # soft_bias
        vid = vid[None,:]

        # -- batching params --
        nbatch,nbatches,ntotal = self.batching_info(vid.shape)

        # -- init & update --
        ifold = self.init_ifold(vid2.shape,vid2.device)
        if not(self.refine_inds):
            self.search.update_flow(vid.shape,vid.device,flows)

        # -- batch across queries --
        for index in range(nbatches):

            # -- batch info --
            qindex = min(nbatch * index,ntotal)
            nbatch_i =  min(nbatch, ntotal - qindex)

            # -- search --
            # print("vid1.shape,vid3.shape: ",vid1.shape,vid3.shape)
            # print(self.search.ws,self.search.wt,self.search.ps,
            #       self.search.k,self.search.chnls,
            #       self.search.stride0,self.search.stride1)
            # print(self.search)
            # print(vid1)
            # print(vid3)

            with TimeIt(timer,"search"):
                with profiler.record_function("search"):
                    dists,inds = self.search.wrap_fwd(vid1,qindex,
                                                      nbatch_i,vid3,inds_pred)
            assert_nonan(dists)
            # print("inds.shape: ",inds.shape)

            # -- subset to only aggregate --
            inds_agg = inds[...,:self.k_a,:].contiguous()
            dists_agg = dists[...,:self.k_a].contiguous()

            # -- softmax --
            yi = F.softmax(dists_agg*self.softmax_scale,2)
            assert_nonan(yi)

            # -- attn mask --
            with TimeIt(timer,"agg"):
                with profiler.record_function("agg"):
                    zi = self.wpsum(vid2[:,None],yi[:,None],inds_agg[:,None])

            # -- ifold --
            with TimeIt(timer,"ifold"):
                with profiler.record_function("ifold"):
                    zi = rearrange(zi,'b H q c h w -> b q H 1 c h w')
                    assert zi.shape[2] == 1
                    # print("zi.shape: ",zi.shape)
                    ifold(zi,qindex)

            # -- update --
            with profiler.record_function("inds"):
                self.update_inds_buffer(inds)

        # -- get post-attn vid --
        y,Z = ifold.vid,ifold.zvid
        y = y / Z
        assert_nonan(y)
        print(timer)

        # -- remove batching --
        vid,y = vid[0],y[0]
        # print("vid.shape: ",vid.shape)
        # print("y.shape: ",y.shape)

        # -- final transform --
        # y = self.W(y)
        # y = vid + y

        # -- get inds --
        inds = self.get_inds_buffer()
        # print("[final] inds.shape: ",inds.shape)
        self.clear_inds_buffer()

        return y,inds

