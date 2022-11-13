
# -- misc --
import dnls
from copy import deepcopy as dcopy
from ..utils import optional

# -- clean code --
from dagl.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)

@register_method
def update_search(self,inds_is_none):
    if self.refine_inds:
        self.search_kwargs["refine_inds"] = not(inds_is_none)
        self.search = self.init_search(**self.search_kwargs)

@register_method
def init_search(self,**kwargs):
    attn_mode = optional(kwargs,"attn_mode","dnls_k")
    refine_inds = optional(kwargs,"refine_inds",False)
    cfg = dcopy(kwargs)
    del cfg["attn_mode"]
    del cfg["refine_inds"]
    if attn_mode == "dnls_k":
        if refine_inds: return self.init_refine(**cfg)
        else: return self.init_dnls_k(**cfg)
    else:
        raise ValueError(f"Uknown attn_mode [{attn_mode}]")

@register_method
def init_refine(self,k=100,ps=7,pt=0,ws=21,ws_r=3,wt=0,
                stride0=4,stride1=1,dilation=1,rbwd=True,nbwd=1,exact=False,
                reflect_bounds=False):
    use_k = k > 0
    search_abs = False
    oh0,ow0,oh1,ow1 = 1,1,3,3
    nheads = 1
    anchor_self = False
    use_self = anchor_self
    search = dnls.search.init("prod_refine", k, ps, pt, ws_r, ws, nheads,
                              chnls=-1,dilation=dilation,
                              stride0=stride0, stride1=stride1,
                              reflect_bounds=reflect_bounds,use_k=use_k,
                              search_abs=search_abs,use_adj=True,
                              anchor_self=anchor_self,use_self=use_self,
                              exact=exact)
    return search

@register_method
def init_dnls_k(self,k=100,ps=7,pt=0,ws=21,ws_r=3,wt=0,stride0=4,stride1=1,
                dilation=1,rbwd=True,nbwd=1,exact=False,
                reflect_bounds=False):
    use_k = k > 0
    search_abs = False
    fflow,bflow = None,None
    oh0,ow0,oh1,ow1 = 1,1,3,3
    anchor_self = True
    use_self = anchor_self
    search = dnls.search.init("prod_with_index", fflow, bflow,
                              k, ps, pt, ws, wt,oh0, ow0, oh1, ow1, chnls=-1,
                              dilation=dilation, stride0=stride0,stride1=stride1,
                              reflect_bounds=reflect_bounds,use_k=use_k,
                              use_adj=True,search_abs=search_abs,
                              rbwd=rbwd,nbwd=nbwd,exact=exact,
                              anchor_self=anchor_self,use_self=use_self)
    return search

@register_method
def init_wpsum(self,ps=7,pt=0,dilation=1,reflect_bounds=False,
               rbwd=True,nbwd=1,exact=False,use_wpsum_patches=False):
    wpsum = dnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=0, w_off=0,
                                                dilation=dilation,
                                                reflect_bounds=reflect_bounds,
                                                adj=0, exact=exact,
                                                rbwd=rbwd,nbwd=nbwd)
    return wpsum

@register_method
def init_ifold(self,vshape,device):
    rbounds = self.search.reflect_bounds
    stride0,dil = self.search.stride0,self.search.dilation
    ifold = dnls.iFoldz(vshape,None,stride=stride0,dilation=dil,
                        adj=0,only_full=False,use_reflect=rbounds,device=device)
    return ifold

@register_method
def init_pfc(self,c_in,c_out,ps,stride):
    pfc = dnls.nn.init("pfc",c_in,c_out,ps,stride)
    return pfc

@register_method
def batching_info(self,vshape):
    B,T,C,H,W = vshape
    stride0 = self.stride0
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    npix = H * W
    ntotal = T * nH * nW
    if self.bs is None:
        div = 2 if npix >= (540 * 960) else 1
        nbatch = ntotal//(T*div)
    elif self.bs == -1:
        nbatch = ntotal
    else:
        nbatch = self.bs
    nbatches = (ntotal-1) // nbatch + 1
    return nbatch,nbatches,ntotal
