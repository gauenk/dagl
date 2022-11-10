
# -- imports --
import numpy as np
import torch as th

# -- separate class and logic --
from dagl.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)

@register_method
def format_inds(self,*inds_list):
    if not(self.return_inds): return None
    else: return th.stack(inds_list,0)

@register_method
def clear_inds_buffer(self):
    self.inds_buffer = []
    # if not(self.return_inds): return
    # self.inds_buffer = []

@register_method
def get_inds_buffer(self):
    if not(self.use_inds_buffer):
        return None
    else:
        ishape = self.inds_buffer.shape
        dim1 = np.product(ishape[:-3])
        ishape = (dim1,)+ishape[-3:]
        # print("dim1,ishape: ",dim1,ishape)
        return self.inds_buffer.view(ishape)

@register_method
def update_inds_buffer(self,inds):
    if not(self.use_inds_buffer): return
    if len(self.inds_buffer) == 0:
        self.inds_buffer = inds[None,:]
    else:
        # print("update: ",self.inds_buffer.shape,inds.shape)
        self.inds_buffer = th.cat([self.inds_buffer,inds[None,:]],0)
