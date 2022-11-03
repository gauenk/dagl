
# -- easy dict --
import random
import numpy as np
import torch as th

def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)

