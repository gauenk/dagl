
# -- code api --
from . import augmented
from . import original
from . import configs
from . import lightning
from . import flow
from . import utils
from .utils import select_sigma
from .utils import optional

# -- publication api --
from . import cvpr23

# -- model api --
# from .original import extract_model_config as extract_model_config_orig
# from .augmented import extract_model_config as extract_model_config_aug
# from .augmented import extract_io_config,extract_search_config
# from .augmented import extract_arch_config,extract_model_config

def load_model(cfg):
    mtype = optional(cfg,'model_type','augmented')
    if mtype == "augmented":
        return augmented.load_model(cfg)
    elif mtype == "original":
        return original.load_model(cfg)
    else:
        raise ValueError(f"Uknown model type [{mtype}]")

def extract_model_config(cfg):
    mtype = optional(cfg,'model_type','augmented')
    if mtype == "augmented":
        return augmented.extract_model_config(cfg)
    elif mtype == "original":
        return original.extract_model_config(cfg)
    else:
        raise ValueError(f"Uknown model type [{mtype}]")
