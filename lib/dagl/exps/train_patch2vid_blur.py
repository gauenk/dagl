"""

Training the DAGL model
on deblurring and using the
"patch2linear" setup so the attn
module is computed as a video


"""

from easydict import EasyDict as edict
import cache_io

def load(mode="base"):
    if mode == "base":
        return load_base()
    elif mode == "train":
        return load_train()
    else:
        return load_test()

def load_base():
    cfg = edict()
    cfg.nsamples_at_testing = 1
    cfg.checkpoint_dir = "./output/checkpoints/"
    cfg.bw = True
    cfg.name = "train_patch2vid_blur"
    return cfg

def load_train():
    expl = {}
    expl['ws'] = [21]
    expl['wt'] = [3]
    expl['k_a'] = [100]
    expl['k_s'] = [100]
    exps = cache_io.mesh_pydicts(expl)
    return exps

def load_test():
    expl = {}
    expl['ws'] = [21]
    expl['wt'] = [3]
    expl['k_a'] = [100]
    expl['k_s'] = [100]
    exps = cache_io.mesh_pydicts(expl)
    return exps

