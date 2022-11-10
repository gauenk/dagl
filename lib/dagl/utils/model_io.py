
import numpy as np

def select_sigma(sigma):
    sigmas = np.array([15, 25, 50])
    msigma = np.argmin((sigmas - sigma)**2)
    return sigmas[msigma]

def remove_state_prefix(prefix,state):
    if prefix == "": return state
    nskip = len(prefix)
    state_new = {}
    for key,val in state.items():
        key_m = key[nskip:]
        state_new[key_m] = val
    return state_new

