"""
Convert model state layers in CE model
   from Linear to Convolution
"""

def convert_lin2conv(state,run_fxn):
    if run_fxn is False: return
    for key,val in state:
        print(key)
