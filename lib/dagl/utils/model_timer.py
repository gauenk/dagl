"""

Add timer for model

"""
from functools import partial
from .timer import ExpTimer,TimeIt

def hook_start(timer,layer_name,module, input):
    timer.sync_start(layer_name)

def hook_stop(timer,layer_name,module, input, output):
    timer.sync_stop(layer_name)

def hook_timer_to_model(model):

    # -- init timer --
    timer = ExpTimer()

    # -- add to registry --
    model.register_forward_pre_hook( partial(hook_start, timer, "total") )
    model.register_forward_hook( partial(hook_stop, timer, "total") )
    for lname,layer in model.named_children():
        # layer.register_forward_pre_hook( partial(hook_start, timer, lname) )
        # layer.register_forward_hook( partial(hook_stop, timer, lname) )
        if lname == "body":
            for lname_b,layer_b in layer.named_children():
                # layer_b.register_forward_pre_hook( partial(hook_start,timer,lname_b) )
                # layer_b.register_forward_hook( partial(hook_stop,timer,lname_b) )
                # print("lname_b: ",lname_b,lname_b == "7")
                if lname_b == "8":
                    for lname_c,layer_c in layer_b.named_children():
                        # print("lname_c: ",lname_c)
                        layer_c.register_forward_pre_hook( partial(hook_start,
                                                                   timer, lname_c) )
                        layer_c.register_forward_hook( partial(hook_stop,
                                                               timer, lname_c) )

    # -- return timer --
    return timer

