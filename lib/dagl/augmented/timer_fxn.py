# -- imports --
import torch as th
from dagl.utils.timer import AggTimer

# -- separate class and logic --
from dagl.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)

@register_method
def update_timer(self,timer):
    # print(timer.names)
    for key in timer.names:
        if not(key in self.times.names):
            self.times[key] = [timer[key]]
        else:
            self.times[key].append(timer[key])

@register_method
def _reset_times(self):
    self.times = AggTimer()
