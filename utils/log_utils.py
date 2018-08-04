import numpy as np
from collections import OrderedDict


class Logger():
  def __init__(self, total_steps):
    self.stats = OrderedDict()
    self.total_steps = total_steps
  
  def log_val(self, key, val):
    if key in self.stats: 
      self.stats[key] = self.stats[key]*0.99 + val*0.01
    else: self.stats[key] = val
  
  def print(self, step):
    print('step: {}/{}'.format(step, self.total_steps))
    for k, v in self.stats.items():
      if isinstance(v, np.ndarray):
        s = "[{}]".format(", ".join("{:.4}".format(e) for e in v))
      else: s = "{:.4f}".format(v)
      print("{}: {}".format(k, s))
    print()

