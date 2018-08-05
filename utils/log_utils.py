import os
from collections import OrderedDict
import numpy as np


LOG_FILENAME = 'info.txt'

class Logger():
  def __init__(self, logdir, total_steps):
    self.stats = OrderedDict()
    self.total_steps = total_steps
    self.logfile_path = os.path.join(logdir, LOG_FILENAME)
  
  def log_val(self, key, val):
    if key in self.stats: 
      self.stats[key] = self.stats[key]*0.99 + val*0.01
    else: self.stats[key] = val
  
  def print(self, step):
    info_str = self._gen_info_str(step)
    print(info_str)

  def save(self, step):
    info_str = self._gen_info_str(step)
    with open(self.logfile_path, 'a') as f: 
      f.write(info_str + '\n')
  
  def _gen_info_str(self, step):
    info_str = "step: {}/{}\n".format(step, self.total_steps)
    for k, v in self.stats.items():
      if isinstance(v, np.ndarray):
        s = "[{}]".format(", ".join("{:.4}".format(e) for e in v))
      else: s = "{:.4f}".format(v)
      info_str += "{}: {}\n".format(k, s)
    return info_str
