from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.nn import functional as F

from utils.log_utils import Logger


def kl_gauss_unag(mu, logvar):
  kld = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
  return kld


def kl_cat_unag(logits):
  """The (unaggregated, i.e. not summed over dimensions yet) 
     KL divergence as per equation 22 in CONCRETE paper."""
  q_z = F.softmax(logits, dim=1)
  return q_z*(torch.log(q_z + 1e-20) - np.log(1 / logits.size(1)))


def sse_loss(source, target):
  return 0.5*(source - target).pow(2).mean(0).sum()


def overrides(interface_class):
  """Convenience decorator for later overriding the Trainer class."""
  def overrider(method):
    assert(method.__name__ in dir(interface_class))
    return method
  return overrider


class BaseTrainer(ABC):
  def __init__(self, args, nets, opt, dataloader, vis, device):
    self.args = args
    self.nets = nets
    self.opt = opt
    self.dataloader = dataloader
    self.vis = vis
    self.device = device
    self.logger = Logger(self.args.steps)
  
  def get_cap_loss(self, kl, step):
    cap = (self.args.cap_max - self.args.cap_min)* \
      step/self.args.cap_iters
    cap = min(cap, self.args.cap_max)
    return self.args.cap_coeff*torch.abs(cap - kl)

  @abstractmethod
  def put_in_work():
    pass

