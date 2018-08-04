from collections import OrderedDict
import numpy as np
import torch
from torchvision.utils import save_image


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


def traverse(model, imgs, save_path):
  """Reconstructs all images in 'imgs' and traverses the first image."""
  model.eval()
  recons, z, dist_params = model(imgs)
  canvas = torch.cat((imgs, recons), dim=1)
  nb_cols = len(imgs)
    
  # traverse continuous 
  if 'cont' in dist_params:
    first_mu = dist_params['cont'][0][0]
    cont_dim = len(first_mu)
    latents = torch.cat((first_mu, z[0, cont_dim:]))
    
    trav_range = torch.linspace(-3, 3, nb_cols)
    for i in range(cont_dim):
      temp_latents = latents.repeat(nb_cols, 1)
      temp_latents[:, i] = trav_range
      recons = model.decoder(temp_latents)
      canvas = torch.cat((canvas, recons), dim=1)
  else:
    cont_dim = 0
    latents = z[0] 
  
  # traverse categorical
  if 'cat' in dist_params:
    for i, d in enumerate(model.cat_dims):
      si = cont_dim + sum(model.cat_dims[:i])
      d -= max(0, d - nb_cols)  # cut traversals off if not enough cols
      temp_latents = latents.repeat(d, 1)
      temp_latents[:, si: si + d] = torch.eye(d)
      recon = model.decoder(temp_latents)
      row = torch.ones(nb_cols, *recon.shape[1:]).to(model.device)
      row[:d] = recon
      canvas = torch.cat((canvas, row), dim=1)
  
  img_size = canvas.shape[2:]
  canvas = canvas.transpose(0, 1).contiguous().view(-1, 1, *img_size)
  save_image(canvas, save_path, nrow=nb_cols, pad_value=1)
  model.train()

