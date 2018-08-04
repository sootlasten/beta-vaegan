import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical 

from utils.model_utils import *


def _encoder(img_dims, nb_latents, norm, nonl, out_nonl):
  enc_layers = []
  enc_layers.extend(conv_block(1, 32, norm, nonl))
  enc_layers.extend(conv_block(32, 32, norm, nonl))
  enc_layers.extend(conv_block(32, 64, norm, nonl))
  if img_dims[1:] == (64, 64): 
    enc_layers.extend(conv_block(64, 64, norm, nonl))
  enc_layers.append(Flatten())
  enc_layers.extend(linear_block(64*4*4, 128, norm, nonl))
  enc_layers.extend(linear_block(128, nb_latents, False, out_nonl))
  return nn.Sequential(*enc_layers)


def _decoder(img_dims, nb_latents, norm, nonl, out_nonl):
  dec_layers = []
  dec_layers.extend(linear_block(nb_latents, 128, norm, nonl)),
  dec_layers.extend(linear_block(128, 64*4*4, norm, nonl)),
  dec_layers.append(Reshape(-1, 64, 4, 4)),
  if img_dims[1:] == (64, 64):
    dec_layers.extend(deconv_block(64, 64, norm, nonl)),
  dec_layers.extend(deconv_block(64, 32, norm, nonl)),
  dec_layers.extend(deconv_block(32, 32, norm, nonl)),
  dec_layers.extend(deconv_block(32, 1, False, out_nonl))
  return nn.Sequential(*dec_layers)


class VAE(nn.Module):
  def __init__(self, img_dims, cont_dim, cat_dims, temp):
    super(VAE, self).__init__()
  
    self.cont_dim = [] if not cont_dim else 2*[cont_dim]
    self.cat_dims = cat_dims
    self.chunk_sizes = self.cont_dim + self.cat_dims
    self.temp = temp

    self.encoder = _encoder(img_dims, sum(self.chunk_sizes), 
      norm=False, nonl=nn.ReLU(), out_nonl=None)
    self.decoder = _decoder(img_dims, sum(self.chunk_sizes)-cont_dim,      norm=False, nonl=nn.ReLU(), out_nonl=nn.Sigmoid())
  
  @property
  def device(self):
    return next(self.parameters()).device
    
  def _get_dists_params(self, x):
    """Returns the parameters that the encoder predicts."""
    out = self.encoder(x).split(self.chunk_sizes, dim=1)
    params = {}; i = 0
    if self.cont_dim: i += 2; params['cont'] = out[:i]
    if self.cat_dims: params['cat'] = out[i:]
    return params
    
  def _reparam_gauss(self, mu, logvar):
    if self.training:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_().requires_grad_()
        return eps.mul(std).add_(mu)
    else:
        return mu
  
  def sample(self, n):
    """Samples n datapoints from the prior and reconstructs them."""
    z = []
    if len(self.cont_dim):
      d = self.cont_dim[0]
      z_cont = self._reparam_gauss(torch.zeros(n, d), torch.ones(n, d))
      z.append(z_cont)
    for cat_dim in self.cat_dims:
      z_cat = torch.zeros(n, cat_dim)
      indices = Categorical(torch.ones(n, cat_dim) / cat_dim).sample()
      z_cat[torch.arange(n, out=torch.LongTensor()), indices] = 1
      # z_cat = F.gumbel_softmax(torch.ones(n, cat_dim), tau=self.temp) 
      z.append(z_cat)
    z = torch.cat(z, dim=1).to(self.device)
    return self.decoder(z)
  
  def forward(self, x, decode=True):
    params = self._get_dists_params(x)
    zs = []
    if 'cont' in params.keys():
      zs = [self._reparam_gauss(*params['cont'])]
    if 'cat' in params.keys():
      for logits in params['cat']:
        if self.training: zs.append(F.gumbel_softmax(logits, tau=self.temp))
        else: zs.append(F.gumbel_softmax(logits, tau=self.temp, hard=True))
    z = torch.cat(zs, dim=1)
    
    if decode: recon = self.decoder(z)
    else: recon = None
    return recon, z, params


class Discriminator(nn.Module):
  def __init__(self, img_dims, nb_latents):
    super(Discriminator, self).__init__()
    
    # shares architecture with VAE for simplicity
    self.encoder = _encoder(img_dims, nb_latents, norm=True, 
      nonl=nn.ReLU(), out_nonl=None)
    self.decoder = _decoder(img_dims, nb_latents, norm=True,
      nonl=nn.ReLU(), out_nonl=None)
      
  def forward(self, x, extract_idx=4, full=True):
    featmap = None
    for i, layer in enumerate(self.encoder):
      x = layer(x)
      if i == extract_idx:
        if not full: return None, x
        else: featmap = x
    return self.decoder(x), featmap

