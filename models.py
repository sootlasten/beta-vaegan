import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical 

from utils.model_utils import *


def _encoder(img_dims, nb_out, norm, nonl, out_nonl):
  enc_layers = []
  enc_layers.extend(conv_block(img_dims[0], 64, norm, nonl))
  enc_layers.extend(conv_block(64, 128, norm, nonl))
  enc_layers.extend(conv_block(128, 256, norm, nonl))
  enc_layers.append(Flatten())
  enc_layers.extend(linear_block(8*8*256, 1024, norm, nonl))
  enc_layers.extend(linear_block(1024, nb_out, norm, nonl))
  return nn.Sequential(*enc_layers)


def _decoder(img_dims, nb_latents, norm, nonl, out_nonl):
  dec_layers = []
  dec_layers.extend(linear_block(nb_latents, 8*8*256, norm, nonl)),
  dec_layers.append(Reshape(-1, 256, 8, 8)),
  dec_layers.extend(deconv_block(256, 256, norm, nonl)),
  dec_layers.extend(deconv_block(256, 128, norm, nonl)),
  dec_layers.extend(deconv_block(128, 32, norm, nonl)),
  dec_layers.extend(conv_block(32, img_dims[0], norm, nonl, stride=1))
  return nn.Sequential(*dec_layers)


class VAE(nn.Module):
  def __init__(self, img_dims, cont_dim, cat_dims, temp):
    super(VAE, self).__init__()
    self.in_dim = img_dims

    self.cont_dim = [] if not cont_dim else 2*[cont_dim]
    self.cat_dims = cat_dims
    self.chunk_sizes = self.cont_dim + self.cat_dims
    self.temp = temp

    # encoder
    self.encoder = _encoder(img_dims, sum(self.chunk_sizes), 
      norm=True, nonl=nn.ReLU(), out_nonl=None)
    self.decoder = _decoder(img_dims, sum(self.chunk_sizes)-cont_dim,   
      norm=True, nonl=nn.ReLU(), out_nonl=nn.Tanh())
  
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
  def __init__(self, img_dims):
    super(Discriminator, self).__init__()
    self.extract_idx = 5 if img_dims[0] == 1 else 8

    # shares architecture with encoder for simplicity
    nonl = nn.ReLU()
    disc_layers = []
    disc_layers.extend(conv_block(img_dims[0], 32, False, nonl, stride=1))
    disc_layers.extend(conv_block(32, 128, True, nonl))
    disc_layers.extend(conv_block(128, 256, True, nonl))
    disc_layers.extend(conv_block(256, 256, True, nonl))
    disc_layers.append(Flatten())
    disc_layers.extend(linear_block(8*8*256, 512, True, nonl))
    disc_layers.extend(linear_block(512, 1, False, nn.Sigmoid()))
    self.layers = nn.Sequential(*disc_layers)
    self._init_parameters()
  
  def _init_parameters(self):
    for m in self.modules():
      if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if hasattr(m, "weight") and m.weight is not None \
            and m.weight.requires_grad:
          # init as original implementation
          scale = 1.0/np.sqrt(np.prod(m.weight.shape[1:]))
          scale /= np.sqrt(3)
          nn.init.uniform_(m.weight, -scale, scale)
        if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
          nn.init.constant_(m.bias, 0.0)

  def forward(self, x, full=True):
    featmap = None
    for i, layer in enumerate(self.layers):
      x = layer(x)
      if i == self.extract_idx:
        if not full: return None, x
        else: featmap = x
    return x, featmap

