from torch import nn


class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)


class Reshape(nn.Module):
  def __init__(self, *args):
    super(Reshape, self).__init__()
    self.shape = args

  def forward(self, x):
    return x.view(self.shape)


def conv_block(in_c, out_c, norm, nonl, kernel_size=5, stride=2, padding=2):
  layers = [nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)]
  if norm: layers.append(nn.BatchNorm2d(out_c))
  if nonl: layers.append(nonl)
  return layers


def deconv_block(in_c, out_c, norm, nonl, kernel_size=5, stride=2, padding=2):
  layers = [nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1)] 
  if norm: layers.append(nn.BatchNorm2d(out_c))
  if nonl: layers.append(nonl)
  return layers


def linear_block(in_dim, out_dim, norm, nonl):
  layers = [nn.Linear(in_dim, out_dim)]
  if norm: layers.append(nn.BatchNorm1d(out_dim))
  if nonl: layers.append(nonl)
  return layers

