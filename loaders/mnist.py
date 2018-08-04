import os
import errno
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.datasets.utils as vision_utils
from torchvision.datasets.mnist import read_image_file, get_int


class MNIST(data.Dataset):
  url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
  raw_folder = 'raw'
  processed_folder = 'processed'
  imgs_file = 'imgs.pt'
  
  def __init__(self, root, transform=None, download=False):
    self.root = os.path.expanduser(root)
    self.transform = transform

    if download:
      self.download()

    if not self._check_exists():
      raise RuntimeError('Dataset not found. Use download=True to download it.')

    filepath = os.path.join(self.root, self.processed_folder, self.imgs_file)
    self.imgs = torch.load(filepath)[0]
    
  def __getitem__(self, idx):
    x = self.imgs[idx]
    x = Image.fromarray(x.numpy(), mode='L')
    if self.transform is not None:
      x = self.transform(x)
    return x.numpy().squeeze()

  def __len__(self):
    return len(self.imgs)
      
  def _check_exists(self):
    path = os.path.join(self.root, self.processed_folder, self.imgs_file)
    return os.path.exists(path)

  def download(self):
    """Download the MNIST data if it doesn't exist in processed_folder already."""
    from six.moves import urllib
    import gzip

    if self._check_exists(): return

    # download files
    try:
      os.makedirs(os.path.join(self.root, self.raw_folder))
      os.makedirs(os.path.join(self.root, self.processed_folder))
    except OSError as e:
      if e.errno == errno.EEXIST:
        pass
      else:
        raise

    filename = self.url.rpartition('/')[2]
    root_path = os.path.join(self.root, self.raw_folder)
    vision_utils.download_url(self.url, root=root_path, filename=filename, md5=None)
    file_path = os.path.join(self.root, self.raw_folder, filename)
    with open(file_path.replace('.gz', ''), 'wb') as out_f, \
            gzip.GzipFile(file_path) as zip_f:
      out_f.write(zip_f.read())
    os.unlink(file_path)

    # process and save as torch files
    print('Processing...')

    raw_path = os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')
    imgs_set = (read_image_file(raw_path),)
    proc_path = os.path.join(self.root, self.processed_folder, self.imgs_file)
    with open(proc_path, 'wb') as f:
      torch.save(imgs_set, f)

    print('Done!')


def get_mnist_dataloader(batch_size, shuffle=True, path_to_data='data'):
  all_transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
  ])
  mnist = MNIST(path_to_data, download=True, transform=all_transforms)
  return DataLoader(mnist, batch_size=batch_size, shuffle=True)


def get_mnist_test():
  all_transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
  ])
  test_data = datasets.MNIST('../mnist', train=False,
    transform=all_transforms, download=True)
  return test_data

