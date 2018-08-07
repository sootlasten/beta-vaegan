import os
import numpy as np
from skimage.io import imread
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import cv2
from skimage import filters, transform

from utils.misc import overrides
from .data_util import DataUtil


IMGDIR = '/home/stensootla/projects/datasets/celeba/img_align_celeba/'
PARTITION_INFO_FILE = '/home/stensootla/projects/datasets/celeba/list_eval_partition.csv'


def _resize(img):
  rescale_size = 64
  bbox = (40, 218 - 30, 15, 178 - 15)
  img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
  # Smooth image before resize to avoid moire patterns
  scale = img.shape[0] / float(rescale_size)
  sigma = np.sqrt(scale) / 2.0
  img = filters.gaussian(img, sigma=sigma, multichannel=True)
  img = transform.resize(img, 
    (rescale_size, rescale_size, 3), order=3, mode="constant")
  img = (img*255).astype(np.uint8)
  return img


class CelebADataset(Dataset):
  def __init__(self, train=True, transform=None):
    self.img_paths = CelebADataset._get_img_paths(train)
    self.transform = transform
    
  @staticmethod
  def _get_img_paths(train):
    img_paths = []
    with open(PARTITION_INFO_FILE, 'r') as f:
      for line in f:
        line = line.strip()
        if '.jpg' in line:
          filename, part_idx = line.split(',')
          filepath = os.path.join(IMGDIR, filename)
          if train and int(part_idx) in (0, 1): 
            img_paths.append(filepath)
          elif not train and int(part_idx) == 2:
            img_paths.append(filepath)
    return img_paths
            
  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    data = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
    data = _resize(data)
    data = data.transpose(2, 0, 1)  # channel first
    data = data.astype("float32") / 127.5 - 1.0  # tanh
    return torch.tensor(data)


class CelebAUtil(DataUtil):
  @overrides(DataUtil)
  def get_trainloader(self, batch_size):
    celeba_data = CelebADataset()
    return DataLoader(celeba_data, batch_size=batch_size, 
      shuffle=True, num_workers=3)
  
  @property
  @overrides(DataUtil)
  def testdata(self):
    return CelebADataset(train=False)

