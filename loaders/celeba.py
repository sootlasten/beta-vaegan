import os
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from utils.misc import overrides
from .data_util import DataUtil


IMGDIR = '/home/stensootla/projects/datasets/celeba/resized/'
PARTITION_INFO_FILE = '/home/stensootla/projects/datasets/celeba/list_eval_partition.csv'

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
    img_path = self.img_paths[idx]
    img = imread(img_path)
    if self.transform:
      img = self.transform(img)
    return img


class CelebAUtil(DataUtil):
  def __init__(self):
    self.transforms = transforms.ToTensor()

  @overrides(DataUtil)
  def get_trainloader(self, batch_size):
    celeba_data = CelebADataset(transform=self.transforms)
    return DataLoader(celeba_data, batch_size=batch_size, shuffle=True)
  
  @property
  @overrides(DataUtil)
  def testdata(self):
    return CelebADataset(train=False, transform=self.transforms)

