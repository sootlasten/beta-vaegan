import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.misc import overrides
from .data_util import DataUtil


DSPRITES_PATH = '/home/stensootla/projects/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'

class SpritesDataset(Dataset):
  def __init__(self):
    self.dataset_zip = np.load(DSPRITES_PATH, encoding='latin1')
    self.imgs = self.dataset_zip['imgs']
  
  def __len__(self):
    return len(self.imgs)
      
  def __getitem__(self, idx):
    img = self.imgs[idx].astype(np.float32)
    return torch.tensor(img).unsqueeze(0)


class SpritesUtil(DataUtil):
  @overrides(DataUtil)
  def get_trainloader(self, batch_size):
    sprite_dataset = SpritesDataset()
    return DataLoader(sprite_dataset, batch_size=batch_size, shuffle=True)

  @property
  @overrides(DataUtil)
  def testdata(self):
    return SpritesDataset()

