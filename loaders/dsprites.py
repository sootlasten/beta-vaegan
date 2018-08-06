import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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

def get_dsprites_dataloader(batch_size):
  sprite_dataset = SpritesDataset()
  return DataLoader(sprite_dataset, batch_size=batch_size, shuffle=True)

def get_dsprites_testdata():
  return SpritesDataset()

