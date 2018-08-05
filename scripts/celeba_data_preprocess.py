"""Resizes and crops CelebA 178x128 images to 64x64."""
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.misc import imresize
from PIL import Image


IMGDIR = '/home/stensootla/projects/celeba/img_align_celeba/'
SAVE_IMGDIR = '/home/stensootla/projects/celeba/resized/'


if not os.path.isdir(SAVE_IMGDIR):
  os.mkdir(SAVE_IMGDIR)
img_list = os.listdir(IMGDIR)

for i, filename in enumerate(tqdm(img_list)):
  img_path = os.path.join(IMGDIR, filename)
  img = Image.open(img_path) \
          .resize((64, 78), Image.ANTIALIAS) \
          .crop((0, 7, 64, 64 + 7))
  
  save_path = os.path.join(SAVE_IMGDIR, filename)
  with open(save_path, 'w') as f:
    img.save(f)
  
