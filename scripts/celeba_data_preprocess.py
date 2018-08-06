"""Resizes and crops CelebA 178x128 images to 64x64."""
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.misc import imresize
from PIL import Image


def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--source-dir', type=str, required=True,
                      help='path to directory where the original images reside')
  parser.add_argument('--target-dir', type=str, required=True,
                      help='path to directory where the resized images will be \
                            saved')
  return parser.parse_args()


def preprocess(args):
  if not os.path.isdir(args.target_dir):
    os.mkdir(args.target_dir)
  img_list = os.listdir(args.source_dir)

  for i, filename in enumerate(tqdm(img_list)):
    src_img_path = os.path.join(args.source_dir, filename)
    img = Image.open(src_img_path) \
            .resize((64, 78), Image.ANTIALIAS) \
            .crop((0, 7, 64, 64 + 7))
  
    target_img_path = os.path.join(args.target_dir, filename)
    with open(target_img_path, 'w') as f:
      img.save(f)


if __name__ == '__main__':
  args = parse()
  preprocess(args)

