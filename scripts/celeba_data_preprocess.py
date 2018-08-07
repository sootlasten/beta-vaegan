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


def _resize(img):
  rescale_size = 64
  bbox = (40, 218 - 30, 15, 178 - 15)
  img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
  # Smooth image before resize to avoid moire patterns
  scale = img.shape[0] / float(rescale_size)
  sigma = numpy.sqrt(scale) / 2.0
  img = filters.gaussian(img, sigma=sigma, multichannel=True)
  img = transform.resize(img, 
    (rescale_size, rescale_size, 3), order=3, mode="constant")
  img = (img*255).astype(numpy.uint8)
  return img


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

