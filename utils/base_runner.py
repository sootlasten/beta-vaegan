"""
A base module that is to be extended by specific run modules
for specific models.
"""
import os
import shutil
import argparse
import random
from functools import reduce, wraps
import torch

from loaders.dsprites import get_dsprites_dataloader
from loaders.mnist import get_mnist_dataloader
from loaders.blobs import get_blobs_dataloader


DATASETS = {'mnist': [(1, 32, 32), get_mnist_dataloader],
            'dsprites': [(1, 64, 64), get_dsprites_dataloader],
            'blobs': [(1, 32, 32), get_blobs_dataloader]}


def _get_dataset(data_arg):
  """Checks if the given dataset is available. If yes, returns
     the input dimensions and dataloader."""
  if data_arg not in DATASETS:
    raise ValueError("Dataset not available!")
  return DATASETS[data_arg]


def _check_dim_args(cont_dim, cat_dims):
  if cat_dims == '0': cat_dims_list = []
  else: cat_dims_list = list(map(lambda x: int(x), cat_dims.split(',')))

  if not len(cat_dims_list) and not cont_dim: 
    raise ValueError("Model should have some latents!")
  if cont_dim < 0:
    raise ValueError("Negative continuous dimensions!")
  if not reduce(lambda x, y: x and y > 1, cat_dims_list, True):
    raise ValueError("Every categorical variable should at least have 2 dims!")

  return cont_dim, cat_dims_list


def get_common_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--eta', type=float, default=1e-4,
                      help='learning rate for VAE')
  parser.add_argument('--cap-coeff', type=float, default=1000,
                      help='capacity constraint coefficient')
  parser.add_argument('--cap-min', type=float, default=0,
                      help='min capacity for KL')
  parser.add_argument('--cap-max', type=float, default=25,
                      help='max capacity for KL')
  parser.add_argument('--cap-iters', type=int, default=100000,
                      help='number of iters to increase the capacity over')
  parser.add_argument('--temp', type=float, default=0.1,
                      help='temperature for gumbel-softmax')
  parser.add_argument('--cont-dim', type=int, default=10,
                      help='dimension of the gaussian latent')
  parser.add_argument('--cat-dims', type=str, default='10',
                      help='A comma-separated list of the \
                            dimensions of categorical variables. \
                            0 if no cat variables.')
  parser.add_argument('--dataset', type=str, required=True,
                      help='The dataset to use for training \
                           (blobs | dsprites | mnist | celeba)')
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--steps', type=int, default=100000,
                      help='number of batches to train for')
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--nb-test', type=int, default=10,
                      help='number of test samples to visualize the recons of')
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--log-interval', type=int, default=10)
  parser.add_argument('--save-interval', type=int, default=200)
  parser.add_argument('--logdir', type=str, default='results')
  return parser


def base_runner(setup_models):
  @wraps(setup_models)
  def wrapper(args, trainer):
    torch.manual_seed(args.seed)
    
    # device placement
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    # data loader
    img_dims, dataloader = _get_dataset(args.dataset)
    dataloader = dataloader(args.batch_size)

    # test images to reconstruct during training
    dataset_size = len(dataloader.dataset)
    indices = [3] + random.sample(range(1, dataset_size), args.nb_test-1)
    testimgs = torch.empty(args.nb_test, *img_dims).to(device)
    for i, img_idx in enumerate(indices): 
      testimgs[i, 0] = torch.tensor(dataloader.dataset[img_idx])
    
    # models and optimizers
    cont_dim, cat_dims = _check_dim_args(args.cont_dim, args.cat_dims)
    nets, optimizers = setup_models(img_dims, cont_dim, cat_dims, args)
    for net in nets.values(): net.to(device)
    
    # log directory
    if os.path.isdir(args.logdir): shutil.rmtree(args.logdir)
    os.makedirs(args.logdir)

    # train
    trainer = trainer(args, nets, optimizers, dataloader, testimgs, device)
    trainer.put_in_work()
  return wrapper
