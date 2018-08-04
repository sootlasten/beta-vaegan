from torch import optim

from utils.base_runner import get_common_parser, base_runner 
from models import VAE
from trainers.train_vae import Trainer


@base_runner
def run(img_dims, cont_dim, cat_dims, args):
  nets = {
    'vae': VAE(img_dims, cont_dim, cat_dims, args.temp)
  }
  optimizers = {
    'vae': optim.Adam(nets['vae'].parameters(), lr=args.eta)
  }
  return nets, optimizers


if __name__ == '__main__':
  parser = get_common_parser()
  args = parser.parse_args()
  run(args, Trainer)

