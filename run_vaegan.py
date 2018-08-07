from torch import optim

from utils.base_runner import get_common_parser, base_runner 
from models import VAE, Discriminator
from trainers.train_vaegan import Trainer


@base_runner
def run(img_dims, cont_dim, cat_dims, args):
  nets = {
    'vae': VAE(img_dims, cont_dim, cat_dims, args.temp),
    'disc': Discriminator(img_dims)
  }
  optimizers = {
    'enc': optim.RMSprop(nets['vae'].encoder.parameters(), lr=args.eta),
    'dec': optim.RMSprop(nets['vae'].decoder.parameters(), lr=args.eta),
    'disc': optim.RMSprop(nets['disc'].parameters(), lr=args.eta)
  }
  return nets, optimizers


def get_args(parser):
  parser.add_argument('--fw-coeff', type=float, default=1e-6)
  return parser.parse_args()


if __name__ == '__main__':
  parser = get_common_parser()
  args = get_args(parser)
  run(args, Trainer)

