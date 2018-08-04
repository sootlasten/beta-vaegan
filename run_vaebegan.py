from torch import optim

from utils.base_runner import get_common_parser, base_runner 
from models import VAE, Discriminator
from trainers.train_vaebegan import Trainer


@base_runner
def run(img_dims, cont_dim, cat_dims, args):
  nets = {
    'vae': VAE(img_dims, cont_dim, cat_dims, args.temp),
    'disc': Discriminator(img_dims, cont_dim + sum(cat_dims))
  }
  optimizers = {
    'enc': optim.Adam(nets['vae'].encoder.parameters(), lr=args.eta),
    'dec': optim.Adam(nets['vae'].decoder.parameters(), lr=args.eta),
    'disc': optim.Adam(nets['disc'].parameters(), lr=args.eta_disc)
  }
  return nets, optimizers


def get_args(parser):
  parser.add_argument('--eta-disc', type=float, default=1e-4)
  parser.add_argument('--fw-coeff', type=float, default=1e-6)
  parser.add_argument('--k', type=float, default=0,
                      help='controls how much emphasis is put on L(G(z_d))')
  parser.add_argument('--lambda_k', type=float, default=1e-3,
                      help='learning rate for k')
  parser.add_argument('--gamma', type=float, default=0.75,
                      help='balances the effort allocated to the \
                            generator and discriminator so that \
                            neither wins over the other. Lower \
                            values lead to lower image diversity.')
  return parser.parse_args()


if __name__ == '__main__':
  parser = get_common_parser()
  args = get_args(parser)
  run(args, Trainer)

