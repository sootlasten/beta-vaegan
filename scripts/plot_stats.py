"""Takes an info file and plots the stats present."""
import ast
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_scals(scals_dict): 
  for key, val in scals_dict.items():
    plt.plot(val)
    plt.xlabel(key)
    plt.show()


def plot_seqs(seqs_dict):
  for key, val in seqs_dict.items():
    plt.plot(np.array(val))
    plt.xlabel(key)
    plt.show()


def populate_dicts(info_path, scals_dict, seqs_dict):
  with open(info_path, 'r') as f:
    for line in f:
      line = "".join(line.strip().split())  # removes all whitespaces
      parts = line.split(":")
      if len(parts) == 1: continue
      key, val = parts
      if key in scals_dict:
        scals_dict[key].append(float(val))
      if key in seqs_dict:
        seqs_dict[key].append(ast.literal_eval(val))
      continue 


def make_dict(keys):
  if len(keys):
    gen_empty_lists = lambda n: [[] for _ in range(n)]
    keys_list = keys.split(',')
    return dict(zip(keys_list, gen_empty_lists(len(keys_list))))
  return {}


def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--info-path', type=str, required=True,
                      help='path to the info.txt file')
  parser.add_argument('--scals-to-plot', type=str, default='',
                      help="a comma separated list of the names of the \
                            scalar-valued stats to plot")
  parser.add_argument('--seqs-to-plot', type=str, default='cont_kl', 
                      help="a comma separated list of the names of the \
                            sequence stats to plot (e.g. dimension-wise \
                            kl-s)")
  return parser.parse_args()


if __name__ == '__main__':
  args = parse()

  scals_dict = make_dict(args.scals_to_plot)
  seqs_dict = make_dict(args.seqs_to_plot)

  populate_dicts(args.info_path, scals_dict, seqs_dict)

  plot_scals(scals_dict)
  plot_seqs(seqs_dict)

