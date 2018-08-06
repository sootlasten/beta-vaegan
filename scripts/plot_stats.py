"""Takes an info file and plots the stats present."""
import argparse
import matplotlib.pyplot as plt
import numpy as np


def parse():
  parser = argparse.ArgumentParser()
  parser.add_arugment('--info-path', type=str, required=True,
                      help='path to the info.txt file')
  return parser.parse_args()


if __name__ == '__main__':
  # TODO!
  pass

