import multiprocessing as mp
import numpy as np
from scipy.stats import multivariate_normal
import torch

from utils.misc import overrides
from .data_util import DataUtil


class BlobsLoader():
  def __init__(self, dataset, batch_size, nb_cores):
    self.dataset = dataset
    self.batch_size = batch_size
    self.nb_cores = nb_cores
  
  def __iter__(self):
    def _producer(queue):
      batch_gen = self._gen_batches(self.batch_size)
      for batch in batch_gen:
        queue.put(batch, block=True)
      queue.put(None)
      queue.close()
  
    queue = mp.Queue(maxsize=50)
    pool = mp.Pool(self.nb_cores, _producer, (queue,))

    batch = queue.get()
    while batch is not None: 
      yield batch
      batch = queue.get()
  
  def _gen_batches(self, batch_size):
    imgs = np.zeros((batch_size, self.dataset.canvas_len, 
      self.dataset.canvas_len), dtype=np.float32)
    while True:
      for i in range(batch_size): 
        imgs[i] = self.dataset.make_rand_img()
      yield torch.tensor(imgs).unsqueeze(1)
  
  def __len__(self):
    return int(np.ceil(len(self.dataset) // self.batch_size))


class BlobsDataset():
  def __init__(self, canvas_len, std, nb_particles):
    self.canvas_len = canvas_len
    self.var = std**2
    self.margin = std
    self.nb_particles = nb_particles
  
  def make_rand_img(self):
    cx, cy = np.random.randint(self.margin, 
      self.canvas_len - self.margin + 1, 2)
    return self._make_img(cx, cy)

  def _make_img(self, cx, cy):
    shades, (xs, ys) = self._make_blob_coords(cx, cy)
    canvas = np.zeros((self.canvas_len, self.canvas_len), dtype=np.float32)
    canvas[xs, ys] = shades[:, np.newaxis]
    canvas = torch.tensor(canvas).unsqueeze(0)
    return canvas
      
  def _make_blob_coords(self, x, y):
    var = multivariate_normal(mean=[x, y], cov=[self.var, self.var])
    blob = var.rvs(self.nb_particles)
    
    blob_coords = np.round(blob).astype(np.int)
    too_small = blob_coords < 0
    too_big = blob_coords > self.canvas_len - 1
    valid_rows = np.logical_not(np.logical_or(too_small, too_big).any(1))

    blob_shades = BlobsDataset._normalize(
      var.pdf(blob))[valid_rows]
    blob_coords = blob_coords[valid_rows]
    return blob_shades, np.hsplit(blob_coords, 2)
  
  def __getitem__(self, idx):
    return self.make_rand_img()
  
  def __len__(self):
    return 1000000  # returns an artificial length
  
  @staticmethod
  def _normalize(x):
    """Normalize array so that all elements are between 0 and 1."""
    return (x - x.min()) / (x.max() - x.min())


class BlobsUtil(DataUtil):
  @overrides(DataUtil)
  def get_trainloader(self, batch_size):
    dataset = BlobsDataset(canvas_len=32, std=4, nb_particles=10000)
    return BlobsLoader(dataset, batch_size, nb_cores=4)

  @property
  @overrides(DataUtil)
  def testdata(self):
    return BlobsDataset(canvas_len=32, std=4, nb_particles=10000)

