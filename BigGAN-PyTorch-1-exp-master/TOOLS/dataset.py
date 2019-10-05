import os
from random import Random

import utils

from template_lib.datasets import CelebA

def get_data_loaders(config):

  if config.dataset.lower() == 'celeba64':
    data_loader = CelebA.CelebA64(
      datadir=os.path.expanduser(config.datadir),
      batch_size=config.batch_size,
      shuffle=config.shuffle,
      num_workers=config.num_workers,
      seed=config.seed)
  elif config.dataset.lower().startswith('celebahq'):
    data_loader = CelebA.CelebaHQ(datadir=config.datadir,
                                  batch_size=config.batch_size,
                                  shuffle=config.shuffle,
                                  num_workers=config.num_workers,
                                  seed=config.seed)
  elif config.dataset == 'C10':
    data_loaders = utils.get_data_loaders(config=config, **config)
    data_loader = data_loaders[0]
  else:
    assert 0
  return data_loader


def get_dataset(config):
  if config.dataset == 'Celeba64':
    dataset = CelebA.CelebA64(
      datadir=os.path.expanduser(config.datadir),
      batch_size=config.batch_size,
      shuffle=config.shuffle,
      num_workers=config.num_workers,
      seed=config.seed,
      get_dataset=True)
  else:
    assert 0
  return dataset


class Partition(object):
  """ Dataset-like object, but only access a subset of it. """

  def __init__(self, data, index):
    self.data = data
    self.index = index

  def __len__(self):
    return len(self.index)

  def __getitem__(self, index):
    data_idx = self.index[index]
    return self.data[data_idx]


class DataPartitioner(object):
  """ Partitions a dataset into different chuncks. """

  def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
    self.data = data
    self.partitions = []
    rng = Random()
    rng.seed(seed)
    data_len = len(data)
    indexes = [x for x in range(0, data_len)]
    rng.shuffle(indexes)

    for idx, frac in enumerate(sizes):
      part_len = int(frac * data_len)
      if idx == len(sizes) - 1:
        self.partitions.append(indexes[0:])
      else:
        self.partitions.append(indexes[0:part_len])
      indexes = indexes[part_len:]

  def use(self, partition):
    return Partition(self.data, self.partitions[partition])
