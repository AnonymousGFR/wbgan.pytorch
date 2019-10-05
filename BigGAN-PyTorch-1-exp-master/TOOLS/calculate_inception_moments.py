import os
from collections import OrderedDict
from pprint import pformat
import torch.nn.functional as F
import numpy as np
import tqdm
import torch
import unittest
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

from template_lib.gans import inception_utils

from . import dataset


def create_inception_moments(args, myargs):
  config = getattr(myargs.config, args.command)
  loaders = dataset.get_data_loaders(config=config.loader)
  # Load inception net
  net = inception_utils.load_inception_net(parallel=config.parallel)
  pool, logits, labels = [], [], []
  device = 'cuda'
  for i, (x, y) in enumerate(tqdm.tqdm(loaders, file=myargs.stdout)):
    x = x.to(device)
    with torch.no_grad():
      pool_val, logits_val = net(x)
      pool += [np.asarray(pool_val.cpu())]
      logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
      labels += [np.asarray(y.cpu())]

  pool, logits, labels = [np.concatenate(item, 0) for item in \
                          [pool, logits, labels]]
  # uncomment to save pool, logits, and labels to disk
  # logger.info('Saving pool, logits, and labels to disk...')
  # np.savez(config['dataset']+'_inception_activations.npz',
  #           {'pool': pool, 'logits': logits, 'labels': labels})
  # Calculate inception metrics and report them
  print('Calculating inception metrics...')
  IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
  print('Training data from dataset %s has IS of %5.5f +/- %5.5f'
        % (config.loader.dataset, IS_mean, IS_std))
  # Prepare mu and sigma, save to disk. Remove "hdf5" by default
  # (the FID code also knows to strip "hdf5")
  print('Calculating means and covariances...')
  mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
  print('Saving calculated means and covariances to: %s' \
        %config.saved_inception_moments)
  saved_inception_moments = os.path.expanduser(config.saved_inception_moments)
  base_dir = os.path.dirname(saved_inception_moments)
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
  np.savez(saved_inception_moments, **{'mu': mu, 'sigma': sigma})


def test_Check_MomentFile_Celeba64():
  old = np.load('/cluster/home/it_stu39/ZhouPeng/code/biggan-pytorch/'
                'results/datasets/Celeba_align64_inception_moments.npz')
  old_mu, old_sigma = old['mu'], old['sigma']
  new = np.load(os.path.expanduser(
    '~/.keras/BigGAN-PyTorch-1/Celeba64_inception_moments.npz'))
  new_mu, new_sigma = new['mu'], new['sigma']
  err_mu, err_sig = np.sum(new_mu - old_mu), np.sum(new_sigma - old_sigma)
  pass





