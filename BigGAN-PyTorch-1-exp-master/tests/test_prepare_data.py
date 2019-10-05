import numpy as np
import os
import sys
import unittest
import argparse

from template_lib import utils
os.chdir('..')


class Prepare_data(unittest.TestCase):
  def test_Calculate_inception_moments_cifar10(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=0
        export PORT=6011
        export TIME_STR=1
        export PYTHONPATH=../../submodule:..
        python -c "import test_prepare_data; \
        test_prepare_data.Prepare_data().test_Calculate_inception_moments_cifar10()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

      # func name
    outdir = os.path.join('results/Prepare_data',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
                --config configs/prepare_data.yaml
                --command Calculate_inception_moments_cifar10
                """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from TOOLS import calculate_inception_moments
    calculate_inception_moments.create_inception_moments(args, myargs)
    input('End %s' % outdir)
    return

  def test_Calculate_inception_moments_Celeba64(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=0
        export PORT=6011
        export TIME_STR=1
        export PYTHONPATH=../../submodule:..
        python -c "import test_DCGAN; \
        test_DCGAN.Prepare_data().test_Calculate_inception_moments_Celeba64()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

      # func name
    outdir = os.path.join('results/Prepare_data',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
                --config ./configs/prepare_data.yaml
                --command Calculate_inception_moments_Celeba64
                --resume False --resume_path None
                --resume_root None
                """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from TOOLS import calculate_inception_moments
    calculate_inception_moments.create_inception_moments(args, myargs)
    input('End %s' % outdir)
    return

  def test_Check_inception_moments_Celeba64(self):
    old = np.load(os.path.expanduser(
      '~/ZhouPeng/code/biggan-pytorch/'
      'results/datasets/Celeba_align64_inception_moments.npz'))
    old_mu, old_sigma = old['mu'], old['sigma']
    new = np.load(os.path.expanduser(
      '~/.keras/BigGAN-PyTorch-1/Celeba64_inception_moments.npz'))
    new_mu, new_sigma = new['mu'], new['sigma']
    err_mu, err_sig = np.sum(new_mu - old_mu), np.sum(new_sigma - old_sigma)

    new1 = np.load(os.path.expanduser(
      '~/.keras/BigGAN-PyTorch-1/Celeba64_inception_moments.npz1.npz'))
    new_mu1, new_sigma1 = new1['mu'], new1['sigma']
    err_mu, err_sig = np.sum(new_mu - new_mu1), np.sum(new_sigma - new_sigma1)
    pass