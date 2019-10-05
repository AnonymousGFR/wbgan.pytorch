import os, pprint, sys
import argparse
from tensorboardX import SummaryWriter
from easydict import EasyDict

sys.path.insert(0, '../submodule')
from template_lib import utils
from template_lib.utils import config, modelarts_utils, logging_utils

from TOOLS import calculate_inception_moments
from DCGAN.trainer import run

exe_dict = {
  'CelebA64_dcgan_wgan_gp': run.train,
  'CelebA64_dcgan_wgan_gp_bound_sinkhorn': run.train,
}


def main():
  myargs = argparse.Namespace()
  parser = utils.args_parser.build_parser()
  args = parser.parse_args()

  config.setup_dirs_and_files(args=args)
  config.setup_logger_and_redirect_stdout(args.logfile, myargs)
  myargs.textlogger = logging_utils.TextLogger(
    log_root=args.textlogdir, reinitialize=(not args.resume),
    logstyle='%10.3f')
  print(pprint.pformat(vars(args)))

  config.setup_config(
    config_file=args.config, saved_config_file=args.configfile,
    myargs=myargs)
  myargs.writer = SummaryWriter(logdir=args.tbdir)

  modelarts_utils.modelarts_setup(args, myargs)

  config.setup_checkpoint(ckptdir=args.ckptdir, myargs=myargs)

  args = EasyDict(vars(args))
  myargs.config = EasyDict(myargs.config)

  exe_dict[args.command](args=args, myargs=myargs)

  return


if __name__ == '__main__':
  main()