import os
import pprint
from easydict import EasyDict
from collections import OrderedDict
from pprint import pformat

from template_lib.utils import seed_utils

from . import exe_dict, parser_dict, run_dict, trainer_dict

def ImageNet128_make_hdf5(args, myargs):
  import make_hdf5
  parser = make_hdf5.prepare_parser()
  config = vars(parser.parse_args())
  config = EasyDict(config)

  config1 = myargs.config.ImageNet128_make_hdf5
  for k, v in config1.items():
    setattr(config, k, v)
  config.data_root = os.path.expanduser(config.data_root)
  myargs.logger.info(pprint.pformat(config))
  make_hdf5.run(config, myargs=myargs)
  pass

def ImageNet128_calculate_inception_moments(args, myargs):
  import calculate_inception_moments
  parser = calculate_inception_moments.prepare_parser()
  config = vars(parser.parse_args())
  config = EasyDict(config)

  config1 = myargs.config.ImageNet128_calculate_inception_moments
  for k, v in config1.items():
    setattr(config, k, v)
  config.data_root = os.path.expanduser(config.data_root)
  print(pprint.pformat(config))
  calculate_inception_moments.run(config, myargs)
  pass


def train(args, myargs):
  parser = parser_dict[args.command]()
  config = vars(parser.parse_args())
  config = EasyDict(config)

  config1 = getattr(myargs.config, args.command)
  for k, v in config1.items():
    if not hasattr(config, k):
      print('=> config does not have attr [%s: %s]' % (k, v))

  for k, v in config1.items():
    setattr(config, k, v)
  run_dict[args.command](config, args, myargs)
  pass


def run_trainer(args, myargs):
  myargs.config = getattr(myargs.config, args.command)
  config = myargs.config
  seed_utils.set_random_seed(config.seed)
  trainer = trainer_dict[args.command](args=args, myargs=myargs)

  if args.evaluate:
    trainer.evaluate()
    return

  if args.resume:
    trainer.resume()
  elif args.finetune:
    trainer.finetune()

  # Load dataset
  trainer.dataset_load()

  trainer.train()


def main(args, myargs):
  exe = exe_dict[args.command]
  exec('%s(args, myargs)'%exe)