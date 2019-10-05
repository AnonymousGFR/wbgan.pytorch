import os
import torch.distributed as dist
from torch.multiprocessing import Process
from collections import OrderedDict
from pprint import pformat
from tensorboardX import SummaryWriter

from template_lib.utils import seed_utils

from . import exe_dict, trainer_dict

# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def train(args, myargs):
  try:
    rank = dist.get_rank()
    size = dist.get_world_size()
    args.rank = rank
    args.size = size
    if rank == 0:
      myargs.writer = SummaryWriter(logdir=args.tbdir)
  except:
    print('Do not use multiprocessing.')
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


def init_processes(rank, size, args, myargs, backend='nccl'):
  """ Initialize the distributed environment. """
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29500'
  dist.init_process_group(backend, rank=rank, world_size=size)
  train(args, myargs)


def train_dist(args, myargs):
  myargs.writer.close()
  size = args.world_size
  processes = []
  for rank in range(size):
    p = Process(target=init_processes, args=(rank, size, args, myargs))
    p.start()
    processes.append(p)

  for p in processes:
    p.join()



