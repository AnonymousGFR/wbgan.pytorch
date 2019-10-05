import collections
from .dict_utils import trainer_dict

def init_train_dict():
  train_dict = collections.OrderedDict()
  train_dict['epoch_done'] = 0
  train_dict['batches_done'] = 0
  train_dict['best_FID'] = 9999
  return train_dict


def trainer_create(args, myargs):
  config = myargs.config.trainer
  myargs.logger.info('Create trainer: %s', config.type)
  trainer = trainer_dict[config.type](args=args, myargs=myargs)
  return trainer


def main(args, myargs):
  config = myargs.config.main
  logger = myargs.logger

  from template_lib.utils import seed_utils
  seed_utils.set_random_seed(config.seed)

  # Create train_dict
  train_dict = init_train_dict()
  myargs.checkpoint_dict['train_dict'] = train_dict

  # Create trainer
  trainer = trainer_create(args=args, myargs=myargs)

  if args.evaluate:
    trainer.evaluate()
    return

  if args.resume:
    logger.info('=> Resume from: %s', args.resume_path)
    loaded_state_dict = myargs.checkpoint.load_checkpoint(
      checkpoint_dict=myargs.checkpoint_dict,
      resumepath=args.resume_path)
    for key in train_dict:
      if key in loaded_state_dict['train_dict']:
        train_dict[key] = loaded_state_dict['train_dict'][key]

  # Load dataset
  trainer.dataset_load()

  for epoch in range(train_dict['epoch_done'], config.epochs):
    logger.info('epoch: [%d/%d]' % (epoch, config.epochs))

    trainer.train_one_epoch()

    train_dict['epoch_done'] += 1
    # test
    trainer.test()
  trainer.finalize()
