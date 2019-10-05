import functools
import collections
import numpy as np
import tqdm
import PIL.Image as Image
import torch
from torch.autograd import grad
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from . import models_64x64
from . import utils

from TOOLS import inception_utils, gan_utils, ema_model


def seed_rng(manualSeed):
  np.random.seed(manualSeed)

  torch.manual_seed(manualSeed)
  # if you are suing GPU
  torch.cuda.manual_seed(manualSeed)
  torch.cuda.manual_seed_all(manualSeed)

  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = False


def init_train_dict():
  train_dict = collections.OrderedDict()
  train_dict['epoch_done'] = 0
  train_dict['batches_done'] = 0
  train_dict['best_FID'] = 9999
  return train_dict


def model_create(train_dict, args, myargs):
  """ model """
  D = models_64x64.DiscriminatorWGANGP(3)
  G = models_64x64.Generator(args.z_dim)
  G_ema = models_64x64.Generator(args.z_dim)
  ema = ema_model.EMA(G, G_ema, decay=0.9999, start_itr=args.ema_start)
  utils.cuda([D, G, G_ema])

  myargs.checkpoint_dict['D'] = D
  myargs.checkpoint_dict['G'] = G
  myargs.checkpoint_dict['G_ema'] = G_ema

  myargs.logger.info_msg('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D]]))

  return G, D, G_ema, ema


def dataset_load(datadir, batch_size, num_workers, logger, **kwargs):
  logger.info('=> Load dataset in: %s', datadir)
  """ data """
  crop_size = 108
  re_size = 64
  offset_height = (218 - crop_size) // 2
  offset_width = (178 - crop_size) // 2
  crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(crop),
     transforms.ToPILImage(),
     transforms.Scale(size=(re_size, re_size), interpolation=Image.BICUBIC),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

  imagenet_data = dsets.ImageFolder(datadir, transform=transform)

  def _init_fn(worker_id):
    np.random.seed(kwargs['seed'] + worker_id)

  data_loader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            worker_init_fn=_init_fn)
  return data_loader


def inception_metrics_func_create(args, myargs):
  myargs.logger.info('=> Load inception moments: %s', args.saved_inception_moments)
  inception_metrics = inception_utils.InceptionMetrics(saved_inception_moments=args.saved_inception_moments)

  inception_metrics = functools.partial(inception_metrics, num_inception_images=args.num_inception_images,
                                        num_splits=10, prints=True)

  return inception_metrics


def train_func_create(args, logger):
  # create train_func
  logger.info('=> Create train func: %s', args.train_func)
  if args.train_func.lower() == 'wgan_gp':
    from . import train_func_wgan_gp
    train = train_func_wgan_gp.train
  elif args.train_func.lower() == 'wgan_gp_sinkhorn':
    from . import train_func_wgan_gp_sinkhorn
    train = train_func_wgan_gp_sinkhorn.train
  elif args.train_func.lower() == 'sinkhorn':
    from . import train_func_sinkhorn
    train = train_func_sinkhorn.train
  else:
    def dummy_train(*args, **kwargs):
      pass
    train = dummy_train
  return train


def main(epochs, batch_size, n_critic, lr, z_dim, args, myargs, **kwargs):
  seed_rng(args.seed)
  logger = myargs.logger

  # create train_dict
  train_dict = init_train_dict()
  myargs.checkpoint_dict['train_dict'] = train_dict

  G, D, G_ema, ema = model_create(train_dict, args, myargs)

  # optimizer
  d_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
  g_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
  myargs.checkpoint_dict['d_optimizer'] = d_optimizer
  myargs.checkpoint_dict['g_optimizer'] = g_optimizer

  """ noise """
  # z_sample = torch.randn(128, z_dim, device='cuda')
  z_sample = gan_utils.z_normal(batch_size=128, dim_z=z_dim, z_mean=0, z_var=1.)
  z_sample.sample_()
  z_train = gan_utils.z_normal(batch_size=args.batch_size, dim_z=args.z_dim, z_mean=0, z_var=1)
  z_test = gan_utils.z_normal(batch_size=args.batch_size, dim_z=args.z_dim, z_mean=0, z_var=1)

  # load inception network
  inception_metrics = inception_metrics_func_create(args=args, myargs=myargs)

  """ load checkpoint """
  if args.evaluate:
    logger.info('=> Evaluate from: %s', args.evaluate_path)
    # load G
    loaded_state_dict = torch.load(args.evaluate_path)
    G.load_state_dict(loaded_state_dict['G'])
    IS_mean, IS_std, FID = inception_metrics(G=G, z=z_test, show_process=True, use_torch=False)
    logger.info_msg('G: IS_mean: %f +- %f, FID: %f', IS_mean, IS_std, FID)
    # G_ema
    G_ema.load_state_dict(loaded_state_dict['G_ema'])
    IS_mean, IS_std, FID = inception_metrics(G=G_ema, z=z_test, show_process=True, use_torch=False)
    logger.info_msg('G_ema: IS_mean: %f +- %f, FID: %f', IS_mean, IS_std, FID)
    return

  if args.resume:
    logger.info('=> Resume from: %s', args.resume_path)
    loaded_state_dict = myargs.checkpoint.load_checkpoint(checkpoint_dict=myargs.checkpoint_dict,
                                                          resumepath=args.resume_path)
    for key in train_dict:
      if key in loaded_state_dict['train_dict']:
        train_dict[key] = loaded_state_dict['train_dict'][key]

  """load dataset"""
  data_loader = dataset_load(batch_size=batch_size, logger=logger, **kwargs)

  # create train_func
  logger.info('=> Create train func: %s', args.train_func)
  if args.train_func.lower() == 'wgan_gp':
    from . import train_func_wgan_gp
    train = train_func_wgan_gp.train_func(data_loader=data_loader, G=G, D=D, G_ema=G_ema, ema=ema,
                                          z_train=z_train,
                                          g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                          z_sample=z_sample, train_dict=train_dict, args=args, myargs=myargs)
  elif args.train_func.lower() == 'wgan_gp_sinkhorn':
    from . import train_func_wgan_gp_sinkhorn
    train = train_func_wgan_gp_sinkhorn.train_func(data_loader=data_loader, G=G, D=D, G_ema=G_ema, ema=ema,
                                                   z_train=z_train,
                                                   g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                                   z_sample=z_sample, train_dict=train_dict, args=args, myargs=myargs)
  elif args.train_func.lower() == 'wgan_gp_d_bound':
    from . import train_func_wgan_gp_d_bound
    train = train_func_wgan_gp_d_bound.train_func(data_loader=data_loader, G=G, D=D, G_ema=G_ema, ema=ema,
                                                  z_train=z_train,
                                                  g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                                  z_sample=z_sample, train_dict=train_dict, args=args, myargs=myargs)
  elif args.train_func.lower() == 'wgan_gp_g_sinkhorn':
    from . import train_func_wgan_gp_G_sinkhorn
    train = train_func_wgan_gp_G_sinkhorn.train_func(data_loader=data_loader, G=G, D=D, G_ema=G_ema, ema=ema,
                                                   z_train=z_train,
                                                   g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                                   z_sample=z_sample, train_dict=train_dict, args=args, myargs=myargs)
  elif args.train_func.lower() == 'sinkhorn':
    from . import train_func_sinkhorn
    train = train_func_sinkhorn.train
  elif args.train_func.lower() == 'wgan_div':
    from . import train_func_wgan_div
    train = train_func_wgan_div.train_func(data_loader=data_loader, G=G, D=D, G_ema=G_ema, ema=ema,
                                           z_train=z_train,
                                           g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                           z_sample=z_sample, train_dict=train_dict, args=args, myargs=myargs)
  elif args.train_func.lower() == 'wgan_div_sinkhorn':
    from . import train_func_wgan_div_sinkhorn
    train = train_func_wgan_div_sinkhorn.train_func(data_loader=data_loader, G=G, D=D, G_ema=G_ema, ema=ema,
                                                    z_train=z_train,
                                                    g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                                    z_sample=z_sample, train_dict=train_dict, args=args, myargs=myargs)
  elif args.train_func.lower() == 'wgan_div_correctness':
    from . import train_func_wgan_div_correctness
    train = train_func_wgan_div_correctness.train_func(data_loader=data_loader, G=G, D=D, G_ema=G_ema, ema=ema,
                                                       z_train=z_train,
                                                       g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                                       z_sample=z_sample, train_dict=train_dict, args=args, myargs=myargs)
  elif args.train_func.lower() == 'gp_real':
    from . import train_func_gp_real
    train = train_func_gp_real.train_func(data_loader=data_loader, G=G, D=D, G_ema=G_ema, ema=ema,
                                          z_train=z_train,
                                          g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                          z_sample=z_sample, train_dict=train_dict, args=args, myargs=myargs)
  elif args.train_func.lower() == 'gp_real_sinkhorn':
    from . import train_func_gp_real_sinkhorn
    train = train_func_gp_real_sinkhorn.train_func(data_loader=data_loader, G=G, D=D, G_ema=G_ema, ema=ema,
                                                    z_train=z_train,
                                                    g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                                    z_sample=z_sample, train_dict=train_dict, args=args, myargs=myargs)
  else:
    def dummy_train(*args, **kwargs):
      pass
    train = dummy_train


  for epoch in range(train_dict['epoch_done'], epochs):
    logger.info('epoch: [%d/%d]' % (epoch, epochs))

    train(data_loader=data_loader, G=G, D=D, z_train=z_train,
          g_optimizer=g_optimizer, d_optimizer=d_optimizer,
          z_sample=z_sample, train_dict=train_dict, args=args, myargs=myargs)

    train_dict['epoch_done'] += 1

    # test
    IS_mean, IS_std, FID = inception_metrics(G=G, z=z_test, show_process=False, use_torch=False)
    logger.info_msg('IS_mean: %f +- %f, FID: %f', IS_mean, IS_std, FID)
    summary = {'IS_mean': IS_mean, 'IS_std': IS_std, 'FID': FID}
    for key in summary:
      myargs.writer.add_scalar('test/' + key, summary[key], train_dict['epoch_done'])
    if train_dict['best_FID'] > FID:
      train_dict['best_FID'] = FID
      myargs.checkpoint.save_checkpoint(checkpoint_dict=myargs.checkpoint_dict,
                                        filename='ckpt_epoch_%d_FID_%f.tar' % (train_dict['epoch_done'], FID))

  logger.info_msg('best_FID: %f', train_dict['best_FID'])
  myargs.checkpoint.save_checkpoint(checkpoint_dict=myargs.checkpoint_dict, filename='ckpt_end.tar')

