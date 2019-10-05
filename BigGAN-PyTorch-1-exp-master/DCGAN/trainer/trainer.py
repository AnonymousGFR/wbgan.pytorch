import os
import functools

import collections
import tqdm
import torch
import torch.nn as nn
import torchvision

from template_lib.models import ema_model
from template_lib.gans import gan_utils, inception_utils, gan_losses
from template_lib.trainer import base_trainer
from template_lib.warmup_scheduler import GradualWarmupScheduler
from template_lib.utils import modelarts_utils

from .. import dataset


class Trainer(base_trainer.Trainer):
  def __init__(self, args, myargs):
    self.args = args
    self.myargs = myargs
    self.logger = myargs.logger
    self.config = myargs.config
    self.train_dict = self.init_train_dict()

    self.G, self.D, self.G_ema, self.ema = self.model_create()
    self.optimizer_create()
    self.scheduler_create()
    self.noise_create()

    # load inception network
    self.inception_metrics = self.inception_metrics_func_create()

  def init_train_dict(self, ):
    train_dict = collections.OrderedDict()
    train_dict['epoch_done'] = 0
    train_dict['batches_done'] = 0
    train_dict['best_FID'] = 9999
    self.myargs.checkpoint_dict['train_dict'] = train_dict
    return train_dict

  def model_create(self):
    from .. import models_64x64
    config = self.myargs.config.model

    D = models_64x64.DiscriminatorWGANGP(3)
    print(D)
    G = models_64x64.Generator(config.z_dim)
    print(G)
    G_ema = models_64x64.Generator(config.z_dim)
    ema = ema_model.EMA(G, G_ema, decay=0.9999, start_itr=config.ema_start)

    D.cuda()
    G.cuda()
    G_ema.cuda()
    if config.parallel:
      D = nn.DataParallel(D)
      G = nn.DataParallel(G)

    self.myargs.checkpoint_dict['D'] = D
    self.myargs.checkpoint_dict['G'] = G
    self.myargs.checkpoint_dict['G_ema'] = G_ema

    self.print_number_params(models=dict(G=G, D=D))


    return G, D, G_ema, ema

  def optimizer_create(self):
    config = self.myargs.config.optimizer
    if config.optim_type == 'adam':
      d_optimizer = torch.optim.Adam(self.D.parameters(), lr=config.d_lr,
                                     betas=(config.d_beta1, config.d_beta2))
      g_optimizer = torch.optim.Adam(self.G.parameters(), lr=config.g_lr,
                                     betas=(config.g_beta1, config.g_beta2))
    elif config.optim_type == 'radam':
      from template_lib.optimizer import radam
      d_optimizer = radam.RAdam(self.D.parameters(), lr=config.d_lr,
                                betas=(config.d_beta1, config.d_beta2))
      g_optimizer = radam.RAdam(self.G.parameters(), lr=config.g_lr,
                                betas=(config.g_beta1, config.g_beta2))
    else:
      raise NotImplementedError

    self.myargs.checkpoint_dict['d_optimizer'] = d_optimizer
    self.myargs.checkpoint_dict['g_optimizer'] = g_optimizer

    self.d_optimizer, self.g_optimizer = d_optimizer, g_optimizer
    return

  def scheduler_create(self):
    return
    scheduler = None
    batch_size = self.config.noise.batch_size_train
    multiplier = batch_size // 128
    self.d_scheduler_warmup = GradualWarmupScheduler(
      self.d_optimizer, multiplier=multiplier,
      total_epoch=10, after_scheduler=scheduler)
    self.g_scheduler_warmup = GradualWarmupScheduler(
      self.g_optimizer, multiplier=multiplier,
      total_epoch=10, after_scheduler=scheduler)

  def scheduler_step(self, epoch):
    return
    self.d_scheduler_warmup.step(epoch=epoch)
    d_lr = self.d_scheduler_warmup.get_lr()[0]
    self.g_scheduler_warmup.step(epoch=epoch)
    g_lr = self.g_scheduler_warmup.get_lr()[0]
    self.summary_scalars_together(summary={'d_lr': d_lr, 'g_lr': g_lr},
                                  prefix='lr', step=epoch)

  def noise_create(self):
    config = self.myargs.config.noise
    # z_sample = torch.randn(128, z_dim, device='cuda')
    z_sample = gan_utils.z_normal(batch_size=128,
                                  dim_z=self.config.model.z_dim,
                                  z_mean=config.z_mean, z_var=config.z_var)
    z_sample.sample_()
    z_train = gan_utils.z_normal(batch_size=config.batch_size_train,
                                 dim_z=self.config.model.z_dim,
                                 z_mean=config.z_mean, z_var=config.z_var)
    z_test = gan_utils.z_normal(batch_size=config.batch_size_test,
                                dim_z=self.config.model.z_dim,
                                z_mean=config.z_mean, z_var=config.z_var)
    self.z_sample, self.z_train, self.z_test = z_sample, z_train, z_test
    return

  def inception_metrics_func_create(self):
    config = self.config.inception_metric
    print('Load inception moments: %s'%config.saved_inception_moments)
    inception_metrics = inception_utils.InceptionMetrics(
      saved_inception_moments=config.saved_inception_moments)

    inception_metrics = functools.partial(
      inception_metrics,
      num_inception_images=config.num_inception_images,
      num_splits=10, prints=True)

    return inception_metrics

  def train(self, ):
    config = self.config
    self.modelarts(join=True)
    for epoch in range(self.train_dict['epoch_done'], config.epochs):
      self.logger.info('epoch: [%d/%d]' % (epoch, config.epochs))
      self.scheduler_step(epoch=epoch)
      self.train_one_epoch()

      self.train_dict['epoch_done'] += 1
      # test
      self.test()
      self.modelarts(join=False)
    self.finalize()
    self.modelarts(join=True, end=True)

  def evaluate(self):
    self.logger.info('Evaluate from: %s', self.args.evaluate_path)
    # load G
    loaded_state_dict = torch.load(self.args.evaluate_path)
    self.G.load_state_dict(loaded_state_dict['G'])
    IS_mean, IS_std, FID = self.inception_metrics(G=self.G, z=self.z_test,
                                                  show_process=True,
                                                  use_torch=False)
    self.logger.info_msg('G: IS_mean: %f +- %f, FID: %f', IS_mean, IS_std, FID)
    # G_ema
    self.G_ema.load_state_dict(loaded_state_dict['G_ema'])
    IS_mean, IS_std, FID = self.inception_metrics(G=self.G_ema, z=self.z_test,
                                                  show_process=True,
                                                  use_torch=False)
    self.logger.info_msg('G_ema: IS_mean: %f +- %f, FID: %f', IS_mean, IS_std, FID)

  def dataset_load(self, ):
    config = self.config.loader
    config.batch_size = self.config.noise.batch_size_train
    config.seed = self.config.seed

    self.data_loader = dataset.get_data_loaders(config)

  # def _summary_create(self):
  #   self.summary = {}
  #   self.summary_logit_mean = {}
  #   self.summary_wd = {}

  def _summary_figures(self):
    myargs = self.myargs
    myargs.textlogger.log_axes(**self.summary)
    myargs.textlogger.log_ax(**self.summary_logit_mean)
    myargs.textlogger.log_ax(**self.summary_wd)
    pass

  def _summary_scalars(self):
    myargs = self.myargs
    itr = self.train_dict['batches_done']
    myargs.textlogger.log(itr, **self.summary)
    myargs.textlogger.log(itr, **self.summary_logit_mean)
    myargs.textlogger.log(itr, **self.summary_wd)
    for key in self.summary:
      myargs.writer.add_scalar(
        'train_one_epoch/%s' % key, self.summary[key], itr)
    myargs.writer.add_scalars(
      'logit_mean', self.summary_logit_mean, itr)
    myargs.writer.add_scalars(
      'wd', self.summary_wd, itr)

  def _summary_images(self, imgs, itr):
    # G
    self.G.eval()
    f_imgs_sample = self.G(self.z_sample)
    merged_img = torchvision.utils.make_grid(f_imgs_sample, normalize=True,
                                             pad_value=1, nrow=16)
    filename = os.path.join(self.args.imgdir, 'G_z_itr_%09d.jpg'%itr)
    torchvision.utils.save_image(
      tensor=merged_img.view(1, *merged_img.shape), filename=filename)
    # myargs.writer.add_images('G_z', merged_img.view(1, *merged_img.shape),
    #                          self.train_dict['batches_done'])

    # G_ema
    self.G_ema.eval()
    G_ema_z = self.G_ema(self.z_sample)
    merged_img = torchvision.utils.make_grid(G_ema_z, normalize=True,
                                             pad_value=1, nrow=16)
    filename = os.path.join(self.args.imgdir, 'G_ema_z_itr_%09d.jpg' % itr)
    torchvision.utils.save_image(
      tensor=merged_img.view(1, *merged_img.shape), filename=filename)
    # myargs.writer.add_images('G_ema_z',
    #                          merged_img.view(1, *merged_img.shape),
    #                          self.train_dict['batches_done'])

    # x
    merged_img = torchvision.utils.make_grid(imgs, normalize=True,
                                             pad_value=1, nrow=16)
    filename = os.path.join(self.args.imgdir, 'X_itr_%09d.jpg' % itr)
    torchvision.utils.save_image(
      tensor=merged_img.view(1, *merged_img.shape), filename=filename)
    # myargs.writer.add_images('x', merged_img.view(1, *merged_img.shape),
    #                          self.train_dict['batches_done'])
    self.G.train()

  def test(self):
    config = self.config.test
    train_dict = self.myargs.checkpoint_dict['train_dict']

    if config.use_ema:
      G = self.G_ema
    else:
      G = self.G
    IS_mean, IS_std, FID = self.inception_metrics(G=G, z=self.z_test,
                                                  show_process=False,
                                                  use_torch=False)
    print('IS_mean: %f +- %f, FID: %f'%(IS_mean, IS_std, FID))
    summary = {'IS_mean': IS_mean, 'IS_std': IS_std, 'FID': FID}
    self.summary_scalars(summary, prefix='test', step=train_dict['epoch_done'])
    self.myargs.textlogger.log_axes(**summary)

    if train_dict['best_FID'] > FID:
      train_dict['best_FID'] = FID
      self.myargs.textlogger.log(train_dict['epoch_done'], best_FID=FID)
      # self.myargs.checkpoint.save_checkpoint(
      #   checkpoint_dict=self.myargs.checkpoint_dict,
      #   filename='ckpt_epoch_%d_FID_%f.tar' % (
      #     train_dict['epoch_done'], FID))

  def finalize(self):
    self.logger.info_msg('best_FID: %f', self.train_dict['best_FID'])
    self.myargs.checkpoint.save_checkpoint(
      checkpoint_dict=self.myargs.checkpoint_dict,
      filename='ckpt_end.tar')

