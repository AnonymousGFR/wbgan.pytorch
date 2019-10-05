import functools, time
import collections
import math
import os

import torch
import torchvision
import torch.nn as nn

from template_lib.models import ema_model
from template_lib.gans import gan_utils, inception_utils, gan_losses, \
  weight_regularity
from template_lib.trainer import base_trainer
from template_lib.utils import modelarts_utils

from utils import get_data_loaders
from TOOLS import interpolation_BigGAN


class BaseTrainer(base_trainer.Trainer):
  def __init__(self, args, myargs):
    self.args = args
    self.myargs = myargs
    self.config = myargs.config
    self.logger = myargs.logger
    self.train_dict = self.init_train_dict()

    # self.dataset_load()
    self.model_create()
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
    train_dict['best_IS'] = 0
    self.myargs.checkpoint_dict['train_dict'] = train_dict
    return train_dict

  def model_create(self):
    import BigGAN as model
    import utils
    config = self.config.model

    Generator = model.Generator
    Discriminator = model.Discriminator
    G_D = model.G_D

    print('Create generator: {}'.format(Generator))
    self.resolution = utils.imsize_dict[self.config.dataset.dataset]
    self.n_classes = utils.nclass_dict[self.config.dataset.dataset]
    G_activation = utils.activation_dict[config.Generator.G_activation]
    self.G = Generator(**{**config.Generator,
                          'resolution': self.resolution,
                          'n_classes': self.n_classes,
                          'G_activation': G_activation},
                       **config.optimizer).cuda()
    optim_type = getattr(config.optimizer, 'optim_type', None)
    if optim_type == 'radam':
      print('Using radam optimizer.')
      from template_lib.optimizer import radam
      self.G.optim = radam.RAdam(
        params=self.G.parameters(), lr=config.optimizer.G_lr,
        betas=(config.optimizer.G_B1, config.optimizer.G_B2), weight_decay=0,
        eps=config.optimizer.adam_eps)

    print('Create discriminator: {}'.format(Discriminator))
    D_activation = utils.activation_dict[config.Discriminator.D_activation]
    self.D = Discriminator(logger=self.logger,
                           **{**config.Discriminator,
                              'resolution': self.resolution,
                              'n_classes': self.n_classes,
                              'D_activation': D_activation},
                           **config.optimizer).cuda()
    if optim_type == 'radam':
      print('Using radam optimizer.')
      from TOOLS import radam
      self.D.optim = radam.RAdam(
        params=self.D.parameters(), lr=config.optimizer.D_lr,
        betas=(config.optimizer.D_B1, config.optimizer.D_B2), weight_decay=0,
        eps=config.optimizer.adam_eps)

    if getattr(self.config.train_one_epoch, 'weigh_loss', False):
      self.create_alpha()

    self.G_ema = Generator(logger=self.logger,
                           **{**config.Generator,
                              'resolution': self.resolution,
                              'n_classes': self.n_classes,
                              'G_activation': G_activation,
                              'skip_init': True,
                              'no_optim': True}).cuda()
    self.ema = ema_model.EMA(
      self.G, self.G_ema, decay=0.9999, start_itr=config.ema_start)

    print('Create G_D: {}'.format(G_D))
    self.GD = G_D(self.G, self.D)
    if config['parallel']:
      self.GD = nn.DataParallel(self.GD)

    self.myargs.checkpoint_dict['G'] = self.G
    self.myargs.checkpoint_dict['G_optim'] = self.G.optim
    self.myargs.checkpoint_dict['D'] = self.D
    self.myargs.checkpoint_dict['D_optim'] = self.D.optim
    self.myargs.checkpoint_dict['G_ema'] = self.G_ema

    models = {'G': self.G, 'D': self.D}
    self.print_number_params(models=models)

  def create_alpha(self):
    config = self.config.model.create_alpha
    # initial at zero
    alpha_params = []
    if hasattr(config, 'weight_gp'):
      alpha_gp = math.log(1. / config.weight_gp)
      self.alpha_gp = nn.Parameter(torch.tensor(alpha_gp).cuda())
      alpha_params.append(self.alpha_gp)
    if hasattr(config, 'weight_wd'):
      alpha_wd = math.log(1. / config.weight_wd)
      self.alpha_wd = nn.Parameter(torch.tensor(alpha_wd).cuda())
      alpha_params.append(self.alpha_wd)
    if hasattr(config, 'weight_g'):
      alpha_g = math.log(1. / config.weight_g)
      self.alpha_g = nn.Parameter(torch.tensor(alpha_g).cuda())
      alpha_params.append(self.alpha_g)
    self.alpha_params = alpha_params

    if config.optim_type == 'adam':
      self.alpha_optim = torch.optim.Adam(
        alpha_params, lr=config.lr, betas=config.betas, eps=config.eps)
    elif config.optim_type == 'sgd':
      self.alpha_optim = torch.optim.SGD(
        alpha_params, lr=config.lr, momentum=0.99
      )
    else:
      raise NotImplementedError
    pass

  def optimizer_create(self):
    pass

  def noise_create(self):
    config = self.myargs.config.noise
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    self.z_ = gan_utils.z_normal(
      batch_size=G_batch_size,
      dim_z=self.config.model.Generator.dim_z,
      z_mean=config.z_mean, z_var=config.z_var)
    self.y_ = gan_utils.y_categorical(batch_size=G_batch_size,
                                      nclasses=self.n_classes)

    self.z_test = gan_utils.z_normal(
      batch_size=G_batch_size,
      dim_z=self.config.model.Generator.dim_z,
      z_mean=config.z_mean, z_var=config.z_var)
    self.y_test = gan_utils.y_categorical(batch_size=G_batch_size,
                                          nclasses=self.n_classes)

    # Prepare a fixed z & y to see individual sample evolution throghout training
    self.fixed_z = gan_utils.z_normal(
      batch_size=G_batch_size,
      dim_z=self.config.model.Generator.dim_z,
      z_mean=config.z_mean, z_var=config.z_var)
    self.fixed_y = gan_utils.y_categorical(batch_size=G_batch_size,
                                           nclasses=self.n_classes)
    self.fixed_z.sample_()
    self.fixed_y.sample_()

  def inception_metrics_func_create(self):
    config = self.config.inception_metric
    self.logger.info('Load inception moments: %s',
                     config.saved_inception_moments)
    inception_metrics = inception_utils.InceptionMetricsCond(
      saved_inception_moments=config.saved_inception_moments)

    inception_metrics = functools.partial(
      inception_metrics,
      num_inception_images=config.num_inception_images,
      num_splits=10, prints=True)

    return inception_metrics

  def dataset_load(self, ):
    if self.config.train_one_epoch.dummy_train:
      return
    config = self.config.dataset
    D_batch_size = (self.config.noise.batch_size
                    * self.config.train_one_epoch.num_D_steps
                    * self.config.train_one_epoch.num_D_accumulations)

    if config.dataset == 'celeba64':
      self.logger.info('Load dataset in: %s', config.datadir)
      config.batch_size = D_batch_size
      config.seed = self.config.seed
      from TOOLS import dataset
      self.loaders = [dataset.get_data_loaders(config)]

    else:
      self.logger.info('Load dataset in: %s', config.data_root)
      self.loaders = get_data_loaders(
        **{**config,
           'batch_size': D_batch_size,
           'start_itr': self.train_dict['batches_done'],
           'config': config})

  def _summary_scalars(self):
    myargs = self.myargs

    for key in self.summary:
      myargs.writer.add_scalar('train_one_epoch/%s' % key, self.summary[key],
                               self.train_dict['batches_done'])
    myargs.writer.add_scalars('logit_mean', self.summary_D,
                              self.train_dict['batches_done'])
    myargs.writer.add_scalars('wd', self.summary_wd,
                              self.train_dict['batches_done'])
    self.myargs.textlogger.log(
      self.train_dict['batches_done'], **self.summary)
    self.myargs.textlogger.log(
      self.train_dict['batches_done'], **self.summary_D)
    self.myargs.textlogger.log(
      self.train_dict['batches_done'], **self.summary_wd)

  def _summary_images(self, imgs):
    myargs = self.myargs
    train_dict = self.train_dict
    config = self.config.summary_images

    bs_log = config.bs_log
    n_row = config.n_row

    self.G.eval()
    self.G_ema.eval()
    itr = train_dict['batches_done']
    # x
    filename = os.path.join(self.args.imgdir, 'x_itr_%09d.jpg'%itr)
    torchvision.utils.save_image(
      imgs[:bs_log], filename=filename, nrow=n_row, normalize=True,
      pad_value=1)
    # merged_img = torchvision.utils.make_grid(imgs[:bs_log], normalize=True,
    #                                          pad_value=1, nrow=n_row)
    # myargs.writer.add_images('imgs', merged_img.view(1, *merged_img.shape),
    #                          train_dict['batches_done'])
    with torch.no_grad():
      # G
      G_z = self.G(self.fixed_z[:bs_log], self.G.shared(self.fixed_y[:bs_log]))
      filename = os.path.join(self.args.imgdir, 'G_z_itr_%09d.jpg' % itr)
      torchvision.utils.save_image(
        G_z, filename=filename, nrow=n_row, normalize=True,
        pad_value=1)
      # merged_img = torchvision.utils.make_grid(G_z, normalize=True,
      #                                          pad_value=1, nrow=n_row)
      # myargs.writer.add_images('G_z', merged_img.view(1, *merged_img.shape),
      #                          train_dict['batches_done'])
      # G_ema
      G_ema_z = self.G_ema(self.fixed_z[:bs_log],
                       self.G_ema.shared(self.fixed_y[:bs_log]))
      filename = os.path.join(self.args.imgdir, 'G_ema_z_itr_%09d.jpg' % itr)
      torchvision.utils.save_image(
        G_ema_z, filename=filename, nrow=n_row, normalize=True,
        pad_value=1)
      # merged_img = torchvision.utils.make_grid(G_z, normalize=True,
      #                                          pad_value=1, nrow=n_row)
      # myargs.writer.add_images('G_ema_z',
      #                          merged_img.view(1, *merged_img.shape),
      #                          train_dict['batches_done'])

    self.G.train()

  def sample_interpolation(self, G, name_prefix):
    config=self.config.test.sample_interpolation
    myargs = self.myargs
    train_dict = self.train_dict
    G.eval()

    image_filenames, images = interpolation_BigGAN.sample_sheet(
      G=G, classes_per_sheet=config.classes_per_sheet,
      num_classes=self.n_classes, samples_per_class=10,
      parallel=True, z_=self.fixed_z)
    for key, value in zip(image_filenames, images):
      key = name_prefix + '/' + key
      myargs.writer.add_images(key, value, train_dict['epoch_done'])

    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
      image_filename, image = interpolation_BigGAN.interp_sheet(
        G, num_per_sheet=config.num_per_sheet,
        num_midpoints=config.num_midpoints, num_classes=self.n_classes,
        parallel=True, sheet_number=0,
        fix_z=fix_z, fix_y=fix_y, device='cuda')
      myargs.writer.add_images(name_prefix + '/' + image_filename, image,
                               train_dict['epoch_done'])
    G.train()

  def finetune(self):
    config = self.config.finetune
    if config.type == 'imagenet':
      self.logger.info('Finetune imagenet model.')
      self.G.load_state_dict(state_dict=torch.load(config.G_path))
      self.G_ema.load_state_dict(state_dict=torch.load(config.G_ema_path))
      self.D.load_state_dict(state_dict=torch.load(config.D_path))

  def train_(self, ):
    config = self.config
    self.modelarts(join=True)
    for epoch in range(self.train_dict['epoch_done'], config.epochs):
      self.logger.info('epoch: [%d/%d]' % (epoch, config.epochs))

      self.train_one_epoch()

      self.train_dict['epoch_done'] += 1
      # test
      self.test()
      self.modelarts()
    self.finalize()
    self.modelarts(join=True, end=True)

  def test(self):
    config = self.config.test
    train_dict = self.myargs.checkpoint_dict['train_dict']

    # Interpolation
    if hasattr(config, 'sample_interpolation'):
      self.sample_interpolation(G=self.G, name_prefix='G')
      self.sample_interpolation(G=self.G_ema, name_prefix='G_ema')

    if config.use_ema:
      G = self.G_ema
    else:
      G = self.G
    IS_mean, IS_std, FID = self.inception_metrics(
      G=G, z=self.z_test, y=self.y_test,
      show_process=False, use_torch=False, parallel=True)
    self.logger.info_msg('IS_mean: %f +- %f, FID: %f', IS_mean, IS_std, FID)
    summary = {'IS_mean': IS_mean, 'IS_std': IS_std, 'FID': FID}

    self.summary_scalars(
      summary=summary, prefix='test', step=train_dict['epoch_done'])
    self.myargs.textlogger.log_axes(**summary)

    if train_dict['best_FID'] > FID:
      train_dict['best_FID'] = FID
      self.myargs.textlogger.log(
        train_dict['epoch_done'], best_FID=FID)
      # self.myargs.checkpoint.save_checkpoint(
      #   checkpoint_dict=self.myargs.checkpoint_dict,
      #   filename='ckpt_epoch_%d_FID_%f.tar' % (
      #     train_dict['epoch_done'], FID))
    if train_dict['best_IS'] < IS_mean:
      self.train_dict['best_IS'] = IS_mean
      self.myargs.textlogger.log(
        train_dict['epoch_done'], best_IS=IS_mean)

  def finalize(self):
    self.logger.info_msg('best_FID: %f', self.train_dict['best_FID'])
    self.myargs.checkpoint.save_checkpoint(
      checkpoint_dict=self.myargs.checkpoint_dict,
      filename='ckpt_end.tar')

  def evaluate(self):
    self.logger.info("Evaluating ...")
    config = self.config.evaluate
    self.G.load_state_dict(torch.load(config.G_path))
    self.G_ema.load_state_dict(torch.load(config.G_ema_path))

    IS_mean, IS_std, FID = self.inception_metrics(
      G=self.G, z=self.z_test, y=self.y_test,
      show_process=True, use_torch=False, parallel=config.parallel)
    self.logger.info("Test G:")
    self.logger.info_msg('IS_mean: %f +- %f, FID: %f', IS_mean, IS_std, FID)

    IS_mean, IS_std, FID = self.inception_metrics(
      G=self.G_ema, z=self.z_test, y=self.y_test,
      show_process=True, use_torch=False, parallel=config.parallel)
    self.logger.info("Test G_ema:")
    self.logger.info_msg('IS_mean: %f +- %f, FID: %f', IS_mean, IS_std, FID)
    pass

