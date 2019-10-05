import os
import functools
import tqdm
import torch
import torchvision

from template_lib.models import ema_model
from template_lib.gans import gan_utils, inception_utils, gan_losses


class Trainer(object):
  def __init__(self, args, myargs):
    self.args = args
    self.myargs = myargs
    self.logger = myargs.logger
    self.config = myargs.config
    self.train_dict = myargs.checkpoint_dict['train_dict']

    self.G, self.D, self.G_ema, self.ema = self.model_create()
    self.d_optimizer, self.g_optimizer = self.optimizer_create()
    self.z_sample, self.z_train, self.z_test = self.noise_create()

    # load inception network
    self.inception_metrics = self.inception_metrics_func_create()


  def model_create(self):
    from .. import models_64x64
    config = self.myargs.config.trainer.model

    D = models_64x64.DiscriminatorWGANGP(3)
    G = models_64x64.Generator(config.z_dim)
    G_ema = models_64x64.Generator(config.z_dim)
    ema = ema_model.EMA(G, G_ema, decay=0.9999, start_itr=config.ema_start)
    D.cuda()
    G.cuda()
    G_ema.cuda()

    self.myargs.checkpoint_dict['D'] = D
    self.myargs.checkpoint_dict['G'] = G
    self.myargs.checkpoint_dict['G_ema'] = G_ema

    self.logger.info_msg('Number of params in G: {} D: {}'.format(
      *[sum([p.data.nelement() for p in net.parameters()]) for net in
        [G, D]]))

    return G, D, G_ema, ema

  def optimizer_create(self):
    config = self.myargs.config.trainer.optimizer
    d_optimizer = torch.optim.Adam(self.D.parameters(), lr=config.d_lr,
                                   betas=(config.d_beta1, config.d_beta2))
    g_optimizer = torch.optim.Adam(self.G.parameters(), lr=config.g_lr,
                                   betas=(config.g_beta1, config.g_beta2))
    self.myargs.checkpoint_dict['d_optimizer'] = d_optimizer
    self.myargs.checkpoint_dict['g_optimizer'] = g_optimizer

    return d_optimizer, g_optimizer

  def noise_create(self):
    config = self.myargs.config.trainer.noise
    # z_sample = torch.randn(128, z_dim, device='cuda')
    z_sample = gan_utils.z_normal(batch_size=128,
                                  dim_z=self.config.trainer.model.z_dim,
                                  z_mean=config.z_mean, z_var=config.z_var)
    z_sample.sample_()
    z_train = gan_utils.z_normal(batch_size=config.batch_size_train,
                                 dim_z=self.config.trainer.model.z_dim,
                                 z_mean=config.z_mean, z_var=config.z_var)
    z_test = gan_utils.z_normal(batch_size=config.batch_size_test,
                                dim_z=self.config.trainer.model.z_dim,
                                z_mean=config.z_mean, z_var=config.z_var)
    return z_sample, z_train, z_test

  def inception_metrics_func_create(self):
    config = self.config.trainer.inception_metric
    self.logger.info('Load inception moments: %s',
                     config.saved_inception_moments)
    inception_metrics = inception_utils.InceptionMetrics(
      saved_inception_moments=config.saved_inception_moments)

    inception_metrics = functools.partial(inception_metrics,
                                          num_inception_images=config.num_inception_images,
                                          num_splits=10, prints=True)

    return inception_metrics

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
    config = self.config.trainer.dataset
    self.logger.info('Load dataset in: %s', config.datadir)

    from template_lib.datasets import CelebA
    data_loader = CelebA.CelebA64(datadir=os.path.expanduser(config.datadir),
                                  batch_size=self.config.trainer.noise.batch_size_train,
                                  num_workers=config.num_workers,
                                  seed=self.config.main.seed)
    self.data_loader = data_loader

  def train_one_epoch(self, ):
    config = self.config.trainer.train_one_epoch
    if config.dummy_train:
      return
    myargs = self.myargs
    train_dict = myargs.checkpoint_dict['train_dict']

    self.G.train()
    self.D.train()

    for i, (imgs, _) in enumerate(tqdm.tqdm(self.data_loader)):
      train_dict['batches_done'] += 1
      self._summary_create()

      imgs = imgs.cuda()
      bs = imgs.size(0)
      self.z_train.sample_()
      f_imgs = self.G(self.z_train[:bs])

      # train D
      D_r_logit = self.D(imgs)
      D_r_logit_mean = D_r_logit.mean()
      D_f_logit = self.D(f_imgs.detach())
      D_f_logit_mean = D_f_logit.mean()
      self.summary_logit_mean['D_r_logit_mean'] = D_r_logit_mean.item()
      self.summary_logit_mean['D_f_logit_mean'] = D_f_logit_mean.item()

      # Wasserstein-1 Distance
      wd = D_r_logit_mean - D_f_logit_mean
      gp = gan_losses.wgan_gp_gradient_penalty(imgs, f_imgs, self.D)
      D_loss = -wd + gp * config.gp_lambda
      self.summary_wd['wd'] = wd.item()
      self.summary['gp'] = gp.item()
      self.summary['D_loss'] = D_loss.item()

      self.d_optimizer.zero_grad()
      D_loss.backward()
      self.d_optimizer.step()

      if i % config.n_critic == 0:
        # train G
        self.z_train.sample_()
        f_imgs = self.G(self.z_train)
        D_f_logit = self.D(f_imgs)
        D_f_logit_mean = D_f_logit.mean()
        g_loss_only = - D_f_logit_mean
        G_loss = g_loss_only
        self.summary_logit_mean['G_f_logit_mean'] = D_f_logit_mean.item()
        self.summary['g_loss_only'] = g_loss_only.item()
        self.summary['G_loss'] = G_loss.item()

        self.g_optimizer.zero_grad()
        G_loss.backward()
        self.g_optimizer.step()

        # end iter
        self.ema.update(train_dict['batches_done'])

      if i % config.sample_every == 0:
        # images
        self._summary_images(imgs=imgs)
        # checkpoint
        myargs.checkpoint.save_checkpoint(
          checkpoint_dict=myargs.checkpoint_dict, filename='ckpt.tar')
        # summary
        self._summary_scalars()

      elif train_dict['batches_done'] <= config.sample_start_iter:
        self._summary_scalars()

  def _summary_create(self):
    self.summary = {}
    self.summary_logit_mean = {}
    self.summary_wd = {}

  def _summary_scalars(self):
    myargs = self.myargs
    for key in self.summary:
      myargs.writer.add_scalar('train_one_epoch/%s' % key, self.summary[key],
                               self.train_dict['batches_done'])
    myargs.writer.add_scalars('logit_mean', self.summary_logit_mean,
                              self.train_dict['batches_done'])
    myargs.writer.add_scalars('wd', self.summary_wd,
                              self.train_dict['batches_done'])

  def _summary_images(self, imgs):
    myargs = self.myargs

    self.G.eval()
    f_imgs_sample = self.G(self.z_sample)
    merged_img = torchvision.utils.make_grid(f_imgs_sample, normalize=True,
                                             pad_value=1, nrow=16)
    myargs.writer.add_images('G_z', merged_img.view(1, *merged_img.shape),
                             self.train_dict['batches_done'])
    # G_ema
    self.G_ema.eval()
    G_ema_z = self.G_ema(self.z_sample)
    merged_img = torchvision.utils.make_grid(G_ema_z, normalize=True,
                                             pad_value=1, nrow=16)
    myargs.writer.add_images('G_ema_z',
                             merged_img.view(1, *merged_img.shape),
                             self.train_dict['batches_done'])
    # x
    merged_img = torchvision.utils.make_grid(imgs, normalize=True,
                                             pad_value=1, nrow=16)
    myargs.writer.add_images('x', merged_img.view(1, *merged_img.shape),
                             self.train_dict['batches_done'])
    self.G.train()

  def test(self):
    config = self.config.trainer.test
    train_dict = self.myargs.checkpoint_dict['train_dict']

    if config.use_ema:
      G = self.G_ema
    else:
      G = self.G
    IS_mean, IS_std, FID = self.inception_metrics(G=G, z=self.z_test,
                                                  show_process=False,
                                                  use_torch=False)
    self.logger.info_msg('IS_mean: %f +- %f, FID: %f', IS_mean, IS_std, FID)
    summary = {'IS_mean': IS_mean, 'IS_std': IS_std, 'FID': FID}
    for key in summary:
      self.myargs.writer.add_scalar('test/' + key, summary[key],
                                    train_dict['epoch_done'])
    if train_dict['best_FID'] > FID:
      train_dict['best_FID'] = FID
      self.myargs.checkpoint.save_checkpoint(
        checkpoint_dict=self.myargs.checkpoint_dict,
        filename='ckpt_epoch_%d_FID_%f.tar' % (
          train_dict['epoch_done'], FID))

  def finalize(self):
    self.logger.info_msg('best_FID: %f', self.train_dict['best_FID'])
    self.myargs.checkpoint.save_checkpoint(
      checkpoint_dict=self.myargs.checkpoint_dict,
      filename='ckpt_end.tar')


