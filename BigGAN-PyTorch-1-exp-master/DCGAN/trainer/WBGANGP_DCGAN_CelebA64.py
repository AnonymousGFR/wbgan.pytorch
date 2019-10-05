import os
import functools
import tqdm
import torch
import torchvision

from template_lib.models import ema_model
from template_lib.gans import gan_utils, inception_utils, gan_losses, \
  sinkhorn_autodiff

from . import WGANGP_DCGAN_CelebA64


class Trainer(WGANGP_DCGAN_CelebA64.Trainer):
  def __init__(self, args, myargs):
    super().__init__(args=args, myargs=myargs)

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
      if config.sinkhorn_bound:
        with torch.no_grad():
          sinkhorn_d = sinkhorn_autodiff.sinkhorn_loss(
            x=imgs.view(bs, -1), y=f_imgs.view(bs, -1).detach(),
            epsilon=config.sinkhorn.sinkhorn_eps,
            niter=config.sinkhorn.sinkhorn_niter,
            cuda=True,
            pi_detach=config.sinkhorn.sinkhorn_pi_detach)
          self.summary_sinkhorn['D_sinkhorn_d'] = sinkhorn_d.item()
        bound = sinkhorn_d.item()
      else:
        bound = config.bound

      D_r_logit = self.D(imgs)
      D_r_logit_mean = D_r_logit.mean()
      D_f_logit = self.D(f_imgs.detach())
      D_f_logit_mean = D_f_logit.mean()
      self.summary_logit_mean['D_r_logit_mean'] = D_r_logit_mean.item()
      self.summary_logit_mean['D_f_logit_mean'] = D_f_logit_mean.item()

      # Wasserstein-1 Distance
      wd = D_r_logit_mean - D_f_logit_mean
      if config.gp_lambda > 0:
        gp = gan_losses.wgan_gp_gradient_penalty(imgs, f_imgs, self.D)
        self.summary['gp'] = gp.item()
        D_loss = -wd + gp * config.gp_lambda + torch.relu(wd - bound)
      else:
        D_loss = -wd + torch.relu(wd - bound)
      self.summary_wd['wd'] = wd.item()
      self.summary['D_loss'] = D_loss.item()

      self.d_optimizer.zero_grad()
      D_loss.backward()
      self.d_optimizer.step()

      if i % config.n_critic == 0:
        # train G
        self.z_train.sample_()
        f_imgs = self.G(self.z_train)

        sinkhorn_d = sinkhorn_autodiff.sinkhorn_loss(
          x=imgs.view(imgs.size(0), -1), y=f_imgs.view(f_imgs.size(0), -1),
          epsilon=config.sinkhorn.sinkhorn_eps,
          niter=config.sinkhorn.sinkhorn_niter,
          cuda=True, pi_detach=config.sinkhorn.sinkhorn_pi_detach)
        self.summary_sinkhorn['G_sinkhorn_d'] = sinkhorn_d.item()

        D_f_logit = self.D(f_imgs)
        D_f_logit_mean = D_f_logit.mean()
        g_loss_only = - D_f_logit_mean
        G_loss = g_loss_only + config.lambda_sinkhorn * sinkhorn_d
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
    super()._summary_create()
    self.summary_sinkhorn = {}

  def _summary_scalars(self):
    super()._summary_scalars()
    myargs = self.myargs
    myargs.writer.add_scalars('sinkhorn', self.summary_sinkhorn,
                              self.train_dict['batches_done'])

