import collections
import torch
import tqdm

from template_lib.gans import gan_losses

from TOOLS import sinkhorn_autodiff
from . import trainer


class Trainer(trainer.Trainer):

  def train_one_epoch(self, ):
    config = self.config.train_one_epoch
    if config.dummy_train:
      return
    myargs = self.myargs
    train_dict = self.train_dict
    pbar = tqdm.tqdm(self.data_loader, file=myargs.stdout)
    for i, (imgs, _) in enumerate(pbar):
      summary_dict = collections.defaultdict(dict)
      self.G.train()
      self.D.train()
      train_dict['batches_done'] += 1

      imgs = imgs.cuda()
      bs = imgs.size(0)
      if bs != self.config.noise.batch_size_train:
        return
      self.z_train.sample_()
      with torch.no_grad():
        f_imgs = self.G(self.z_train[:bs])

      # train D
      D_r_logit = self.D(imgs)
      D_r_logit_mean = D_r_logit.mean()
      D_f_logit = self.D(f_imgs.detach())
      D_f_logit_mean = D_f_logit.mean()
      summary_dict['D_logit_mean']['D_r_logit_mean'] = \
        D_r_logit_mean.item()
      summary_dict['D_logit_mean']['D_f_logit_mean'] = \
        D_f_logit_mean.item()

      self.d_optimizer.zero_grad()

      # Wasserstein-1 Distance
      wd = D_r_logit_mean - D_f_logit_mean
      # Backward gp loss in this func
      gp = gan_losses.wgan_gp_gradient_penalty(
        imgs, f_imgs, self.D, gp_lambda=config.gp_lambda)

      if config.bound_type == 'constant':
        D_loss = -wd + torch.relu(wd - float(config.bound))
        # D_loss = -wd + gp * config.gp_lambda + \
        #          torch.relu(wd - float(config.bound))
        summary_dict['wd']['bound'] = config.bound
      elif config.bound_type == 'sinkhorn':
        with torch.no_grad():
          sinkhorn_d = sinkhorn_autodiff.sinkhorn_loss(
            x=imgs.view(bs, -1), y=f_imgs.view(bs, -1).detach(),
            epsilon=config.sinkhorn_eps, niter=config.sinkhorn_niter,
            cuda=True, pi_detach=config.sinkhorn_pi_detach)
          summary_dict['wd']['D_sinkhorn_d'] = sinkhorn_d.item()
        D_loss = -wd + torch.relu(wd - sinkhorn_d.item())
      else:
        D_loss = -wd
        # D_loss = -wd + gp * config.gp_lambda
      summary_dict['wd']['wd'] = wd.item()
      summary_dict['scalars']['gp'] = gp.item()
      summary_dict['scalars']['D_loss'] = D_loss.item()

      D_loss.backward()
      self.d_optimizer.step()

      if i % config.n_critic == 0:
        # train G
        self.z_train.sample_()
        f_imgs = self.G(self.z_train)

        D_f_logit = self.D(f_imgs)
        D_f_logit_mean = D_f_logit.mean()
        g_loss_only = - D_f_logit_mean

        if getattr(config, 'min_sinkhorn_in_g', False):
          sinkhorn_d = sinkhorn_autodiff.sinkhorn_loss(
            x=imgs.view(imgs.size(0), -1), y=f_imgs.view(f_imgs.size(0), -1),
            epsilon=config.sinkhorn_eps, niter=config.sinkhorn_niter,
            cuda=True, pi_detach=config.sinkhorn_pi_detach)
          summary_dict['wd']['G_sinkhorn_d'] = sinkhorn_d.item()
          G_loss = g_loss_only + config.sinkhorn_lambda * sinkhorn_d
        else:
          G_loss = g_loss_only
        summary_dict['D_logit_mean']['G_f_logit_mean'] = D_f_logit_mean.item()
        summary_dict['scalars']['g_loss_only'] = g_loss_only.item()
        summary_dict['scalars']['G_loss'] = G_loss.item()

        self.g_optimizer.zero_grad()
        G_loss.backward()
        self.g_optimizer.step()

        # end iter
        self.ema.update(train_dict['batches_done'])

      if i % config.sample_every == 0:

        # checkpoint
        myargs.checkpoint.save_checkpoint(
          checkpoint_dict=myargs.checkpoint_dict, filename='ckpt.tar')
        # summary
        self.summary_dicts(summary_dicts=summary_dict, prefix='train_one_epoch',
                           step=train_dict['batches_done'])


      elif train_dict['batches_done'] <= config.sample_start_iter:
        self.summary_dicts(summary_dicts=summary_dict, prefix='train_one_epoch',
                           step=train_dict['batches_done'])

      if (i + 1) % (len(self.data_loader)//4) == 0:
        # save images
        self.summary_figures(summary_dicts=summary_dict, prefix='train_one_epoch')
        self._summary_images(imgs=imgs, itr=self.train_dict['batches_done'])

