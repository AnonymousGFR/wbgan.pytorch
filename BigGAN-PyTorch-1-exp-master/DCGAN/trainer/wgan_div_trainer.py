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
    self._summary_create()
    for i, (imgs, _) in enumerate(pbar):
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

      imgs.requires_grad_()
      f_imgs.requires_grad_()
      # train D
      D_r_logit = self.D(imgs)
      D_r_logit_mean = D_r_logit.mean()
      D_f_logit = self.D(f_imgs)
      D_f_logit_mean = D_f_logit.mean()
      self.summary_logit_mean['D_r_logit_mean'] = D_r_logit_mean.item()
      self.summary_logit_mean['D_f_logit_mean'] = D_f_logit_mean.item()

      self.d_optimizer.zero_grad()

      wd =  D_f_logit_mean - D_r_logit_mean
      # Backward gp loss in this func
      gp = gan_losses.wgan_div_gradient_penalty(
        real_imgs=imgs, fake_imgs=f_imgs, real_validity=D_r_logit,
        fake_validity=D_f_logit, backward=True, retain_graph=True
      )

      if config.bound_type == 'constant':
        raise NotImplemented
        D_loss = -wd + torch.relu(wd - float(config.bound))
        # D_loss = -wd + gp * config.gp_lambda + \
        #          torch.relu(wd - float(config.bound))
        self.summary_wd['bound'] = config.bound
      elif config.bound_type == 'sinkhorn':
        with torch.no_grad():
          sinkhorn_d = sinkhorn_autodiff.sinkhorn_loss(
            x=imgs.view(bs, -1), y=f_imgs.view(bs, -1).detach(),
            epsilon=config.sinkhorn_eps, niter=config.sinkhorn_niter,
            cuda=True, pi_detach=config.sinkhorn_pi_detach)
          self.summary_wd['D_sinkhorn_d'] = sinkhorn_d.item()
        D_loss = -wd + torch.relu(wd - sinkhorn_d.item())
      else:
        D_loss = -wd

      self.summary_wd['wd'] = wd.item()
      self.summary['gp'] = gp.item()
      self.summary['D_loss'] = D_loss.item()

      D_loss.backward()
      self.d_optimizer.step()

      if i % config.n_critic == 0:
        # train G
        self.z_train.sample_()
        f_imgs = self.G(self.z_train)

        D_f_logit = self.D(f_imgs)
        D_f_logit_mean = D_f_logit.mean()
        g_loss_only = D_f_logit_mean

        if config.bound_type == 'sinkhorn':
          sinkhorn_d = sinkhorn_autodiff.sinkhorn_loss(
            x=imgs.view(imgs.size(0), -1), y=f_imgs.view(f_imgs.size(0), -1),
            epsilon=config.sinkhorn_eps, niter=config.sinkhorn_niter,
            cuda=True, pi_detach=config.sinkhorn_pi_detach)
          self.summary_wd['G_sinkhorn_d'] = sinkhorn_d.item()
          G_loss = g_loss_only + config.sinkhorn_lambda * sinkhorn_d
        else:
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

        # checkpoint
        myargs.checkpoint.save_checkpoint(
          checkpoint_dict=myargs.checkpoint_dict, filename='ckpt.tar')
        # summary
        self._summary_scalars()

      elif train_dict['batches_done'] <= config.sample_start_iter:
        self._summary_scalars()

      if i % (len(self.data_loader)//4) == 0:
        # save images
        self._summary_figures()
        self._summary_images(imgs=imgs, itr=self.train_dict['batches_done'])

