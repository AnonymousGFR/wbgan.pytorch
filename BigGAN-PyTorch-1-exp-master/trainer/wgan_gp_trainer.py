import collections
import tqdm
import torch
import torch.nn.functional as F

from template_lib.gans import gan_utils, gan_losses, weight_regularity

from . import trainer
from TOOLS import utils_BigGAN, sinkhorn_autodiff


class Trainer(trainer.BaseTrainer):

  def train_one_epoch(self, ):
    config = self.config.train_one_epoch
    if config.dummy_train:
      return
    myargs = self.myargs
    train_dict = myargs.checkpoint_dict['train_dict']
    batch_size = self.config.noise.batch_size
    pbar = tqdm.tqdm(self.loaders[0], file=self.myargs.stdout)
    for i, (images, labels) in enumerate(pbar):
      if len(images) % batch_size != 0:
        return
      summary_dict = collections.defaultdict(dict)
      # Make sure G and D are in training mode, just in case they got set to eval
      # For D, which typically doesn't have BN, this shouldn't matter much.
      self.G.train()
      self.D.train()
      self.GD.train()
      self.G_ema.train()
      images, labels = images.cuda(), labels.cuda()

      # Optionally toggle D and G's "require_grad"
      if config.toggle_grads:
        gan_utils.toggle_grad(self.D, True)
        gan_utils.toggle_grad(self.G, False)

      # How many chunks to split x and y into?
      x = torch.split(images, batch_size)
      y = torch.split(labels, batch_size)
      counter = 0
      for step_index in range(config.num_D_steps):
        self.G.optim.zero_grad()
        self.D.optim.zero_grad()
        if getattr(config, 'weigh_loss', False):
          self.alpha_optim.zero_grad()
        # Increment the iteration counter
        train_dict['batches_done'] += 1
        # If accumulating gradients, loop multiple times before an optimizer step
        for accumulation_index in range(config.num_D_accumulations):
          self.z_.sample_()
          dy = y[counter]
          gy = dy
          # self.y_.sample_()
          D_fake, D_real, G_z = self.GD(
            z=self.z_[:batch_size], gy=gy, x=x[counter], dy=dy, train_G=False,
            split_D=config.split_D, return_G_z=True)

          # Compute components of D's loss, average them, and divide by
          # the number of gradient accumulations
          D_real_mean, D_fake_mean, wd, _ = gan_losses.wgan_discriminator_loss(
            r_logit=D_real, f_logit=D_fake)
          # entropy loss
          if getattr(config, 'weigh_loss', False) and config.use_entropy:
            weight_logit = 1. / torch.stack(self.alpha_params).exp()
            weight = F.softmax(weight_logit)
            entropy = (- weight * F.log_softmax(weight_logit)).sum()
            (- config.entropy_lambda * entropy).backward()
          # gp
          if getattr(config, 'weigh_loss', False) and \
                  hasattr(self, 'alpha_gp'):
            if config.use_entropy:
              weight_logit = 1. / torch.stack(self.alpha_params).exp()
              alpha_gp_logit = 1. / self.alpha_gp.exp()
              weight = alpha_gp_logit.exp() / weight_logit.exp().sum()
            else:
              weight = 1. / self.alpha_gp.exp()
            weight_gp = weight
            sigma_gp = torch.exp(self.alpha_gp / 2.)
            gp_loss, gp = gan_losses.wgan_gp_gradient_penalty_cond(
              x=x[counter], G_z=G_z, gy=gy, f=self.D,
              backward=True, gp_lambda=weight_gp, return_gp=True)
            (self.alpha_gp / 2.).backward()
            summary_dict['sigma']['sigma_gp'] = sigma_gp.item()
            summary_dict['weight']['weight_gp'] = weight_gp.item()
            summary_dict['gp']['gp_loss'] = gp_loss.item()
            summary_dict['gp']['gp_raw'] = gp.item()
          else:
            gp_loss = gan_losses.wgan_gp_gradient_penalty_cond(
              x=x[counter], G_z=G_z, gy=gy, f=self.D,
            backward=True, gp_lambda=config.gp_lambda)
            summary_dict['gp']['gp_loss'] = gp_loss.item()

          # wd
          if getattr(config, 'weigh_loss', False) and \
                  hasattr(self, 'alpha_wd'):
            if config.use_entropy:
              weight_logit = 1. / torch.stack(self.alpha_params).exp()
              alpha_wd_logit = 1. / self.alpha_wd.exp()
              weight = alpha_wd_logit.exp() / weight_logit.exp().sum()
            else:
              weight = 1. / self.alpha_wd.exp()
            weight_wd = weight
            sigma_wd = torch.exp(self.alpha_wd / 2.)
            D_loss = weight_wd.item() * (-wd) + \
              weight_wd * wd.abs().item() + self.alpha_wd / 2.
            summary_dict['sigma']['sigma_wd'] = sigma_wd.item()
            summary_dict['weight']['weight_wd'] = weight_wd.item()
            summary_dict['wd']['wd_loss'] = \
              weight_wd.item() * wd.abs().item()
            summary_dict['wd']['wd_raw'] = wd.item()
          elif getattr(config, 'd_sinkhorn', False):
            sinkhorn_c = config.sinkhorn_c
            with torch.no_grad():
              sinkhorn_d = sinkhorn_autodiff.sinkhorn_loss(
                x=x[counter].view(batch_size, -1),
                y=G_z.view(batch_size, -1).detach(),
                epsilon=sinkhorn_c.sinkhorn_eps,
                niter=sinkhorn_c.sinkhorn_niter,
                cuda=True, pi_detach=sinkhorn_c.sinkhorn_pi_detach)
              summary_dict['wd']['D_sinkhorn_d'] = sinkhorn_d.item()
            D_loss = -wd + torch.relu(wd - sinkhorn_d.item())
            summary_dict['wd']['wd_raw'] = wd.item()
          else:
            D_loss = -wd
            summary_dict['wd']['wd_raw'] = wd.item()

          D_losses = D_loss / float(config.num_D_accumulations)
          # Accumulate gradients
          D_losses.backward()
          counter += 1
          summary_dict['D_logit_mean']['D_real_mean'] = D_real_mean.item()
          summary_dict['D_logit_mean']['D_fake_mean'] = D_fake_mean.item()
          summary_dict['scalars']['D_loss'] = D_loss.item()

        # End accumulation
        # Optionally apply ortho reg in D
        if config.D_ortho > 0.0:
          # Debug print to indicate we're using ortho reg in D.
          print('using modified ortho reg in D')
          weight_regularity.ortho(self.D, config.D_ortho)

        # for name, value in self.D.named_parameters():
        #   self.myargs.writer.add_histogram(
        #     name, value.grad, train_dict['batches_done'])
        self.D.optim.step()
        if getattr(config, 'weigh_loss', False):
          self.alpha_optim.step()

      # Optionally toggle "requires_grad"
      if config.toggle_grads:
        gan_utils.toggle_grad(self.D, False)
        gan_utils.toggle_grad(self.G, True)

      # Zero G's gradients by default before training G, for safety
      self.G.optim.zero_grad()
      if getattr(config, 'weigh_loss', False):
        self.alpha_optim.zero_grad()
      # If accumulating gradients, loop multiple times
      for accumulation_index in range(config.num_G_accumulations):
        self.z_.sample_()
        gy = dy
        # self.y_.sample_()
        D_fake, G_z = self.GD(
          z=self.z_, gy=gy, train_G=True, return_G_z=True,
          split_D=config.split_D)

        G_fake_mean, _ = gan_losses.wgan_generator_loss(f_logit=D_fake)
        # wd for generator
        if getattr(config, 'weigh_loss', False) and \
                  hasattr(self, 'alpha_g'):
          if config.use_entropy:
            weight_logit = 1. / torch.stack(self.alpha_params).exp()
            alpha_g_logit = 1. / self.alpha_g.exp()
            weight = alpha_g_logit.exp() / weight_logit.exp().sum()
          else:
            weight = 1. / self.alpha_g.exp()
          weight_g = weight
          sigma_g = torch.exp(self.alpha_g / 2.)
          wd_g = D_real_mean.item() - G_fake_mean
          G_loss = weight_g.item() * (wd_g) + \
                   weight_g * wd_g.abs().item() + self.alpha_g / 2.
          summary_dict['sigma']['sigma_g'] = sigma_g.item()
          summary_dict['weight']['weight_g'] = weight_g.item()
          summary_dict['wd']['wd_g_raw'] = wd_g.item()
          summary_dict['wd']['wd_g_loss'] = weight_g.item() * wd_g.abs().item()
        elif getattr(config, 'g_sinkhorn', False):
          sinkhorn_c = config.sinkhorn_c
          sinkhorn_d = sinkhorn_autodiff.sinkhorn_loss(
            x=x[-1].view(x[-1].size(0), -1),
            y=G_z.view(G_z.size(0), -1),
            epsilon=sinkhorn_c.sinkhorn_eps,
            niter=sinkhorn_c.sinkhorn_niter,
            cuda=True, pi_detach=sinkhorn_c.sinkhorn_pi_detach)
          summary_dict['wd']['G_sinkhorn_d'] = sinkhorn_d.item()
          g_loss_only = - G_fake_mean
          summary_dict['scalars']['g_loss_only'] = g_loss_only.item()
          G_loss = g_loss_only + sinkhorn_c.sinkhorn_lambda * sinkhorn_d
        else:
          G_loss = - G_fake_mean
        # Accumulate gradients
        G_loss = G_loss / float(config['num_G_accumulations'])
        G_loss.backward()

        summary_dict['D_logit_mean']['G_fake_mean'] = G_fake_mean.item()
        summary_dict['scalars']['G_loss'] = G_loss.item()

      # Optionally apply modified ortho reg in G
      if config.G_ortho > 0.0:
        # Debug print to indicate we're using ortho reg in G
        print('using modified ortho reg in G')
        # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
        weight_regularity.ortho(
          self.G, config.G_ortho,
          blacklist=[param for param in self.G.shared.parameters()])
      self.G.optim.step()
      if getattr(config, 'weigh_loss', False):
        self.alpha_optim.step()

      # If we have an ema, update it, regardless of if we test with it or not
      self.ema.update(train_dict['batches_done'])
      pbar.set_description('wd=%f'%wd.item())

      if i % config.sample_every == 0:
        # weights
        self.save_checkpoint(filename='ckpt.tar')
        self.summary_dicts(summary_dicts=summary_dict, prefix='train_one_epoch',
                           step=train_dict['batches_done'])

        # singular values
        # G_sv_dict = utils_BigGAN.get_SVs(self.G, 'G')
        # D_sv_dict = utils_BigGAN.get_SVs(self.D, 'D')
        # myargs.writer.add_scalars('sv/G_sv_dict', G_sv_dict,
        #                           train_dict['batches_done'])
        # myargs.writer.add_scalars('sv/D_sv_dict', D_sv_dict,
        #                           train_dict['batches_done'])



      elif train_dict['batches_done'] <= config.sample_start_iter:
        # scalars
        self.summary_dicts(summary_dicts=summary_dict, prefix='train_one_epoch',
                           step=train_dict['batches_done'])

      if (i + 1) % (len(self.loaders[0]) // 2) == 0:
        # save images
        # self.summary_figures(summary_dicts=summary_dict,
        #                      prefix='train_one_epoch')
        # samples
        self._summary_images(imgs=x[0])




