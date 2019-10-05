import tqdm
import torch
import torchvision
from torch.autograd import grad
from torch.autograd import Variable

from template_lib.models import ema_model
from template_lib.gans import gan_utils, inception_utils, gan_losses, \
  sinkhorn_autodiff

from . import WBGANGP_DCGAN_CelebA64


def wgan_kgp_gradient_penalty(x, y, f):
  # interpolation
  shape = [x.size(0)] + [1] * (x.dim() - 1)
  alpha = torch.rand(shape, device='cuda')
  z = x + alpha * (y - x)

  # gradient penalty
  z = Variable(z, requires_grad=True).cuda()
  o = f(z)
  g = grad(o, z, grad_outputs=torch.ones(o.size(), device='cuda'), create_graph=True)[0].view(z.size(0), -1)
  with torch.no_grad():
    g_norm_mean = g.norm(p=2, dim=1).mean().item()
  gp = ((g.norm(p=2, dim=1) - g_norm_mean)**2).mean()

  return gp


class Trainer(WBGANGP_DCGAN_CelebA64.Trainer):
  def __init__(self, args, myargs):
    super().__init__(args=args, myargs=myargs)

  def model_create(self):
    from .. import models_64x64
    config = self.myargs.config.trainer.model

    if config.d_use_lp:
      self.logger.info("Use LayerNorm in D.")
      D = models_64x64.DiscriminatorWGANGPLN(3)
    else:
      self.logger.info("Use InstanceNorm in D.")
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
        gp, g_norm_mean = gan_losses.wgan_agp_gradient_penalty(
          imgs, f_imgs, self.D)
        self.summary['gp'] = gp.item()
        self.summary['g_norm_mean'] = g_norm_mean
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

        if config.lambda_sinkhorn > 0:
          sinkhorn_d = sinkhorn_autodiff.sinkhorn_loss(
            x=imgs.view(imgs.size(0), -1), y=f_imgs.view(f_imgs.size(0), -1),
            epsilon=config.sinkhorn.sinkhorn_eps,
            niter=config.sinkhorn.sinkhorn_niter,
            cuda=True, pi_detach=config.sinkhorn.sinkhorn_pi_detach)
          self.summary_sinkhorn['G_sinkhorn_d'] = sinkhorn_d.item()
        else:
          sinkhorn_d = 0

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
