import torch

import torchvision
import tqdm

from TOOLS import gan_losses
from TOOLS import sinkhorn_autodiff


def train_func(data_loader, G, D, G_ema, ema, z_train, g_optimizer, d_optimizer, z_sample, train_dict, args, myargs):
  def train(**kwargs):
    for i, (imgs, _) in enumerate(tqdm.tqdm(data_loader)):
      train_dict['batches_done'] += 1
      step = train_dict['batches_done']
      summary = {}
      summary_d_logits_mean = {}
      summary_wd_sinkhornd = {}

      G.train()
      G_ema.train()

      imgs = imgs.cuda()
      bs = imgs.size(0)

      z_train.sample_()
      with torch.no_grad():
        f_imgs = G(z_train[:bs])

      # train D
      # with torch.no_grad():
      #   sinkhorn_d = sinkhorn_autodiff.sinkhorn_loss(x=imgs.view(bs, -1), y=f_imgs.view(bs, -1).detach(),
      #                                                epsilon=args.sinkhorn_eps, niter=args.sinkhorn_niter,
      #                                                cuda=True, pi_detach=args.sinkhorn_pi_detach)
      #   summary_wd_sinkhornd['D_sinkhorn_d'] = sinkhorn_d.item()

      r_logit = D(imgs)
      r_logit_mean = r_logit.mean()
      f_logit = D(f_imgs.detach())
      f_logit_mean = f_logit.mean()
      summary_d_logits_mean['D_r_logit_mean'] = r_logit_mean.item()
      summary_d_logits_mean['D_f_logit_mean'] = f_logit_mean.item()

      # Wasserstein-1 Distance
      wd = r_logit_mean - f_logit_mean
      gp = gan_losses.wgan_gp_gradient_penalty(imgs.data, f_imgs.data, D)
      d_loss = -wd + gp * 10.0
      summary_wd_sinkhornd['wd'] = wd.item()
      summary['gp'] = gp.item()
      summary['d_loss'] = d_loss.item()

      D.zero_grad()
      d_loss.backward()
      d_optimizer.step()

      if step % args.n_critic == 0:
        # train G
        z_train.sample_()
        f_imgs = G(z_train)

        sinkhorn_d = sinkhorn_autodiff.sinkhorn_loss(x=imgs.view(imgs.size(0), -1), y=f_imgs.view(f_imgs.size(0), -1),
                                                     epsilon=args.sinkhorn_eps, niter=args.sinkhorn_niter,
                                                     cuda=True, pi_detach=args.sinkhorn_pi_detach)
        summary_wd_sinkhornd['G_sinkhorn_d'] = sinkhorn_d.item()

        f_logit = D(f_imgs)
        f_logit_mean = f_logit.mean()
        g_loss = - f_logit_mean + args.lambda_sinkhorn * sinkhorn_d
        summary_d_logits_mean['G_f_logit_mean'] = f_logit_mean.item()
        summary['g_loss'] = g_loss.item()

        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

      # end iter
      ema.update(train_dict['batches_done'])

      if i % args.sample_every == 0:
        # sample images
        G.eval()
        G_ema.eval()
        G_z = G(z_sample)
        merged_img = torchvision.utils.make_grid(G_z, normalize=True, pad_value=1, nrow=16)
        myargs.writer.add_images('G_z', merged_img.view(1, *merged_img.shape), train_dict['batches_done'])
        # G_ema
        G_ema_z = G_ema(z_sample)
        merged_img = torchvision.utils.make_grid(G_ema_z, normalize=True, pad_value=1, nrow=16)
        myargs.writer.add_images('G_ema_z', merged_img.view(1, *merged_img.shape), train_dict['batches_done'])
        # x
        merged_img = torchvision.utils.make_grid(imgs, normalize=True, pad_value=1, nrow=16)
        myargs.writer.add_images('x', merged_img.view(1, *merged_img.shape), train_dict['batches_done'])
        # checkpoint
        myargs.checkpoint.save_checkpoint(checkpoint_dict=myargs.checkpoint_dict, filename='ckpt.tar')
        # summary
        for key in summary:
          myargs.writer.add_scalar('train_vs_batch/%s'%key, summary[key], train_dict['batches_done'])
        myargs.writer.add_scalars('train_vs_batch', summary_d_logits_mean, train_dict['batches_done'])
        myargs.writer.add_scalars('dist_vs_batch', summary_wd_sinkhornd, train_dict['batches_done'])

        G.train()

  return train