import torch

import torchvision
import tqdm

from TOOLS import gan_losses

def train_func(data_loader, G, D, G_ema, ema, z_train, g_optimizer, d_optimizer, z_sample, train_dict, args, myargs):
  def train(**kwargs):
    for i, (imgs, _) in enumerate(tqdm.tqdm(data_loader)):
      train_dict['batches_done'] += 1
      step = train_dict['batches_done']
      summary = {}
      summary_scalars = {}

      G.train()

      imgs = imgs.cuda()
      imgs.requires_grad_()
      bs = imgs.size(0)

      z_train.sample_()
      with torch.no_grad():
        f_imgs = G(z_train[:bs])

      # train D
      r_logit = D(imgs)
      r_logit_mean = r_logit.mean()
      f_logit = D(f_imgs)
      f_logit_mean = f_logit.mean()
      summary_scalars['D_r_logit_mean'] = r_logit_mean.item()
      summary_scalars['D_f_logit_mean'] = f_logit_mean.item()

      # Wasserstein-1 Distance
      wd = r_logit_mean - f_logit_mean
      gp_real, gp_real_mean = gan_losses.compute_grad2(d_out=r_logit, x_in=imgs)

      d_loss = -wd + 10. * gp_real_mean
      summary['wd'] = wd.item()
      summary['gp_real_mean'] = gp_real_mean.item()
      summary['d_loss'] = d_loss.item()

      D.zero_grad()
      d_loss.backward()
      d_optimizer.step()

      if step % args.n_critic == 0:
        # train G
        z_train.sample_()
        f_imgs = G(z_train)
        f_logit = D(f_imgs)
        f_logit_mean = f_logit.mean()
        g_loss = - f_logit_mean
        summary_scalars['G_f_logit_mean'] = f_logit_mean.item()
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
        f_imgs_sample = G(z_sample)
        merged_img = torchvision.utils.make_grid(f_imgs_sample, normalize=True, pad_value=1, nrow=16)
        myargs.writer.add_images('G_z', merged_img.view(1, *merged_img.shape), train_dict['batches_done'])
        # G_ema
        G_ema.eval()
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
        myargs.writer.add_scalars('train_vs_batch', summary_scalars, train_dict['batches_done'])

        G.train()
  return train