import torch
import torchvision
import tqdm

from TOOLS import gan_losses
from TOOLS import sinkhorn_autodiff


def train(data_loader, G, z_train, g_optimizer, z_sample, train_dict, args, myargs, **kwargs):
  for i, (imgs, _) in enumerate(tqdm.tqdm(data_loader)):
    train_dict['batches_done'] += 1
    summary = {}

    G.train()
    imgs = imgs.cuda()

    # train G
    z_train.sample_()
    f_imgs = G(z_train)

    sinkhorn_d = sinkhorn_autodiff.sinkhorn_loss(x=imgs.view(imgs.size(0), -1), y=f_imgs.view(f_imgs.size(0), -1),
                                                 epsilon=args.sinkhorn_eps, niter=args.sinkhorn_niter,
                                                 cuda=True, pi_detach=args.sinkhorn_pi_detach)
    summary['sinkhorn_d'] = sinkhorn_d.item()

    g_loss = sinkhorn_d
    summary['g_loss'] = g_loss.item()

    G.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    if i % args.sample_every == 0:
      # sample images
      G.eval()
      f_imgs_sample = G(z_sample)
      merged_img = torchvision.utils.make_grid(f_imgs_sample, normalize=True, pad_value=1, nrow=16)
      myargs.writer.add_images('G_z', merged_img.view(1, *merged_img.shape), train_dict['batches_done'])
      merged_img = torchvision.utils.make_grid(imgs, normalize=True, pad_value=1, nrow=16)
      myargs.writer.add_images('x', merged_img.view(1, *merged_img.shape), train_dict['batches_done'])
      # checkpoint
      myargs.checkpoint.save_checkpoint(checkpoint_dict=myargs.checkpoint_dict, filename='ckpt.tar')
      # summary
      for key in summary:
        myargs.writer.add_scalar('train_vs_batch/%s'%key, summary[key], train_dict['batches_done'])

      G.train()
