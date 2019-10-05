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
          r_logit_mean, f_logit_mean, D_loss = \
            gan_losses.hinge_loss_discriminator(r_logit=D_real, f_logit=D_fake)

          D_losses = D_loss / float(config.num_D_accumulations)
          # Accumulate gradients
          D_losses.backward()
          counter += 1
          summary_dict['D_logit_mean']['r_logit_mean'] = r_logit_mean.item()
          summary_dict['D_logit_mean']['f_logit_mean'] = f_logit_mean.item()
          summary_dict['scalars']['D_loss'] = D_loss.item()

        # End accumulation
        # Optionally apply ortho reg in D
        if config.D_ortho > 0.0:
          # Debug print to indicate we're using ortho reg in D.
          print('using modified ortho reg in D')
          weight_regularity.ortho(self.D, config.D_ortho)

        self.D.optim.step()

      # Optionally toggle "requires_grad"
      if config.toggle_grads:
        gan_utils.toggle_grad(self.D, False)
        gan_utils.toggle_grad(self.G, True)

      # Zero G's gradients by default before training G, for safety
      self.G.optim.zero_grad()
      # If accumulating gradients, loop multiple times
      for accumulation_index in range(config.num_G_accumulations):
        self.z_.sample_()
        self.y_.sample_()
        gy = self.y_
        # self.y_.sample_()
        D_fake, G_z = self.GD(
          z=self.z_, gy=gy, train_G=True, return_G_z=True,
          split_D=config.split_D)

        f_logit_mean, G_loss = gan_losses.hinge_loss_generator(f_logit=D_fake)

        # Accumulate gradients
        G_loss = G_loss / float(config['num_G_accumulations'])
        G_loss.backward()

        summary_dict['D_logit_mean']['G_f_logit_mean'] = f_logit_mean.item()
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

      # If we have an ema, update it, regardless of if we test with it or not
      self.ema.update(train_dict['batches_done'])

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
        self.summary_figures(summary_dicts=summary_dict,
                             prefix='train_one_epoch')
        # samples
        self._summary_images(imgs=x[0])




