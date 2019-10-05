import torch
import tqdm
import torch.autograd as autograd

from template_lib.gans import gan_losses

from . import trainer


def compute_grad2(d_out, x_in, backward=False, gp_lambda=10.,
                  retain_graph=True):
  batch_size = x_in.size(0)
  grad_dout = autograd.grad(
    outputs=d_out.sum(), inputs=x_in,
    create_graph=True, retain_graph=True, only_inputs=True
  )[0]
  grad_dout2 = grad_dout.pow(2)
  assert (grad_dout2.size() == x_in.size())
  reg = grad_dout2.view(batch_size, -1).sum(1)
  reg_mean = reg.mean()
  reg_mean = gp_lambda * reg_mean
  if backward:
    reg_mean.backward(retain_graph=retain_graph)
  return reg_mean, grad_dout.detach()


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
      f_imgs = self.G(self.z_train[:bs])

      imgs.requires_grad_()
      # train D
      D_r_logit = self.D(imgs)
      D_r_logit_mean = D_r_logit.mean()
      D_f_logit = self.D(f_imgs.detach())
      D_f_logit_mean = D_f_logit.mean()
      self.summary_logit_mean['D_r_logit_mean'] = D_r_logit_mean.item()
      self.summary_logit_mean['D_f_logit_mean'] = D_f_logit_mean.item()

      self.d_optimizer.zero_grad()

      # Wasserstein-1 Distance
      wd = D_r_logit_mean - D_f_logit_mean
      # Backward gp loss in this func
      gp, gp_img = compute_grad2(
        d_out=D_r_logit, x_in=imgs,
        backward=True, gp_lambda=config.gp_lambda, retain_graph=True)

      if config.adv_train:
        adv_lr = 0.01
        D_r_logit_adv = self.D(imgs - adv_lr * gp_img.sign())
        # D_r_logit_adv = self.D(imgs + adv_lr * gp_img.sign())
        D_r_logit_mean_adv = D_r_logit_adv.mean()
        adv_loss = D_r_logit_mean_adv.pow(2)
        adv_loss.backward()
        self.summary_logit_mean['D_r_logit_mean_adv'] = D_r_logit_mean_adv.item()
        pbar.set_description('D_r_logit_mean_adv: %f'%D_r_logit_mean_adv)

      if config.bound_type == 'constant':
        D_loss = -wd + torch.relu(wd - float(config.bound))
        # D_loss = -wd + gp * config.gp_lambda + \
        #          torch.relu(wd - float(config.bound))
        self.summary_wd['bound'] = config.bound
      else:
        D_loss = -wd
        # D_loss = -wd + gp * config.gp_lambda
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
        # self._summary_images(imgs=imgs)
        # checkpoint
        myargs.checkpoint.save_checkpoint(
          checkpoint_dict=myargs.checkpoint_dict, filename='ckpt.tar')
        # summary
        self._summary_scalars()

      elif train_dict['batches_done'] <= config.sample_start_iter:
        self._summary_scalars()

      if i % (len(self.data_loader) // 2) == 0:
        # save images
        self._summary_figures()
        self._summary_images(imgs=imgs, itr=self.train_dict['epoch_done'])
