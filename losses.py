import torch.autograd as autograd
from torch.autograd import grad
from torch.autograd import Variable
import torch
import torch.nn.functional as F

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss


def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss


def wgan_discriminator_loss(r_logit, f_logit):
  """
  d_loss = -wd + gp * 10.0
  :param r_logit:
  :param f_logit:
  :return:
  """
  r_logit_mean = r_logit.mean()
  f_logit_mean = f_logit.mean()

  # Wasserstein-1 Distance
  wd = r_logit_mean - f_logit_mean
  # wd = (r_logit - f_logit).mean()
  D_loss = -wd
  return r_logit_mean, f_logit_mean, wd, D_loss


def compute_grad2(d_out, x_in):
  batch_size = x_in.size(0)
  grad_dout = autograd.grad(
    outputs=d_out.sum(), inputs=x_in,
    create_graph=True, retain_graph=True, only_inputs=True
  )[0]
  grad_dout2 = grad_dout.pow(2)
  assert (grad_dout2.size() == x_in.size())
  reg = grad_dout2.view(batch_size, -1).sum(1)
  reg_mean = reg.mean()
  return reg, reg_mean


def wgan_gpreal_gradient_penalty(x, dy, f):
  x.requires_grad_()
  D_real = f(x=x, dy=dy)
  gpreal, gpreal_mean = compute_grad2(d_out=D_real, x_in=x)
  gpreal_mean.backward()
  return gpreal_mean


def wgan_generator_loss(f_logit):
  G_f_logit_mean = f_logit.mean()
  G_loss = - G_f_logit_mean
  return G_f_logit_mean, G_loss


def adv_loss(netD, img, y, gp_img, adv_lr=0.01, retain_graph=False):
  d_real_adv = netD(x=img + adv_lr * gp_img, dy=y)
  # D_r_logit_adv = self.D(imgs + adv_lr * gp_img.sign())
  d_real_mean_adv = d_real_adv.mean()
  adv_loss = d_real_mean_adv.pow(2)
  adv_loss.backward(retain_graph=retain_graph)
  return d_real_mean_adv.item()


# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis