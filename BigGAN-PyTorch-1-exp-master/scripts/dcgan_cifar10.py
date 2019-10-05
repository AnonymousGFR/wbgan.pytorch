from __future__ import print_function
from torch.autograd import grad
import torch.autograd as autograd
from torch.autograd import Variable
import argparse
import functools
import os
import pprint
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models


from networks import inception_v3, resnet, darts
from template_lib.gans import gan_losses, inception_utils

# python dcgan.py --dataset cifar10 --dataroot /scratch/users/vision/yu_dl/raaz.rsk/data/cifar10 --imageSize 32 --cuda --outf out_cifar --manualSeed 13 --niter 100

# custom weights initialization called on netG and netD
import tqdm


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


class Generator(nn.Module):
  def __init__(self, ngpu, nc=3, nz=100, ngf=64):
    super(Generator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
      # input is Z, going into a convolution
      nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 8),
      nn.ReLU(True),
      # state size. (ngf*8) x 4 x 4
      nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      # state size. (ngf*4) x 8 x 8
      nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),
      # state size. (ngf*2) x 16 x 16
      nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0,
                         bias=False),
      nn.Tanh()
    )

  def forward(self, input):
    if input.is_cuda and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
      output = self.main(input)
    return output


class Discriminator(nn.Module):
  def __init__(self, ngpu, nc=3, ndf=64):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
      # input is (nc) x 64 x 64
      nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf) x 32 x 32
      nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*2) x 16 x 16
      nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*4) x 8 x 8
      nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 8),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*8) x 4 x 4
      nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
    )

  def forward(self, input):
    if input.is_cuda and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
      output = self.main(input)

    return output.view(-1, 1).squeeze(1)


class StrongDisc(nn.Module):
  def __init__(self, net_type, image_size=32, args=None):
    super(StrongDisc, self).__init__()
    self.net_type = net_type
    if net_type == 'inception_v3':
      self.net = inception_v3.inception_v3(pretrained=True, image_size=image_size)
    elif net_type == 'resnet18':
      self.net = resnet.resnet18(pretrained=True)
    elif net_type == 'resnet34':
      self.net = resnet.resnet34(pretrained=True)
    elif net_type == 'resnet50':
      self.net = resnet.resnet50(pretrained=True)
    elif net_type == 'resnet101':
      self.net = resnet.resnet101(pretrained=True)
    elif net_type == 'darts':
      self.net = darts.AugmentCNNOneOutput(model_path=args.darts_model)
    else:
      assert 0

  def forward(self, x):
    x = self.net(x)
    return x


def build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset',
                      help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
  parser.add_argument('--dataroot', help='path to dataset')
  parser.add_argument('--workers', type=int,
                      help='number of data loading workers', default=2)
  parser.add_argument('--batchSize', type=int, default=64,
                      help='input batch size')
  parser.add_argument('--imageSize', type=int, default=64,
                      help='the height / width of the input image to network')
  parser.add_argument('--nz', type=int, default=100,
                      help='size of the latent z vector')
  parser.add_argument('--ngf', type=int, default=64)
  parser.add_argument('--ndf', type=int, default=64)
  parser.add_argument('--niter', type=int, default=25,
                      help='number of epochs to train for')
  parser.add_argument('--lr', type=float, default=0.0002,
                      help='learning rate, default=0.0002')
  parser.add_argument('--beta1', type=float, default=0.5,
                      help='beta1 for adam. default=0.5')
  parser.add_argument('--cuda', action='store_true', help='enables cuda')
  parser.add_argument('--ngpu', type=int, default=1,
                      help='number of GPUs to use')
  parser.add_argument('--netG', default='',
                      help="path to netG (to continue training)")
  parser.add_argument('--netD', default='',
                      help="path to netD (to continue training)")
  parser.add_argument('--outf', default='.',
                      help='folder to output images and model checkpoints')
  parser.add_argument('--manualSeed', type=int, help='manual seed')
  parser.add_argument('--adv_train', action='store_true')
  parser.add_argument('--netD_type', type=str, default='DCGAN')
  parser.add_argument('--n_critic', type=int, default=1)
  return parser


def write_summary(summary, summary_d, summary_wd, writer, step):
  for k, v in summary.items():
    writer.add_scalar('train/%s'%k, v, step)
  writer.add_scalars('wd', summary_wd, step)
  writer.add_scalars('logit', summary_d, step)


def inception_metrics_func_create(config):
  config = config.inception_metric
  print('Load inception moments: %s' % config.saved_inception_moments)
  inception_metrics = inception_utils.InceptionMetrics(
    saved_inception_moments=config.saved_inception_moments)

  inception_metrics = functools.partial(
    inception_metrics,
    num_inception_images=config.num_inception_images,
    num_splits=10, prints=True)

  return inception_metrics

def sample_func(G, z_test):
  with torch.no_grad():
    G.eval()
    z = z_test.sample()
    G_z = G(z)

    G.train()
    return G_z

def test_inception_moments(inception_metrics, G, writer, epoch, config):
  nz = config.nz
  config = config.inception_metric
  z_test = torch.distributions.Normal(
    loc=torch.full((config.batch_size, nz, 1, 1), 0, device='cuda'),
    scale=torch.full((config.batch_size, nz, 1, 1), 1, device='cuda'))
  sample = functools.partial(sample_func, G=G, z_test=z_test)

  IS_mean, IS_std, FID = inception_metrics(
    G=G, z=z_test, show_process=False, use_torch=False,
    sample_func=sample)

  print('IS_mean: %f +- %f, FID: %f' % (IS_mean, IS_std, FID))
  summary = {'IS_mean': IS_mean, 'IS_std': IS_std, 'FID': FID}
  for key in summary:
    writer.add_scalar('test/' + key, summary[key], epoch)



def main(opt, args=None, myargs=None):
  opt.outimg = os.path.join(opt.outf, 'imgs')
  os.makedirs(opt.outimg, exist_ok=True)
  try:
    os.makedirs(opt.outf)
  except OSError:
    pass

  if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
  print("Random Seed: ", opt.manualSeed)
  random.seed(opt.manualSeed)
  torch.manual_seed(opt.manualSeed)

  cudnn.benchmark = True

  if torch.cuda.is_available() and not opt.cuda:
    print(
      "WARNING: You have a CUDA device, so you should probably run with --cuda")

  dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                         transform=transforms.Compose([
                           transforms.Resize(opt.imageSize),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5)),
                         ]))
  assert dataset
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                           shuffle=True,
                                           num_workers=int(opt.workers))

  device = torch.device("cuda:0" if opt.cuda else "cpu")
  ngpu = int(opt.ngpu)
  nz = int(opt.nz)

  netG = Generator(ngpu).to(device)
  netG.apply(weights_init)
  if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
  print(netG)

  if opt.netD_type == 'DCGAN':
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
  else:
    netD = StrongDisc(net_type=opt.netD_type, args=opt).to(device)

  if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
  print(netD)

  fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)

  # setup optimizer
  optimizerD = optim.Adam(netD.parameters(), lr=opt.lr,
                          betas=(opt.beta1, 0.999))
  optimizerG = optim.Adam(netG.parameters(), lr=opt.lr,
                          betas=(opt.beta1, 0.999))

  # load inception network
  inception_metrics = inception_metrics_func_create(opt)

  pbar = tqdm.tqdm(dataloader, file=myargs.stdout)
  step = 0
  for epoch in range(opt.niter):
    if opt.dummy_train:
      pbar = []
    for i, data in enumerate(pbar, 0):
      summary = {}
      summary_wd = {}
      summary_d = {}
      netD.train()
      netG.train()
      ############################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ###########################
      # train with real
      netD.zero_grad()
      real = data[0].to(device)
      batch_size = real.size(0)

      optimizerD.zero_grad()
      if opt.loss == 'wgan_gp':
        d_real = netD(real)
        d_real_mean = d_real.mean()
        summary_d['d_real_mean'] = d_real_mean.item()

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        d_fake = netD(fake.detach())
        d_fake_mean = d_fake.mean()
        summary_d['d_fake_mean'] = d_fake_mean.item()

        img_interp, gp_img, gp = wgan_gp_gradient_penalty(
          x=real, y=fake, f=netD, backward=True, gp_lambda=10.,
          return_grad=True)
        summary['gp'] = gp.item()
        if opt.adv_train:
          adv_lr = opt.adv_lr
          d_real_adv = netD(img_interp + adv_lr * gp_img)
          # d_real_adv = netD(img_interp - adv_lr * gp_img.sign())
          # D_r_logit_adv = self.D(imgs + adv_lr * gp_img.sign())
          d_real_mean_adv = d_real_adv.mean()
          adv_loss = d_real_mean_adv.pow(2)
          adv_loss.backward()
          summary_d['d_real_mean_adv'] = d_real_mean_adv.item()
          pbar.set_description('d_real_mean_adv: %f' % d_real_mean_adv)

        wd = d_real_mean - d_fake_mean
        summary_wd['wd'] = wd.item()
        if opt.use_bound:
          d_loss = -wd + torch.relu(wd - float(opt.bound))
          summary_wd['bound'] = opt.bound
        else:
          d_loss = -wd

        d_loss.backward()
        optimizerD.step()
        summary['d_loss'] = d_loss.item()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        d_fake_g = netD(fake)
        d_fake_g_mean = d_fake_g.mean()
        summary_d['d_fake_g_mean'] = d_fake_g_mean.item()
        g_loss = -d_fake_g_mean
        g_loss.backward()
        optimizerG.step()
        summary['g_loss'] = g_loss.item()
      elif opt.loss == 'wgan_div':
        d_real = netD(real)
        d_real_mean = d_real.mean()
        summary_d['d_real_mean'] = d_real_mean.item()

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        d_fake = netD(fake)
        d_fake_mean = d_fake.mean()
        summary_d['d_fake_mean'] = d_fake_mean.item()
        gp = wgan_div_gradient_penalty(
          x=real, y=fake, f=netD, gp_lambda=1., backward=True)

        wd = d_fake_mean - d_real_mean
        # wd = d_real_mean - d_fake_mean
        summary_wd['wd'] = wd.item()
        if opt.use_bound:
          d_loss = -wd + torch.relu(wd - float(opt.bound))
          summary_wd['bound'] = opt.bound
        else:
          d_loss = -wd

        d_loss.backward()
        optimizerD.step()
        summary['d_loss'] = d_loss.item()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        d_fake_g = netD(fake)
        d_fake_g_mean = d_fake_g.mean()
        summary_d['d_fake_g_mean'] = d_fake_g_mean.item()
        g_loss = d_fake_g_mean
        # g_loss = -d_fake_g_mean
        g_loss.backward()
        optimizerG.step()
        summary['g_loss'] = g_loss.item()
      elif opt.loss == 'wgan_gpreal':
        real.requires_grad_()
        d_real = netD(real)
        d_real_mean = d_real.mean()
        summary_d['d_real_mean'] = d_real_mean.item()

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        d_fake = netD(fake.detach())
        d_fake_mean = d_fake.mean()
        summary_d['d_fake_mean'] = d_fake_mean.item()
        if opt.adaptive_gp:
          gp_img, gp, g_norm_mean = gan_losses.compute_grad2_adaptive(
            d_out=d_real, x_in=real,
            backward=True, gp_lambda=opt.gp_lambda,
            retain_graph=True, return_grad=True)
          summary['gp'] = gp.item()
          summary['g_norm_mean'] = g_norm_mean
        else:
          gp_img, gp = gan_losses.compute_grad2(
            d_out=d_real, x_in=real,
            backward=True, gp_lambda=opt.gp_lambda,
            retain_graph=True, return_grad=True)
          summary['gp'] = gp.item()

        if opt.adv_train:
          adv_lr = opt.adv_lr
          adv_value = opt.adv_value
          # d_real_adv = netD(real - adv_lr * gp_img)
          # d_real_adv = netD(real - adv_lr * gp_img.sign())
          d_real_adv = netD(real + adv_lr * gp_img)
          # D_r_logit_adv = self.D(imgs + adv_lr * gp_img.sign())
          d_real_mean_adv = d_real_adv.mean()
          adv_loss = (d_real_mean_adv - adv_value).pow(2)
          adv_loss.backward()
          summary_d['d_real_mean_adv'] = d_real_mean_adv.item()
          pbar.set_description('d_real_mean_adv: %f' % d_real_mean_adv)

        wd = d_real_mean - d_fake_mean
        summary_wd['wd'] = wd.item()
        if opt.use_bound:
          d_loss = -wd + torch.relu(wd - float(opt.bound))
          summary_wd['bound'] = opt.bound
        else:
          d_loss = -wd

        d_loss.backward()
        optimizerD.step()
        summary['d_loss'] = d_loss.item()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if i % opt.n_critic == 0:
          netG.zero_grad()
          d_fake_g = netD(fake)
          d_fake_g_mean = d_fake_g.mean()
          summary_d['d_fake_g_mean'] = d_fake_g_mean.item()
          g_loss = -d_fake_g_mean
          g_loss.backward()
          optimizerG.step()
          summary['g_loss'] = g_loss.item()

      else:
        assert 0

      step += 1
      write_summary(summary, summary_d, summary_wd, writer=myargs.writer,
                    step=step)

      if i % 100 == 0:
        vutils.save_image(real,
                          '%s/real_samples.png' % opt.outf,
                          normalize=True)
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(),
                          '%s/fake_samples_epoch_%03d.png' \
                          % (opt.outimg, epoch),
                          normalize=True)
        merged_img = vutils.make_grid(fake.detach(), normalize=True)
        myargs.writer.add_images(
          'G_z', merged_img.view(1, *merged_img.shape), step)

    # end epoch
    test_inception_moments(
      inception_metrics=inception_metrics, G=netG, writer=myargs.writer,
      epoch=epoch, config=opt)
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch.pth' % (args.ckptdir, ))
    torch.save(netD.state_dict(), '%s/netD_epoch.pth' % (args.ckptdir, ))


def wgan_gp_gradient_penalty(x, y, f, backward=True,
                             gp_lambda=10., retain_graph=False,
                             return_grad=False):
  # interpolation
  device = x.device
  shape = [x.size(0)] + [1] * (x.dim() - 1)
  alpha = torch.rand(shape, device=device)
  z = x + alpha * (y - x)

  # gradient penalty
  z = Variable(z, requires_grad=True).cuda(device)
  o = f(z)
  img_gp = grad(o, z, grad_outputs=torch.ones(o.size(), device=device),
           create_graph=True)[0]
  g = img_gp.view(img_gp.size(0), -1)
  gp = ((g.norm(p=2, dim=1) - 1)**2).mean()

  gp_loss = gp * gp_lambda
  if backward:
    gp_loss.backward(retain_graph=retain_graph)
  if return_grad:
    return z.requires_grad_(False), img_gp.detach(), gp_loss
  else:
    return gp_loss


def wgan_div_gradient_penalty(x, y, f, backward=False,
                              gp_lambda=10., retain_graph=False):
  # Compute W-div gradient penalty
  k = 2
  p = 6
  device = x.device
  shape = [x.size(0)] + [1] * (x.dim() - 1)
  alpha = torch.rand(shape, device=device)
  z = x + alpha * (y - x)

  # gradient penalty
  z = Variable(z, requires_grad=True).cuda(device)
  o = f(z)
  g = grad(o, z, grad_outputs=torch.ones(o.size(), device=device),
           create_graph=True)[0].view(z.size(0), -1)

  grad_norm = g.view(g.size(0), -1).pow(2).sum(1) ** (p / 2)
  div_gp = torch.mean(grad_norm) * k / 2

  if backward:
    div_gp = gp_lambda * div_gp
    div_gp.backward(retain_graph=retain_graph)
  return div_gp


def run(args, myargs):
  parser = build_parser()
  opt = parser.parse_args()

  config = getattr(myargs.config, args.command)
  for k, v in config.items():
    setattr(opt, k, v)
  opt.outf = args.outdir
  opt.dataroot = os.path.expanduser(opt.dataroot)
  print(pprint.pformat(vars(opt)))
  main(opt, args, myargs)


if __name__ == '__main__':
  parser = build_parser()
  opt = parser.parse_args()
  print(opt)
  main(opt)
