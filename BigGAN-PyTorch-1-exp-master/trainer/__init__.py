import utils
import train


exe_dict = {
  'BigGAN_bs256x8': 'train',
  'BigGAN_bs256x8_wgan_gpreal': 'train',
  'BigGAN_bs256x8_wbgan_gpreal': 'train',
  'wgan_gp_celebaHQ1024': 'run_trainer'
}

parser_dict = {
  'BigGAN_bs256x8': utils.prepare_parser,
  'BigGAN_bs256x8_wgan_gpreal': utils.prepare_parser,
  'BigGAN_bs256x8_wgan_gpreal_adv': utils.prepare_parser,
  'BigGAN_bs256x8_wbgan_gpreal': utils.prepare_parser
}

run_dict = {
  'BigGAN_bs256x8': train.run,
  'BigGAN_bs256x8_wgan_gpreal': train.run,
  'BigGAN_bs256x8_wgan_gpreal_adv': train.run,
  'BigGAN_bs256x8_wbgan_gpreal': train.run
}

from . import (wgan_gp_trainer, wgan_gpreal_trainer, hingeloss_trainer)

trainer_dict = {
  'wgan_gp_celebaHQ1024': wgan_gp_trainer.Trainer,
  'cifar10_wgan_gpreal': wgan_gpreal_trainer.Trainer,
  'cifar10_wgan_gpreal_adv': wgan_gpreal_trainer.Trainer,
  'celebahq128_wgan_gp': wgan_gp_trainer.Trainer,
  'celebahq128_wgan_gpreal': wgan_gpreal_trainer.Trainer,
  'celebahq128_wgan_gpreal_bound': wgan_gpreal_trainer.Trainer,
  'celebahq128_wgan_gpreal_adv': wgan_gpreal_trainer.Trainer,
  'celebahq1024_wgan_gp': wgan_gp_trainer.Trainer,
  'celebahq1024_wgan_gpreal': wgan_gpreal_trainer.Trainer,
  'celebahq1024_wgan_gpreal_adv': wgan_gpreal_trainer.Trainer,

  'celeba64_wgan_gp': wgan_gp_trainer.Trainer,
  'celeba64_wgan_gp_wl': wgan_gp_trainer.Trainer,

  'cifar10_wgan_gp': wgan_gp_trainer.Trainer,
  'cifar10_wgan_gp_sinkhorn': wgan_gp_trainer.Trainer,
  'cifar10_hingeloss': hingeloss_trainer.Trainer,
}
