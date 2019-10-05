from .trainer import WGANGP_DCGAN_CelebA64
from .trainer import WBGANGP_DCGAN_CelebA64
from .trainer import WGAN_AGP_DCGAN_CelebA64
from .trainer import WBGAN_AGP_DCGAN_CelebA64


trainer_dict = {
  'WGANGP_DCGAN_CelebA64': WGANGP_DCGAN_CelebA64.Trainer,
  'WBGANGP_DCGAN_CelebA64': WBGANGP_DCGAN_CelebA64.Trainer,
  'WGAN_AGP_DCGAN_CelebA64': WGAN_AGP_DCGAN_CelebA64.Trainer,
  'WBGAN_AGP_DCGAN_CelebA64': WBGAN_AGP_DCGAN_CelebA64.Trainer
}