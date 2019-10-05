#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# wgan-gp on celeba64
python main.py \
  --config DCGAN/configs/dcgan_celeba64.yaml \
  --command CelebA64_dcgan_wgan_gp \
  --outdir results/temp/CelebA64_dcgan_wgan_gp

## wbgan-gp on celeba64
#python main.py \
#  --config DCGAN/configs/dcgan_celeba64.yaml \
#  --command CelebA64_dcgan_wgan_gp_bound_sinkhorn \
#  --outdir results/temp/CelebA64_dcgan_wgan_gp_bound_sinkhorn
