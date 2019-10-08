# Environment
- python3.6
```bash
pip install -r requirements.txt
```
# Prepare inception moment file
* Download CelebA (Align&Cropped Images) dataset [url](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and save *img_align_celeba* into *~/.keras/celeba/*.

You'll need the Inception moments to calculate FID:
```bash
export CUDA_VISIBLE_DEVICES=0
cd BigGAN-PyTorch-1-exp-master
export PYTHONPATH=../submodule:..
python main.py \
  --config ./configs/prepare_data.yaml \
  --command Calculate_inception_moments_Celeba64 \
  --outdir results/temp/Calculate_inception_moments_Celeba64
```
# Train
## wgan-gp on celeba64
```bash
export CUDA_VISIBLE_DEVICES=0
cd BigGAN-PyTorch-1-exp-master
export PYTHONPATH=../submodule:..
python main.py \
  --config DCGAN/configs/dcgan_celeba64.yaml \
  --command CelebA64_dcgan_wgan_gp \
  --outdir results/temp/CelebA64_dcgan_wgan_gp
```
## wbgan-gp on celeba64
```bash
export CUDA_VISIBLE_DEVICES=0
cd BigGAN-PyTorch-1-exp-master
export PYTHONPATH=../submodule:..
python main.py \
  --config DCGAN/configs/dcgan_celeba64.yaml \
  --command CelebA64_dcgan_wgan_gp_bound_sinkhorn \
  --outdir results/temp/CelebA64_dcgan_wgan_gp_bound_sinkhorn
```
# Acknowledgments
Thanks BigGAN code from https://github.com/ajbrock/BigGAN-PyTorch.
DCGAN code from https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch.







