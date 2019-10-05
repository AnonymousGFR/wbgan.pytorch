import os
import sys
import unittest
import argparse
os.chdir('..')
from template_lib import utils


class TestingPrepareData(unittest.TestCase):

  def test_ImageNet128_make_hdf5(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../submodule:..
        python -c "import test_BigGAN; \
        test_BigGAN.TestingPrepareData().test_ImageNet128_make_hdf5()"
    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/dataset', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/dataset.yaml 
            --command ImageNet128_make_hdf5
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from trainer import run
    run.main(args, myargs)
    input('End %s' % outdir)
    return

  def test_ImageNet128_calculate_inception_moments(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../submodule:..
        python -c "import test_BigGAN; \
        test_BigGAN.TestingPrepareData().test_ImageNet128_calculate_inception_moments()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/dataset', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/dataset.yaml 
            --command ImageNet128_calculate_inception_moments
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from trainer import run
    run.main(args, myargs)
    input('End %s' % outdir)
    return

  def test_Calculate_inception_moments_CelebaHQ1024(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6011
        export TIME_STR=1
        export PYTHONPATH=../submodule:.
        python -c "import test_BigGAN; \
        test_BigGAN.Prepare_data().test_Calculate_inception_moments_CelebaHQ1024()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6012'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

      # func name
    outdir = os.path.join('results/dataset',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
                --config configs/biggan_celebahq.yaml
                --command CelebaHQ1024
                --resume False --resume_path None
                --resume_root None
                """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from TOOLS import calculate_inception_moments
    calculate_inception_moments.create_inception_moments(args, myargs)
    print('End %s'%outdir)
    return

  def test_Calculate_inception_moments_CelebaHQ128(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=5
        export PORT=6011
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..:..
        python -c "import test_BigGAN; \
        test_BigGAN.TestingPrepareData().test_Calculate_inception_moments_CelebaHQ128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

      # func name
    outdir = os.path.join('results/dataset',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
                --config configs/biggan_celebahq.yaml
                --command CelebaHQ128
                --resume False --resume_path None
                --resume_root None
                """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from TOOLS import calculate_inception_moments
    calculate_inception_moments.create_inception_moments(args, myargs)
    input('End %s' % outdir)
    return


class TestingTrainBigGAN(unittest.TestCase):

  def test_ImageNet128_train(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6011
        export TIME_STR=1
        export PYTHONPATH=../../submodule:..
        python -c "import test_BigGAN; \
        test_BigGAN.TestingTrainBigGAN().test_ImageNet128_train()"
    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6211'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_imagenet128.yaml 
            --command BigGAN_bs256x8
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.train(args, myargs)
    input('End %s' % outdir)
    return


class TestingTrainBigGAN_WGAN_GPReal(unittest.TestCase):

  def test_ImageNet128_train_wgan_gpreal(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6011
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..
        python -c "import test_BigGAN; \
        test_BigGAN.TestingTrainBigGAN_WGAN_GPReal().test_ImageNet128_train_wgan_gpreal()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_imagenet128.yaml 
            --command BigGAN_bs256x8_wgan_gpreal
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.train(args, myargs)
    input('End %s' % outdir)
    return

  def test_ImageNet128_train_wgan_gpreal_adv(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6011
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..
        python -c "import test_BigGAN; \
        test_BigGAN.TestingTrainBigGAN_WGAN_GPReal().test_ImageNet128_train_wgan_gpreal_adv()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_imagenet128.yaml 
            --command BigGAN_bs256x8_wgan_gpreal_adv
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.train(args, myargs)
    input('End %s' % outdir)
    return

  def test_ImageNet128_train_wbgan_gpreal(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6011
        export TIME_STR=1
        export PYTHONPATH=../submodule:..
        python -c "import test_BigGAN; \
        test_BigGAN.TestingTrainBigGAN_WGAN_GPReal().test_ImageNet128_train_wbgan_gpreal()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_imagenet128.yaml 
            --command BigGAN_bs256x8_wbgan_gpreal
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from trainer import run
    run.main(args, myargs)
    input('End %s' % outdir)
    return

  def test_CelebaHQ1024_biggan_wgan_gp(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6111
        export TIME_STR=1
        export PYTHONPATH=../submodule:.
        python -c "import test_BigGAN; \
          test_BigGAN.TestingTrainBigGAN_WGAN_GPReal().test_CelebaHQ1024_biggan_wgan_gp()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config configs/biggan_celebahq.yaml
            --command wgan_gp_celebaHQ1024
            --resume False 
            --resume_path None 
            --resume_root None 
            --evaluate False --evaluate_path None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from trainer import run
    run.main(args=args, myargs=myargs)
    input('End %s' % outdir)

    return


class Testing_train_BigGAN_cifar10(unittest.TestCase):

  def test_cifar10_wgan_gp(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=3
        export PORT=6009
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..:../
        python -c "import test_BigGAN; \
        test_BigGAN.Testing_train_BigGAN_cifar10().test_cifar10_wgan_gp()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN_cifar10',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_cifar10.yaml 
            --command cifar10_wgan_gp
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.run_trainer(args, myargs)
    input('End %s' % outdir)
    return

  def test_cifar10_wgan_gp_sinkhorn(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=0
        export PORT=6007
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..:../
        python -c "import test_BigGAN; \
        test_BigGAN.Testing_train_BigGAN_cifar10().test_cifar10_wgan_gp_sinkhorn()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN_cifar10',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_cifar10.yaml 
            --command cifar10_wgan_gp_sinkhorn
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.run_trainer(args, myargs)
    input('End %s' % outdir)
    return

  def test_cifar10_hingeloss(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=2
        export PORT=6008
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..:../
        python -c "import test_BigGAN; \
        test_BigGAN.Testing_train_BigGAN_cifar10().test_cifar10_hingeloss()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN_cifar10',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_cifar10.yaml 
            --command cifar10_hingeloss
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.run_trainer(args, myargs)
    input('End %s' % outdir)
    return

  def test_cifar10_wgan_gpreal(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=1
        export PORT=6107
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..:../
        python -c "import test_BigGAN; \
        test_BigGAN.Testing_train_BigGAN_cifar10().test_cifar10_wgan_gpreal()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN_cifar10',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_cifar10.yaml 
            --command cifar10_wgan_gpreal
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.run_trainer(args, myargs)
    input('End %s' % outdir)
    return

  def test_cifar10_wgan_gpreal_adv(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=2
        export PORT=6108
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..:../
        python -c "import test_BigGAN; \
        test_BigGAN.Testing_train_BigGAN_cifar10().test_cifar10_wgan_gpreal_adv()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN_cifar10',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_cifar10.yaml 
            --command cifar10_wgan_gpreal_adv
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.run_trainer(args, myargs)
    input('End %s' % outdir)
    return


class Testing_train_BigGAN_Celeba64(unittest.TestCase):

  def test_celeba64_wgan_gp(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=3
        export PORT=6009
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..:../
        python -c "import test_BigGAN; \
        test_BigGAN.Testing_train_BigGAN_Celeba64().test_celeba64_wgan_gp()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6013'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN_celeba64',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_celeba64.yaml 
            --command celeba64_wgan_gp
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.run_trainer(args, myargs)
    input('End %s' % outdir)
    return

  def test_celeba64_wgan_gp_wl(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=2
        export PORT=6008
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..:../
        python -c "import test_BigGAN; \
        test_BigGAN.Testing_train_BigGAN_Celeba64().test_celeba64_wgan_gp_wl()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6012'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN_celeba64',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_celeba64.yaml 
            --command celeba64_wgan_gp_wl
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.run_trainer(args, myargs)
    input('End %s' % outdir)
    return


class Testing_train_BigGAN_CelebaHQ128(unittest.TestCase):

  def test_celebahq128_wgan_gp(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=5
        export PORT=6011
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..:../
        python -c "import test_BigGAN; \
        test_BigGAN.Testing_train_BigGAN_CelebaHQ128().test_celebahq128_wgan_gp()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN_celebahq128',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_celebahq.yaml 
            --command celebahq128_wgan_gp
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.run_trainer(args, myargs)
    input('End %s' % outdir)
    return

  def test_celebahq128_wgan_gpreal(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=5
        export PORT=6011
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..:../
        python -c "import test_BigGAN; \
        test_BigGAN.Testing_train_BigGAN_CelebaHQ128().test_celebahq128_wgan_gpreal()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6012'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN_celebahq128',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_celebahq.yaml 
            --command celebahq128_wgan_gpreal
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.run_trainer(args, myargs)
    input('End %s' % outdir)
    return

  def test_celebahq128_wgan_gpreal_bound(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=3
        export PORT=6009
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..:../
        python -c "import test_BigGAN; \
        test_BigGAN.Testing_train_BigGAN_CelebaHQ128().test_celebahq128_wgan_gpreal_bound()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6012'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN_celebahq128',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_celebahq.yaml 
            --command celebahq128_wgan_gpreal_bound
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.run_trainer(args, myargs)
    input('End %s' % outdir)
    return

  def test_celebahq128_wgan_gpreal_adv(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=2
        export PORT=6008
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..:../
        python -c "import test_BigGAN; \
        test_BigGAN.Testing_train_BigGAN_CelebaHQ128().test_celebahq128_wgan_gpreal_adv()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6012'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN_celebahq128',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_celebahq.yaml 
            --command celebahq128_wgan_gpreal_adv
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.run_trainer(args, myargs)
    input('End %s' % outdir)
    return


class Testing_train_BigGAN_CelebaHQ1024(unittest.TestCase):

  def test_celebahq1024_wgan_gp(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=2
        export PORT=6008
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..:../
        python -c "import test_BigGAN; \
        test_BigGAN.Testing_train_BigGAN_CelebaHQ1024().test_celebahq1024_wgan_gp()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6008'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN_celebahq1024',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_celebahq.yaml 
            --command celebahq1024_wgan_gp
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.run_trainer(args, myargs)
    input('End %s' % outdir)
    return

  def test_celebahq1024_wgan_gpreal(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=2
        export PORT=6008
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..:../
        python -c "import test_BigGAN; \
        test_BigGAN.Testing_train_BigGAN_CelebaHQ1024().test_celebahq1024_wgan_gpreal()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6008'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN_celebahq1024',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_celebahq.yaml 
            --command celebahq1024_wgan_gpreal
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.run_trainer(args, myargs)
    input('End %s' % outdir)
    return

  def test_celebahq1024_wgan_gpreal_adv(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=2
        export PORT=6008
        export TIME_STR=1
        export PYTHONPATH=../../submodule:../..:../
        python -c "import test_BigGAN; \
        test_BigGAN.Testing_train_BigGAN_CelebaHQ1024().test_celebahq1024_wgan_gpreal_adv()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6008'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN_celebahq1024',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_celebahq.yaml 
            --command celebahq1024_wgan_gpreal_adv
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from trainer import run
    run.run_trainer(args, myargs)
    input('End %s' % outdir)
    return