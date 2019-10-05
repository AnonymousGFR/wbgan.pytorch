import argparse

def none_or_str(value):
    if value == 'None':
        return None
    return value

def true_or_false(value):
    if value == 'True':
        return True
    return False

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_critic', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--datadir', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--resume', type=true_or_false, default=False)
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--sample_every', type=int, default=100)
    parser.add_argument('--saved_inception_moments', type=str, default='')
    parser.add_argument('--num_inception_images', type=int, default=50000)
    parser.add_argument('--evaluate', type=true_or_false, default=False)
    parser.add_argument('--evaluate_path', type=str, default='')
    parser.add_argument('--train_func', type=str, default='')
    parser.add_argument('--sinkhorn_eps', type=float, default=0.01)
    parser.add_argument('--sinkhorn_niter', type=int, default=100)
    parser.add_argument('--sinkhorn_pi_detach', type=true_or_false, default=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ema_start', type=int, default=5000)
    parser.add_argument('--lambda_sinkhorn', type=float, default=1.)
    return parser
