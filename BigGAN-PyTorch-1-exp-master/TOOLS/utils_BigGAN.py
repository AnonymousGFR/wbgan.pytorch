import torch




def get_SVs(net, prefix):
  """
  # Get singular values to log. This will use the state dict to find them
  # and substitute underscores for dots.
  :param net:
  :param prefix:
  :return:
  """
  d = net.state_dict()
  return {('%s_%s' % (prefix, key)).replace('.', '_') :
            float(d[key].item())
            for key in d if 'sv' in key}


def load_weights(G, D, G_ema=None, state_dict=None,
                 resume_dir=None,
                 name_suffix=None,
                 strict=True, logger=None,
                 load_G=True, load_G_optim=True, load_D=True, load_D_optim=True, load_state=True, load_config=False,
                 **kwargs):
  """
  # Load a model's weights, optimizer, and the state_dict

  :return:
  """
  def join_strings(base_string, strings):
    return base_string.join([item for item in strings if item])

  root = resume_dir
  if name_suffix:
    logger.info_msg('Loading %s weights from %s...' % (name_suffix, root))
  else:
    logger.info_msg('Loading weights from %s...' % root)
  if G is not None and load_G:
    G.load_state_dict(
      torch.load('%s/%s.pth' % (root, join_strings('_', ['G', name_suffix]))),
      strict=strict)
    if load_G_optim:
      G.optim.load_state_dict(
        torch.load('%s/%s.pth' % (root, join_strings('_', ['G_optim', name_suffix]))))
  if D is not None and load_D:
    D.load_state_dict(
      torch.load('%s/%s.pth' % (root, join_strings('_', ['D', name_suffix]))),
      strict=strict)
    if load_D_optim:
      D.optim.load_state_dict(
        torch.load('%s/%s.pth' % (root, join_strings('_', ['D_optim', name_suffix]))))
  # Load state dict
  if load_state:
    for item in state_dict:
      if item == 'config' and not load_config:
        continue
      state_dict[item] = torch.load('%s/%s.pth' % (root, join_strings('_', ['state_dict', name_suffix])))[item]
  if G_ema is not None:
    G_ema.load_state_dict(
      torch.load('%s/%s.pth' % (root, join_strings('_', ['G_ema', name_suffix]))),
      strict=strict)
  return