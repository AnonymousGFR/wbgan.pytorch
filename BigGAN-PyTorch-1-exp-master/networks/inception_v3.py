import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import \
  (BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionAux, InceptionD,
   InceptionE, model_zoo, model_urls)


def inception_v3(pretrained=False, **kwargs):
  r"""Inception v3 model architecture from
  `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  if pretrained:
    if 'transform_input' not in kwargs:
      kwargs['transform_input'] = True
    model = Inception3(**kwargs)
    model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']),
                          strict=False)
    return model

  return Inception3(**kwargs)


class Inception3(nn.Module):

  def __init__(self, num_classes=1000, aux_logits=False, transform_input=False,
               image_size=32):
    super(Inception3, self).__init__()
    self.image_size = image_size
    self.aux_logits = aux_logits
    self.transform_input = transform_input
    self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
    self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
    self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
    self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
    self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
    self.Mixed_5b = InceptionA(192, pool_features=32)
    self.Mixed_5c = InceptionA(256, pool_features=64)
    self.Mixed_5d = InceptionA(288, pool_features=64)
    self.Mixed_6a = InceptionB(288)
    self.Mixed_6b = InceptionC(768, channels_7x7=128)
    self.Mixed_6c = InceptionC(768, channels_7x7=160)
    self.Mixed_6d = InceptionC(768, channels_7x7=160)
    self.Mixed_6e = InceptionC(768, channels_7x7=192)
    if aux_logits:
      self.AuxLogits = InceptionAux(768, num_classes)
    self.Mixed_7a = InceptionD(768)
    self.Mixed_7b = InceptionE(1280)
    self.Mixed_7c = InceptionE(2048)

    if self.image_size:
      self.fc_disc = nn.Linear(288, 1)
    self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        import scipy.stats as stats
        stddev = m.stddev if hasattr(m, 'stddev') else 0.1
        X = stats.truncnorm(-2, 2, scale=stddev)
        values = torch.Tensor(X.rvs(m.weight.data.numel()))
        values = values.view(m.weight.data.size())
        m.weight.data.copy_(values)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def forward(self, x):
    if self.transform_input:
      x = x.clone()
      x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
      x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
      x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    # 299 x 299 x 3
    x = self.Conv2d_1a_3x3(x)
    # 149 x 149 x 32
    x = self.Conv2d_2a_3x3(x)
    # 147 x 147 x 32
    x = self.Conv2d_2b_3x3(x)
    # 147 x 147 x 64
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 73 x 73 x 64
    x = self.Conv2d_3b_1x1(x)
    # 73 x 73 x 80
    x = self.Conv2d_4a_3x3(x)
    # 71 x 71 x 192
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 35 x 35 x 192
    x = self.Mixed_5b(x)
    # 35 x 35 x 256
    x = self.Mixed_5c(x)
    # 35 x 35 x 288
    x = self.Mixed_5d(x)
    # 35 x 35 x 288
    if self.image_size == 32:
      x = self.global_pool(x).view(x.size(0), -1)
      x = self.fc_disc(x)
      return x
    x = self.Mixed_6a(x)
    # 17 x 17 x 768
    x = self.Mixed_6b(x)
    # 17 x 17 x 768
    x = self.Mixed_6c(x)
    # 17 x 17 x 768
    x = self.Mixed_6d(x)
    # 17 x 17 x 768
    x = self.Mixed_6e(x)
    # 17 x 17 x 768
    x = self.Mixed_7a(x)
    # 8 x 8 x 1280
    x = self.Mixed_7b(x)
    # 8 x 8 x 2048
    x = self.Mixed_7c(x)
    # 8 x 8 x 2048
    x = F.avg_pool2d(x, kernel_size=8)
    # 1 x 1 x 2048
    x = F.dropout(x, training=self.training)
    # 1 x 1 x 2048
    x = x.view(x.size(0), -1)
    # 2048
    return x