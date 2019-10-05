import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import \
  (model_urls, model_zoo, BasicBlock, Bottleneck)


def resnet18(pretrained=False, **kwargs):
  """Constructs a ResNet-18 model.

  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2], net_type='resnet18', **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),
                          strict=False)
  return model


def resnet34(pretrained=False, **kwargs):
  """Constructs a ResNet-34 model.

  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3], net_type='resnet34', **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']),
                          strict=False)
  return model

def resnet50(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model.

  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3], net_type='resnet50' , **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),
                          strict=False)
  return model


def resnet101(pretrained=False, **kwargs):
  """Constructs a ResNet-101 model.

  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3], net_type='resnet101', **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']),
                          strict=False)
  return model


class ResNet(nn.Module):

  def __init__(self, block, layers, net_type):
    self.net_type = net_type
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    self.avgpool = nn.AvgPool2d(7, stride=1)
    # self.fc = nn.Linear(512 * block.expansion, num_classes)
    if self.net_type == 'resnet18':
      self.fc_disc = nn.Linear(512, 1)
    elif self.net_type == 'resnet34':
      self.fc_disc = nn.Linear(512, 1)
    elif self.net_type == 'resnet50':
      self.fc_disc = nn.Linear(2048, 1)
    elif self.net_type == 'resnet101':
      self.fc_disc = nn.Linear(2048, 1)
    else:
      assert 0
    self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.global_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc_disc(x)
    # x = x.mean(axis=1)
    return x