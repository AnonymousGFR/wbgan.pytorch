import sys

import torch
import torch.nn as nn

DARTSPATH='../../pt.darts'


class AugmentCNNOneOutput(nn.Module):
  def __init__(self, model_path):
    sys.path.insert(0, DARTSPATH)
    model = torch.load(model_path)
    sys.path.pop(0)
    model.module.linear = nn.Linear(model.module.linear.in_features, 1)
    model.module.aux_pos = -1
    super(AugmentCNNOneOutput, self).__init__()
    self.model = model

  def forward(self, *input):
    logits, aux_logits = self.model(*input)
    return logits