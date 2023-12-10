"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json


class LogitAdjustment(_Loss):
    """
    LogitAd justment Loss
    """

    def __init__(self, freq_path):
        super(LogitAdjustment, self).__init__()
        with open(freq_path, "r") as fd:
            freq = json.load(fd)
        freq = torch.tensor(freq)
        self.sample_per_class = freq

    def forward(self, input, label, reduction="mean", **kwargs):
        return logit_adjustment_loss(label, input, self.sample_per_class, reduction, **kwargs)


def logit_adjustment_loss(labels, logits, sample_per_class, reduction, **kwargs):
    """Compute the Logit Adjustment Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Logit Adjustment Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits - spc.log()  # `-` instead of `+` in Balanced Softmax Loss
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def create_loss(freq_path):
    print("Loading Logit Adjustment Loss.")
    return LogitAdjustment(freq_path)
