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


import torch.nn as nn
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json


class Softmax(_Loss):
    """
    Softmax Loss
    """

    def __init__(self, **kwargs):
        super(Softmax, self).__init__()

    def forward(self, input, label, reduction="mean", **kwargs):
        return nn.CrossEntropyLoss()(input=input, target=label)


def create_loss():
    print("Loading Softmax Loss.")
    return Softmax()
