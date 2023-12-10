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
import optuna


class GammaSoftmax(_Loss):
    """
    Gamma Softmax Loss
    """

    def __init__(self, freq_path, tau, iter):
        super(GammaSoftmax, self).__init__()
        with open(freq_path, "r") as fd:
            freq = json.load(fd)
        freq = torch.tensor(freq)
        self.sample_per_class = freq
        self.tau = tau
        self.iter = iter

    def forward(self, input, label, reduction="mean", **kwargs):
        if isinstance(self.tau, list):
            print("Start to auto training")
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: self.objective(trial, label, input, reduction, **kwargs), n_trials=100)
            best_params = study.best_params
            print("Best parameters: ", best_params)

        return gamma_softmax_loss(label, input, self.sample_per_class, reduction, best_params['tau'], best_params['iter'], **kwargs)
    
    # 定义目标函数
    def objective(self, trial, label, input, reduction, **kwargs):
        tau = trial.suggest_uniform('tau', self.tau[0], self.tau[1])
        iter = trial.suggest_int('iter', self.iter[0], self.iter[1])
        loss = gamma_softmax_loss(label, input, self.sample_per_class, reduction, tau, iter, **kwargs)
        # trial.report(loss, step=trial.number)
        return loss


def gamma_softmax_loss(labels, logits, sample_per_class, reduction, tau, iter, **kwargs):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    num_a = len(labels)
    sample_per_class = sample_per_class.type_as(logits)
    spc = torch.bincount(labels, minlength=sample_per_class.shape[0])
    spc = spc.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)

    uniform = torch.ones(num_a).to(logits.device) / num_a
    ratio_per_class = sample_per_class / sample_per_class.sum()
    logits = logits + spc.log()
    bd = ((1 - tau) * ratio_per_class).to(logits.device)
    bu = ((1 + tau) * ratio_per_class).to(logits.device)

    gamma = gammasinkhorn(logits, uniform, bd, bu, epsilon=1, iter=iter)
    # sum_of_col = torch.sum(gamma, dim=0).to(logits.device)
    # tar = torch.min(torch.max(sum_of_col, bd), bu)

    # loss = F.cross_entropy(input=sum_of_col, target=tar, reduction=reduction) * 0.4 - torch.log(gamma[range(num_a), labels]).mean() * 0.6
    # loss = F.cross_entropy(input=logits, target=labels, reduction=reduction) + F.cross_entropy(input=sum_of_col, target=tar, reduction=reduction)
    loss = - torch.log(gamma[range(num_a), labels]).mean()
    return loss


def create_loss(freq_path, tau, iter):
    print("Loading Gamma Softmax Loss.")
    return GammaSoftmax(freq_path, tau, iter)


def gammasinkhorn(C, a, bd, bu, epsilon=1, iter=1):
    """Solve the entropic regularization optimal transport problem.
    Args:
        C: cost matrix, of shape (batch_size, num_class)
        a: a tensor of shape (batch_size, )
        b: a tensor of shape (num_class, )
        epsilon: a float, the regularization parameter
        iter: number of iterations
    """
    device = C.device
    batch_size = C.shape[0]
    num_class = C.shape[1]

    a = a.reshape(1, batch_size)  # (1, B)
    bd = bd.reshape(1, num_class)  # (1, N)
    bu = bu.reshape(1, num_class)  # (1, N)
    gamma = torch.exp(C / epsilon).to(device)  # (B, N)

    u = torch.ones_like(a).to(device)  # (1, B)
    v = torch.ones_like(bu).to(device)  # (1, N)
    ea = torch.full_like(a, 1e-10).to(device)
    eb = torch.full_like(bu, 1e-10).to(device)

    for _ in range(iter):
        gamma = (gamma.T * a / torch.max(torch.sum(gamma, dim=1), ea)).T
        # gamma = gamma * torch.max(bd / torch.max(torch.sum(gamma, dim=0), eb), v)
        sum_of_col = torch.sum(gamma, dim=0)
        b = torch.min(torch.max(sum_of_col, bd), bu)
        # gamma = gamma * torch.min(bu / torch.max(torch.sum(gamma, dim=0), eb), v)
        gamma = gamma * b / torch.max(torch.sum(gamma, dim=0), eb)

    return gamma


