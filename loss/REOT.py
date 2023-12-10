import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json


class OTSoftmax(_Loss):
    """
    OT Softmax Loss
    """

    def __init__(self, freq_path):
        super(OTSoftmax, self).__init__()
        with open(freq_path, "r") as fd:
            freq = json.load(fd)
        freq = torch.tensor(freq)
        self.sample_per_class = freq

    def forward(self, input, label, reduction="mean", **kwargs):
        return ot_softmax_loss(label, input, self.sample_per_class, reduction, **kwargs)


def ot_softmax_loss(labels, logits, sample_per_class, reduction="mean", **kwargs):
    """Compute the OT Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. OT Softmax Loss.
    """
    epoch = kwargs.get("epoch", 0)
    B = len(labels)
    uniform = torch.ones(B).to(logits.device) / B
    crit_def = kwargs.get("crit_def", dict())
    t1 = crit_def.get("t1", 500)
    t2 = crit_def.get("t2", 680)
    t3 = crit_def.get("t3", 790)
    lamb = crit_def.get("lamb", 0.3)
    gama = crit_def.get("gama", 0.0)
    rowcol = crit_def.get("rowcol", False)

    sample_per_class = sample_per_class.type_as(logits)
    spc = torch.bincount(labels, minlength=sample_per_class.shape[0])
    spc = spc.type_as(logits)
    ratio_spc = spc / spc.sum()
    ratio_per_class = sample_per_class / sample_per_class.sum()
    spc = (1 - lamb) * ratio_spc + lamb * ratio_per_class
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1).contiguous()

    size = logits.shape
    U = torch.ones(size).to(logits.device)
    U = U / U.sum()

    sample_wilse = (1 - gama) * U + gama * F.softmax(logits, dim=1).detach()
    spc *= sample_wilse
    if rowcol:
        spc = sinkhorn(spc, uniform, ratio_spc, exp=False, epsilon=1, iter=5)

    if epoch <= t1:
        Q = U
    elif epoch <= t2:
        Q = (t2 - epoch) * U + (epoch - t1) * spc
    elif epoch <= t3:
        Q = 1 * spc
    else:
        Q = torch.pow(spc, 1.03)
    Q = Q / Q.sum()
    logits = logits + Q.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)

    return loss


def create_loss(freq_path):
    print("Loading OT Softmax Loss.")
    return OTSoftmax(freq_path)


def sinkhorn(S, a, b, exp=True, epsilon=1, iter=2):
    """Solve the entropic regularization optimal transport problem.
    Args:
        S: negative cost matrix, of shape (batch_size, num_class)
        a: a tensor of shape (batch_size, )
        b: a tensor of shape (num_class, )
        epsilon: a float, the regularization parameter
        iter: number of iterations
    """
    device = S.device
    batch_size = S.shape[0]
    num_class = S.shape[1]

    a = a.reshape(batch_size, 1)  # (B, 1)
    b = b.reshape(num_class, 1)  # (N, 1)
    if exp:
        K = torch.exp(S / epsilon)  # (B, N)
    else:
        K = S
    u = torch.ones_like(a).to(device)  # (B, 1)
    v = torch.ones_like(b).to(device)  # (N, 1)
    for _ in range(iter):
        u = a / (torch.matmul(K, v) + 1e-8)  # (B, 1)
        v = b / (torch.matmul(K.t(), u) + 1e-8)  # (N, 1)
    P = torch.diag(u.reshape(-1)) @ K @ torch.diag(v.reshape(-1))  # (B, N)
    return P
