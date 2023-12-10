import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json


class FocalLoss(_Loss):
    """
    Focal Loss
    """

    def __init__(self, freq_path):
        super(FocalLoss, self).__init__()
        with open(freq_path, "r") as fd:
            freq = json.load(fd)
        freq = torch.tensor(freq)
        self.sample_per_class = freq

    def forward(self, input, label, reduction="mean", **kwargs):
        return focal_loss(label, input, self.sample_per_class, reduction, **kwargs)


def focal_loss(labels, logits, sample_per_class, reduction, **kwargs):
    """Compute the Focal Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes]. Not used in focal loss.
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Focal Loss.
    """
    gamma = kwargs.get("gamma", 2.0)
    output = F.cross_entropy(logits, labels, reduction="none")
    p = torch.exp(-output)
    loss = (1 - p) ** gamma * output
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def create_loss(freq_path):
    print("Loading Focal Loss.")
    return FocalLoss(freq_path)
