import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json


class ClassBalancedLoss(_Loss):
    """
    ClassBalanced Loss
    """

    def __init__(self, freq_path):
        super(ClassBalancedLoss, self).__init__()
        with open(freq_path, "r") as fd:
            freq = json.load(fd)
        freq = torch.tensor(freq)
        self.sample_per_class = freq

    def forward(self, input, label, reduction="mean", **kwargs):
        return classbalanced_loss(label, input, self.sample_per_class, reduction, **kwargs)


def classbalanced_loss(labels, logits, sample_per_class, reduction, **kwargs):
    """Compute the ClassBalanced Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. ClassBalanced Loss.
    """
    C = len(sample_per_class)
    device = logits.device
    beta = kwargs.get("beta", 0.9999)
    loss_type = kwargs.get("loss_type", "cross_entropy")

    effective_num = 1.0 - torch.pow(beta, sample_per_class)
    weights = (1.0 - beta) / effective_num
    weights = weights / torch.sum(weights) * C
    weights = weights.to(device)

    labels_one_hot = F.one_hot(labels, C).float()
    if loss_type != "cross_entropy":
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, C)

    if loss_type == "cross_entropy":
        loss = F.cross_entropy(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "focal":
        gamma = kwargs.get("gamma", 2.0)
        loss = focal_loss(logits, labels_one_hot, alpha=weights, gamma=gamma)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def create_loss(freq_path):
    print("Loading ClassBalanced Loss.")
    return ClassBalancedLoss(freq_path)


def focal_loss(logits, labels, alpha=None, gamma=2):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bc_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bc_loss

    if alpha is not None:
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    else:
        focal_loss = torch.sum(loss)

    focal_loss /= torch.sum(labels)
    return focal_loss
