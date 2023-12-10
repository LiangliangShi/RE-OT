import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json


class LDAMLoss(_Loss):
    """
    LDAM Loss
    """

    def __init__(self, freq_path):
        super(LDAMLoss, self).__init__()
        with open(freq_path, "r") as fd:
            freq = json.load(fd)
        freq = torch.tensor(freq)
        self.sample_per_class = freq

    def forward(self, input, label, reduction="mean", **kwargs):
        return ldam_loss(label, input, self.sample_per_class, reduction, **kwargs)


def ldam_loss(labels, logits, sample_per_class, reduction, **kwargs):
    """Compute the LDAM Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes]
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. LDAM Loss.
    """
    s = kwargs.get("s", 30)
    m_max = kwargs.get("m_max", 0.5)
    m = 1 / torch.sqrt(torch.sqrt(sample_per_class))
    m = m * (m_max / torch.max(m))
    m = m.to(logits.device)

    idx = torch.zeros_like(logits, dtype=torch.uint8)
    idx.scatter_(1, labels.data.view(-1, 1), 1)
    idx_float = idx.type_as(logits)

    batch_m = torch.matmul(m[None, :], idx_float.transpose(0, 1)).view((-1, 1))
    logits_m = logits - batch_m
    output = torch.where(idx, logits_m, logits)
    loss = F.cross_entropy(s * output, labels, reduction=reduction)
    return loss


def create_loss(freq_path):
    print("Loading LDAM Loss.")
    return LDAMLoss(freq_path)
