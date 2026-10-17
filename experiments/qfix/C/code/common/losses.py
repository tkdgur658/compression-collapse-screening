"""Pixel losses shared by every model (identical arithmetic to the legacy utils.py)."""
import torch
import torch.nn as nn

EPS = 1e-7
DS_WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5)  # legacy GT_BceDiceLoss weights for 5 side outputs


def _prob(logit):
    return torch.clamp(torch.sigmoid(logit), EPS, 1 - EPS)


def bce_dice(logit, target):
    """BCE + soft Dice (smooth 1, per-sample mean), on sigmoid probabilities."""
    p = _prob(logit)
    n = p.size(0)
    p_, t_ = p.view(n, -1), target.view(n, -1)
    bce = nn.functional.binary_cross_entropy(p_, t_)
    dice = 1 - ((2 * (p_ * t_).sum(1) + 1) / (p_.sum(1) + t_.sum(1) + 1)).sum() / n
    return bce + dice


def seg_loss(out, side, target):
    """Main BCE+Dice plus the model's native deep supervision, if it has any.

    side: None, or a sequence of 5 side-output logits ordered as the legacy GT tuple."""
    loss = bce_dice(out, target)
    if side is not None:
        if len(side) != len(DS_WEIGHTS):
            raise ValueError(f'expected {len(DS_WEIGHTS)} side outputs, got {len(side)}')
        for w, s in zip(DS_WEIGHTS, side):
            loss = loss + w * bce_dice(s, target)
    return loss
