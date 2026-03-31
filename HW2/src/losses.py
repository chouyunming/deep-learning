import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) /
                 (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1.):
        inputs = torch.sigmoid(inputs)
        return dice_loss(inputs, targets, smooth)


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1.):
        inputs_sig = torch.sigmoid(inputs)
        dice = dice_loss(inputs_sig, targets, smooth)
        bce = F.binary_cross_entropy(inputs_sig, targets, reduction='mean')
        return dice + bce
