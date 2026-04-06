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


class SoftSkeletonRecallLoss(nn.Module):
    """Soft Skeleton Recall loss for binary segmentation.

    Measures how well the predicted probability map covers the ground-truth
    skeleton. Minimising this drives the network to assign high probability
    to skeleton pixels.

    Args:
        smooth: Laplace smoothing constant to avoid division by zero.
    """

    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, skel, loss_mask=None):
        """
        pred      : (B, 1, H, W) probabilities after sigmoid
        skel      : (B, 1, H, W) binary skeleton mask
        loss_mask : (B, 1, H, W) optional valid-pixel mask (1 = valid)
        """
        axes = tuple(range(2, pred.ndim))
        if loss_mask is not None:
            inter = (pred * skel * loss_mask).sum(axes)
            sum_skel = (skel * loss_mask).sum(axes)
        else:
            inter = (pred * skel).sum(axes)
            sum_skel = skel.sum(axes)
        rec = (inter + self.smooth) / (sum_skel + self.smooth).clamp(min=1e-8)
        return 1 - rec.mean()


class DC_SkelREC_and_CE_loss(nn.Module):
    """Compound Dice + Skeleton-Recall + BCE loss for binary segmentation.

    Equivalent to DC_SkelREC_and_CE_loss from the Skeleton-Recall paper
    (https://github.com/MIC-DKFZ/Skeleton-Recall), adapted for binary
    segmentation with sigmoid activation instead of softmax.

    Usage::

        criterion = DC_SkelREC_and_CE_loss()
        loss = criterion(pred, target, skel)

    Args:
        weight_ce   : weight for the BCE term (default 1.0)
        weight_dice : weight for the Dice term (default 1.0)
        weight_srec : weight for the Skeleton-Recall term (default 1.0)
        smooth      : Laplace smoothing shared by Dice and Skeleton-Recall
        ignore_label: optional label value to mask out from all three losses
    """

    def __init__(self, weight_ce=1., weight_dice=1., weight_srec=1.,
                 smooth=1., ignore_label=None):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_srec = weight_srec
        self.smooth = smooth
        self.ignore_label = ignore_label
        self.srec = SoftSkeletonRecallLoss(smooth=smooth)

    def forward(self, pred, target, skel):
        """
        pred   : (B, 1, H, W) raw logits
        target : (B, 1, H, W) binary segmentation ground-truth
        skel   : (B, 1, H, W) binary skeleton of the ground-truth
        """
        if self.ignore_label is not None:
            mask = (target != self.ignore_label).float()
            target = torch.where(mask.bool(), target, torch.zeros_like(target))
            skel = torch.where(mask.bool(), skel, torch.zeros_like(skel))
        else:
            mask = None

        prob = torch.sigmoid(pred)

        dc_loss = dice_loss(prob, target, smooth=self.smooth) \
            if self.weight_dice != 0 else 0
        srec_loss = self.srec(prob, skel, loss_mask=mask) \
            if self.weight_srec != 0 else 0
        ce_loss = F.binary_cross_entropy_with_logits(
            pred if mask is None else pred * mask,
            target if mask is None else target * mask
        ) if self.weight_ce != 0 else 0

        return self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_srec * srec_loss
