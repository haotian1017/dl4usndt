# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        inter = (p * target).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        loss = 1.0 - (2 * inter + self.smooth) / (denom + self.smooth)
        return loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        tp = (p * target).sum(dim=(1, 2, 3))
        fp = (p * (1 - target)).sum(dim=(1, 2, 3))
        fn = ((1 - p) * target).sum(dim=(1, 2, 3))
        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - ti).mean()


class FocalTverskyLoss(nn.Module):
    """(1 - TI)^(1/gamma)，gamma>1 时放大难样本梯度"""

    def __init__(self, alpha=0.3, beta=0.7, gamma=1.3333, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth
        self.gamma = gamma

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        tp = (p * target).sum(dim=(1, 2, 3))
        fp = (p * (1 - target)).sum(dim=(1, 2, 3))
        fn = ((1 - p) * target).sum(dim=(1, 2, 3))
        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1.0 - ti, 1.0 / self.gamma).mean()


class DiceBCELoss(nn.Module):
    """Dice + 带权 BCE。warmup 阶段与常规训练都好用"""

    def __init__(self, pos_weight=20.0, dice_weight=1.0, smooth=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth)

    def forward(self, logits, target):
        pw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        return bce + self.dice_weight * self.dice(logits, target)


class ComboLoss(nn.Module):
    """FocalTversky + 带权 BCE：兼顾区域重叠与逐像素监督"""

    def __init__(self, pos_weight=20.0, alpha=0.3, beta=0.7, gamma=1.3333,
                 ft_weight=1.0, bce_weight=1.0, smooth=1.0):
        super().__init__()
        self.ft = FocalTverskyLoss(alpha, beta, gamma, smooth)
        self.ft_weight, self.bce_weight, self.pos_weight = ft_weight, bce_weight, pos_weight

    def forward(self, logits, target):
        pw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        return self.ft_weight * self.ft(logits, target) + self.bce_weight * bce


def build_loss(name, cfg):
    if name == "combo":
        return ComboLoss(cfg.pos_weight, cfg.tversky_alpha, cfg.tversky_beta,
                         cfg.tversky_gamma, smooth=cfg.dice_smooth)
    if name == "dice_bce":
        return DiceBCELoss(cfg.pos_weight, smooth=cfg.dice_smooth)
    if name == "focal_tversky":
        return FocalTverskyLoss(cfg.tversky_alpha, cfg.tversky_beta,
                                cfg.tversky_gamma, cfg.dice_smooth)
    if name == "tversky":
        return TverskyLoss(cfg.tversky_alpha, cfg.tversky_beta, cfg.dice_smooth)
    if name == "dice":
        return DiceLoss(cfg.dice_smooth)
    raise ValueError(f"未知损失: {name}")


def warmup_loss(cfg):
    """前几个 epoch 用的稳定版本"""
    return DiceBCELoss(cfg.pos_weight, smooth=cfg.dice_smooth)
