import torch
import torch.nn as nn
import torch.nn.functional as F

from piq import LPIPS, DISTS


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_pred, x):
        return (x_pred - x).abs().mean()


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_pred, x):
        return ((x_pred - x) ** 2).mean()


class LPIPSLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = LPIPS()

    def forward(self, x_pred, x):
        x_pred = F.interpolate(x_pred, size=224, mode="bilinear")
        x = F.interpolate(x.float(), size=224, mode="bilinear")
        
        x_pred = (x_pred + 1) / 2
        x = (x + 1) / 2

        return self.loss(x_pred, x)


class DISTSLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = DISTS()

    def forward(self, x_pred, x):
        x_pred = F.interpolate(x_pred, size=224, mode="bilinear")
        x = F.interpolate(x.float(), size=224, mode="bilinear")
        
        x_pred = (x_pred + 1) / 2
        x = (x + 1) / 2

        return self.loss(x_pred, x)


loss_dict = {
        'l1': L1Loss,
        'l2': L2Loss,
        'lpips': LPIPSLoss,
        'dists': DISTSLoss
        }
