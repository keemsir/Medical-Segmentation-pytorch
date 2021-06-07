import torch
import torch.nn as nn
import torch.nn.functional as F


## Loss Function


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEWithLogitsLoss, self).__init__()

    def forward(self, input, target):

        BCE = F.binary_cross_entropy_with_logits(input, target, reduction='mean')

        return BCE



class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


# # TverskyLoss # # a + b = 1.0


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha_t=0.5, beta_t=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha_t = alpha_t
        self.beta_t = beta_t


    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha_t * FP + self.beta_t * FN + smooth)
        print(self.alpha_t, self.beta_t)

        return 1 - Tversky



# FocalLoss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha_f=0.8, gamma_f=2.0):
        super(FocalLoss, self).__init__()
        self.alpha_f = alpha_f
        self.gamma_f = gamma_f

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha_f * (1 - BCE_EXP) ** self.gamma_f * BCE

        return focal_loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha_ft=0.5, beta_ft=0.5, gamma_ft=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha_ft = 0.5
        self.beta_ft = 0.5
        self.gamma_ft = 1.0

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha_ft * FP + self.beta_ft * FN + smooth)
        FocalTversky = (1 - Tversky) ** self.gamma_ft

        return FocalTversky


