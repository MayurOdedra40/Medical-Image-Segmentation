import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyOnlyLoss(nn.Module):
    """Standard cross-entropy, wrapped so all losses share the same interface."""

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets)


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss.
    Operates on raw logits; applies softmax internally.
    """

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits:  (B, C, H, W)
            targets: (B, H, W) integer class labels
        """
        num_classes = logits.shape[1]
        probs       = F.softmax(logits, dim=1)

        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        union        = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DiceCELoss(nn.Module):
    """
    Combination of Cross-Entropy and Dice Loss (equal weight).
    CE handles per-pixel calibration; Dice improves region overlap.
    """

    def __init__(self):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


class WeightedDiceLoss(nn.Module):
    """
    Dice Loss with fixed per-class weights.

    Default weights [0, 1, 1, 1] give equal importance to each foreground
    class (RV, Myo, LV) regardless of how many voxels each occupies.
    Pass class_weights to override (must match num_classes).
    """

    def __init__(self, class_weights=None, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        # stored as buffer so it moves with .to(device) automatically
        if class_weights is None:
            class_weights = torch.tensor([0.0, 1.0, 1.0, 1.0])
        self.register_buffer('class_weights', class_weights.float())

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs       = F.softmax(logits, dim=1)

        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        intersection = (probs * targets_onehot).sum(dim=(2, 3))  # (B, C)
        union        = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)  # (B, C)

        w = self.class_weights.to(logits.device)
        weighted_dice = (dice * w).sum(dim=1) / (w.sum() + self.smooth)
        return 1 - weighted_dice.mean()


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss (Sudre et al. 2017).

    Per-class weight = 1 / (sum_GT_c)^2, computed per batch.
    Strongly up-weights small structures (RV, Myo) relative to background.
    """

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs       = F.softmax(logits, dim=1)

        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # GT volume per class, averaged over batch: (C,)
        gt_vol = targets_onehot.sum(dim=(0, 2, 3))
        # weight = 1 / (vol^2), clipped to avoid infinity for empty classes
        w = 1.0 / (gt_vol ** 2 + self.smooth)

        intersection = (probs * targets_onehot).sum(dim=(2, 3))  # (B, C)
        union        = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))

        # Weighted sums over classes for each batch element
        num   = (w * intersection).sum(dim=1)          # (B,)
        denom = (w * union).sum(dim=1)                 # (B,)

        gdl = 1 - 2 * (num + self.smooth) / (denom + self.smooth)
        return gdl.mean()
