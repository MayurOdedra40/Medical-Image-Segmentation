import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss for semantic segmentation.

    Computes overlap between predicted probability maps and
    one-hot encoded ground truth masks.

    Designed to directly work on raw logits from the model.
    """

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) raw model outputs
            targets: (B, H, W) ground truth class labels

        Returns:
            scalar Dice loss
        """

        num_classes = logits.shape[1]

        probs = F.softmax(logits, dim=1)

        targets_onehot = F.one_hot(
            targets,
            num_classes=num_classes,
        ).permute(0, 3, 1, 2).float()

        intersection = (
                probs * targets_onehot
        ).sum(dim=(2, 3))

        union = (
            probs.sum(dim=(2, 3))
            + targets_onehot.sum(dim=(2, 3))
        )

        dice = (
            2 * intersection + self.smooth
        ) / (
            union + self.smooth
        )

        return 1 - dice.mean()

class DiceCELoss(nn.Module):
    """
    Combination of Cross-Entropy Loss and Dice Loss.

    This loss improves both:
    - pixel-wise classification (CrossEntropy)
    - region overlap quality (Dice)
    """

    def __init__(self):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)

        Returns:
            combined loss (scalar)
        """
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return ce_loss + dice_loss