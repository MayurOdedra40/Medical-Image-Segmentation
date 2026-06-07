import torch


def dice_score(preds, targets, num_classes=4):
    """
    Computes mean Dice coefficient over foreground classes.

    For each class:
    A = predicted binary mask for the class
    B = ground truth binary mask for the class

    Dice = (2 * |A ∩ B|) / (|A| + |B|)

    Returns average Dice over all classes (excluding background).
    """
    dice_per_class = []

    for cls in range(1, num_classes):
        pred_cls = (preds == cls).float()
        targets_cls = (targets == cls).float()

        intersection = (pred_cls * targets_cls).sum()

        dice = (
            2 * intersection + 1e-5
        ) / (
            pred_cls.sum()
            + targets_cls.sum()
            + 1e-5
        )

        dice_per_class.append(dice.item())

    return sum(dice_per_class) / len(dice_per_class)

def iou_score(preds, targets, num_classes=4):
    """
    Computes mean Intersection over Union (IoU).

    For each class:
    A = predicted binary mask for the class
    B = ground truth binary mask for the class

    IoU = |A ∩ B| / |A ∪ B|

    Returns average IoU over all foreground classes.
    """
    ious = []

    for cls in range(1, num_classes):
        pred_cls = (preds == cls)
        targets_cls = (targets == cls)

        intersection = (
            pred_cls & targets_cls
        ).sum()

        union = (
            pred_cls | targets_cls
        ).sum()

        iou = (
            intersection.float() + 1e-5
        ) / (
            union.float() + 1e-5
        )

        ious.append(iou.item())

    return sum(ious) / len(ious)

def pixel_accuracy(preds, targets):
    """
    Computes pixel-wise accuracy.

    Accuracy = number of correctly classified pixels / total number of pixels

    Compares predicted segmentation mask with ground truth
    on a per-pixel basis (ignores class structure).
    """

    correct = (preds == targets).sum()

    total = targets.numel()

    return (
        correct.float() / total
    ).item()