import torch
import numpy as np


def per_class_dice(preds, targets, num_classes=4):
    """
    Compute per-class Dice for foreground classes (1..num_classes-1).

    Returns a dict {class_idx: dice_float}. When both pred and GT are
    empty for a class the entry is float('nan') so callers can skip it.
    """
    results = {}
    for cls in range(1, num_classes):
        pred_cls = (preds == cls).float()
        gt_cls   = (targets == cls).float()

        intersection = (pred_cls * gt_cls).sum()
        denom        = pred_cls.sum() + gt_cls.sum()

        if denom == 0:
            results[cls] = float('nan')
        else:
            results[cls] = (2 * intersection / denom).item()

    return results


def mean_dice(preds, targets, num_classes=4):
    """Mean Dice over foreground classes, ignoring nan (absent classes)."""
    per_cls = per_class_dice(preds, targets, num_classes)
    values  = [v for v in per_cls.values() if not (v != v)]  # filter nan
    return sum(values) / len(values) if values else 0.0


def iou_score(preds, targets, num_classes=4):
    """Mean IoU over foreground classes."""
    ious = []
    for cls in range(1, num_classes):
        pred_cls    = (preds == cls)
        targets_cls = (targets == cls)

        intersection = (pred_cls & targets_cls).sum()
        union        = (pred_cls | targets_cls).sum()

        iou = (intersection.float() + 1e-5) / (union.float() + 1e-5)
        ious.append(iou.item())

    return sum(ious) / len(ious)


def pixel_accuracy(preds, targets):
    """Pixel-wise accuracy across all classes."""
    correct = (preds == targets).sum()
    total   = targets.numel()
    return (correct.float() / total).item()
