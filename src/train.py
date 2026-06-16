import torch

from utils.metrics import per_class_dice, mean_dice, iou_score, pixel_accuracy

_CLASS_NAMES = {1: 'rv', 2: 'myo', 3: 'lv'}


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(data_loader):
        if batch_idx % 10 == 0:
            print(f"Batch: {batch_idx + 1}/{len(data_loader)}")

        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(data_loader)


def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    class_sum   = {1: 0.0, 2: 0.0, 3: 0.0}
    class_count = {1: 0,   2: 0,   3: 0}

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)
            loss    = criterion(outputs, masks)
            preds   = torch.argmax(outputs, dim=1)

            total_loss += loss.item()

            cls_dice = per_class_dice(preds.cpu(), masks.cpu())
            for cls, val in cls_dice.items():
                if val == val:   # not nan
                    class_sum[cls]   += val
                    class_count[cls] += 1

    n = len(data_loader)

    def safe_avg(cls):
        return class_sum[cls] / class_count[cls] if class_count[cls] > 0 else float('nan')

    dice_rv  = safe_avg(1)
    dice_myo = safe_avg(2)
    dice_lv  = safe_avg(3)

    valid     = [v for v in [dice_rv, dice_myo, dice_lv] if v == v]
    dice_mean = sum(valid) / len(valid) if valid else float('nan')

    return {
        'loss':      total_loss / n,
        'dice_mean': dice_mean,
        'dice_rv':   dice_rv,
        'dice_myo':  dice_myo,
        'dice_lv':   dice_lv,
    }


def train(
    model, train_loader, test_loader,
    optimizer, criterion, device,
    epochs=20, use_wandb=False,
):
    print("START TRAINING")

    if use_wandb:
        import wandb

    history = {
        'train_loss': [],
        'val_loss':   [],
        'dice_mean':  [],
        'dice_rv':    [],
        'dice_myo':   [],
        'dice_lv':    [],
    }

    for epoch in range(epochs):
        print(f"EPOCH {epoch + 1}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics    = evaluate(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(metrics['loss'])
        history['dice_mean'].append(metrics['dice_mean'])
        history['dice_rv'].append(metrics['dice_rv'])
        history['dice_myo'].append(metrics['dice_myo'])
        history['dice_lv'].append(metrics['dice_lv'])

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train loss: {train_loss:.4f}  Val loss: {metrics['loss']:.4f}  "
            f"Dice mean: {metrics['dice_mean']:.4f}  "
            f"RV: {metrics['dice_rv']:.4f}  "
            f"Myo: {metrics['dice_myo']:.4f}  "
            f"LV: {metrics['dice_lv']:.4f}"
        )

        if use_wandb:
            wandb.log({
                "train/loss":    train_loss,
                "val/loss":      metrics["loss"],
                "val/dice_mean": metrics["dice_mean"],
                "val/dice_rv":   metrics["dice_rv"],
                "val/dice_myo":  metrics["dice_myo"],
                "val/dice_lv":   metrics["dice_lv"],
            }, step=epoch + 1)

    return history
