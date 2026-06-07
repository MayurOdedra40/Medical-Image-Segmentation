import torch

from utils.metrics import (dice_score, iou_score, pixel_accuracy)

def train_one_epoch(
        model,
        data_loader,
        optimizer,
        criterion,
        device
):
    model.train()

    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(data_loader):

        if batch_idx % 10 ==0:
            print(f"Batch: {batch_idx + 1}/{len(data_loader)}")

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(data_loader)

def evaluate(
        model,
        data_loader,
        criterion,
        device
):
    model.eval()

    total_loss = 0.0

    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for images, masks in data_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item()

            total_dice += dice_score(
                preds.cpu(),
                masks.cpu()
            )

            total_iou += iou_score(
                preds.cpu(),
                masks.cpu()
            )

            total_acc += pixel_accuracy(
                preds.cpu(),
                masks.cpu()
            )

    n = len(data_loader)

    return {
        'loss': total_loss / n,
        'dice': total_dice / n,
        'iou': total_iou / n,
        'accuracy': total_acc / n,
    }

def train(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        device,
        epochs=20
):

    print("START TRAINING")

    history = {
        'train_loss': [],
        'val_loss': [],
        'dice': [],
        'iou': [],
        'accuracy': []
    }

    for epoch in range(epochs):
        print(f"EPOCH {epoch + 1}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        metrics = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(metrics['loss'])
        history['dice'].append(metrics['dice'])
        history['iou'].append(metrics['iou'])
        history['accuracy'].append(metrics['accuracy'])

        print(
            f'Epoch [{epoch + 1}/{epochs}] '
            f'Train loss: {train_loss:.4f} '
            f'Val loss: {metrics["loss"]:.4f} '
            f'Dice: {metrics["dice"]:.4f} '
            f'IoU: {metrics["iou"]:.4f} '
            f'Acc: {metrics["accuracy"]:.4f}'
        )

    return history