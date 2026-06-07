import os
import random

import torch
import matplotlib.pyplot as plt


def show_sample(image, mask, title=None):
    image = image.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Image")
    ax[0].axis('off')

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Ground Truth")
    ax[1].axis('off')

    if title:
        fig.suptitle(title)

    return fig

def show_prediction(image, mask, pred, title=None):
    image = image.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Image")
    ax[0].axis('off')

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Ground Truth")
    ax[1].axis('off')

    ax[2].imshow(pred, cmap='gray')
    ax[2].set_title("Prediction")
    ax[2].axis('off')

    if title:
        fig.suptitle(title)

    return fig


def visualize_batch(images, masks, preds=None, n=3, save_path=None):
    images = images[:n]
    masks = masks[:n]
    preds = preds[:n] if preds is not None else None

    figs = []

    for i in range(n):
        if preds is None:
            fig = show_sample(images[i], masks[i], title=f"Sample {i}")
        else:
            fig = show_prediction(images[i], masks[i], preds[i], title=f"Prediction {i}")

        figs.append(fig)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        base = save_path.replace(".png", "")
        for i, fig in enumerate(figs):
            fig.savefig(f"{base}_{i}.png", dpi=300, bbox_inches='tight')

    for fig in figs:
        plt.close(fig)

def show_prediction_overlay(image, mask, pred, alpha=0.4, save_path=None):
    image = image.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Image")
    ax[0].axis('off')

    ax[1].imshow(image, cmap='gray')
    ax[1].imshow(mask, alpha=alpha, cmap='Reds')
    ax[1].set_title("Ground Truth Overlay")
    ax[1].axis('off')

    ax[2].imshow(image, cmap='gray')
    ax[2].imshow(pred, alpha=alpha, cmap='Blues')
    ax[2].set_title("Prediction Overlay")
    ax[2].axis('off')

    fig.suptitle("Overlay")
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close(fig)


def show_random_prediction(model, dataloader, device, save_path=None):
    model.eval()

    with torch.no_grad():
        images, masks = next(iter(dataloader))

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        idx = random.randint(0, images.shape[0] - 1)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        image = images[idx].squeeze().cpu().numpy()
        mask = masks[idx].squeeze().cpu().numpy()
        pred = preds[idx].squeeze().cpu().numpy()

        ax[0].imshow(image, cmap='gray')
        ax[0].set_title("Image")
        ax[0].axis('off')

        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title("Ground Truth")
        ax[1].axis('off')

        ax[2].imshow(pred, cmap='gray')
        ax[2].set_title("Prediction")
        ax[2].axis('off')

        fig.suptitle(f"Random Sample {idx}")
        fig.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)
