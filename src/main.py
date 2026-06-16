import os
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import wandb

torch.backends.cudnn.enabled   = False
torch.backends.cudnn.benchmark = False

from torch.utils.data import DataLoader

from experiments_config import EXPERIMENTS
from utils.preprocessing import load_split, preprocess_patient, ACDCDataset
from train import train


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _make_prediction_figures(model, dataloader, device, n=4):
    """Return n matplotlib Figure objects (Image | GT | Prediction) from random test batches."""
    model.eval()
    figs = []

    loader_iter = iter(dataloader)
    for _ in range(n):
        try:
            images, masks = next(loader_iter)
        except StopIteration:
            break

        with torch.no_grad():
            outputs = model(images.to(device))
            preds   = torch.argmax(outputs, dim=1).cpu()

        idx   = random.randint(0, images.shape[0] - 1)
        image = images[idx].squeeze().cpu().numpy()
        mask  = masks[idx].squeeze().cpu().numpy()
        pred  = preds[idx].squeeze().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(image, cmap='gray');  ax[0].set_title("Image");       ax[0].axis('off')
        ax[1].imshow(mask,  cmap='tab10', vmin=0, vmax=3); ax[1].set_title("Ground Truth"); ax[1].axis('off')
        ax[2].imshow(pred,  cmap='tab10', vmin=0, vmax=3); ax[2].set_title("Prediction");   ax[2].axis('off')
        fig.tight_layout()
        figs.append(fig)

    return figs


def main():
    BASE_DIR     = os.path.dirname(os.path.dirname(__file__))
    DATASET_ROOT = os.path.join(BASE_DIR, "data", "ACDC")
    RESULTS_DIR  = os.path.join(BASE_DIR, "results")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load + preprocess dataset once (shared across all experiments)
    # ------------------------------------------------------------------
    print("Loading training set...")
    train_patients = load_split(os.path.join(DATASET_ROOT, "training"))
    print("Loading test set...")
    test_patients  = load_split(os.path.join(DATASET_ROOT, "testing"))

    train_processed = [preprocess_patient(p) for p in train_patients]
    test_processed  = [preprocess_patient(p) for p in test_patients]

    train_dataset = ACDCDataset(train_processed, augment=True)
    test_dataset  = ACDCDataset(test_processed,  augment=False)

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    for idx, exp in enumerate(EXPERIMENTS):
        print(f"\n[{idx + 1}/{len(EXPERIMENTS)}] Running: {exp['name']}")

        set_seed(exp["seed"])

        # Config dict (serialisable — used for both wandb and the JSON file)
        config = {
            "name":         exp["name"],
            "block":        exp.get("block", ""),
            "seed":         exp["seed"],
            "model":        exp["model"].__name__,
            "loss_func":    exp["loss_func"].__name__,
            "optimizer":    exp["optimizer"].__name__,
            "lr":           exp["lr"],
            "weight_decay": exp["weight_decay"],
            "epochs":       exp["epochs"],
            "batch_size":   exp["batch_size"],
            "train_slices": len(train_dataset),
            "test_slices":  len(test_dataset),
        }

        # --- wandb run ---
        run = wandb.init(
            project="acdc-segmentation",
            name=exp["name"],
            group=exp.get("block", "misc"),
            tags=[
                exp["model"].__name__,
                exp["loss_func"].__name__,
                f"seed{exp['seed']}",
            ],
            config=config,
            reinit=True,
        )

        model     = exp["model"]().to(device)
        criterion = exp["loss_func"]()
        optimizer = exp["optimizer"](
            model.parameters(),
            lr=exp["lr"],
            weight_decay=exp["weight_decay"],
        )

        # Track gradients and parameter histograms automatically
        wandb.watch(model, log="gradients", log_freq=200)

        train_loader = DataLoader(train_dataset, batch_size=exp["batch_size"], shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=exp["batch_size"], shuffle=False)

        history = train(
            model, train_loader, test_loader,
            optimizer, criterion, device,
            epochs=exp["epochs"],
            use_wandb=True,
        )

        # ------------------------------------------------------------------
        # Save results locally
        # ------------------------------------------------------------------
        exp_dir = os.path.join(RESULTS_DIR, exp["name"])
        os.makedirs(exp_dir, exist_ok=True)

        pd.DataFrame(history).to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)

        model_path = os.path.join(exp_dir, "model.pth")
        torch.save(model.state_dict(), model_path)

        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # ------------------------------------------------------------------
        # wandb: best-epoch summary
        # ------------------------------------------------------------------
        wandb.run.summary["best_dice_mean"] = max(history["dice_mean"])
        wandb.run.summary["best_dice_rv"]   = max(history["dice_rv"])
        wandb.run.summary["best_dice_myo"]  = max(history["dice_myo"])
        wandb.run.summary["best_dice_lv"]   = max(history["dice_lv"])
        wandb.run.summary["best_val_loss"]  = min(history["val_loss"])

        # ------------------------------------------------------------------
        # wandb: prediction images (4 random test samples)
        # ------------------------------------------------------------------
        figs = _make_prediction_figures(model, test_loader, device, n=4)
        wandb.log({"predictions": [wandb.Image(f) for f in figs]})
        for f in figs:
            plt.close(f)

        # Also save one locally for quick inspection
        figs_local = _make_prediction_figures(model, test_loader, device, n=1)
        if figs_local:
            figs_local[0].savefig(
                os.path.join(exp_dir, "random_prediction.png"),
                dpi=150, bbox_inches='tight',
            )
            plt.close(figs_local[0])

        # ------------------------------------------------------------------
        # wandb: model artifact
        # ------------------------------------------------------------------
        artifact = wandb.Artifact(name=exp["name"], type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)

        wandb.finish()
        print(f"Saved to {exp_dir}")

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
