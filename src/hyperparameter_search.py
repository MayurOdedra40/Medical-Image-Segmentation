"""
Block C — Hyperparameter tuning via random search across all 3 models.

Searches over: lr (log-uniform), weight_decay, batch_size, optimizer, loss_func.
Trains each trial for PROXY_EPOCHS epochs; saves results under results/blockC_*/

Run from the project root:
    conda run -n base python src/hyperparameter_search.py --gpu-id 0

For parallel execution across N GPUs:
    conda run -n base python src/hyperparameter_search.py --gpu-id 0 --total-gpus 3 --shard 0
    conda run -n base python src/hyperparameter_search.py --gpu-id 1 --total-gpus 3 --shard 1
    conda run -n base python src/hyperparameter_search.py --gpu-id 2 --total-gpus 3 --shard 2
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb

from torch.utils.data import DataLoader

from experiments_config import (
    MODELS, LOSSES, OPTIMIZERS,
    HP_SEARCH_SPACE, N_TRIALS, PROXY_EPOCHS, BLOCK_C_SEED,
)
from utils.preprocessing import load_split, preprocess_patient, ACDCDataset
from train import train

torch.backends.cudnn.enabled    = False
torch.backends.cudnn.benchmark  = False

BLOCK_C_MODELS = ["UNet", "AttentionUNet", "TransUNet"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# HP sampling
# ---------------------------------------------------------------------------

def sample_hp_config(rng: np.random.Generator) -> dict:
    """Sample one random hyperparameter configuration from HP_SEARCH_SPACE."""
    kind, lo, hi = HP_SEARCH_SPACE["lr"]
    lr = float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))

    weight_decay = float(rng.choice(HP_SEARCH_SPACE["weight_decay"][1]))
    batch_size   = int(rng.choice(HP_SEARCH_SPACE["batch_size"][1]))
    optimizer    = str(rng.choice(HP_SEARCH_SPACE["optimizer"][1]))
    loss_func    = str(rng.choice(HP_SEARCH_SPACE["loss_func"][1]))

    return {
        "lr":           lr,
        "weight_decay": weight_decay,
        "batch_size":   batch_size,
        "optimizer":    optimizer,
        "loss_func":    loss_func,
    }


def generate_all_configs(model_names: list[str], n_trials: int, rng: np.random.Generator) -> list[dict]:
    """
    Pre-generate all (model, trial) HP configs upfront so the full search
    table is fixed before any training begins (reproducible regardless of
    which shard runs first).
    """
    configs = []
    for model_name in model_names:
        for trial_idx in range(n_trials):
            hp = sample_hp_config(rng)
            configs.append({
                "name":        f"blockC_{model_name}_trial{trial_idx:02d}",
                "block":       "C",
                "model":       model_name,
                "trial":       trial_idx,
                "seed":        trial_idx,   # unique init seed per trial
                **hp,
            })
    return configs


# ---------------------------------------------------------------------------
# Prediction figure helper (mirrors main.py)
# ---------------------------------------------------------------------------

def _make_prediction_figures(model, dataloader, device, n=4):
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
        ax[0].imshow(image, cmap="gray");                      ax[0].set_title("Image");        ax[0].axis("off")
        ax[1].imshow(mask, cmap="tab10", vmin=0, vmax=3);     ax[1].set_title("Ground Truth"); ax[1].axis("off")
        ax[2].imshow(pred, cmap="tab10", vmin=0, vmax=3);     ax[2].set_title("Prediction");   ax[2].axis("off")
        fig.tight_layout()
        figs.append(fig)
    return figs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Block C — random HP search for ACDC segmentation")
    parser.add_argument("--gpu-id",    type=int, default=None, help="GPU index (e.g. 0). Defaults to cuda:0 or cpu.")
    parser.add_argument("--total-gpus", type=int, default=1,   help="Total parallel processes for sharding.")
    parser.add_argument("--shard",     type=int, default=0,    help="Which shard this process runs (0-indexed).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    BASE_DIR     = os.path.dirname(os.path.dirname(__file__))
    DATASET_ROOT = os.path.join(BASE_DIR, "data", "ACDC")
    RESULTS_DIR  = os.path.join(BASE_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Generate all 45 configs upfront (same RNG seed → same configs always)
    # ------------------------------------------------------------------
    rng         = np.random.default_rng(BLOCK_C_SEED)
    all_configs = generate_all_configs(BLOCK_C_MODELS, N_TRIALS, rng)
    print(f"Total Block C trials: {len(all_configs)}  ({N_TRIALS} per model × {len(BLOCK_C_MODELS)} models)")

    # Shard: distribute trials across parallel processes
    my_configs = [c for i, c in enumerate(all_configs) if i % args.total_gpus == args.shard]
    print(f"Shard {args.shard}/{args.total_gpus}: running {len(my_configs)}/{len(all_configs)} trials")

    # ------------------------------------------------------------------
    # Load + preprocess dataset once (shared across all trials)
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
    # Run trials
    # ------------------------------------------------------------------
    summary_rows = []

    for run_idx, cfg in enumerate(my_configs):
        print(f"\n[{run_idx + 1}/{len(my_configs)}] {cfg['name']}")
        print(f"  lr={cfg['lr']:.2e}  weight_decay={cfg['weight_decay']}  "
              f"batch_size={cfg['batch_size']}  optimizer={cfg['optimizer']}  "
              f"loss_func={cfg['loss_func']}")

        set_seed(cfg["seed"])

        model_cls  = MODELS[cfg["model"]]
        loss_cls   = LOSSES[cfg["loss_func"]]
        optim_cls  = OPTIMIZERS[cfg["optimizer"]]

        serialisable_config = {
            "name":         cfg["name"],
            "block":        "C",
            "trial":        cfg["trial"],
            "seed":         cfg["seed"],
            "model":        cfg["model"],
            "loss_func":    loss_cls.__name__,
            "optimizer":    cfg["optimizer"],
            "lr":           cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "epochs":       PROXY_EPOCHS,
            "batch_size":   cfg["batch_size"],
            "train_slices": len(train_dataset),
            "test_slices":  len(test_dataset),
        }

        # --- wandb ---
        run = wandb.init(
            project="acdc-segmentation",
            name=cfg["name"],
            group="C_search",
            tags=[cfg["model"], cfg["loss_func"], f"trial{cfg['trial']:02d}"],
            config=serialisable_config,
            reinit=True,
        )

        model     = model_cls().to(device)
        criterion = loss_cls()
        optimizer = optim_cls(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        wandb.watch(model, log="gradients", log_freq=200)

        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=cfg["batch_size"], shuffle=False)

        history = train(
            model, train_loader, test_loader,
            optimizer, criterion, device,
            epochs=PROXY_EPOCHS,
            use_wandb=True,
        )

        # ------------------------------------------------------------------
        # Save results locally
        # ------------------------------------------------------------------
        exp_dir = os.path.join(RESULTS_DIR, cfg["name"])
        os.makedirs(exp_dir, exist_ok=True)

        pd.DataFrame(history).to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)
        torch.save(model.state_dict(), os.path.join(exp_dir, "model.pth"))
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(serialisable_config, f, indent=4)

        # ------------------------------------------------------------------
        # wandb summary + artifacts
        # ------------------------------------------------------------------
        best_dice_mean = max(history["dice_mean"])
        best_dice_rv   = max(history["dice_rv"])
        best_dice_myo  = max(history["dice_myo"])
        best_dice_lv   = max(history["dice_lv"])

        wandb.run.summary["best_dice_mean"] = best_dice_mean
        wandb.run.summary["best_dice_rv"]   = best_dice_rv
        wandb.run.summary["best_dice_myo"]  = best_dice_myo
        wandb.run.summary["best_dice_lv"]   = best_dice_lv
        wandb.run.summary["best_val_loss"]  = min(history["val_loss"])

        figs = _make_prediction_figures(model, test_loader, device, n=4)
        wandb.log({"predictions": [wandb.Image(f) for f in figs]})
        for f in figs:
            plt.close(f)

        figs_local = _make_prediction_figures(model, test_loader, device, n=1)
        if figs_local:
            figs_local[0].savefig(
                os.path.join(exp_dir, "random_prediction.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(figs_local[0])

        model_path = os.path.join(exp_dir, "model.pth")
        artifact   = wandb.Artifact(name=cfg["name"], type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)

        wandb.finish()

        summary_rows.append({
            "model":        cfg["model"],
            "trial":        cfg["trial"],
            "lr":           cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "batch_size":   cfg["batch_size"],
            "optimizer":    cfg["optimizer"],
            "loss_func":    cfg["loss_func"],
            "best_dice_mean": best_dice_mean,
            "best_dice_rv":   best_dice_rv,
            "best_dice_myo":  best_dice_myo,
            "best_dice_lv":   best_dice_lv,
        })

        print(f"  → best_dice_mean={best_dice_mean:.4f}  Saved to {exp_dir}")

    # ------------------------------------------------------------------
    # Write / update summary CSV (append-safe for multi-shard runs)
    # ------------------------------------------------------------------
    summary_path = os.path.join(RESULTS_DIR, "blockC_summary.csv")
    new_df = pd.DataFrame(summary_rows)

    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["model", "trial"], keep="last")
        combined.to_csv(summary_path, index=False)
    else:
        new_df.to_csv(summary_path, index=False)

    print(f"\nSummary written to {summary_path}")
    print("\nAll Block C trials finished.")


if __name__ == "__main__":
    main()
