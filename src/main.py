import os
import pandas as pd
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
import random
import json
import numpy as np

from torch.utils.data import DataLoader

from experiments_config import EXPERIMENTS
from utils.preprocessing import (
load_split,
preprocess_patient,
ACDCDataset
)
from train import train
from utils.visualization import (
show_random_prediction
)

def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATASET_ROOT = os.path.join(BASE_DIR, "data", "ACDC")
    RESULTS_DIR = "results"

    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # ----------------------------
    # Load dataset
    # ----------------------------

    train_patients = load_split(
        os.path.join(DATASET_ROOT, "training"),
    )

    test_patients = load_split(
        os.path.join(DATASET_ROOT, "testing"),
    )

    train_processed = [
        preprocess_patient(p)
        for p in train_patients
    ]

    test_processed = [
        preprocess_patient(p)
        for p in test_patients
    ]

    train_dataset = ACDCDataset(
        train_processed,
        augment=True
    )

    test_dataset = ACDCDataset(
        test_processed,
        augment=False
    )

    # ----------------------------
    # Experiments
    # ----------------------------
    for idx, exp in enumerate(EXPERIMENTS):
        print(f"\nRunning: {exp['name']} - experiment: {idx}/{len(EXPERIMENTS)}")

        model = exp["model"]().to(device)
        criterion = exp["loss_func"]()
        optimizer = exp["optimizer"](
            model.parameters(),
            lr=exp["lr"],
            weight_decay=exp["weight_decay"],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=exp["batch_size"],
            shuffle=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=exp["batch_size"],
            shuffle=False,
        )

        history = train(
            model,
            train_loader,
            test_loader,
            optimizer,
            criterion,
            device,
            epochs=exp["epochs"]
        )

        # ------------------------
        # Results path
        # ------------------------
        exp_dir = os.path.join(
            RESULTS_DIR,
            exp["name"]
        )
        os.makedirs(exp_dir, exist_ok=True)

        # ------------------------
        # Save metrics
        # ------------------------
        csv_path = os.path.join(
            exp_dir,
            "metrics.csv"
        )

        pd.DataFrame(history).to_csv(
            csv_path,
            index=False
        )

        # ------------------------
        # Save model
        # ------------------------
        model_path = os.path.join(
            exp_dir,
            "model.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        # ------------------------
        # Save config
        # ------------------------
        config_path = os.path.join(
            exp_dir,
            "config.json"
        )

        config_to_save = {
        "name": exp["name"],
        "lr": exp["lr"],
        "weight_decay": exp["weight_decay"],
        "epochs": exp["epochs"],
        "batch_size": exp["batch_size"],
        "model": exp["model"].__name__,
        "loss_func": exp["loss_func"].__name__,
        "optimizer": exp["optimizer"].__name__,
    }

        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=4)

        # ------------------------
        # Visualization
        # ------------------------
        show_random_prediction(
            model,
            test_loader,
            device,
            save_path=os.path.join(exp_dir, "random_prediction.png")
        )

        print(f"Finished: {exp['name']}")

    print("\nAll experiments finished.")

if __name__ == "__main__":
    main()