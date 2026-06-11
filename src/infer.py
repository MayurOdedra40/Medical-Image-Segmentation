"""
Random inference check: load a saved model.pth and run it on a random test slice.

Usage (from the src/ directory):
    python infer.py                                  # auto-discover first results/ experiment
    python infer.py --exp results/unet_dice_adam     # specific experiment folder
    python infer.py --exp results/unet_dice_adam --save  # also save the plot as PNG
"""

import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe; switch to "TkAgg" / "Qt5Agg" if you want a window
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Make src/ imports work whether the script is run from src/ or the repo root
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.transunet import TransUNet
from utils.preprocessing import load_split, preprocess_patient, ACDCDataset

MODEL_REGISTRY = {
    "UNet": UNet,
    "AttentionUNet": AttentionUNet,
    "TransUNet": TransUNet,
}

CLASS_COLORS = ["black", "green", "yellow", "red"]   # bg, RV, myo, LV
CLASS_NAMES  = ["Background", "RV", "Myocardium", "LV"]


def find_first_exp(results_root: str) -> str:
    for entry in sorted(os.listdir(results_root)):
        candidate = os.path.join(results_root, entry)
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "model.pth")):
            return candidate
    raise FileNotFoundError(f"No experiment folder with model.pth found in {results_root}")


def load_model(exp_dir: str, device: torch.device) -> torch.nn.Module:
    config_path = os.path.join(exp_dir, "config.json")
    model_path  = os.path.join(exp_dir, "model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pth not found in {exp_dir}")

    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        model_cls = MODEL_REGISTRY[cfg["model"]]
        print(f"Model class from config: {cfg['model']}")
    else:
        print("config.json not found — defaulting to UNet")
        model_cls = UNet

    model = model_cls().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights from: {model_path}")
    return model


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert integer label mask (H,W) to an RGB image for display."""
    palette = np.array([
        [0,   0,   0],    # 0 background
        [0,   200, 0],    # 1 RV      — green
        [255, 200, 0],    # 2 myo     — yellow
        [200, 0,   0],    # 3 LV      — red
    ], dtype=np.uint8)
    return palette[mask.clip(0, 3)]


def run_inference(model, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))   # (1, C, H, W)
        pred   = torch.argmax(logits, dim=1).squeeze(0)        # (H, W)
    return pred.cpu().numpy()


def plot_result(image_np, gt_np, pred_np, title: str, save_path: str | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("MRI Slice")
    axes[0].axis("off")

    axes[1].imshow(colorize_mask(gt_np))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(colorize_mask(pred_np))
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255)
        for c in [[0,0,0],[0,200,0],[255,200,0],[200,0,0]]
    ]
    fig.legend(legend_patches, CLASS_NAMES, loc="lower center",
               ncol=4, frameon=False, fontsize=9)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default=None,
                        help="Path to experiment folder (contains model.pth). "
                             "Defaults to the first folder found in results/.")
    parser.add_argument("--save", action="store_true",
                        help="Save the output plot as inference_result.png inside the exp folder.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible slice selection.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # ---- paths ---------------------------------------------------------------
    repo_root   = os.path.dirname(SRC_DIR)
    results_dir = os.path.join(repo_root, "results")

    exp_dir = args.exp if args.exp else find_first_exp(results_dir)
    exp_dir = os.path.abspath(exp_dir)
    print(f"Experiment: {exp_dir}")

    # ---- device + model ------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(exp_dir, device)

    # ---- dataset (test split) ------------------------------------------------
    data_root  = os.path.join(repo_root, "data", "ACDC")
    test_dir   = os.path.join(data_root, "testing")

    print("Loading test patients...")
    test_patients   = load_split(test_dir)
    test_processed  = [preprocess_patient(p) for p in test_patients]
    test_dataset    = ACDCDataset(test_processed, augment=False)

    # ---- random slice --------------------------------------------------------
    idx         = random.randint(0, len(test_dataset) - 1)
    image, mask = test_dataset[idx]          # (1, H, W), (H, W)
    print(f"Slice index: {idx}/{len(test_dataset)-1}  "
          f"image shape: {tuple(image.shape)}  unique labels: {mask.unique().tolist()}")

    # ---- inference -----------------------------------------------------------
    pred = run_inference(model, image, device)

    image_np = image.squeeze().numpy()
    gt_np    = mask.numpy()

    # quick per-class dice
    dice_per_class = []
    for c in range(1, 4):
        inter = ((pred == c) & (gt_np == c)).sum()
        union = (pred == c).sum() + (gt_np == c).sum()
        dice_per_class.append(2 * inter / (union + 1e-8) if union > 0 else float("nan"))

    print("Dice (fg classes):")
    for name, d in zip(CLASS_NAMES[1:], dice_per_class):
        print(f"  {name:12s}: {d:.4f}")

    # ---- plot ----------------------------------------------------------------
    save_path = os.path.join(exp_dir, "inference_result.png") if args.save else None
    exp_name  = os.path.basename(exp_dir)
    plot_result(image_np, gt_np, pred,
                title=f"{exp_name} — slice {idx}", save_path=save_path)


if __name__ == "__main__":
    main()
