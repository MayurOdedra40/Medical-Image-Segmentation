import torch.optim as optim

from models.unet import UNet
from models.transunet import TransUNet

from utils.loss_funcs import (
    CrossEntropyOnlyLoss,
    DiceLoss,
    DiceCELoss,
    WeightedDiceLoss,
    GeneralizedDiceLoss,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODELS = {
    "UNet":     UNet,
    "TransUNet": TransUNet,
}

LOSSES = {
    "CE":                 CrossEntropyOnlyLoss,
    "DiceLoss":           DiceLoss,
    "DiceCELoss":         DiceCELoss,
    "WeightedDiceLoss":   WeightedDiceLoss,
    "GeneralizedDiceLoss": GeneralizedDiceLoss,
}

OPTIMIZERS = {
    "Adam":  optim.Adam,
    "AdamW": optim.AdamW,
}

# ---------------------------------------------------------------------------
# Shared training hyperparameters (frozen — do not vary between blocks)
# ---------------------------------------------------------------------------

LEARNING_PARAMS_DEFAULT = {
    "lr":           1e-3,
    "weight_decay": 0.0,
    "epochs":       1,
    "batch_size":   8,
}

SEEDS = [42, 123, 456]

# ---------------------------------------------------------------------------
# Block A — Architecture comparison: UNet vs TransUNet
# Identical everything; 3 seeds; headline result.
# ---------------------------------------------------------------------------

BLOCK_A = [
    {
        **LEARNING_PARAMS_DEFAULT,
        "name":      f"blockA_{model}_seed{seed}",
        "block":     "A",
        "model":     MODELS[model],
        "loss_func": LOSSES["DiceCELoss"],
        "optimizer": OPTIMIZERS["Adam"],
        "seed":      seed,
    }
    for model in ["UNet", "TransUNet"]
    for seed  in SEEDS
]

# ---------------------------------------------------------------------------
# Block B sweep — Loss study: 5 loss functions × 2 models, single seed.
# Inspect results to pick the winning loss, then fill BLOCK_B_WINNER below.
# ---------------------------------------------------------------------------

BLOCK_B_SWEEP = [
    {
        **LEARNING_PARAMS_DEFAULT,
        "name":      f"blockB_{model}_{loss_key}_seed42",
        "block":     "B_sweep",
        "model":     MODELS[model],
        "loss_func": LOSSES[loss_key],
        "optimizer": OPTIMIZERS["Adam"],
        "seed":      42,
    }
    for model    in ["UNet", "TransUNet"]
    for loss_key in ["CE", "DiceLoss", "DiceCELoss", "WeightedDiceLoss", "GeneralizedDiceLoss"]
]

# ---------------------------------------------------------------------------
# Block B winner — run the best loss with the remaining 2 seeds.
# After inspecting BLOCK_B_SWEEP results, set WINNING_LOSS to the key of
# the best-performing loss and uncomment BLOCK_B_WINNER.
# ---------------------------------------------------------------------------

WINNING_LOSS = "DiceCELoss"   # update after sweep

BLOCK_B_WINNER = [
    {
        **LEARNING_PARAMS_DEFAULT,
        "name":      f"blockB_{model}_{WINNING_LOSS}_seed{seed}",
        "block":     "B_winner",
        "model":     MODELS[model],
        "loss_func": LOSSES[WINNING_LOSS],
        "optimizer": OPTIMIZERS["Adam"],
        "seed":      seed,
    }
    for model in ["UNet", "TransUNet"]
    for seed  in [123, 456]          # seed 42 already run in sweep
]

# ---------------------------------------------------------------------------
# Active experiment list — edit to run only the block you need
# ---------------------------------------------------------------------------

EXPERIMENTS = BLOCK_A + BLOCK_B_SWEEP
# After the sweep, append BLOCK_B_WINNER:
# EXPERIMENTS = BLOCK_A + BLOCK_B_SWEEP + BLOCK_B_WINNER
