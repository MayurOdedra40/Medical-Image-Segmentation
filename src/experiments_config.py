import torch
import torch.nn as nn
import torch.optim as optim

from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.transunet import TransUNet

from utils.loss_funcs import  DiceLoss, DiceCELoss

MODELS = {
    "UNet": UNet,
    "AttentionUNet": AttentionUNet,
    "TransUNet": TransUNet,
}

LOSSES = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "DiceLoss": DiceLoss,
    "DiceCELoss": DiceCELoss,
}

OPTIMIZERS = {
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "SGD": optim.SGD,
}

LEARNING_PARAMS_DEFAULT = {
    "lr": 1e-3,
    "weight_decay": 0.0,
    "epochs": 2,
    "batch_size": 8
}

EXPERIMENTS = [
    # --------------------------------------------------
    # Baseline
    # --------------------------------------------------
    {
        **LEARNING_PARAMS_DEFAULT,

        "name": "unet_ce_adam",
        "model": MODELS["UNet"],
        "loss_func": LOSSES["CrossEntropyLoss"],
        "optimizer": OPTIMIZERS["Adam"],
    },
    # --------------------------------------------------
    # Loss comparison
    # --------------------------------------------------
    {
        **LEARNING_PARAMS_DEFAULT,

        "name": "unet_dice_adam",
        "model": MODELS["UNet"],
        "loss_func": LOSSES["DiceLoss"],
        "optimizer": OPTIMIZERS["Adam"],
    },
    {
        **LEARNING_PARAMS_DEFAULT,

        "name": "unet_diceCE_adam",
        "model": MODELS["UNet"],
        "loss_func": LOSSES["DiceCELoss"],
        "optimizer": OPTIMIZERS["Adam"],
    },
    # --------------------------------------------------
    # Architecture comparison
    # --------------------------------------------------
    {
        **LEARNING_PARAMS_DEFAULT,

        "name": "attention_diceCE_adam",
        "model": MODELS["AttentionUNet"],
        "loss_func": LOSSES["DiceCELoss"],
        "optimizer": OPTIMIZERS["Adam"],
    },
    {
        **LEARNING_PARAMS_DEFAULT,

        "name": "transunet_diceCE_adam",
        "model": MODELS["TransUNet"],
        "loss_func": LOSSES["DiceCELoss"],
        "optimizer": OPTIMIZERS["Adam"],
    },
    # --------------------------------------------------
    # Optimizer comparison
    # --------------------------------------------------
    {
        **LEARNING_PARAMS_DEFAULT,

        "name": "unet_diceCE_adamW",
        "model": MODELS["UNet"],
        "loss_func": LOSSES["DiceCELoss"],
        "optimizer": OPTIMIZERS["AdamW"],
    },
    {
        **LEARNING_PARAMS_DEFAULT,

        "name": "unet_diceCE_sgd",
        "model": MODELS["UNet"],
        "loss_func": LOSSES["DiceCELoss"],
        "optimizer": OPTIMIZERS["SGD"],
    },
    # --------------------------------------------------
    # Hyperparameter comparison
    # --------------------------------------------------
    {
        **LEARNING_PARAMS_DEFAULT,

        "name": "unet_ce_adam_lr_1e4",
        "model": MODELS["UNet"],
        "loss_func": LOSSES["CrossEntropyLoss"],
        "optimizer": OPTIMIZERS["Adam"],

        "lr": 1e-4
    }
]