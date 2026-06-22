#!/usr/bin/env python3
"""
Generate all result plots and tables for the ACDC segmentation project.

Run from the project root:
    conda run -n base python src/generate_results.py

Outputs (saved to docs/results/):
    blockA_bar_chart.png       – grouped bar chart, mean +/- std per model per class
    blockA_learning_curves.png – val loss + dice curves with std bands over seeds
    blockA_table.png           – formatted table, mean +/- std, best value highlighted
    blockB_heatmap.png         – 4-panel heatmap, model x loss for each class
    blockB_per_class_bars.png  – grouped bars, per model subplot, grouped by loss
    blockB_learning_curves.png – dice_mean training curves, per model subplot
    blockB_table.png           – formatted table, best value highlighted
"""

import csv
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR     = os.path.join(BASE_DIR, "docs", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Display names ─────────────────────────────────────────────────────────────
MODEL_NAMES = {
    "UNet":          "U-Net",
    "AttentionUNet": "Attention U-Net",
    "TransUNet":     "TransU-Net",
}
LOSS_NAMES = {
    "CrossEntropyOnlyLoss": "Cross Entropy",
    "DiceLoss":             "Dice Loss",
    "DiceCELoss":           "Dice + CE Loss",
}
CLASS_LABELS = ["Mean", "RV", "MYO", "LV"]
CLASS_KEYS   = ["dice_mean", "dice_rv", "dice_myo", "dice_lv"]

# ── Colour palette ────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "UNet":          "#2196F3",
    "AttentionUNet": "#FF9800",
    "TransUNet":     "#4CAF50",
}
LOSS_COLORS = {
    "CrossEntropyOnlyLoss": "#E91E63",
    "DiceLoss":             "#9C27B0",
    "DiceCELoss":           "#00ACC1",
}

# ── Experiment ordering ───────────────────────────────────────────────────────
BLOCK_A_MODELS = ["UNet", "AttentionUNet", "TransUNet"]
BLOCK_A_SEEDS  = [42, 123, 456]
BLOCK_B_MODELS = ["UNet", "AttentionUNet", "TransUNet"]
BLOCK_B_LOSSES = ["CrossEntropyOnlyLoss", "DiceLoss", "DiceCELoss"]

# ── Global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "figure.dpi":         150,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_experiments():
    """Return dict name -> {config, epochs, best, best_epoch}."""
    exps = {}
    for name in sorted(os.listdir(RESULTS_DIR)):
        exp_dir  = os.path.join(RESULTS_DIR, name)
        cfg_path = os.path.join(exp_dir, "config.json")
        met_path = os.path.join(exp_dir, "metrics.csv")
        if not (os.path.isdir(exp_dir) and os.path.exists(cfg_path) and os.path.exists(met_path)):
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)

        epoch_rows = []
        with open(met_path) as f:
            for row in csv.DictReader(f):
                try:
                    epoch_rows.append({
                        "train_loss": float(row["train_loss"]),
                        "val_loss":   float(row["val_loss"]),
                        "dice_mean":  float(row["dice_mean"]),
                        "dice_rv":    float(row["dice_rv"]),
                        "dice_myo":   float(row["dice_myo"]),
                        "dice_lv":    float(row["dice_lv"]),
                    })
                except (ValueError, KeyError):
                    pass  # skip corrupted / incomplete rows

        if not epoch_rows:
            continue

        best_idx = max(range(len(epoch_rows)), key=lambda i: epoch_rows[i]["dice_mean"])
        exps[name] = {
            "config":     cfg,
            "epochs":     epoch_rows,
            "best":       epoch_rows[best_idx],
            "best_epoch": best_idx + 1,
        }
    return exps


def blockA_summary(exps):
    """Return {model: {metric_key: [val_seed42, val_seed123, val_seed456]}}."""
    out = {}
    for model in BLOCK_A_MODELS:
        out[model] = {k: [] for k in CLASS_KEYS}
        for seed in BLOCK_A_SEEDS:
            name = f"blockA_{model}_seed{seed}"
            if name in exps:
                for k in CLASS_KEYS:
                    out[model][k].append(exps[name]["best"][k])
    return out


def blockB_summary(exps):
    """Return {(model, loss_func): best_metrics_dict}."""
    out = {}
    for exp in exps.values():
        cfg = exp["config"]
        if cfg.get("block", "").startswith("B"):
            out[(cfg["model"], cfg["loss_func"])] = exp["best"]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK A — ARCHITECTURE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def plot_blockA_bar_chart(a_data, path):
    """Grouped bar chart: one group per class, one bar per model, error = std."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(CLASS_LABELS))
    w = 0.22
    n = len(BLOCK_A_MODELS)

    for i, model in enumerate(BLOCK_A_MODELS):
        means = [np.mean(a_data[model][k]) for k in CLASS_KEYS]
        stds  = [np.std( a_data[model][k]) for k in CLASS_KEYS]
        off   = (i - (n - 1) / 2) * w

        bars = ax.bar(
            x + off, means, w,
            yerr=stds, capsize=5,
            color=MODEL_COLORS[model], alpha=0.85,
            label=MODEL_NAMES[model],
            error_kw={"elinewidth": 1.5, "ecolor": "#333", "capthick": 1.5},
            zorder=3,
        )
        # Value labels inside bars
        for bar, mean_val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.016,
                f"{mean_val:.3f}",
                ha="center", va="top",
                fontsize=7.5, color="white", fontweight="bold",
            )

    ax.set_xlabel("Segmentation Class", labelpad=8)
    ax.set_ylabel("Dice Score", labelpad=8)
    ax.set_title(
        "Block A: Architecture Comparison — Best-Epoch Dice Score\n"
        "(Mean ± Std over Seeds: 42 / 123 / 456, Loss: Dice + CE)",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_ylim(0.55, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.legend(loc="lower right", title="Model Architecture")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {os.path.basename(path)}")


def plot_blockA_learning_curves(exps, path):
    """Side-by-side: (1) val loss on log scale, (2) val mean dice — both with std band."""
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(14, 5))

    for model in BLOCK_A_MODELS:
        loss_curves, dice_curves = [], []
        for seed in BLOCK_A_SEEDS:
            name = f"blockA_{model}_seed{seed}"
            if name in exps:
                loss_curves.append([ep["val_loss"]  for ep in exps[name]["epochs"]])
                dice_curves.append([ep["dice_mean"] for ep in exps[name]["epochs"]])
        if not loss_curves:
            continue

        n_ep  = min(len(c) for c in loss_curves)
        ep    = np.arange(1, n_ep + 1)
        color = MODEL_COLORS[model]
        label = MODEL_NAMES[model]

        for ax, curves in [(ax_loss, loss_curves), (ax_dice, dice_curves)]:
            arr  = np.array([c[:n_ep] for c in curves])
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            ax.plot(ep, mean, color=color, label=label, linewidth=2, zorder=3)
            ax.fill_between(ep, mean - std, mean + std, color=color, alpha=0.15)

    ax_loss.set_xlabel("Epoch", labelpad=8)
    ax_loss.set_ylabel("Validation Loss (log scale)", labelpad=8)
    ax_loss.set_title("Validation Loss over Training")
    ax_loss.set_yscale("log")
    ax_loss.set_xlim(1, None)
    ax_loss.legend(title="Model")

    ax_dice.set_xlabel("Epoch", labelpad=8)
    ax_dice.set_ylabel("Mean Dice Score (Validation)", labelpad=8)
    ax_dice.set_title("Mean Dice Score over Training")
    ax_dice.set_ylim(0.0, 1.0)
    ax_dice.set_xlim(1, None)
    ax_dice.legend(title="Model")

    fig.suptitle(
        "Block A: Training Dynamics — Mean ± Std over 3 Seeds",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {os.path.basename(path)}")


def make_blockA_table(a_data, path):
    """Matplotlib table: rows = models, cols = per-class dice (mean +/- std)."""
    col_labels = ["Model"] + [f"Dice {c}" for c in CLASS_LABELS]

    best_per_col = [
        max(np.mean(a_data[m][k]) for m in BLOCK_A_MODELS if a_data[m][k])
        for k in CLASS_KEYS
    ]

    rows = []
    for model in BLOCK_A_MODELS:
        row = [MODEL_NAMES[model]]
        for k in CLASS_KEYS:
            v = a_data[model][k]
            row.append(f"{np.mean(v):.4f} +/- {np.std(v):.4f}" if v else "—")
        rows.append(row)

    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 2.3)

    # Header row
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#37474F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Data rows: alternating background + green highlight for column winner
    for ri, model in enumerate(BLOCK_A_MODELS):
        alt_bg = "#ECEFF1" if ri % 2 == 0 else "white"
        for ci in range(len(col_labels)):
            tbl[ri + 1, ci].set_facecolor(alt_bg)
        for ci, (k, bv) in enumerate(zip(CLASS_KEYS, best_per_col)):
            v = a_data[model][k]
            if v and abs(np.mean(v) - bv) < 1e-5:
                tbl[ri + 1, ci + 1].set_facecolor("#C8E6C9")
                tbl[ri + 1, ci + 1].set_text_props(fontweight="bold")

    ax.set_title(
        "Block A: Architecture Comparison  —  Best-Epoch Dice Score  (Mean ± Std over Seeds 42 / 123 / 456)\n"
        "Green cell = best value per column",
        pad=10, fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK B — LOSS FUNCTION ABLATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_blockB_heatmap(b_data, path):
    """4-panel heatmap: one subplot per class, rows = models, cols = losses."""
    cmap = LinearSegmentedColormap.from_list("dice_cmap", ["#EF5350", "#FFEE58", "#66BB6A"])
    models_disp = [MODEL_NAMES[m] for m in BLOCK_B_MODELS]
    losses_disp = [LOSS_NAMES[l]  for l in BLOCK_B_LOSSES]

    fig, axes = plt.subplots(1, len(CLASS_LABELS), figsize=(16, 4.2), sharey=True)
    im_ref = None

    for ax, ckey, clabel in zip(axes, CLASS_KEYS, CLASS_LABELS):
        mat = np.array([
            [b_data.get((m, l), {}).get(ckey, 0.0) for l in BLOCK_B_LOSSES]
            for m in BLOCK_B_MODELS
        ])
        im = ax.imshow(mat, cmap=cmap, vmin=0.5, vmax=0.95, aspect="auto")
        im_ref = im

        ax.set_xticks(range(len(BLOCK_B_LOSSES)))
        ax.set_xticklabels(losses_disp, rotation=35, ha="right", fontsize=10)
        ax.set_yticks(range(len(BLOCK_B_MODELS)))
        ax.set_yticklabels(models_disp, fontsize=10)
        ax.set_title(f"{clabel} Dice", fontsize=12, pad=6)
        ax.set_xlabel("Loss Function", labelpad=6)
        ax.grid(False)

        for i in range(len(BLOCK_B_MODELS)):
            for j in range(len(BLOCK_B_LOSSES)):
                v = mat[i, j]
                text_color = "white" if v < 0.68 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=text_color)

    cbar = fig.colorbar(im_ref, ax=axes[-1], fraction=0.055, pad=0.04)
    cbar.set_label("Dice Score", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Block B: Loss Function Ablation — Best-Epoch Dice Score (Seed 42)\n"
        "Colour scale: 0.50 (red) → 0.95 (green)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {os.path.basename(path)}")


def plot_blockB_per_class_bars(b_data, path):
    """One subplot per model; within each, bars grouped by loss function."""
    x = np.arange(len(CLASS_LABELS))
    w = 0.22
    n = len(BLOCK_B_LOSSES)

    fig, axes = plt.subplots(1, len(BLOCK_B_MODELS), figsize=(16, 5.5), sharey=True)

    for ax, model in zip(axes, BLOCK_B_MODELS):
        for j, loss in enumerate(BLOCK_B_LOSSES):
            vals = [b_data.get((model, loss), {}).get(k, 0.0) for k in CLASS_KEYS]
            off  = (j - (n - 1) / 2) * w
            ax.bar(
                x + off, vals, w,
                color=LOSS_COLORS[loss], alpha=0.85,
                label=LOSS_NAMES[loss], zorder=3,
            )
            # Annotate bars whose value is notably low (< 0.70) to flag failures
            for xi, v in zip(x + off, vals):
                if v < 0.70:
                    ax.text(
                        xi, v + 0.012, f"{v:.3f}",
                        ha="center", va="bottom",
                        fontsize=7.5, color="#B71C1C", fontweight="bold",
                    )

        ax.set_title(MODEL_NAMES[model], fontsize=12, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_LABELS)
        ax.set_xlabel("Segmentation Class", labelpad=6)
        ax.set_ylim(0.0, 1.08)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
        ax.set_axisbelow(True)
        if model == BLOCK_B_MODELS[0]:
            ax.set_ylabel("Dice Score", labelpad=8)

    # Shared legend placed outside the axes
    handles = [
        mpatches.Patch(color=LOSS_COLORS[l], label=LOSS_NAMES[l], alpha=0.85)
        for l in BLOCK_B_LOSSES
    ]
    fig.legend(
        handles=handles, loc="upper right",
        bbox_to_anchor=(0.99, 0.96),
        title="Loss Function", frameon=True, fontsize=10,
    )
    fig.suptitle(
        "Block B: Per-Class Dice Score by Loss Function (Seed 42, Best Epoch)\n"
        "Red labels = values below 0.70",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {os.path.basename(path)}")


def plot_blockB_learning_curves(exps, path):
    """One subplot per model; within each, one curve per loss function."""
    fig, axes = plt.subplots(1, len(BLOCK_B_MODELS), figsize=(16, 5), sharey=True)

    for ax, model in zip(axes, BLOCK_B_MODELS):
        max_ep = 0
        for loss in BLOCK_B_LOSSES:
            match = None
            for exp in exps.values():
                cfg = exp["config"]
                if (cfg["model"] == model
                        and cfg["loss_func"] == loss
                        and cfg.get("block", "").startswith("B")):
                    match = exp
                    break
            if match is None:
                continue

            curve = [ep["dice_mean"] for ep in match["epochs"]]
            ep    = np.arange(1, len(curve) + 1)
            max_ep = max(max_ep, len(curve))

            ax.plot(
                ep, curve,
                color=LOSS_COLORS[loss], label=LOSS_NAMES[loss],
                linewidth=1.8, alpha=0.9,
            )
            # Dotted vertical line at best epoch
            ax.axvline(
                match["best_epoch"],
                color=LOSS_COLORS[loss], linestyle=":", alpha=0.55, linewidth=1.3,
            )

        ax.set_title(MODEL_NAMES[model], fontsize=12, pad=6)
        ax.set_xlabel("Epoch", labelpad=6)
        ax.set_xlim(1, max_ep if max_ep > 0 else 100)
        ax.set_ylim(-0.05, 1.05)
        if model == BLOCK_B_MODELS[0]:
            ax.set_ylabel("Mean Dice Score (Validation)", labelpad=8)
        ax.legend(title="Loss Function", fontsize=9)

    fig.suptitle(
        "Block B: Validation Dice Score over Training — Loss Function Comparison (Seed 42)\n"
        "Dotted vertical lines mark the best epoch per run",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {os.path.basename(path)}")


def make_blockB_table(b_data, path):
    """Matplotlib table: rows = (loss, model), cols = per-class dice."""
    col_labels = ["Loss Function", "Model"] + [f"Dice {c}" for c in CLASS_LABELS]

    # Row colour per loss to visually group rows
    loss_bg = {
        "CrossEntropyOnlyLoss": "#FCE4EC",
        "DiceLoss":             "#F3E5F5",
        "DiceCELoss":           "#E0F7FA",
    }
    best_per_col = [
        max(b_data.get((m, l), {}).get(k, 0.0)
            for m in BLOCK_B_MODELS for l in BLOCK_B_LOSSES)
        for k in CLASS_KEYS
    ]

    rows_meta = [
        (model, loss)
        for loss  in BLOCK_B_LOSSES
        for model in BLOCK_B_MODELS
    ]
    cell_rows = []
    for model, loss in rows_meta:
        best = b_data.get((model, loss), {})
        cell_rows.append(
            [LOSS_NAMES[loss], MODEL_NAMES[model]]
            + [f"{best.get(k, 0.0):.4f}" for k in CLASS_KEYS]
        )

    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.95)

    # Header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#37474F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Data rows
    for ri, (model, loss) in enumerate(rows_meta):
        bg = loss_bg[loss]
        for ci in range(len(col_labels)):
            tbl[ri + 1, ci].set_facecolor(bg)
        for ci, (k, bv) in enumerate(zip(CLASS_KEYS, best_per_col)):
            v = b_data.get((model, loss), {}).get(k, 0.0)
            if bv > 0 and abs(v - bv) < 1e-5:
                tbl[ri + 1, ci + 2].set_facecolor("#C8E6C9")
                tbl[ri + 1, ci + 2].set_text_props(fontweight="bold")

    # Legend patches for row colours
    legend_handles = [
        mpatches.Patch(color=loss_bg[l], label=LOSS_NAMES[l], edgecolor="#aaa")
        for l in BLOCK_B_LOSSES
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right", bbox_to_anchor=(1.0, 1.15),
        title="Row colour = loss", fontsize=9, frameon=True,
    )
    ax.set_title(
        "Block B: Loss Function Ablation — Best-Epoch Dice Scores (Seed 42)\n"
        "Green cell = best value per class across all experiments",
        pad=10, fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE TABLES (copy-paste into report)
# ══════════════════════════════════════════════════════════════════════════════

def print_blockA_table(a_data):
    sep = "=" * 74
    print(f"\n{sep}")
    print("BLOCK A — Architecture Comparison (Best-Epoch Dice, Mean +/- Std over Seeds 42/123/456)")
    print(sep)
    print(f"{'Model':<22} {'Dice Mean':>16} {'Dice RV':>16} {'Dice MYO':>16} {'Dice LV':>16}")
    print("-" * 74)
    for model in BLOCK_A_MODELS:
        row = MODEL_NAMES[model].ljust(22)
        for k in CLASS_KEYS:
            v = a_data[model][k]
            row += (f"{np.mean(v):.4f}+/-{np.std(v):.4f}".rjust(16) if v else "—".rjust(16))
        print(row)
    print(sep)


def print_blockB_table(b_data):
    sep = "=" * 68
    print(f"\n{sep}")
    print("BLOCK B — Loss Ablation (Best-Epoch Dice, Seed 42)")
    print(sep)
    print(f"{'Loss':<18} {'Model':<20} {'Mean':>8} {'RV':>8} {'MYO':>8} {'LV':>8}")
    print("-" * 68)
    for loss in BLOCK_B_LOSSES:
        for model in BLOCK_B_MODELS:
            best = b_data.get((model, loss), {})
            vals = [f"{best.get(k, 0.0):.4f}" for k in CLASS_KEYS]
            print(f"{LOSS_NAMES[loss]:<18} {MODEL_NAMES[model]:<20} "
                  + "  ".join(v.rjust(8) for v in vals))
        print()
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading experiments...")
    exps   = load_experiments()
    a_data = blockA_summary(exps)
    b_data = blockB_summary(exps)
    print(f"  Loaded {len(exps)} experiments.")

    print(f"\nSaving plots to: {OUT_DIR}\n")

    print("Block A — Architecture Comparison:")
    plot_blockA_bar_chart(a_data,        os.path.join(OUT_DIR, "blockA_bar_chart.png"))
    plot_blockA_learning_curves(exps,    os.path.join(OUT_DIR, "blockA_learning_curves.png"))
    make_blockA_table(a_data,            os.path.join(OUT_DIR, "blockA_table.png"))

    print("\nBlock B — Loss Function Ablation:")
    plot_blockB_heatmap(b_data,          os.path.join(OUT_DIR, "blockB_heatmap.png"))
    plot_blockB_per_class_bars(b_data,   os.path.join(OUT_DIR, "blockB_per_class_bars.png"))
    plot_blockB_learning_curves(exps,    os.path.join(OUT_DIR, "blockB_learning_curves.png"))
    make_blockB_table(b_data,            os.path.join(OUT_DIR, "blockB_table.png"))

    # Console summary tables
    print_blockA_table(a_data)
    print_blockB_table(b_data)

    print(f"\nDone. All outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
