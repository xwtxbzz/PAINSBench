"""Visualization utilities for PAINSBench."""

import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIGURES_DIR

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


def plot_pains_comparison(results_df, filename="pains_comparison.png"):
    """Bar plot comparing PAINS+ vs PAINS- RMSE per model."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(results_df))
    w = 0.35
    ax.bar(x - w / 2, results_df["pains_pos_RMSE"], w, label="PAINS+", color="#e74c3c")
    ax.bar(x + w / 2, results_df["pains_neg_RMSE"], w, label="PAINS−", color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["model"], rotation=45, ha="right")
    ax.set_ylabel("RMSE (pChEMBL)")
    ax.legend()
    ax.set_title("PAINS+ vs PAINS− Prediction Error")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, filename))
    plt.close(fig)


def plot_delta_rmse(results_df, filename="delta_rmse.png"):
    """ΔRMSE bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in results_df["delta_rmse"]]
    ax.barh(results_df["model"], results_df["delta_rmse"], color=colors)
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("ΔRMSE (PAINS+ − PAINS−)")
    ax.set_title("PAINS Robustness: Positive = Worse on PAINS Compounds")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, filename))
    plt.close(fig)


def plot_fp_ratio(results_df, filename="fp_ratio.png"):
    """False-positive ratio bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(results_df["model"], results_df["fp_ratio"], color="#e67e22")
    ax.axvline(1, color="gray", ls="--", lw=1)
    ax.set_xlabel("FP Ratio (PAINS+ residual / PAINS− residual)")
    ax.set_title("False-Positive Susceptibility to PAINS Interference")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, filename))
    plt.close(fig)


def plot_scatter(y_true, y_pred, pains_status, model_name, filename=None):
    """Predicted vs actual, colored by PAINS status."""
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {0: "#3498db", 1: "#e74c3c"}
    labels = {0: "PAINS−", 1: "PAINS+"}
    for status in [0, 1]:
        mask = pains_status == status
        ax.scatter(y_true[mask], y_pred[mask], c=colors[status],
                   label=labels[status], alpha=0.3, s=8, edgecolors="none")
    ax.plot([2, 11], [2, 11], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("True pChEMBL")
    ax.set_ylabel("Predicted pChEMBL")
    ax.set_title(f"{model_name} — Predicted vs Actual")
    ax.legend()
    ax.set_xlim(2, 11)
    ax.set_ylim(2, 11)
    fig.tight_layout()
    if filename:
        fig.savefig(os.path.join(FIGURES_DIR, filename))
    else:
        fig.savefig(os.path.join(FIGURES_DIR, f"scatter_{model_name}.png"))
    plt.close(fig)


def plot_residual_distribution(residuals_pos, residuals_neg, model_name, filename=None):
    """Residual distribution comparison."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals_neg, bins=50, alpha=0.6, label="PAINS−", color="#3498db", density=True)
    ax.hist(residuals_pos, bins=50, alpha=0.6, label="PAINS+", color="#e74c3c", density=True)
    ax.set_xlabel("|Residual| (pChEMBL)")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name} — Residual Distribution by PAINS Status")
    ax.legend()
    fig.tight_layout()
    fn = filename or f"residuals_{model_name}.png"
    fig.savefig(os.path.join(FIGURES_DIR, fn))
    plt.close(fig)
