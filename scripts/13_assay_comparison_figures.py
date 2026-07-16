"""Generate Ki vs IC50 comparison figures for PAINSBench."""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, FIGURES_DIR

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ========== Load data ==========
ki = pd.read_csv(os.path.join(RESULTS_DIR, "dta_ki_results.csv"))
ic50 = pd.read_csv(os.path.join(RESULTS_DIR, "dta_ic50_results.csv"))

ki["assay"] = "Ki"
ic50["assay"] = "IC50"
both = pd.concat([ki, ic50], ignore_index=True)
both["delta_rmse"] = both["delta_rmse"].fillna(0)
both["fp_ratio"] = both["fp_ratio"].fillna(1.0)

models = sorted(both["model"].unique())
n_models = len(models)

print(f"Ki models: {len(ki)}  IC50 models: {len(ic50)}  Total: {len(both)}")
print(f"Models: {models}")

# ========== Figure 1: Paired bar chart - RMSE ==========
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(n_models)
w = 0.32

ki_sorted = ki.sort_values("overall_RMSE")["model"].values
order = {m: i for i, m in enumerate(ki_sorted)}
both["order"] = both["model"].map(order)
both_sorted = both.sort_values("order")

ki_vals = both_sorted[both_sorted["assay"] == "Ki"].set_index("model").loc[ki_sorted, "overall_RMSE"]
ic50_vals = both_sorted[both_sorted["assay"] == "IC50"].set_index("model").loc[ki_sorted, "overall_RMSE"]

bars1 = ax.bar(x - w/2, ki_vals.values, w, label="Ki",
               color="#2166ac", alpha=0.85, edgecolor="white", linewidth=0.3)
bars2 = ax.bar(x + w/2, ic50_vals.values, w, label="IC50",
               color="#d6604d", alpha=0.85, edgecolor="white", linewidth=0.3)

ax.set_xticks(x)
ax.set_xticklabels(ki_sorted, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Overall RMSE")
ax.set_title("DTA Model Accuracy: Ki vs IC50")
ax.legend(fontsize=8)
ax.set_xlim(-0.5, n_models - 0.5)

# Add value labels
for i, (k, ic) in enumerate(zip(ki_vals.values, ic50_vals.values)):
    ax.text(i - w/2, k + 0.008, f"{k:.3f}", ha="center", va="bottom", fontsize=5, rotation=90)
    ax.text(i + w/2, ic + 0.008, f"{ic:.3f}", ha="center", va="bottom", fontsize=5, rotation=90)

fig.savefig(os.path.join(FIGURES_DIR, "assay_comp_rmse.png"))
plt.close()

# ========== Figure 2: Paired bar chart - ΔRMSE ==========
fig, ax = plt.subplots(figsize=(9, 4))

ki_drmse = both_sorted[both_sorted["assay"] == "Ki"].set_index("model").loc[ki_sorted, "delta_rmse"]
ic50_drmse = both_sorted[both_sorted["assay"] == "IC50"].set_index("model").loc[ki_sorted, "delta_rmse"]

bars1 = ax.bar(x - w/2, ki_drmse.values, w, label="Ki",
               color="#2166ac", alpha=0.85, edgecolor="white", linewidth=0.3)
bars2 = ax.bar(x + w/2, ic50_drmse.values, w, label="IC50",
               color="#d6604d", alpha=0.85, edgecolor="white", linewidth=0.3)

ax.axhline(0, color="gray", lw=0.8, ls="-", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(ki_sorted, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("ΔRMSE (PAINS+ − PAINS−)")
ax.set_title("PAINS Robustness: Ki vs IC50")
ax.legend(fontsize=8)
ax.set_xlim(-0.5, n_models - 0.5)

for i, (k, ic) in enumerate(zip(ki_drmse.values, ic50_drmse.values)):
    offset_k = 0.005 if k >= 0 else -0.015
    offset_ic = 0.005 if ic >= 0 else -0.015
    ax.text(i - w/2, k + offset_k, f"{k:.3f}", ha="center", va="bottom" if k >= 0 else "top", fontsize=5, rotation=90)
    ax.text(i + w/2, ic + offset_ic, f"{ic:.3f}", ha="center", va="bottom" if ic >= 0 else "top", fontsize=5, rotation=90)

fig.savefig(os.path.join(FIGURES_DIR, "assay_comp_delta_rmse.png"))
plt.close()

# ========== Figure 3: Scatter plot - Ki ΔRMSE vs IC50 ΔRMSE ==========
fig, ax = plt.subplots(figsize=(6, 5))

ki_dict = ki.set_index("model")["delta_rmse"].to_dict()
ic50_dict = ic50.set_index("model")["delta_rmse"].to_dict()

x_vals = [ki_dict[m] for m in models if m in ki_dict]
y_vals = [ic50_dict[m] for m in models if m in ic50_dict]
valid_models = [m for m in models if m in ki_dict and m in ic50_dict]

sc = ax.scatter(x_vals, y_vals, c=range(len(valid_models)), cmap="viridis",
                s=120, alpha=0.8, edgecolors="white", linewidth=0.5, zorder=5)

for m, xv, yv in zip(valid_models, x_vals, y_vals):
    ax.annotate(m, (xv, yv), fontsize=6, ha="center", va="bottom", alpha=0.8)

# Identity line
lims = [min(x_vals + y_vals) - 0.02, max(x_vals + y_vals) + 0.02]
ax.plot(lims, lims, "gray", ls="--", lw=0.8, alpha=0.5, label="y = x")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("ΔRMSE (Ki)")
ax.set_ylabel("ΔRMSE (IC50)")
ax.set_title("PAINS Robustness Correlation: Ki vs IC50")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Correlation
corr = np.corrcoef(x_vals, y_vals)[0, 1]
ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
        fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

fig.savefig(os.path.join(FIGURES_DIR, "assay_comp_correlation.png"))
plt.close()

# ========== Figure 4: RMSE ΔRMSE trade-off (both assays) ==========
fig, ax = plt.subplots(figsize=(7.5, 5))

markers = {"Ki": "o", "IC50": "^"}
colors = {"Ki": "#2166ac", "IC50": "#d6604d"}

for assay_name, marker in markers.items():
    subset = both[both["assay"] == assay_name]
    ax.scatter(subset["overall_RMSE"], subset["delta_rmse"],
               c=colors[assay_name], marker=marker, s=100,
               alpha=0.7, edgecolors="white", linewidth=0.5,
               label=assay_name, zorder=5)

for _, row in both.iterrows():
    ax.annotate(row["model"], (row["overall_RMSE"], row["delta_rmse"]),
                fontsize=5.5, ha="center", va="bottom", alpha=0.7,
                color=colors[row["assay"]])

ax.axhline(0, color="gray", lw=0.8, ls="-", alpha=0.3)
ax.set_xlabel("Overall RMSE")
ax.set_ylabel("ΔRMSE (PAINS robustness)")
ax.set_title("Accuracy–Robustness Trade-off: Ki vs IC50")
ax.legend(fontsize=9, loc="lower left")
ax.set_xlim(both["overall_RMSE"].min() - 0.05, both["overall_RMSE"].max() + 0.05)

fig.savefig(os.path.join(FIGURES_DIR, "assay_comp_tradeoff.png"))
plt.close()

# ========== Figure 5: FP Ratio comparison ==========
fig, ax = plt.subplots(figsize=(9, 4))

ki_fp = both_sorted[both_sorted["assay"] == "Ki"].set_index("model").loc[ki_sorted, "fp_ratio"]
ic50_fp = both_sorted[both_sorted["assay"] == "IC50"].set_index("model").loc[ki_sorted, "fp_ratio"]

bars1 = ax.bar(x - w/2, ki_fp.values, w, label="Ki",
               color="#2166ac", alpha=0.85, edgecolor="white", linewidth=0.3)
bars2 = ax.bar(x + w/2, ic50_fp.values, w, label="IC50",
               color="#d6604d", alpha=0.85, edgecolor="white", linewidth=0.3)

ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(ki_sorted, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("FP Ratio")
ax.set_title("False-Positive Susceptibility: Ki vs IC50")
ax.legend(fontsize=8)
ax.set_xlim(-0.5, n_models - 0.5)

for i, (k, ic) in enumerate(zip(ki_fp.values, ic50_fp.values)):
    ax.text(i - w/2, k + 0.008, f"{k:.3f}", ha="center", va="bottom", fontsize=5, rotation=90)
    ax.text(i + w/2, ic + 0.008, f"{ic:.3f}", ha="center", va="bottom", fontsize=5, rotation=90)

fig.savefig(os.path.join(FIGURES_DIR, "assay_comp_fp_ratio.png"))
plt.close()

# ========== Figure 6: ΔRMSE rank comparison (dot plot) ==========
fig, ax = plt.subplots(figsize=(8, 5))

ki_ranked = ki.sort_values("delta_rmse", ascending=False)
ic50_ranked = ic50.sort_values("delta_rmse", ascending=False)

ki_order = {m: i for i, m in enumerate(ki_ranked["model"])}
ic50_order = {m: i for i, m in enumerate(ic50_ranked["model"])}

y_ki = np.arange(len(ki_ranked))
y_ic50 = np.arange(len(ic50_ranked))

ax.scatter(ki_ranked["delta_rmse"], y_ki, c="#2166ac", s=100,
           marker="o", label="Ki", zorder=5, edgecolors="white", linewidth=0.5)
ax.scatter(ic50_ranked["delta_rmse"], y_ic50, c="#d6604d", s=100,
           marker="^", label="IC50", zorder=5, edgecolors="white", linewidth=0.5)

# Connect same models
for m in models:
    if m in ki_order and m in ic50_order:
        kr = ki_ranked[ki_ranked["model"] == m].iloc[0]
        ir = ic50_ranked[ic50_ranked["model"] == m].iloc[0]
        ax.plot([kr["delta_rmse"], ir["delta_rmse"]],
                [ki_order[m], ic50_order[m]],
                "gray", lw=0.5, alpha=0.4)
        ax.annotate(m, (kr["delta_rmse"], ki_order[m]),
                    fontsize=5.5, ha="right" if kr["delta_rmse"] > ir["delta_rmse"] else "left",
                    va="center", alpha=0.7)

ax.axvline(0, color="gray", lw=0.8, ls="-", alpha=0.3)
ax.set_yticks(y_ki)
ax.set_yticklabels(ki_ranked["model"], fontsize=7)
ax.set_xlabel("ΔRMSE")
ax.set_title("ΔRMSE Ranking: Ki vs IC50")
ax.legend(fontsize=9, loc="lower right")
ax.invert_yaxis()

fig.savefig(os.path.join(FIGURES_DIR, "assay_comp_ranking.png"))
plt.close()

# ========== Summary table ==========
print("\n" + "=" * 80)
print(f"{'Model':20s} {'Ki_RMSE':>9s} {'IC50_RMSE':>10s} {'Ki_ΔRMSE':>9s} {'IC50_ΔRMSE':>10s} {'Ki_FP':>7s} {'IC50_FP':>8s}")
print("=" * 80)
for m in sorted(set(ki["model"]) & set(ic50["model"]), key=lambda x: (ki[ki["model"]==x]["overall_RMSE"].values[0])):
    kr = ki[ki["model"]==m].iloc[0]
    ir = ic50[ic50["model"]==m].iloc[0]
    print(f"{m:20s} {kr['overall_RMSE']:9.4f} {ir['overall_RMSE']:10.4f} "
          f"{kr['delta_rmse']:9.4f} {ir['delta_rmse']:10.4f} "
          f"{kr['fp_ratio']:7.4f} {ir['fp_ratio']:8.4f}")

print(f"\nFigures saved to figures/assay_comp_*.png")
print("Done.")
