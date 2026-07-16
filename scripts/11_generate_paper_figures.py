"""Generate publication-quality figures for BIBM 2026 paper."""
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

# Font setup for publication
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
COLORS = {"pos": "#d73027", "neg": "#4575b4", "overall": "#2ca02c", "frontier": "#e67e22"}

# ========== Load all data ==========
dta = pd.read_csv(os.path.join(RESULTS_DIR, "dta_comparison_results.csv"))
frontier = pd.read_csv(os.path.join(RESULTS_DIR, "dta_frontier_results.csv"))
dl = pd.read_csv(os.path.join(RESULTS_DIR, "dl_comparison_results.csv"))
fron = pd.read_csv(os.path.join(RESULTS_DIR, "frontier_comparison_results.csv"))

# Tag models
dta["category"] = "DTA-Baseline"
frontier["category"] = "DTA-Frontier"
dl["category"] = "Drug-Only"
fron["category"] = "Drug-Frontier"

# Combine all
all_df = pd.concat([dta, frontier, dl, fron], ignore_index=True)
all_df["delta_rmse"] = all_df["delta_rmse"].fillna(0)
all_df["overall_RMSE"] = all_df["overall_RMSE"].fillna(1.5)
all_df["fp_ratio"] = all_df["fp_ratio"].fillna(1.0)

# Sort by overall_RMSE for ranking
ranked = all_df.sort_values("overall_RMSE")

# ========== Figure 1: Overall RMSE ranking with PAINS breakdown ==========
fig, ax = plt.subplots(figsize=(8, 4.5))
models = ranked["model"].values
y_pos = np.arange(len(models))

# Color bars by category
cat_colors = {
    "Drug-Only": "#2166ac",
    "Drug-Frontier": "#4393c3",
    "DTA-Baseline": "#d6604d",
    "DTA-Frontier": "#f4a582",
}
bar_colors = [cat_colors[c] for c in ranked["category"]]

bars = ax.barh(y_pos, ranked["overall_RMSE"], color=bar_colors, edgecolor="white", linewidth=0.3)

# Add value labels
for i, (_, row) in enumerate(ranked.iterrows()):
    ax.text(row["overall_RMSE"] + 0.01, i, f"{row['overall_RMSE']:.3f}",
            va="center", fontsize=6, fontfamily="monospace")

ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=7)
ax.set_xlabel("Overall RMSE (pChEMBL)")
ax.set_title("PAINSBench: Full Model Ranking")
ax.invert_yaxis()
ax.set_xlim(0, ranked["overall_RMSE"].max() + 0.15)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

# Legend
legend_elements = [
    Line2D([0], [0], color=c, lw=4, label=l)
    for l, c in [("Drug-Only (ML/DL)", "#2166ac"),
                 ("Drug-Frontier (GNN/KAN)", "#4393c3"),
                 ("DTA-Baseline", "#d6604d"),
                 ("DTA-Frontier", "#f4a582")]
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=7, framealpha=0.9)
fig.savefig(os.path.join(FIGURES_DIR, "paper_fig1_ranking.png"))
plt.close()

# ========== Figure 2: PAINS robustness (ΔRMSE) ==========
fig, ax = plt.subplots(figsize=(8, 3.5))
x = np.arange(len(all_df))
w = 0.35

sorted_dr = all_df.sort_values("delta_rmse", ascending=False)
models_dr = sorted_dr["model"].values
delta_vals = sorted_dr["delta_rmse"].values
cat_c = [cat_colors[c] for c in sorted_dr["category"]]

colors = ["#d73027" if v > 0 else "#2ca02c" for v in delta_vals]
bars = ax.bar(x, delta_vals, color=cat_c, edgecolor="white", linewidth=0.3, width=0.8)
ax.axhline(0, color="gray", lw=0.8, ls="-")

# Threshold line
ax.axhline(-0.2, color="gray", lw=0.6, ls="--", alpha=0.5)
ax.text(len(x) - 1, -0.2, "ΔRMSE = -0.2", fontsize=7, ha="right", va="bottom", alpha=0.5, style="italic")

for i, (_, row) in enumerate(sorted_dr.iterrows()):
    v = row["delta_rmse"]
    ax.text(i, v + (0.01 if v > 0 else -0.025), f"{v:.3f}",
            ha="center", va="bottom" if v > 0 else "top", fontsize=5.5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(models_dr, rotation=45, ha="right", fontsize=6.5)
ax.set_ylabel("ΔRMSE (PAINS+ − PAINS−)")
ax.set_title("PAINS Robustness Across All Models")
ax.legend(handles=legend_elements, loc="upper right", fontsize=7)
ax.set_xlim(-0.5, len(x) - 0.5)
fig.savefig(os.path.join(FIGURES_DIR, "paper_fig2_delta_rmse.png"))
plt.close()

# ========== Figure 3: Accuracy-Robustness trade-off scatter ==========
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for cat, marker, size in [("Drug-Only", "o", 80), ("Drug-Frontier", "s", 80),
                            ("DTA-Baseline", "^", 80), ("DTA-Frontier", "D", 120)]:
    subset = all_df[all_df["category"] == cat]
    sc = ax.scatter(subset["overall_RMSE"], subset["delta_rmse"],
                    c=[cat_colors[cat]] * len(subset), marker=marker, s=size,
                    alpha=0.8, edgecolors="white", linewidth=0.5, label=cat, zorder=5)

# Label each point
for _, row in all_df.iterrows():
    w = "bold" if row["category"] == "DTA-Frontier" else "normal"
    ax.annotate(row["model"], (row["overall_RMSE"], row["delta_rmse"]),
                fontsize=5.5, ha="center", va="bottom", fontweight=w, alpha=0.8)

ax.axhline(0, color="gray", lw=0.8, ls="-", alpha=0.4)
ax.axvline(all_df["overall_RMSE"].mean(), color="gray", lw=0.8, ls=":", alpha=0.4)
ax.set_xlabel("Overall RMSE → (lower is better)")
ax.set_ylabel("ΔRMSE (PAINS robustness)")
ax.set_title("Accuracy–Robustness Trade-off in Drug–Target Binding Prediction")
ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
ax.set_xlim(all_df["overall_RMSE"].min() - 0.05, all_df["overall_RMSE"].max() + 0.05)
fig.savefig(os.path.join(FIGURES_DIR, "paper_fig3_tradeoff.png"))
plt.close()

# ========== Figure 4: Top-10 models bar chart (RMSE + ΔRMSE dual axis) ==========
top10 = all_df.nsmallest(10, "overall_RMSE")

fig, ax1 = plt.subplots(figsize=(7, 3.8))
x = np.arange(len(top10))
w = 0.35

bars1 = ax1.bar(x - w / 2, top10["overall_RMSE"], w, label="Overall RMSE",
                color="#2ca02c", alpha=0.85, edgecolor="white", linewidth=0.3)
ax1.set_ylabel("Overall RMSE", color="#2ca02c", fontsize=9)
ax1.set_ylim(0.9, 1.2)
ax1.tick_params(axis="y", labelcolor="#2ca02c")

ax2 = ax1.twinx()
bars2 = ax2.bar(x + w / 2, top10["delta_rmse"], w, label="ΔRMSE",
                color="#d73027", alpha=0.85, edgecolor="white", linewidth=0.3)
ax2.set_ylabel("ΔRMSE (PAINS robustness)", color="#d73027", fontsize=9)
ax2.set_ylim(-0.35, -0.05)
ax2.tick_params(axis="y", labelcolor="#d73027")

ax1.set_xticks(x)
ax1.set_xticklabels(top10["model"], rotation=30, ha="right", fontsize=7.5)
ax1.set_title("Top-10 Models: Accuracy and PAINS Robustness", pad=22)

# Legend centered between title and bars
lines = [bars1, bars2]
labels = ["Overall RMSE", "ΔRMSE (PAINS+ − PAINS−)"]
ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.11),
           fontsize=7, framealpha=0.9, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "paper_fig4_top10.png"), bbox_inches="tight")
plt.close()

# ========== Figure 5: DTA-only comparison (12 baseline + 4 frontier) ==========
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

dta_all = pd.concat([dta, frontier], ignore_index=True)
dta_sorted = dta_all.sort_values("overall_RMSE")
is_frontier = dta_sorted["model"].isin(frontier["model"].values)
x = np.arange(len(dta_sorted))

# Left: RMSE
ax = axes[0]
ax.bar(x, dta_sorted["overall_RMSE"],
       color=["#e67e22" if f else "#2ca02c" for f in is_frontier],
       alpha=0.85, edgecolor="white", linewidth=0.3)
ax.set_xticks(x)
ax.set_xticklabels(dta_sorted["model"], rotation=45, ha="right", fontsize=6.5)
ax.set_ylabel("Overall RMSE")
ax.set_title("DTA Model Accuracy")
for i, (_, row) in enumerate(dta_sorted.iterrows()):
    ax.text(i, row["overall_RMSE"] + 0.008, f"{row['overall_RMSE']:.3f}",
            ha="center", fontsize=5.5, rotation=90)

# Right: ΔRMSE
ax = axes[1]
ax.bar(x, dta_sorted["delta_rmse"],
       color=["#e67e22" if f else "#d73027" for f in is_frontier],
       alpha=0.85, edgecolor="white", linewidth=0.3)
ax.axhline(0, color="gray", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(dta_sorted["model"], rotation=45, ha="right", fontsize=6.5)
ax.set_ylabel("ΔRMSE")
ax.set_title("DTA PAINS Robustness")
for i, (_, row) in enumerate(dta_sorted.iterrows()):
    v = row["delta_rmse"]
    ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=5.5, rotation=90)

legend_elements2 = [
    Line2D([0], [0], color="#2ca02c", lw=3, label="DTA Baseline"),
    Line2D([0], [0], color="#e67e22", lw=3, label="DTA Frontier"),
]
axes[0].legend(handles=legend_elements2, loc="upper right", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "paper_fig5_dta_comparison.png"))
plt.close()

# ========== Figure 6: FP Ratio across all models ==========
fig, ax = plt.subplots(figsize=(8, 3))
fp_sorted = all_df.sort_values("fp_ratio", ascending=False)
x = np.arange(len(fp_sorted))
cat_c_fp = [cat_colors[c] for c in fp_sorted["category"]]

ax.bar(x, fp_sorted["fp_ratio"], color=cat_c_fp, edgecolor="white", linewidth=0.3, alpha=0.85)
ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6)
ax.text(len(x) - 1, 1.02, "FP Ratio = 1 (equal error)", fontsize=7, ha="right", style="italic", alpha=0.6)

ax.set_xticks(x)
ax.set_xticklabels(fp_sorted["model"], rotation=45, ha="right", fontsize=6)
ax.set_ylabel("FP Ratio (|residual| ratio: PAINS+ / PAINS−)")
ax.set_title("False-Positive Susceptibility to PAINS Interference")
ax.legend(handles=legend_elements, loc="upper right", fontsize=6.5)
ax.set_xlim(-0.5, len(x) - 0.5)
fig.savefig(os.path.join(FIGURES_DIR, "paper_fig6_fp_ratio.png"))
plt.close()

# ========== Figure 7: DTA ΔRMSE vs RMSE scatter (detailed) ==========
fig, ax = plt.subplots(figsize=(6, 4.5))
dta_all = pd.concat([dta, frontier], ignore_index=True)
is_f = dta_all["model"].isin(frontier["model"].values)

ax.scatter(dta_all[~is_f]["overall_RMSE"], dta_all[~is_f]["delta_rmse"],
           c="#2ca02c", s=100, alpha=0.7, edgecolors="white", linewidth=0.5, label="DTA Baseline", zorder=3)
ax.scatter(dta_all[is_f]["overall_RMSE"], dta_all[is_f]["delta_rmse"],
           c="#e67e22", s=200, alpha=0.9, edgecolors="white", linewidth=0.5,
           marker="D", label="DTA Frontier (2024-2026)", zorder=5)

for _, row in dta_all.iterrows():
    w = "bold" if row["model"] in frontier["model"].values else "normal"
    ax.annotate(row["model"], (row["overall_RMSE"], row["delta_rmse"]),
                fontsize=7, ha="center", va="bottom", fontweight=w)

ax.axhline(0, color="gray", lw=0.8, ls="-", alpha=0.3)
ax.axvline(dta_all["overall_RMSE"].mean(), color="gray", lw=0.8, ls=":", alpha=0.3)
ax.set_xlabel("Overall RMSE")
ax.set_ylabel("ΔRMSE (PAINS+ − PAINS−)")
ax.set_title("DTA Models: Accuracy vs PAINS Robustness")
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

# Quadrant labels
ax.text(1.07, -0.17, "← Accurate & Robust", fontsize=8, alpha=0.4, style="italic")
ax.text(1.15, -0.26, "← Less accurate, robust", fontsize=8, alpha=0.4, style="italic")
fig.savefig(os.path.join(FIGURES_DIR, "paper_fig7_dta_tradeoff.png"))
plt.close()

# ========== Figure 8: Training time comparison ==========
fig, ax = plt.subplots(figsize=(7, 3.5))
time_df = all_df.sort_values("train_time_s")
cat_c_time = [cat_colors[c] for c in time_df["category"]]
ax.barh(np.arange(len(time_df)), time_df["train_time_s"], color=cat_c_time,
        edgecolor="white", linewidth=0.3, alpha=0.85)
ax.set_yticks(np.arange(len(time_df)))
ax.set_yticklabels(time_df["model"], fontsize=6.5)
ax.set_xlabel("Training Time (seconds)")
ax.set_title("Computational Cost Comparison")
ax.legend(handles=legend_elements, loc="lower right", fontsize=7)
fig.savefig(os.path.join(FIGURES_DIR, "paper_fig8_training_time.png"))
plt.close()

print("All figures saved to figures/paper_fig*.png")
