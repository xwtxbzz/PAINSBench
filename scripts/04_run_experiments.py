"""
Step 4: Run baseline models + PAINS-aware evaluation.
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR, TEST_SPLIT, RANDOM_SEED
from src.models import MODEL_REGISTRY
from src.evaluation import evaluate_pains_aware
from src.visualization import (
    plot_pains_comparison, plot_delta_rmse, plot_fp_ratio,
    plot_scatter, plot_residual_distribution,
)

np.random.seed(RANDOM_SEED)

# Speed / debug settings
SUBSAMPLE = None       # set to int for faster debugging; None = full data
FORCE_CPU = False      # set True to avoid GPU OOM


def main():
    sys.stdout.flush()
    print("=" * 60, flush=True)
    print("Step 4: Running baseline experiments", flush=True)
    print("=" * 60, flush=True)

    # ---- load features ----
    data = np.load(os.path.join(PROCESSED_DIR, "features_full.npz"), allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    pains_status = data["pains_status"]
    print(f"Loaded: X {X.shape}, y {y.shape}", flush=True)

    # optional subsample for quick testing
    if SUBSAMPLE:
        idx = np.random.RandomState(RANDOM_SEED).choice(len(y), size=SUBSAMPLE, replace=False)
        X, y, pains_status = X[idx], y[idx], pains_status[idx]
        print(f"Subsampled to {SUBSAMPLE} for fast debugging", flush=True)

    # ---- train/test split ----
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, ps_train, ps_test = \
        train_test_split(X, y, pains_status,
                         test_size=TEST_SPLIT, random_state=RANDOM_SEED,
                         stratify=pains_status)
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}", flush=True)

    # ---- run each model ----
    results_rows = []

    for name, train_fn in MODEL_REGISTRY.items():
        print(f"\n{'─' * 50}", flush=True)
        print(f"Training {name}...", flush=True)
        sys.stdout.flush()
        t0 = time.time()

        model = train_fn(X_train, y_train)
        y_pred = model.predict(X_test)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s", flush=True)

        eval_dict = evaluate_pains_aware(y_test, y_pred, ps_test)
        eval_dict["model"] = name
        eval_dict["train_time_s"] = elapsed
        results_rows.append(eval_dict)

        print(f"  Overall RMSE:  {eval_dict['overall_RMSE']:.4f}", flush=True)
        print(f"  PAINS+ RMSE:   {eval_dict['pains_pos_RMSE']:.4f}", flush=True)
        print(f"  PAINS- RMSE:   {eval_dict['pains_neg_RMSE']:.4f}", flush=True)
        print(f"  ΔRMSE:         {eval_dict['delta_rmse']:.4f}", flush=True)
        print(f"  FP Ratio:      {eval_dict['fp_ratio']:.4f}", flush=True)

        # plots
        plot_scatter(y_test, y_pred, ps_test, name)
        res_pos = np.abs(y_test[ps_test == 1] - y_pred[ps_test == 1])
        res_neg = np.abs(y_test[ps_test == 0] - y_pred[ps_test == 0])
        plot_residual_distribution(res_pos, res_neg, name)

    # ---- PAINS-aware variant: XGBoost + PAINS_status as feature ----
    print(f"\n{'─' * 50}", flush=True)
    print("PAINS-aware: XGBoost + PAINS_status as feature", flush=True)
    sys.stdout.flush()
    X_train_aug = np.column_stack([X_train, ps_train])
    X_test_aug = np.column_stack([X_test, ps_test])

    model_aware = MODEL_REGISTRY["XGBoost"](X_train_aug, y_train)
    y_pred_aware = model_aware.predict(X_test_aug)
    eval_aware = evaluate_pains_aware(y_test, y_pred_aware, ps_test)
    eval_aware["model"] = "XGBoost+PAINSflag"
    results_rows.append(eval_aware)
    print(f"  Overall RMSE:  {eval_aware['overall_RMSE']:.4f}", flush=True)
    print(f"  PAINS+ RMSE:   {eval_aware['pains_pos_RMSE']:.4f}", flush=True)
    print(f"  ΔRMSE:         {eval_aware['delta_rmse']:.4f}", flush=True)

    # ---- compile results ----
    results_df = pd.DataFrame(results_rows)
    cols = ["model"] + [c for c in results_df.columns if c != "model"]
    results_df = results_df[cols]

    results_df.to_csv(os.path.join(RESULTS_DIR, "results_summary.csv"), index=False)
    print(f"\n{'=' * 60}", flush=True)
    print("Final Results:", flush=True)
    # rename columns with special chars for safe print
    print_df = results_df.rename(columns={
        "delta_rmse": "Delta_RMSE", "fp_ratio": "FP_Ratio",
        "pains_pos_RMSE": "PAINS+_RMSE", "pains_neg_RMSE": "PAINS-_RMSE",
    })
    # only print key columns
    key_cols = ["model", "overall_RMSE", "PAINS+_RMSE", "PAINS-_RMSE",
                "Delta_RMSE", "FP_Ratio", "train_time_s"]
    print(print_df[key_cols].to_string(index=False), flush=True)

    # figures
    plot_pains_comparison(results_df)
    plot_delta_rmse(results_df)
    plot_fp_ratio(results_df)

    print(f"\nFigures → {os.path.dirname(os.path.abspath(__file__))}/../figures/", flush=True)
    print(f"Results → {RESULTS_DIR}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
