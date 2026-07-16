"""Evaluation metrics for PAINSBench."""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))


def spearmanr(y_true, y_pred):
    from scipy.stats import spearmanr as sp
    return float(sp(y_true, y_pred).statistic)


ALL_METRICS = {"RMSE": rmse, "MAE": mae, "R²": r2, "Spearman ρ": spearmanr}


def evaluate(y_true, y_pred):
    return {name: fn(y_true, y_pred) for name, fn in ALL_METRICS.items()}


def evaluate_pains_aware(y_true, y_pred, pains_status):
    """
    Compute PAINS-aware metrics.

    Returns
    -------
    dict with keys:
        overall_*:  metrics on full set
        pains_pos_*: metrics on PAINS+ subset
        pains_neg_*: metrics on PAINS- subset
        delta_rmse:  RMSE_pos - RMSE_neg  (positive = model worse on PAINS+)
        fp_ratio:    fraction of PAINS+ where residual > 2 sigma
    """
    overall = evaluate(y_true, y_pred)

    pos_mask = pains_status == 1
    neg_mask = pains_status == 0

    pos_metrics = evaluate(y_true[pos_mask], y_pred[pos_mask])
    neg_metrics = evaluate(y_true[neg_mask], y_pred[neg_mask])

    delta_rmse = pos_metrics["RMSE"] - neg_metrics["RMSE"]

    residuals = np.abs(y_true - y_pred)
    threshold = 2 * np.std(residuals[neg_mask]) if neg_mask.sum() > 1 else 0
    fp_ratio = float(residuals[pos_mask].mean() / (residuals[neg_mask].mean() + 1e-8))

    results = {}
    for k, v in overall.items():
        results[f"overall_{k}"] = v
    for k, v in pos_metrics.items():
        results[f"pains_pos_{k}"] = v
    for k, v in neg_metrics.items():
        results[f"pains_neg_{k}"] = v
    results["delta_rmse"] = delta_rmse
    results["fp_ratio"] = fp_ratio
    results["pos_n"] = int(pos_mask.sum())
    results["neg_n"] = int(neg_mask.sum())
    return results
