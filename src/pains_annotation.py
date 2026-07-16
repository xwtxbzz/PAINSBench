"""PAINS annotation logic."""

import pandas as pd
import numpy as np

from src.data_utils import load_structural_alerts, load_compound_alerts

PAINS_SET_IDS = [4]                    # PAINS (Baell & Holloway)
BROAD_SET_IDS = [1, 2, 3, 4, 6]       # Glaxo, Dundee, BMS, PAINS, MLSMR


def get_pains_molregnos(use_broad=False):
    """Return set[int] of molregnos flagged by structural alerts."""
    alerts_df, _ = load_structural_alerts()
    compound_alerts = load_compound_alerts()

    merged = compound_alerts.merge(alerts_df[["alert_id", "alert_set_id"]], on="alert_id")
    target = BROAD_SET_IDS if use_broad else PAINS_SET_IDS
    flagged = merged[merged["alert_set_id"].isin(target)]
    return set(flagged["molregno"].unique())


def annotate_pains_status(df, molregno_col="molregno", use_broad=False):
    """Add PAINS_status column (1 = PAINS+)."""
    pains_set = get_pains_molregnos(use_broad=use_broad)
    df = df.copy()
    df["PAINS_status"] = df[molregno_col].isin(pains_set).astype(np.int8)
    return df


def pains_stats():
    """Print PAINS coverage statistics."""
    narrow = get_pains_molregnos(use_broad=False)
    broad = get_pains_molregnos(use_broad=True)

    alerts_df, sets_df = load_structural_alerts()
    compound_alerts = load_compound_alerts()
    merged = compound_alerts.merge(alerts_df[["alert_id", "alert_set_id"]], on="alert_id")

    print(f"PAINS-set flagged:  {len(narrow):>8,} compounds")
    print(f"Broad-set flagged:  {len(broad):>8,} compounds")
    print()

    brk = merged.groupby("alert_set_id").agg(n=("molregno", "nunique")).reset_index()
    brk = brk.merge(sets_df, on="alert_set_id").sort_values("n", ascending=False)
    print("Per alert-set coverage:")
    for _, r in brk.iterrows():
        print(f"  {r['set_name']:12s} (id={r['alert_set_id']}): {r['n']:>8,}")
