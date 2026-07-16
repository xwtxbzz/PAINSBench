"""
Step 1: Load & filter activity data, annotate PAINS status.
Output: processed/activity_filtered.csv
"""
import os, sys, gc
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, TARGET_SIZE
from src.data_utils import load_filtered_activity
from src.pains_annotation import annotate_pains_status, pains_stats


def main():
    # ---- load + filter activity data ----
    print("=" * 60)
    print("Step 1: Loading & filtering activity data")
    print("=" * 60)

    # Load all joined activity (IC50/Ki/Kd with valid pChEMBL)
    # This gives roughly 800K-1M rows from 4.5M total
    df = load_filtered_activity()
    print(f"\nFiltered activity records: {len(df):,}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # ---- annotate PAINS ----
    print("\nAnnotating PAINS status...")
    df = annotate_pains_status(df)
    pains_count = df["PAINS_status"].sum()
    print(f"  PAINS+: {pains_count:,} ({pains_count / len(df) * 100:.1f}%)")
    print(f"  PAINS-: {len(df) - pains_count:,} ({(1 - pains_count / len(df)) * 100:.1f}%)")

    # ---- downsample to keep ALL PAINS+, sample PAINS- for ~200K target ----
    if len(df) > 200_000:
        pains_pos = df[df["PAINS_status"] == 1]
        pains_neg = df[df["PAINS_status"] == 0]
        # Keep all PAINS+ (they are the minority class)
        # Sample PAINS- to get ~200K total after dedup/target filtering
        n_neg_target = max(200_000, TARGET_SIZE - len(pains_pos))
        n_neg = min(len(pains_neg), n_neg_target)
        df = pd.concat([
            pains_pos,
            pains_neg.sample(n=n_neg, random_state=42),
        ], ignore_index=True)
        print(f"\nDownsampled to {len(df):,} (pos={len(pains_pos):,}, neg={n_neg:,})")

    # ---- remove duplicates (same compound-target, keep max pChEMBL) ----
    before = len(df)
    df = df.sort_values("pchembl_value", ascending=False).drop_duplicates(
        subset=["molregno", "target_chembl_id"], keep="first"
    )
    print(f"Removed {before - len(df):,} duplicate compound-target pairs")

    # ---- save ----
    out_path = os.path.join(PROCESSED_DIR, "activity_filtered.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df):,} rows)")

    # ---- print stats ----
    pains_stats()
    return df


if __name__ == "__main__":
    main()
