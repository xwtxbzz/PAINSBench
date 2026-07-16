"""
Step 2: Build stratified PAINS+/PAINS- benchmark dataset.
Output: processed/benchmark_dataset.csv
"""
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RANDOM_SEED, MIN_MEASUREMENTS_PER_TARGET, STRATIFIED_SAMPLE_SIZE

np.random.seed(RANDOM_SEED)


def main():
    print("=" * 60)
    print("Step 2: Building stratified PAINS+ / PAINS- benchmark")
    print("=" * 60)

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "activity_filtered.csv"))
    print(f"Loaded {len(df):,} records")

    # ---- filter targets with too few measurements ----
    target_counts = df.groupby("target_chembl_id")["molregno"].count()
    valid_targets = target_counts[target_counts >= MIN_MEASUREMENTS_PER_TARGET].index
    df = df[df["target_chembl_id"].isin(valid_targets)].copy()
    print(f"After per-target filter (≥{MIN_MEASUREMENTS_PER_TARGET}): "
          f"{len(df):,} records, {len(valid_targets):,} targets")

    # ---- Merge with compound properties ----
    props_df = pd.read_csv(os.path.join(PROCESSED_DIR, "compound_properties.csv"))
    merged = df.merge(props_df[["molregno", "mw_freebase", "alogp", "hba", "hbd",
                                "psa", "rtb", "qed_weighted", "canonical_smiles"]],
                      on="molregno", how="left")
    before = len(merged)
    merged = merged.dropna(subset=["mw_freebase", "alogp", "hba", "hbd", "canonical_smiles"])
    print(f"Dropped {before - len(merged):,} rows with missing properties/SMILES")

    # ---- Stratified sampling by PAINS status ----
    pains_pos = merged[merged["PAINS_status"] == 1]
    pains_neg = merged[merged["PAINS_status"] == 0]
    print(f"\nTotal available — PAINS+: {len(pains_pos):,}  PAINS-: {len(pains_neg):,}")

    # Include ALL PAINS+ compounds (they're the minority)
    n_pos = min(len(pains_pos), STRATIFIED_SAMPLE_SIZE // 2)
    n_neg = min(len(pains_neg), STRATIFIED_SAMPLE_SIZE - n_pos)

    benchmark = pd.concat([
        pains_pos.sample(n=n_pos, random_state=RANDOM_SEED),
        pains_neg.sample(n=n_neg, random_state=RANDOM_SEED),
    ], ignore_index=True)
    print(f"\nBenchmark dataset: {len(benchmark):,} rows")
    print(f"  PAINS+: {(benchmark['PAINS_status'] == 1).sum():,} "
          f"({(benchmark['PAINS_status'] == 1).mean() * 100:.1f}%)")
    print(f"  PAINS-: {(benchmark['PAINS_status'] == 0).sum():,} "
          f"({(benchmark['PAINS_status'] == 0).mean() * 100:.1f}%)")

    # ---- Property balance check ----
    feature_cols = ["mw_freebase", "alogp", "hba", "hbd", "psa", "rtb"]
    print("\nProperty balance (mean ± std):")
    for col in feature_cols:
        pos_mean = benchmark[benchmark["PAINS_status"] == 1][col].mean()
        neg_mean = benchmark[benchmark["PAINS_status"] == 0][col].mean()
        std_diff = (pos_mean - neg_mean) / benchmark[col].std()
        print(f"  {col:20s}  PAINS+: {pos_mean:8.2f}  PAINS-: {neg_mean:8.2f}  StdDiff: {std_diff:.3f}")

    # ---- Drop duplicate compound-target pairs (keep highest pChEMBL) ----
    before = len(benchmark)
    benchmark = benchmark.sort_values("pchembl_value", ascending=False).drop_duplicates(
        subset=["molregno", "target_chembl_id"], keep="first"
    )
    print(f"\nDeduplicated: {before} → {len(benchmark)}")

    benchmark.to_csv(os.path.join(PROCESSED_DIR, "benchmark_dataset.csv"), index=False)
    print(f"\nSaved: {os.path.join(PROCESSED_DIR, 'benchmark_dataset.csv')} ({len(benchmark):,} rows)")


if __name__ == "__main__":
    main()
