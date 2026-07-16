"""
Step 3: Generate molecular features (Morgan fingerprints + properties).
Output: processed/features.npz + processed/benchmark_with_targets.csv
"""
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR
from src.data_utils import load_compound_properties
from src.features import extract_properties


def main():
    print("=" * 60)
    print("Step 3: Generating molecular features")
    print("=" * 60)

    benchmark = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_dataset.csv"))
    print(f"Loaded benchmark: {len(benchmark):,} rows")

    # ---- add extra RDKit properties that are not already in benchmark ----
    # (benchmark already has canonical_smiles + basic props from step 2)
    before = len(benchmark)
    benchmark = benchmark.dropna(subset=["canonical_smiles"])
    print(f"Dropped {before - len(benchmark):,} rows without SMILES")

    # ---- extract Morgan fingerprints ----
    print("\nComputing Morgan fingerprints (this may take a while)...")
    from src.features import morgan_from_df
    fp_matrix = morgan_from_df(benchmark)
    print(f"Fingerprint matrix: {fp_matrix.shape}")

    # ---- extract physicochemical properties ----
    print("Extracting physicochemical properties...")
    prop_matrix = extract_properties(benchmark)
    print(f"Property matrix: {prop_matrix.shape}")

    # ---- feature columns for ML models (FP + properties) ----
    X = np.concatenate([fp_matrix, prop_matrix], axis=1)
    print(f"Combined feature matrix: {X.shape}")

    # ---- save features ----
    np.savez_compressed(
        os.path.join(PROCESSED_DIR, "features.npz"),
        X=X.astype(np.float32),
        y=benchmark["pchembl_value"].values.astype(np.float32),
        pains_status=benchmark["PAINS_status"].values.astype(np.int8),
        molregnos=benchmark["molregno"].values,
        target_ids=benchmark["target_chembl_id"].values,
    )
    print(f"\nSaved: {os.path.join(PROCESSED_DIR, 'features.npz')}")

    # ---- also save the benchmark with target info ----
    keep_cols = ["molregno", "target_chembl_id", "pchembl_value", "PAINS_status",
                 "standard_type", "organism"]
    benchmark[keep_cols].to_csv(
        os.path.join(PROCESSED_DIR, "benchmark_labels.csv"), index=False
    )
    print(f"Saved: {os.path.join(PROCESSED_DIR, 'benchmark_labels.csv')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
