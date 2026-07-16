"""
Step 8: Filter to SINGLE PROTEIN targets, add protein sequences.
Output: Single-protein benchmark with sequences for DTA tasks.
"""
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR

# Load benchmark
print("Loading benchmark...")
bench = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_dataset.csv"))
print(f"  Full benchmark: {len(bench):,} rows")

# Filter to SINGLE PROTEIN
sp = bench[bench["target_type"] == "SINGLE PROTEIN"].copy()
print(f"  SINGLE PROTEIN: {len(sp):,} rows ({len(sp)/len(bench)*100:.1f}%)")
print(f"  PAINS+: {sp['PAINS_status'].sum():,}  PAINS-: {(sp['PAINS_status']==0).sum():,}")
print(f"  Unique targets: {sp['target_chembl_id'].nunique()}")

# Merge sequences from ChEMBL
td = pd.read_csv(os.path.join(os.path.dirname(PROCESSED_DIR), "..", "csv_output", "tables", "target_dictionary.csv"))
tc = pd.read_csv(os.path.join(os.path.dirname(PROCESSED_DIR), "..", "csv_output", "tables", "target_components.csv"))
cs = pd.read_csv(os.path.join(os.path.dirname(PROCESSED_DIR), "..", "csv_output", "tables", "component_sequences.csv"))

sp = sp.merge(td[["chembl_id", "tid"]], left_on="target_chembl_id", right_on="chembl_id", how="left")
sp = sp.merge(tc[["tid", "component_id"]], on="tid", how="left")
sp = sp.merge(cs[["component_id", "sequence"]], on="component_id", how="left")

before = len(sp)
sp = sp.dropna(subset=["sequence"])
print(f"  Dropped {before - len(sp)} without sequence → {len(sp):,} remaining")

# Key stats
seq_lens = sp["sequence"].str.len()
print(f"  Sequence length: min={seq_lens.min()}, median={seq_lens.median():.0f}, max={seq_lens.max()}")

# Save
out_path = os.path.join(PROCESSED_DIR, "benchmark_dta.csv")
sp.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")
print(f"  Columns: {list(sp.columns)}")
print("Done.")
