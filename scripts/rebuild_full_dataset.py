"""
Rebuild PAINSBench dataset from raw ChEMBL 36 CSV files.
No data caps — keeps everything after quality filters.
Outputs: benchmark_full.csv + features.npz + benchmark_dta_ki/ic50.csv (with sequences)
"""
import os, sys, time, gc, glob, warnings
import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, PATHS

warnings.filterwarnings("ignore")
t_start = time.time()
CHEMBL_DIR = r"D:\ChemBL\csv_output"

# ==================== 1. Load & Filter Activity Data ====================
print("=" * 60)
print("1. Loading & filtering activity data...")
print("=" * 60)

joined_files = sorted(glob.glob(os.path.join(CHEMBL_DIR, "joined", "activity_target_joined_part*.csv")))
cols = ["molregno", "standard_type", "standard_value", "standard_units",
        "pchembl_value", "standard_flag", "target_chembl_id", "target_name",
        "target_type", "organism"]

records = []
total_raw = 0
for f in joined_files:
    for chunk in pd.read_csv(f, usecols=cols, chunksize=200_000):
        ok = (
            chunk["standard_type"].isin(["IC50", "Ki", "Kd"])
            & chunk["standard_units"].str.upper().str.contains("NM", na=False)
            & chunk["pchembl_value"].notna()
            & chunk["pchembl_value"].between(2, 11)
            & (chunk["standard_flag"] == 1)
        )
        records.append(chunk[ok])
        total_raw += len(chunk[ok])
    print(f"  Loaded {os.path.basename(f)}, {total_raw:,} filtered so far")

df = pd.concat(records, ignore_index=True)
df["pchembl_value"] = df["pchembl_value"].astype(np.float32)
del records
print(f"\nTotal filtered: {len(df):,}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
t1 = time.time()
print(f"  Time: {t1-t_start:.1f}s")

# ==================== 2. PAINS Annotation ====================
print("\n" + "=" * 60)
print("2. Annotating PAINS status...")
print("=" * 60)

# Load structural alerts + alert sets
alerts_df = pd.read_csv(PATHS["structural_alerts"])
sets_df = pd.read_csv(PATHS["alert_sets"])
print(f"  Structural alerts: {len(alerts_df):,}")
print(f"  Alert set mapping: alert_set_id 4 = PAINS (narrow)")

# Load compound-alert mappings
alert_files = PATHS["compound_alerts"]
ca_records = []
for f in alert_files:
    ca = pd.read_csv(f)
    ca_records.append(ca)
ca_all = pd.concat(ca_records, ignore_index=True)
print(f"  Compound-alert mappings: {len(ca_all):,}")

# Merge to get alert_set_id per compound
ca_merged = ca_all.merge(alerts_df[["alert_id", "alert_set_id"]], on="alert_id")
# PAINS+ = at least one alert in set_id=4
pains_compounds = set(ca_merged[ca_merged["alert_set_id"] == 4]["molregno"].unique())
print(f"  PAINS+ compounds: {len(pains_compounds):,}")

df["PAINS_status"] = df["molregno"].isin(pains_compounds).astype(np.int8)
pains_count = df["PAINS_status"].sum()
print(f"  PAINS+: {pains_count:,} ({pains_count/len(df)*100:.1f}%)")
print(f"  PAINS-: {len(df)-pains_count:,} ({(1-pains_count/len(df))*100:.1f}%)")
del ca_all, ca_merged
gc.collect()
t2 = time.time()
print(f"  Time: {t2-t1:.1f}s")

# ==================== 3. Deduplicate ====================
print("\n" + "=" * 60)
print("3. Deduplicating compound-target pairs...")
print("=" * 60)
before = len(df)
df = df.sort_values("pchembl_value", ascending=False).drop_duplicates(
    subset=["molregno", "target_chembl_id"], keep="first")
print(f"  Before: {before:,} → After: {len(df):,} (removed {before-len(df):,})")

# ==================== 4. Per-target ≥30 Filter ====================
print("\n" + "=" * 60)
print("4. Per-target ≥30 measurements filter...")
print("=" * 60)
target_counts = df["target_chembl_id"].value_counts()
valid_targets = target_counts[target_counts >= 30].index
before = len(df)
df = df[df["target_chembl_id"].isin(valid_targets)]
print(f"  Targets ≥30: {len(valid_targets):,} / {len(target_counts):,}")
print(f"  Entries: {before:,} → {len(df):,}")
print(f"  Unique targets: {df['target_chembl_id'].nunique():,}")
print(f"  Unique compounds: {df['molregno'].nunique():,}")

t3 = time.time()
print(f"  Time: {t3-t2:.1f}s")

# ==================== 5. Merge Compound Properties ====================
print("\n" + "=" * 60)
print("5. Merging compound properties...")
print("=" * 60)

prop_files = PATHS["compound_properties"]
prop_records = []
for f in prop_files:
    p = pd.read_csv(f)
    prop_records.append(p)
props = pd.concat(prop_records, ignore_index=True)
print(f"  Properties table: {len(props):,} compounds")

# Columns needed for molecular properties + DTA
prop_cols = ["molregno", "canonical_smiles", "mw_freebase", "alogp", "hba",
             "hbd", "psa", "rtb", "qed_weighted", "aromatic_rings", "heavy_atoms",
             "chembl_id", "full_mwt"]

props = props[prop_cols].drop_duplicates(subset=["molregno"])
print(f"  Unique compounds with properties: {len(props):,}")

before = len(df)
df = df.merge(props, on="molregno", how="left")
df = df.dropna(subset=["canonical_smiles"])
print(f"  After merge + dropna(smiles): {before:,} → {len(df):,}")

t4 = time.time()
print(f"  Time: {t4-t3:.1f}s")

# ==================== 6. Load & Parse Fingerprints ====================
print("\n" + "=" * 60)
print("6. Loading & parsing molecular fingerprints...")
print("=" * 60)

fp_path = PATHS["fingerprints"]
print(f"  Loading pre-computed fingerprints from {fp_path}")

# Map compound chembl_id → molregno
chembl_to_molregno = dict(zip(props["chembl_id"], props["molregno"]))
molregno_to_chembl = dict(zip(props["molregno"], props["chembl_id"]))

# Get unique compounds that need fingerprints
unique_molregnos = df["molregno"].unique()
print(f"  Unique compounds needing fingerprints: {len(unique_molregnos):,}")

# Load fingerprints in chunks, parse hex → bit vector
def hex_to_bits(hex_str, bits=2048):
    """Convert hex string to 2048-bit numpy array (fast vectorized)."""
    hex_str = str(hex_str).strip()
    # Pad to correct length if leading zeros were trimmed
    needed_chars = bits // 4
    if len(hex_str) < needed_chars:
        hex_str = hex_str.zfill(needed_chars)
    byte_arr = bytes.fromhex(hex_str[:needed_chars])
    return np.unpackbits(np.frombuffer(byte_arr, dtype=np.uint8)).astype(np.uint8)[:bits]

# Build fingerprint matrix for our compounds
molregno_to_fp = {}
chunk_size = 50000
reader = pd.read_csv(fp_path, chunksize=chunk_size)
processed = 0
for chunk in reader:
    for _, row in chunk.iterrows():
        cid = row["chembl_id"]
        if cid in chembl_to_molregno:
            mr = chembl_to_molregno[cid]
            molregno_to_fp[mr] = hex_to_bits(row["morgan_fingerprint_2048"])
    processed += len(chunk)
    if processed % 200000 == 0:
        print(f"    Processed {processed:,} fingerprint rows...")

print(f"  Fingerprints loaded: {len(molregno_to_fp):,} / {len(unique_molregnos):,}")

# Build feature matrix aligned with df rows (vectorized merge — NO iterrows)
print("  Building feature matrix (vectorized)...")
unique_mr_list = list(molregno_to_fp.keys())
fp_matrix = np.array([molregno_to_fp[mr] for mr in unique_mr_list], dtype=np.uint8)

# Use pandas merge for fast C-level alignment
fp_df = pd.DataFrame({"molregno": unique_mr_list, "_fp_idx": np.arange(len(unique_mr_list))})
df = df.merge(fp_df, on="molregno", how="inner")

X = fp_matrix[df["_fp_idx"].values]
y = df["pchembl_value"].values.astype(np.float32)
ps = df["PAINS_status"].values

missing_fp = len(unique_molregnos) - len(unique_mr_list)
if missing_fp > 0:
    print(f"  Dropped entries for {missing_fp:,} compounds without fingerprints")

print(f"  Feature matrix: {X.shape}")
print(f"  Labels: {len(y):,}")
print(f"  PAINS+: {ps.sum():,} / PAINS-: {(ps==0).sum():,}")

t5 = time.time()
print(f"  Time: {t5-t4:.1f}s")

# ==================== 7. Stratified Sampling → 30% PAINS+ ====================
print("\n" + "=" * 60)
print("7. Stratified sampling — target 30% PAINS+...")
print("=" * 60)

# Keep all PAINS+, downsample PAINS- to achieve target ratio
pains_pos_mask = ps == 1
pains_neg_mask = ~pains_pos_mask

n_pos = pains_pos_mask.sum()
n_total_target = int(n_pos / 0.3)
n_neg_target = n_total_target - n_pos
print(f"  PAINS+: {n_pos:,} (keep all)")
print(f"  PAINS-: {pains_neg_mask.sum():,} → sample {n_neg_target:,}")
print(f"  Target total: {n_total_target:,} ({n_pos/n_total_target*100:.1f}% PAINS+)")

# Sample indices
neg_indices = np.where(pains_neg_mask)[0]
sampled_neg = np.random.RandomState(42).choice(neg_indices, size=min(n_neg_target, len(neg_indices)), replace=False)
keep_mask = np.zeros(len(df), dtype=bool)
keep_mask[pains_pos_mask] = True
keep_mask[sampled_neg] = True

# Apply sampling
df = df.iloc[keep_mask].reset_index(drop=True)
X = X[keep_mask]
y = y[keep_mask]
ps = ps[keep_mask]

print(f"  Result: {len(df):,} total")
print(f"  PAINS+: {ps.sum():,} ({ps.mean()*100:.1f}%)")
print(f"  PAINS-: {(ps==0).sum():,}")

t_samp = time.time()
print(f"  Time: {t_samp-t5:.1f}s")

# ==================== 8. Save Full Benchmark ====================
print("\n" + "=" * 60)
print("8. Saving full benchmark...")
print("=" * 60)

csv_out = os.path.join(PROCESSED_DIR, "benchmark_full.csv")
df.to_csv(csv_out, index=False)
print(f"  Saved: {csv_out} ({len(df):,} rows)")

npz_out = os.path.join(PROCESSED_DIR, "features_full.npz")
np.savez_compressed(npz_out, X=X, y=y, pains_status=ps)
print(f"  Saved: {npz_out} ({X.shape})")

# ==================== 8. Create DTA Subsets ====================
print("\n" + "=" * 60)
print("8. Creating DTA subsets (with protein sequences)...")
print("=" * 60)

# Focus on SINGLE PROTEIN targets
sp = df[df["target_type"] == "SINGLE PROTEIN"].copy()
print(f"  SINGLE PROTEIN rows: {len(sp):,}")

# Load protein sequences
td = pd.read_csv(os.path.join(CHEMBL_DIR, "tables", "target_dictionary.csv"))
tc = pd.read_csv(os.path.join(CHEMBL_DIR, "tables", "target_components.csv"))
cs = pd.read_csv(os.path.join(CHEMBL_DIR, "tables", "component_sequences.csv"))

sp = sp.merge(td[["chembl_id", "tid"]], left_on="target_chembl_id", right_on="chembl_id", how="left")
sp = sp.merge(tc[["tid", "component_id"]], on="tid", how="left")
sp = sp.merge(cs[["component_id", "sequence"]], on="component_id", how="left")
before = len(sp)
sp = sp.dropna(subset=["sequence"])
print(f"  With sequences: {len(sp):,} (dropped {before-len(sp):,} without)")

# Save full DTA
dta_out = os.path.join(PROCESSED_DIR, "benchmark_dta_full.csv")
sp.to_csv(dta_out, index=False)
print(f"  Saved: {dta_out} ({len(sp):,} rows)")

# Save Ki and IC50 subsets
for t in ["Ki", "IC50"]:
    subset = sp[sp["standard_type"] == t].copy()
    fname = f"benchmark_dta_{t.lower()}.csv"
    subset.to_csv(os.path.join(PROCESSED_DIR, fname), index=False)
    print(f"  {t}: {len(subset):,} rows → {fname}")
    print(f"    PAINS+: {subset['PAINS_status'].sum():,} / PAINS-: {(subset['PAINS_status']==0).sum():,}")

# Save molecular property prediction subsets (no sequences)
for t in ["Ki", "IC50"]:
    subset = sp[sp["standard_type"] == t].copy()
    # Drop sequence-related columns for molecular task
    cols = [c for c in subset.columns if c not in ["tid", "component_id", "sequence", "chembl_id_y"]]
    subset = subset[cols]
    fname = f"benchmark_sp_{t.lower()}.csv"
    subset.to_csv(os.path.join(PROCESSED_DIR, fname), index=False)
    print(f"  SP-{t}: {len(subset):,} rows → {fname}")

t6 = time.time()
print(f"  Time: {t6-t5:.1f}s")

# ==================== Summary ====================
print("\n" + "=" * 60)
print("BUILD COMPLETE")
print("=" * 60)
print(f"Total time: {t6-t_start:.1f}s")
print()

# Print final dataset stats
print("Final Datasets:")
print(f"  benchmark_full.csv:      {len(pd.read_csv(csv_out)):>8,} rows  (full unrestricted)")
print(f"  benchmark_dta_full.csv:  {len(pd.read_csv(dta_out)):>8,} rows  (single-protein with seq)")
print(f"  benchmark_dta_ki.csv:    {len(pd.read_csv(os.path.join(PROCESSED_DIR,'benchmark_dta_ki.csv'))):>8,} rows")
print(f"  benchmark_dta_ic50.csv:  {len(pd.read_csv(os.path.join(PROCESSED_DIR,'benchmark_dta_ic50.csv'))):>8,} rows")
print(f"  benchmark_sp_ki.csv:     {len(pd.read_csv(os.path.join(PROCESSED_DIR,'benchmark_sp_ki.csv'))):>8,} rows")
print(f"  benchmark_sp_ic50.csv:   {len(pd.read_csv(os.path.join(PROCESSED_DIR,'benchmark_sp_ic50.csv'))):>8,} rows")
print(f"  features_full.npz:       {X.shape}")
