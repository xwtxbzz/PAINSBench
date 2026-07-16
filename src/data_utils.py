"""Data loading utilities for PAINSBench."""

import os, sys, gc
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS, VALID_ASSAY_TYPES, PCHEMBL_MIN, PCHEMBL_MAX


def load_structural_alerts():
    """Load structural alert definitions + alert sets."""
    alerts = pd.read_csv(PATHS["structural_alerts"])
    sets = pd.read_csv(PATHS["alert_sets"])
    return alerts, sets


def load_compound_alerts():
    """Load all compound–alert mappings."""
    chunks = []
    for f in tqdm(PATHS["compound_alerts"], desc="compound alerts"):
        chunks.append(pd.read_csv(f))
    return pd.concat(chunks, ignore_index=True)


def load_filtered_activity(max_rows=None):
    """Load & filter joined activity to high-quality IC50/Ki/Kd with pChEMBL."""
    cols = ["molregno", "standard_type", "standard_value", "standard_units",
            "pchembl_value", "standard_flag", "target_chembl_id", "target_name",
            "target_type", "organism"]
    records = []
    total = 0
    for f in tqdm(PATHS["joined_activity"], desc="activity"):
        for chunk in pd.read_csv(f, usecols=cols, chunksize=200_000):
            ok = (
                chunk["standard_type"].isin(VALID_ASSAY_TYPES)
                & chunk["standard_units"].str.upper().str.contains("NM", na=False)
                & chunk["pchembl_value"].notna()
                & chunk["pchembl_value"].between(PCHEMBL_MIN, PCHEMBL_MAX)
                & (chunk["standard_flag"] == 1)
            )
            records.append(chunk[ok])
            total += len(chunk[ok])
            if max_rows and total >= max_rows:
                break
        if max_rows and total >= max_rows:
            break
    df = pd.concat(records, ignore_index=True)
    df["pchembl_value"] = df["pchembl_value"].astype(np.float32)
    return df


def load_compound_properties():
    """Load compound physicochemical properties."""
    chunks = []
    for f in tqdm(PATHS["compound_properties"], desc="props"):
        chunks.append(pd.read_csv(f))
    return pd.concat(chunks, ignore_index=True)


def load_fingerprints():
    """Load Morgan fingerprints CSV."""
    return pd.read_csv(PATHS["fingerprints"])


def memory_usage(df):
    """Return rough MB usage of a DataFrame."""
    return df.memory_usage(deep=True).sum() / 1e6
