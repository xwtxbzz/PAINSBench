"""Global configuration for PAINSBench."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHEMBL_DIR = os.path.normpath(r"D:\ChemBL\csv_output")

# ---------- data sources ----------
PATHS = {
    "structural_alerts": os.path.join(CHEMBL_DIR, "tables", "structural_alerts.csv"),
    "alert_sets": os.path.join(CHEMBL_DIR, "tables", "structural_alert_sets.csv"),
    "compound_alerts": sorted([
        os.path.join(CHEMBL_DIR, "tables", f)
        for f in os.listdir(os.path.join(CHEMBL_DIR, "tables"))
        if f.startswith("compound_structural_alerts_part")
    ]),
    "compound_properties": sorted([
        os.path.join(CHEMBL_DIR, "analysis", f)
        for f in os.listdir(os.path.join(CHEMBL_DIR, "analysis"))
        if f.startswith("compound_with_properties_part")
    ]),
    "joined_activity": sorted([
        os.path.join(CHEMBL_DIR, "joined", f)
        for f in os.listdir(os.path.join(CHEMBL_DIR, "joined"))
        if f.startswith("activity_target_joined_part")
    ]),
    "fingerprints": os.path.join(CHEMBL_DIR, "chembl_36_fingerprints.csv"),
}

# ---------- output ----------
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
for d in [RESULTS_DIR, FIGURES_DIR, PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------- filtering ----------
VALID_ASSAY_TYPES = ["IC50", "Ki", "Kd"]
PCHEMBL_MIN = 2.0
PCHEMBL_MAX = 11.0
MIN_MEASUREMENTS_PER_TARGET = 30

# ---------- benchmark ----------
TARGET_SIZE = 229_566            # final benchmark size (~230K)
PSM_CALIPER = 0.05               # propensity-score matching caliper (secondary analysis)
TEST_SPLIT = 0.2                 # hold-out fraction
VAL_SPLIT = 0.1                  # validation fraction from train
STRATIFIED_SAMPLE_SIZE = 200_000 # stratified sample target (up to available data)

# ---------- model ----------
RANDOM_SEED = 42
N_JOBS = 4
