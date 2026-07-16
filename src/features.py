"""Molecular feature engineering (fingerprints + physicochemical props)."""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from tqdm import tqdm


# Physicochemical property names (must match compound_with_properties columns)
PROP_NAMES = ["mw_freebase", "alogp", "hba", "hbd", "psa", "rtb",
              "num_ro5_violations", "aromatic_rings", "heavy_atoms", "qed_weighted"]


def smiles_to_morgan(smiles_list, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint matrix (n_compounds × n_bits)."""
    fps = []
    valid_idx = []
    for i, smi in enumerate(tqdm(smiles_list, desc="Morgan FP")):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.uint8))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(np.array(fp, dtype=np.uint8))
        valid_idx.append(i)
    return np.array(fps), valid_idx


def morgan_from_df(df, smiles_col="canonical_smiles", radius=2, n_bits=2048):
    """Compute Morgan fingerprints from a DataFrame column."""
    fps, valid = smiles_to_morgan(df[smiles_col].tolist(), radius, n_bits)
    return fps


def extract_properties(df, prop_names=None):
    """Extract physicochemical properties as float32 array."""
    if prop_names is None:
        prop_names = PROP_NAMES
    available = [c for c in prop_names if c in df.columns]
    return df[available].values.astype(np.float32)


def compute_additional_properties(smiles_list):
    """Compute extra RDKit descriptors beyond those in the CSV."""
    results = []
    for smi in tqdm(smiles_list, desc="RDKit desc"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append([np.nan] * 4)
            continue
        results.append([
            Descriptors.FractionCSP3(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
        ])
    return np.array(results, dtype=np.float32)
