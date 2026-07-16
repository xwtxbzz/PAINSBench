"""
Step 5 (optional): Adversarial PAINS robustness test.
Add PAINS substructures to clean compounds — measure prediction shift.
"""
import os, sys
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED
from src.models import MODEL_REGISTRY

np.random.seed(RANDOM_SEED)

# Simple SMARTS that trigger PAINS alerts (used for adversarial injection)
ADVERSARIAL_FRAGMENTS = [
    "c1ccccc1C#N",            # benzonitrile → PAINS alert
    "c1ccc2c(c1)nc(-c3ccccc3)o2",  # 2-phenylbenzoxazole
    "O=C1Nc2ccccc2C1=O",      # phthalimide
    "c1ccsc1-c1cccs1",         # dithiophene (PAINS B)
    "C=C(C(=O)O)c1ccccc1",    # alpha,beta-unsaturated carbonyl
]


def get_clean_smiles(benchmark_df, n=500):
    """Get unique SMILES of PAINS- compounds."""
    clean = benchmark_df[
        benchmark_df["PAINS_status"] == 0
    ]["canonical_smiles"].dropna().unique()
    n = min(n, len(clean))
    return np.random.choice(clean, size=n, replace=False)


def inject_fragment(smiles):
    """Append a PAINS fragment to a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    from rdkit.Chem import MolFromSmiles, MolToSmiles, CombineMols, SanitizeMol
    try:
        frag = MolFromSmiles(np.random.choice(ADVERSARIAL_FRAGMENTS))
        if frag is None:
            return None
        combo = CombineMols(mol, frag)
        return MolToSmiles(combo)
    except Exception:
        return None


def main():
    print("=" * 60, flush=True)
    print("Step 5: Adversarial PAINS robustness test", flush=True)
    print("=" * 60, flush=True)

    # Load benchmark (has SMILES from step 2 merge)
    benchmark = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_dataset.csv"))
    print(f"Loaded benchmark: {len(benchmark):,} rows", flush=True)

    # Get clean SMILES
    clean_smiles = get_clean_smiles(benchmark, n=500)
    print(f"Found {len(clean_smiles)} clean PAINS- compounds", flush=True)

    # Load model (train on FP only for adversarial compatibility)
    from sklearn.model_selection import train_test_split
    data = np.load(os.path.join(PROCESSED_DIR, "features.npz"))
    X, y, ps = data["X"], data["y"], data["pains_status"]
    # Use only Morgan fingerprint part (first 2048 cols)
    X_fp = X[:, :2048]
    X_train, _, y_train, _ = train_test_split(
        X_fp, y, test_size=0.2, random_state=RANDOM_SEED, stratify=ps
    )
    model = MODEL_REGISTRY["XGBoost"](X_train, y_train)

    # Adversarial test: predict before/after fragment injection
    from src.features import smiles_to_morgan, extract_properties
    shifts = []
    for smi in tqdm(clean_smiles[:200], desc="Adversarial"):
        smi_adv = inject_fragment(smi)
        if smi_adv is None:
            continue
        fps, _ = smiles_to_morgan([smi, smi_adv], radius=2, n_bits=2048)
        if fps.shape[0] < 2:
            continue
        X_orig = fps[0:1].astype(np.float32)
        X_adv = fps[1:2].astype(np.float32)
        pred_orig = model.predict(X_orig)[0]
        pred_adv = model.predict(X_adv)[0]
        shifts.append({
            "smiles_orig": smi,
            "smiles_adv": smi_adv,
            "pred_orig": float(pred_orig),
            "pred_adv": float(pred_adv),
            "shift": float(pred_adv - pred_orig),
        })

    shifts_df = pd.DataFrame(shifts)
    if len(shifts_df) == 0:
        print("No adversarial pairs generated", flush=True)
        return

    print(f"\nGenerated {len(shifts_df)} adversarial pairs", flush=True)
    print(f"Mean prediction shift: {shifts_df['shift'].mean():+.4f} pChEMBL", flush=True)
    print(f"Max positive shift:    {shifts_df['shift'].max():+.4f}", flush=True)
    print(f"Max negative shift:    {shifts_df['shift'].min():+.4f}", flush=True)
    print(f"Std of shift:          {shifts_df['shift'].std():.4f}", flush=True)

    n_up = (shifts_df["shift"] > 0.3).sum()
    n_down = (shifts_df["shift"] < -0.3).sum()
    print(f"Shift > +0.3: {n_up}/{len(shifts_df)} ({n_up/len(shifts_df)*100:.1f}%)", flush=True)
    print(f"Shift < -0.3: {n_down}/{len(shifts_df)} ({n_down/len(shifts_df)*100:.1f}%)", flush=True)

    shifts_df.to_csv(os.path.join(RESULTS_DIR, "adversarial_results.csv"), index=False)
    print(f"\nSaved: {os.path.join(RESULTS_DIR, 'adversarial_results.csv')}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
