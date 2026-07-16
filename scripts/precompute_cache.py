"""
Precompute all caches for PUDA training.
Skips steps whose cache already exists.
"""
import os, sys, pickle, time, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR

warnings.filterwarnings("ignore")

# ========== 1. PAINS MASKS & FP ==========
pm_file = os.path.join(PROCESSED_DIR, "pains_masks_puda.npz")
if os.path.exists(pm_file):
    pm = np.load(pm_file)
    print(f"[1/3] OK: PAINS masks ({pm['masks'].shape}, {pm['fp'].shape})")
else:
    print("[1/3] Computing PAINS masks (126K molecules)...")
    from rdkit import Chem
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    catalog = FilterCatalog(params)
    b = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_dta_full.csv"))
    sl = b["canonical_smiles"].values
    dd = np.load(os.path.join(PROCESSED_DIR, "features_full.npz"), allow_pickle=True)
    ps = dd["pains_status"]
    masks, fps = [], []
    for smi in tqdm(sl, desc="PAINS", ncols=80):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: masks.append(np.zeros(128)); fps.append(np.zeros(7)); continue
        m, pf = np.zeros(128), np.zeros(7)
        for e in catalog.GetMatches(mol):
            if not(isinstance(e,tuple)&len(e)>1): continue
            mt = e[-1]
            if isinstance(mt,tuple):
                for idx in mt:
                    if idx<128: m[idx]=1.0
            if hasattr(e[0],'GetDescription'):
                d=e[0].GetDescription()
                if 'PAINS_A' in d: pf[0]=1.0
                elif 'PAINS_B' in d: pf[1]=1.0
                elif 'PAINS_C' in d: pf[2]=1.0
        pf[3]=float(ps[len(masks)]); pf[4]=m.mean()
        masks.append(m); fps.append(pf)
    m_arr, f_arr = np.array(masks), np.array(fps)
    np.savez(pm_file, masks=m_arr, fp=f_arr)
    print(f"  Saved: {m_arr.shape}, {f_arr.shape}")

# ========== 2. GRAPHS ==========
gc_file = os.path.join(PROCESSED_DIR, "graph_cache.pkl")
if os.path.exists(gc_file):
    print(f"[2/3] OK: Graphs ({os.path.getsize(gc_file)//1024//1024}MB)")
else:
    print("[2/3] Building molecular graphs...")
    from rdkit import Chem
    from torch_geometric.data import Data
    b = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_dta_full.csv"))
    sl = b["canonical_smiles"].values
    AT=[5,6,7,8,9,15,16,17,35,53]; BT=[1,2,3,12]
    def g(smi):
        m=Chem.MolFromSmiles(smi)
        if m is None: return None
        at=[a.GetAtomicNum() for a in m.GetAtoms()]
        x=torch.zeros(len(at),len(AT)+3)
        for i,a in enumerate(at):
            x[i,AT.index(a) if a in AT else -3]=1
            x[i,-2]=m.GetAtomWithIdx(i).GetDegree()/4.0
            x[i,-1]=m.GetAtomWithIdx(i).GetTotalNumHs()/3.0
        ei,ea=[],[]
        for bnd in m.GetBonds():
            i,j=bnd.GetBeginAtomIdx(),bnd.GetEndAtomIdx(); ei+=[[i,j],[j,i]]
            bt=bnd.GetBondTypeAsDouble(); f=[1.0 if bt==b else 0.0 for b in BT]; ea+=[f,f]
        if not ei: ei=[[0,0]]; ea=[[1.0,0,0,0]]
        return Data(x=x,edge_index=torch.tensor(ei,dtype=torch.long).t().contiguous(),edge_attr=torch.tensor(ea,dtype=torch.float))
    gc={}
    for smi in tqdm(sl, desc="Graphs", ncols=80):
        gg=g(smi)
        if gg is not None: gc[str(smi)]=gg
    with open(gc_file,"wb") as f: pickle.dump(gc,f)
    print(f"  Saved {len(gc)} graphs")

# ========== 3. ENV LABELS ==========
env_file = os.path.join(PROCESSED_DIR, "env_labels_puda.npy")
if os.path.exists(env_file):
    el = np.load(env_file)
    print(f"[3/3] OK: Env labels ({len(el)} samples, {len(np.unique(el))} clusters, dist={np.bincount(el).tolist()})")
else:
    print("[3/3] Computing env labels (PCA + quantization)...")
    dd = np.load(os.path.join(PROCESSED_DIR, "features_full.npz"), allow_pickle=True)
    fp = dd["X"][:,:256]  # Use first 256 FP bits (avoids sklearn threading bug)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5, random_state=42)
    proj = pca.fit_transform(fp)
    el = np.argmax(proj, axis=1)  # Quantize to 5 bins
    np.save(env_file, el)
    print(f"  Done: dist={np.bincount(el).tolist()}")

print("\nAll caches ready. Train with: python scripts/19_puda.py")
