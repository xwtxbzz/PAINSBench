"""
ID-DTA: Invariant Debiasing DTA.
Environment-invariant learning + MI bottleneck + counterfactual consistency + bias subtraction.
With tqdm training display and data preloading.
"""
import os, sys, time, gc, warnings, pickle
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED
from src.evaluation import evaluate_pains_aware
warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

BATCH_SIZE = 128; EPOCHS = 30; LR = 1e-3; WEIGHT_DECAY = 1e-5
PROTEIN_MAX_LEN = 1200; DRUG_DIM = 128; PROT_DIM = 128; FUSION_DIM = 128
N_ENVS = 5
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"; AA_TO_IDX = {aa:i+1 for i,aa in enumerate(AA_ORDER)}

def tokenize(s, ml=PROTEIN_MAX_LEN):
    t=[AA_TO_IDX.get(a,len(AA_ORDER)+1) for a in s[:ml]]
    t+=[0]*(ml-len(t)); return np.array(t,dtype=np.int64)

# ====================== DATA PRELOADING ======================
print("="*60+"\n[1/5] Loading benchmark data...\n"+"="*60, flush=True)
b=pd.read_csv(os.path.join(PROCESSED_DIR,"benchmark_dta_full.csv"))
bf=pd.read_csv(os.path.join(PROCESSED_DIR,"benchmark_full.csv"))
bf["_oi"]=np.arange(len(bf))
sp=b.merge(bf[["molregno","target_chembl_id","_oi"]],on=["molregno","target_chembl_id"],how="left")["_oi"].values
d=np.load(os.path.join(PROCESSED_DIR,"features_full.npz"),allow_pickle=True)
y_label=d["y"][sp].astype(np.float32); ps=d["pains_status"][sp]
sl=b["canonical_smiles"].values; seqs=b["sequence"].values
prot_tok=np.array([tokenize(s) for s in seqs],dtype=np.int64)

from sklearn.model_selection import train_test_split
ai=np.arange(len(y_label)); ti,te=train_test_split(ai,test_size=0.2,random_state=RANDOM_SEED,stratify=ps)
ti,vi=train_test_split(ti,test_size=0.125,random_state=RANDOM_SEED,stratify=ps[ti])
print(f"  Train {len(ti)} Val {len(vi)} Test {len(te)}", flush=True)

# ====================== ENVIRONMENT LABELS ======================
print("="*60+"\n[2/5] Computing ECFP environment clusters...\n"+"="*60, flush=True)
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
ENV_CACHE = os.path.join(PROCESSED_DIR, "env_labels.npy")
if os.path.exists(ENV_CACHE):
    env_labels = np.load(ENV_CACHE)
    print(f"  Loaded cached env labels", flush=True)
else:
    ecfp_list = []
    for smi in tqdm(sl, desc="ECFP", ncols=80):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            arr = np.zeros((1024,), dtype=np.float32)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            ecfp_list.append(arr)
        else:
            ecfp_list.append(np.zeros(1024, dtype=np.float32))
    ecfp_mat = np.array(ecfp_list)
    kmeans = KMeans(n_clusters=N_ENVS, random_state=RANDOM_SEED, n_init=10)
    env_labels = kmeans.fit_predict(ecfp_mat)
    np.save(ENV_CACHE, env_labels)
    print(f"  Env distribution: {np.bincount(env_labels)}", flush=True)

# ====================== PAINS DETECTION & MASKING ======================
print("="*60+"\n[3/5] Detecting PAINS substructures...\n"+"="*60, flush=True)
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
MASK_CACHE = os.path.join(PROCESSED_DIR, "pains_masks.npz")
SUBFP_CACHE = os.path.join(PROCESSED_DIR, "pains_subfp.npy")
if os.path.exists(MASK_CACHE) and os.path.exists(SUBFP_CACHE):
    pm = np.load(MASK_CACHE)
    pains_masks_arr = pm["masks"]
    pains_subfp = np.load(SUBFP_CACHE)
    print(f"  Loaded cached: masks {pains_masks_arr.shape}, subfp {pains_subfp.shape}", flush=True)
else:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    catalog = FilterCatalog(params)
    pains_subfp_list, pains_masks_list = [], []
    for smi in tqdm(sl, desc="PAINS detection", ncols=80):
        mol = Chem.MolFromSmiles(smi)
        spf = np.zeros(3, dtype=np.float32)
        mask = np.zeros(mol.GetNumAtoms(), dtype=np.float32) if mol else np.zeros(1)
        if mol:
            for entry in catalog.GetMatches(mol):
                if isinstance(entry, tuple) and len(entry) > 1:
                    matched = entry[-1]
                    if isinstance(matched, tuple):
                        for idx in matched:
                            if idx < len(mask): mask[idx] = 1.0
        if len(mask) < 128: mask = np.pad(mask, (0, 128-len(mask)), 'constant')[:128]
        else: mask = mask[:128]
        pains_subfp_list.append(spf); pains_masks_list.append(mask)
    pains_subfp = np.array(pains_subfp_list, dtype=np.float32)
    pains_masks_arr = np.array(pains_masks_list, dtype=np.float32)
    np.savez(MASK_CACHE, masks=pains_masks_arr)
    np.save(SUBFP_CACHE, pains_subfp)
    print(f"  Avg PAINS atoms: {pains_masks_arr.mean():.3f}", flush=True)

# ====================== GRAPHS ======================
print("="*60+"\n[4/5] Loading/building molecular graphs...\n"+"="*60, flush=True)
CACHE_PATH = os.path.join(PROCESSED_DIR, "graph_cache.pkl")
gcache = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f: gcache = pickle.load(f)
    print(f"  Loaded {len(gcache)} cached graphs", flush=True)

from torch_geometric.data import Data
AT = [5,6,7,8,9,15,16,17,35,53]; BT = [1,2,3,12]

def mol_to_graph(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    at = [a.GetAtomicNum() for a in m.GetAtoms()]
    x = torch.zeros(len(at), len(AT)+3)
    for i,a in enumerate(at):
        x[i, AT.index(a) if a in AT else -3] = 1
        x[i,-2] = m.GetAtomWithIdx(i).GetDegree()/4.0
        x[i,-1] = m.GetAtomWithIdx(i).GetTotalNumHs()/3.0
    ei, ea = [], []
    for bnd in m.GetBonds():
        i,j = bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx(); ei += [[i,j],[j,i]]
        bt = bnd.GetBondTypeAsDouble(); f = [1.0 if bt==b else 0.0 for b in BT]; ea += [f,f]
    if not ei: ei = [[0,0]]; ea = [[1.0,0,0,0]]
    return Data(x=x, edge_index=torch.tensor(ei,dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(ea,dtype=torch.float))

def get_graphs(indices):
    gs, va = [], []
    for i, ix in enumerate(indices):
        sm = str(sl[ix])
        g = gcache.get(sm) or mol_to_graph(sm)
        if g is not None: gcache[sm] = g; gs.append(g); va.append(ix)
        if (i+1) % 20000 == 0: print(f"  {i+1}/{len(indices)}", flush=True)
    return gs, np.array(va, dtype=int)

tg, tv = get_graphs(ti); vg, vv = get_graphs(vi); sg, sv = get_graphs(te)
print(f"  Graphs: {len(tg)}/{len(vg)}/{len(sg)}", flush=True)
if not os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "wb") as f: pickle.dump(gcache, f)

# ====================== MASKED GRAPHS (Counterfactual) ======================
print("Building counterfactual masked graphs...", flush=True)
def make_masked_graph(graph, atom_mask, strength=0.9):
    g = deepcopy(graph); m = torch.zeros(g.x.size(0))
    if atom_mask is not None:
        m = torch.tensor(atom_mask[:g.x.size(0)], dtype=torch.float)
    g.x = g.x * (1 - m.unsqueeze(-1) * strength)
    return g

tg_masked = [make_masked_graph(tg[i], pains_masks_arr[tv[i]] if ps[tv[i]]==1 else None)
             for i in tqdm(range(len(tg)), desc="Mask train", ncols=80)]
vg_masked = [make_masked_graph(vg[i], pains_masks_arr[vv[i]] if ps[vv[i]]==1 else None)
             for i in range(len(vg))]

# ====================== TENSOR ALIGNMENT ======================
yt = torch.tensor(y_label[tv], dtype=torch.float)
yv_ = torch.tensor(y_label[vv], dtype=torch.float)
ye = torch.tensor(y_label[sv], dtype=torch.float)
pse = ps[sv]
st_ = torch.tensor(prot_tok[tv]); sv_ = torch.tensor(prot_tok[vv]); se_ = torch.tensor(prot_tok[sv])
et_ = torch.tensor(env_labels[tv], dtype=torch.long); ev_ = torch.tensor(env_labels[vv], dtype=torch.long)
sft_ = torch.tensor(pains_subfp[tv], dtype=torch.float); sfv_ = torch.tensor(pains_subfp[vv], dtype=torch.float)
sfe_ = torch.tensor(pains_subfp[sv], dtype=torch.float)

for g,l in zip(tg,yt): g.y=l
for g,l in zip(vg,yv_): g.y=l
for g,l in zip(sg,ye): g.y=l
for g,l in zip(tg_masked,yt): g.y=l
for g,l in zip(vg_masked,yv_): g.y=l

# ====================== DATASET ======================
class IDDataset(Dataset):
    def __init__(self, g, gm, s, y, e, sf):
        self.g=g; self.gm=gm; self.s=s; self.y=y; self.e=e; self.sf=sf
    def __len__(self): return len(self.g)
    def __getitem__(self,i):
        return self.g[i], self.gm[i], self.s[i], self.y[i], self.e[i], self.sf[i]

class IDDatasetTe(Dataset):
    def __init__(self, g, s, y, sf): self.g=g; self.s=s; self.y=y; self.sf=sf
    def __len__(self): return len(self.g)
    def __getitem__(self,i): return self.g[i], self.s[i], self.y[i], self.sf[i]

def collate(b):
    from torch_geometric.data import Batch
    g,gm,s,y,e,sf = zip(*b)
    return (Batch.from_data_list(list(g)), Batch.from_data_list(list(gm)),
            torch.stack(list(s)), torch.stack(list(y)), torch.tensor(e, dtype=torch.long),
            torch.stack(list(sf)))

def collate_te(b):
    from torch_geometric.data import Batch
    g,s,y,sf = zip(*b)
    return Batch.from_data_list(list(g)), torch.stack(list(s)), torch.stack(list(y)), torch.stack(list(sf))

tr_ds = IDDataset(tg, tg_masked, st_, yt, et_, sft_)
va_ds = IDDataset(vg, vg_masked, sv_, yv_, ev_, sfv_)
te_ds = IDDatasetTe(sg, se_, ye, sfe_)
tr_loader = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, collate_fn=collate)
va_loader = DataLoader(va_ds, BATCH_SIZE, collate_fn=collate)
te_loader = DataLoader(te_ds, BATCH_SIZE, collate_fn=collate_te)

print(f"  Data preloading complete: {len(tr_ds)}/{len(va_ds)}/{len(te_ds)}", flush=True)

# ====================== MODEL ======================
class ProteinCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(25, 32, padding_idx=0)
        self.c1 = nn.Conv1d(32, 64, 5, padding=2); self.b1 = nn.BatchNorm1d(64)
        self.c2 = nn.Conv1d(64, 128, 5, padding=2); self.b2 = nn.BatchNorm1d(128)
        self.c3 = nn.Conv1d(128, PROT_DIM, 5, padding=2); self.b3 = nn.BatchNorm1d(PROT_DIM)
        self.pool = nn.AdaptiveMaxPool1d(1); self.do = nn.Dropout(0.2)
    def forward(self, s):
        x = self.emb(s).permute(0,2,1)
        x = self.do(F.relu(self.b1(self.c1(x))))
        x = self.do(F.relu(self.b2(self.c2(x))))
        x = F.relu(self.b3(self.c3(x)))
        return self.pool(x).squeeze(-1)

class GCNDrug(nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
        self.c1 = GCNConv(len(AT)+3, 128); self.c2 = GCNConv(128, 128); self.c3 = GCNConv(128, DRUG_DIM)
        self.pool = global_mean_pool
    def forward(self, data):
        x = F.relu(self.c1(data.x, data.edge_index))
        x = F.relu(self.c2(x, data.edge_index))
        x = F.relu(self.c3(x, data.edge_index))
        return self.pool(x, data.batch)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a): ctx.a = a; return x
    @staticmethod
    def backward(ctx, g): return -ctx.a * g, None
def gr(x, a): return GradReverse.apply(x, a)

class IDDTA(nn.Module):
    def __init__(self, n_envs=N_ENVS):
        super().__init__()
        self.drug_enc = GCNDrug()
        self.prot_enc = ProteinCNN()
        self.fusion = nn.Linear(DRUG_DIM+PROT_DIM, FUSION_DIM)
        self.predictor = nn.Sequential(nn.Linear(FUSION_DIM, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        self.bias_net = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
        self.env_cls = nn.Linear(FUSION_DIM, n_envs)
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))
        self.criterion_mse = nn.MSELoss()

    def forward(self, full_graph, masked_graph, prot_seq, pains_subfp, alpha_grl=0.0, return_all=False):
        h_drug = self.drug_enc(full_graph)
        h_drug_m = self.drug_enc(masked_graph)
        h_prot = self.prot_enc(prot_seq)
        phi = F.relu(self.fusion(torch.cat([h_drug_m, h_prot], dim=1)))
        pred_main = self.predictor(phi).squeeze(-1)
        bias = self.bias_net(pains_subfp).squeeze(-1)
        pred_total = pred_main + bias

        phi_noisy = phi + torch.randn_like(phi) * torch.exp(self.log_sigma)
        phi_rev = gr(phi_noisy, float(alpha_grl))
        env_logits = self.env_cls(phi_rev)

        if return_all: return pred_total, pred_main, bias, phi, env_logits, h_drug
        return pred_total

# ====================== TRAINING ======================
def train_iddta(model, train_loader, val_loader, te_loader, epochs=EPOCHS,
                lambda_env=0.5, lambda_mi=0.1, lambda_cf=1.0, lambda_bias=0.01, model_name="ID-DTA"):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val = float("inf"); best_state = None
    warmup = epochs // 3

    pbar = tqdm(range(1, epochs+1), desc=model_name, ncols=100)
    for ep in pbar:
        alpha_grl = min(1.0, ep / max(warmup, 1))
        model.train()
        losses = {"mse":0, "env":0, "mi":0, "cf":0, "bias_reg":0, "total":0}; n = 0

        for batch in train_loader:
            bg, bgm, seq, yb, eb, sfb = [x.to(device) for x in batch]
            opt.zero_grad()
            pred_t, pred_m, bias, phi, env_lg, h_drug = model.forward(bg, bgm, seq, sfb, alpha_grl=alpha_grl, return_all=True)

            l_mse = F.mse_loss(pred_t, yb)

            # VREx
            env_ls = []
            for e in range(N_ENVS):
                m = eb == e
                if m.sum() > 1: env_ls.append(F.mse_loss(pred_t[m], yb[m]).unsqueeze(0))
            l_env = torch.cat(env_ls).var() if len(env_ls) > 1 else torch.tensor(0.0, device=device)

            l_mi = F.cross_entropy(env_lg, eb)
            l_cf = F.mse_loss(pred_m, yb) + (1 - F.cosine_similarity(h_drug, model.drug_enc(bgm), dim=1).mean())
            l_br = bias.pow(2).mean()

            loss = l_mse + lambda_env*l_env + lambda_mi*l_mi + lambda_cf*l_cf + lambda_bias*l_br
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = yb.size(0)
            losses["mse"] += l_mse.item()*bs; losses["env"] += l_env.item()*bs if isinstance(l_env, torch.Tensor) else 0
            losses["mi"] += l_mi.item()*bs; losses["cf"] += l_cf.item()*bs; losses["bias_reg"] += l_br.item()*bs
            losses["total"] += loss.item()*bs; n += bs

        sch.step()

        model.eval(); vl, nv = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                bg, bgm, seq, yb, eb, sfb = [x.to(device) for x in batch]
                pt, *_ = model.forward(bg, bgm, seq, sfb, 0.0, True)
                vl += F.mse_loss(pt, yb).item()*yb.size(0); nv += yb.size(0)
        vl /= nv
        if vl < best_val: best_val = vl; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

        pbar.set_postfix({"L":f"{losses['mse']/n:.4f}", "E":f"{losses['env']/n:.4f}",
                          "M":f"{losses['mi']/n:.4f}", "C":f"{losses['cf']/n:.4f}",
                          "val":f"{vl:.4f}", "α":f"{alpha_grl:.2f}"})

    model.load_state_dict(best_state)
    return model

@torch.no_grad()
def predict_iddta(model, loader):
    model.eval(); pr = []
    for batch in loader:
        bg, seq, _, _ = [x.to(device) for x in batch]
        h_drug = model.drug_enc(bg); h_prot = model.prot_enc(seq)
        phi = F.relu(model.fusion(torch.cat([h_drug, h_prot], dim=1)))
        pred = model.predictor(phi).squeeze(-1)
        pr.append(pred.cpu().numpy())
    return np.concatenate(pr)

# ====================== RUN ======================
print("="*60+"\n[5/5] Training\n"+"="*60, flush=True)
results = []

# Baseline: GCN-DTA (no debiasing)
print("\n--- GCN-DTA Baseline ---", flush=True)
gc.collect(); torch.cuda.empty_cache(); t0 = time.time()
m_base = IDDTA(); m_base = train_iddta(m_base, tr_loader, va_loader, te_loader,
    lambda_env=0, lambda_mi=0, lambda_cf=0, lambda_bias=0, model_name="Baseline ")
yp = predict_iddta(m_base, te_loader); et = time.time()-t0
ed = evaluate_pains_aware(ye.numpy(), yp, pse)
print(f"  ✓ RMSE={ed['overall_RMSE']:.4f} ΔRMSE={ed['delta_rmse']:.4f} FP={ed['fp_ratio']:.4f} [{et:.0f}s]", flush=True)
results.append({"variant":"baseline","RMSE":ed["overall_RMSE"],"D":ed["delta_rmse"],"FP":ed["fp_ratio"]})
pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"iddta_results.csv"),index=False)

# ID-DTA Full
print("\n--- ID-DTA Full ---", flush=True)
gc.collect(); torch.cuda.empty_cache(); t0 = time.time()
m_full = IDDTA(); m_full = train_iddta(m_full, tr_loader, va_loader, te_loader,
    lambda_env=0.5, lambda_mi=0.1, lambda_cf=1.0, lambda_bias=0.01, model_name="ID-DTA  ")
yp = predict_iddta(m_full, te_loader); et = time.time()-t0
ed = evaluate_pains_aware(ye.numpy(), yp, pse)
print(f"  ✓ RMSE={ed['overall_RMSE']:.4f} ΔRMSE={ed['delta_rmse']:.4f} FP={ed['fp_ratio']:.4f} [{et:.0f}s]", flush=True)
results.append({"variant":"ID-DTA_full","RMSE":ed["overall_RMSE"],"D":ed["delta_rmse"],"FP":ed["fp_ratio"]})
pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"iddta_results.csv"),index=False)

print("\n"+"="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"{'Variant':20s} {'RMSE':>8s} {'ΔRMSE':>8s} {'FP':>8s} {'Improvement':>12s}")
print("-"*60)
bl_d = results[0]["D"]
for r in results:
    better = abs(r["D"]) < abs(bl_d)
    pct = (1 - abs(r["D"])/abs(bl_d))*100 if bl_d != 0 else 0
    imp = f"✓ {pct:.1f}%" if better else "✗"
    print(f"{r['variant']:20s} {r['RMSE']:8.4f} {r['D']:8.4f} {r['FP']:8.4f} {imp:>12s}")
print("Done.", flush=True)
