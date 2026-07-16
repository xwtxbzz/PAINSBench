"""
PR-DTA: PAINS-Resistant Drug-Target Affinity Prediction.
  Module 1: Data Filtering — remove PAINS+ from training set
  Module 2: Dual-Branch Drug Encoder — GIN main trunk + rule-based PAINS gate
  Module 3: Protein-Drug Interaction Head — CNN(protein) + concat + MLP
"""
import os, sys, time, gc, warnings, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED
from src.evaluation import evaluate_pains_aware

warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
PROTEIN_MAX_LEN = 1200
DRUG_DIM = 128
PROT_DIM = 128

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i+1 for i, aa in enumerate(AA_ORDER)}
AT = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
BT = [1, 2, 3, 12]


def tokenize(seq, ml=PROTEIN_MAX_LEN):
    """Fast protein tokenization, returns int32 array."""
    t = np.zeros(ml, dtype=np.int32)
    n = min(len(seq), ml)
    for i in range(n):
        t[i] = AA_TO_IDX.get(seq[i], len(AA_ORDER)+1)
    return t


def mol_to_graph(smi):
    """SMILES → PyG Data object."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    at = [a.GetAtomicNum() for a in m.GetAtoms()]
    n = len(at)
    x = torch.zeros(n, len(AT)+3)
    for i, a in enumerate(at):
        x[i, AT.index(a) if a in AT else -3] = 1
        x[i, -2] = m.GetAtomWithIdx(i).GetDegree() / 4.0
        x[i, -1] = m.GetAtomWithIdx(i).GetTotalNumHs() / 3.0
    ei, ea = [], []
    for bnd in m.GetBonds():
        i, j = bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx()
        ei += [[i, j], [j, i]]
        bt = bnd.GetBondTypeAsDouble()
        f = [1.0 if bt == b else 0.0 for b in BT]
        ea += [f, f]
    if not ei:  # single-atom molecule
        ei = [[0, 0]]
        ea = [[1.0, 0, 0, 0]]
    return Data(
        x=x,
        edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(ea, dtype=torch.float),
    )


# ====================== MODULE 1: DATA FILTERING ======================
print("=" * 60, flush=True)
print("[MODULE 1] Data Loading & PAINS Filtering", flush=True)
print("=" * 60, flush=True)

b = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_dta_full.csv"))
bf = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_full.csv"))
bf["_oi"] = np.arange(len(bf))
sp = b.merge(bf[["molregno", "target_chembl_id", "_oi"]],
             on=["molregno", "target_chembl_id"], how="left")["_oi"].values

d = np.load(os.path.join(PROCESSED_DIR, "features_full.npz"), allow_pickle=True)
y_label = d["y"][sp].astype(np.float32)
ps = d["pains_status"][sp]
sl = b["canonical_smiles"].values
seqs = b["sequence"].values

# Tokenize proteins — int32 saves 50% memory vs int64
prot_tok = np.array([tokenize(s) for s in seqs], dtype=np.int32)
del seqs
print(f"  Total: {len(y_label)}, PAINS+: {ps.mean():.3f} ({int(ps.sum())})", flush=True)

ai = np.arange(len(y_label))
ti, te = train_test_split(ai, test_size=0.2, random_state=RANDOM_SEED, stratify=ps)
ti, vi = train_test_split(ti, test_size=0.125, random_state=RANDOM_SEED, stratify=ps[ti])
# Filter PAINS+ from TRAINING only
train_mask = ps[ti] == 0
ti = ti[train_mask]
print(f"  Train: {len(ti)} (filtered), Val: {len(vi)}, Test: {len(te)}", flush=True)

# ====================== PRELOADED CACHES ======================
print("Loading precomputed PAINS masks...", flush=True)
pm = np.load(os.path.join(PROCESSED_DIR, "pains_masks_puda.npz"))
pains_masks_all = pm["masks"]  # (N, 128), float32
print(f"  PAINS masks: {pains_masks_all.shape}", flush=True)

print("Loading graph cache...", flush=True)
CACHE_PATH = os.path.join(PROCESSED_DIR, "graph_cache.pkl")
gcache = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        gcache = pickle.load(f)
print(f"  Graphs cached: {len(gcache)}", flush=True)


def get_graphs(indices, desc="Graphs"):
    """Get PyG graphs from cache (no fallback — avoids OOM from bulk mol creation)."""
    gs, valid_orig_indices = [], []
    for orig_idx in tqdm(indices, desc=desc, ncols=80):
        smi = str(sl[orig_idx])
        g = gcache.get(smi)
        if g is not None:
            gs.append(g)
            valid_orig_indices.append(orig_idx)
    arr = np.array(valid_orig_indices, dtype=int)
    return gs, arr


tg, tv = get_graphs(ti, "Train graphs")
vg, vv = get_graphs(vi, "Val graphs")
sg, sv = get_graphs(te, "Test graphs")
print(f"  Graphs: {len(tg)}/{len(vg)}/{len(sg)}", flush=True)

# Free graph cache — keep only used graphs alive
del gcache
gc.collect()

# Align tensors to actually-available graphs
yt = torch.tensor(y_label[tv], dtype=torch.float)
yv_ = torch.tensor(y_label[vv], dtype=torch.float)
ye = torch.tensor(y_label[sv], dtype=torch.float)
pse = ps[sv].copy()

st_ = torch.tensor(prot_tok[tv], dtype=torch.long)
sv_ = torch.tensor(prot_tok[vv], dtype=torch.long)
se_ = torch.tensor(prot_tok[sv], dtype=torch.long)
del prot_tok

pm_train = torch.tensor(pains_masks_all[tv], dtype=torch.float)
pm_val = torch.tensor(pains_masks_all[vv], dtype=torch.float)
pm_test = torch.tensor(pains_masks_all[sv], dtype=torch.float)
del pains_masks_all

ratio_train = pm_train.mean(dim=1)
ratio_val = pm_val.mean(dim=1)
ratio_test = pm_test.mean(dim=1)

for g, l in zip(tg, yt):   g.y = l
for g, l in zip(vg, yv_):  g.y = l
for g, l in zip(sg, ye):   g.y = l

ps_train_t = torch.tensor(ps[tv], dtype=torch.float)
del y_label, ps, sl, d, sp
gc.collect()

# ====================== DATASET ======================
class PRDS(Dataset):
    def __init__(self, g, s, y, pm, pr, ps_=None):
        self.g = g
        self.s = s
        self.y = y
        self.pm = pm
        self.pr = pr
        self.ps = ps_

    def __len__(self):
        return len(self.g)

    def __getitem__(self, i):
        ret = (self.g[i], self.s[i], self.y[i], self.pm[i], self.pr[i])
        if self.ps is not None:
            ret += (self.ps[i],)
        return ret


def collate_fn(batch):
    has_ps = len(batch[0]) == 6
    g, s, y, pm, pr, *rest = zip(*batch)
    bg = Batch.from_data_list(list(g))
    st = torch.stack(list(s))
    yt_ = torch.stack(list(y))
    pmt = torch.stack(list(pm))
    prt = torch.stack(list(pr))
    if has_ps:
        return bg, st, yt_, pmt, prt, torch.tensor([x.item() for x in rest[0]], dtype=torch.float)
    return bg, st, yt_, pmt, prt


tr_ds = PRDS(tg, st_, yt, pm_train, ratio_train, ps_train_t)
va_ds = PRDS(vg, sv_, yv_, pm_val, ratio_val)
te_ds = PRDS(sg, se_, ye, pm_test, ratio_test)
tr_loader = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
va_loader = DataLoader(va_ds, BATCH_SIZE, collate_fn=collate_fn)
te_loader = DataLoader(te_ds, BATCH_SIZE, collate_fn=collate_fn)

# ====================== MODULE 2 & 3: MODEL ======================
class GINEncoder(nn.Module):
    """GIN main trunk: learns genuine drug-target features."""
    def __init__(self, in_dim=len(AT)+3, hid=128, out=DRUG_DIM):
        super().__init__()
        n1 = nn.Sequential(nn.Linear(in_dim, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Linear(hid, hid))
        self.c1 = GINConv(n1)
        n2 = nn.Sequential(nn.Linear(hid, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Linear(hid, hid))
        self.c2 = GINConv(n2)
        n3 = nn.Sequential(nn.Linear(hid, out), nn.BatchNorm1d(out), nn.ReLU(), nn.Linear(out, out))
        self.c3 = GINConv(n3)
        self.pool = global_mean_pool

    def forward(self, data):
        x = F.relu(self.c1(data.x, data.edge_index))
        x = F.relu(self.c2(x, data.edge_index))
        x = F.relu(self.c3(x, data.edge_index))
        return self.pool(x, data.batch)


class PAINSGate(nn.Module):
    """Rule-based PAINS detection → gating signal s_pains ∈ [0, 1]."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(129, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, mask, ratio):
        return self.net(torch.cat([mask.float(), ratio.unsqueeze(-1).float()], dim=1)).squeeze(-1)


class ProteinCNN(nn.Module):
    """Standard 3-layer CNN protein encoder."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(25, 32, padding_idx=0)
        self.c1 = nn.Conv1d(32, 64, 5, padding=2)
        self.b1 = nn.BatchNorm1d(64)
        self.c2 = nn.Conv1d(64, 128, 5, padding=2)
        self.b2 = nn.BatchNorm1d(128)
        self.c3 = nn.Conv1d(128, PROT_DIM, 5, padding=2)
        self.b3 = nn.BatchNorm1d(PROT_DIM)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.do = nn.Dropout(0.2)

    def forward(self, s):
        x = self.emb(s).permute(0, 2, 1)
        x = self.do(F.relu(self.b1(self.c1(x))))
        x = self.do(F.relu(self.b2(self.c2(x))))
        x = F.relu(self.b3(self.c3(x)))
        return self.pool(x).squeeze(-1)


class PRDTA(nn.Module):
    """PR-DTA: h_drug = h_main * (1 - s_pains)."""
    def __init__(self):
        super().__init__()
        self.gin = GINEncoder()
        self.pains_gate = PAINSGate()
        self.prot_cnn = ProteinCNN()
        self.head = nn.Sequential(
            nn.Linear(DRUG_DIM+PROT_DIM, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1))

    def forward(self, bg, seq, mask, ratio):
        h_main = self.gin(bg)
        s_pains = self.pains_gate(mask, ratio)
        h_drug = h_main * (1 - s_pains.unsqueeze(-1))
        h_prot = self.prot_cnn(seq)
        return self.head(torch.cat([h_drug, h_prot], dim=1)).squeeze(-1)


# ====================== TRAINING ======================
print("=" * 60, flush=True)
print("[MODULES 2 & 3] Training PR-DTA", flush=True)
print("=" * 60, flush=True)


def run_epoch(model, loader, opt=None):
    is_train = opt is not None
    model.train() if is_train else model.eval()
    total_loss, n = 0.0, 0
    desc = "  train" if is_train else "  val"
    pbar = tqdm(loader, desc=desc, ncols=80, leave=False)
    with torch.set_grad_enabled(is_train):
        for batch in pbar:
            bg, seq, yb, mask, ratio = [x.to(device) for x in batch[:5]]
            if is_train:
                opt.zero_grad()
            pred = model(bg, seq, mask, ratio)
            loss = F.mse_loss(pred, yb)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            bs = len(yb)
            total_loss += loss.item() * bs
            n += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / n if n > 0 else 0.0


def run_epoch_base(model, loader, opt=None):
    is_train = opt is not None
    model.train() if is_train else model.eval()
    total_loss, n = 0.0, 0
    desc = "  train" if is_train else "  val"
    pbar = tqdm(loader, desc=desc, ncols=80, leave=False)
    with torch.set_grad_enabled(is_train):
        for batch in pbar:
            bg, seq, yb = [x.to(device) for x in batch[:3]]
            if is_train:
                opt.zero_grad()
            pred = model(bg, seq)
            loss = F.mse_loss(pred, yb)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            bs = len(yb)
            total_loss += loss.item() * bs
            n += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / n if n > 0 else 0.0


def train_prdta(model, train_loader, val_loader):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val = float("inf")
    best_state = None
    ep_pbar = tqdm(range(1, EPOCHS + 1), desc="PR-DTA", ncols=95)
    for ep in ep_pbar:
        tr_loss = run_epoch(model, train_loader, opt)
        sched.step()
        val_loss = run_epoch(model, val_loader)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        ep_pbar.set_postfix({"tr": f"{tr_loss:.4f}", "val": f"{val_loss:.4f}"})
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, loader):
    model.eval()
    pr = []
    for batch in tqdm(loader, desc="  predict", ncols=80, leave=False):
        bg, seq, _, mask, ratio = [x.to(device) for x in batch[:5]]
        pr.append(model(bg, seq, mask, ratio).cpu().numpy())
    return np.concatenate(pr)


@torch.no_grad()
def predict_base(model, loader):
    model.eval()
    pr = []
    for batch in tqdm(loader, desc="  predict", ncols=80, leave=False):
        bg, seq = [x.to(device) for x in batch[:2]]
        pr.append(model(bg, seq).cpu().numpy())
    return np.concatenate(pr)


# ====================== RUN ======================
results = []

# --- Baseline: GCN-DTA ---
print("\n--- GCN-DTA Baseline ---", flush=True)
gc.collect()
torch.cuda.empty_cache()
t0 = time.time()


class GCNDTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = GCNConv(len(AT)+3, 128)
        self.c2 = GCNConv(128, 128)
        self.c3 = GCNConv(128, DRUG_DIM)
        self.pool = global_mean_pool
        self.pe = ProteinCNN()
        self.head = nn.Sequential(nn.Linear(DRUG_DIM+PROT_DIM, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, bg, seq):
        x = F.relu(self.c1(bg.x, bg.edge_index))
        x = F.relu(self.c2(x, bg.edge_index))
        x = F.relu(self.c3(x, bg.edge_index))
        h_drug = self.pool(x, bg.batch)
        h_prot = self.pe(seq)
        return self.head(torch.cat([h_drug, h_prot], dim=1)).squeeze(-1)


bm = GCNDTA().to(device)
bo = torch.optim.AdamW(bm.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
bs_sched = torch.optim.lr_scheduler.CosineAnnealingLR(bo, T_max=EPOCHS)
best_val = float("inf")
best_state = None

for ep in tqdm(range(1, EPOCHS + 1), desc="Baseline", ncols=95):
    run_epoch_base(bm, tr_loader, bo)
    bs_sched.step()
    vl = run_epoch_base(bm, va_loader)
    if vl < best_val:
        best_val = vl
        best_state = {k: v.cpu().clone() for k, v in bm.state_dict().items()}

bm.load_state_dict(best_state)
bp = predict_base(bm, te_loader)
et = time.time() - t0
ed = evaluate_pains_aware(ye.numpy(), bp, pse)
print(f"  RMSE={ed['overall_RMSE']:.4f}  P+={ed['pains_pos_RMSE']:.4f}  "
      f"P-={ed['pains_neg_RMSE']:.4f}  ΔRMSE={ed['delta_rmse']:.4f}  "
      f"FP={ed['fp_ratio']:.4f}  [{et:.0f}s]", flush=True)
results.append({"v": "GCN-DTA_baseline", "RMSE": ed["overall_RMSE"],
                "P+": ed["pains_pos_RMSE"], "P-": ed["pains_neg_RMSE"],
                "Δ": ed["delta_rmse"], "FP": ed["fp_ratio"]})
pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "prdta_results.csv"), index=False)

# --- PR-DTA Full ---
gc.collect()
torch.cuda.empty_cache()
print("\n--- PR-DTA (GIN + PAINS gating + filtered training) ---", flush=True)
t0 = time.time()
m = PRDTA()
m = train_prdta(m, tr_loader, va_loader)
yp = predict(m, te_loader)
et = time.time() - t0
ed = evaluate_pains_aware(ye.numpy(), yp, pse)
print(f"  RMSE={ed['overall_RMSE']:.4f}  P+={ed['pains_pos_RMSE']:.4f}  "
      f"P-={ed['pains_neg_RMSE']:.4f}  ΔRMSE={ed['delta_rmse']:.4f}  "
      f"FP={ed['fp_ratio']:.4f}  [{et:.0f}s]", flush=True)
results.append({"v": "PR-DTA_full", "RMSE": ed["overall_RMSE"],
                "P+": ed["pains_pos_RMSE"], "P-": ed["pains_neg_RMSE"],
                "Δ": ed["delta_rmse"], "FP": ed["fp_ratio"]})
pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "prdta_results.csv"), index=False)

# ====================== SUMMARY ======================
rf = pd.DataFrame(results)
print(f"\n{'=' * 60}")
print(f"{'Variant':25s} {'RMSE':>8s} {'P+':>8s} {'P-':>8s} {'ΔRMSE':>8s} {'FP':>8s}")
print("-" * 60)
for _, r in rf.iterrows():
    print(f"{r['v']:25s}  {r['RMSE']:8.4f}  {r['P+']:8.4f}  {r['P-']:8.4f}  "
          f"{r['Δ']:8.4f}  {r['FP']:8.4f}")
print("Done.", flush=True)
