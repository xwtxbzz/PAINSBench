"""
Step 15: PAINS-Adversarial Debiasing (Model-Level Mitigation).
Uses Gradient Reversal Layer (GRL) to force drug encoder
to learn PAINS-agnostic representations.
Tests GCN-DTA with GRL strengths λ = 0.1, 0.5, 1.0.
"""
import os, sys, time, gc, warnings, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED

warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

BATCH_SIZE = 128; EPOCHS = 30; LR = 1e-3; WEIGHT_DECAY = 1e-5
PROTEIN_MAX_LEN = 1200; DRUG_EMBED_DIM = 128; PROTEIN_EMBED_DIM = 128

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"; AA_TO_IDX = {aa: i+1 for i, aa in enumerate(AA_ORDER)}

def tokenize_sequence(seq, max_len=PROTEIN_MAX_LEN):
    tokens = [AA_TO_IDX.get(aa, len(AA_ORDER)+1) for aa in seq[:max_len]]
    if len(tokens) < max_len: tokens += [0]*(max_len-len(tokens))
    return np.array(tokens, dtype=np.int64)

# ====== Load data ======
print("Loading benchmark...", flush=True)
bench = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_dta_full.csv"))
bench_full = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_full.csv"))
bench_full["_orig_idx"] = np.arange(len(bench_full))
sp = bench.merge(bench_full[["molregno","target_chembl_id","_orig_idx"]],
                 on=["molregno","target_chembl_id"],how="left")["_orig_idx"].values
data = np.load(os.path.join(PROCESSED_DIR, "features_full.npz"), allow_pickle=True)
X_fp = data["X"][sp,:2048]; y = data["y"][sp].astype(np.float32); ps = data["pains_status"][sp]
smiles_list = bench["canonical_smiles"].values; sequences = bench["sequence"].values
protein_tokens = np.array([tokenize_sequence(s) for s in sequences], dtype=np.int64)

from sklearn.model_selection import train_test_split
all_idx = np.arange(len(y))
train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=RANDOM_SEED, stratify=ps)
train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=RANDOM_SEED, stratify=ps[train_idx])
print(f"Train {len(train_idx)} Val {len(val_idx)} Test {len(test_idx)}", flush=True)

# ====== Graphs ======
CACHE_PATH = os.path.join(PROCESSED_DIR, "graph_cache.pkl")
print("Loading graphs...", flush=True)
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f: graph_cache = pickle.load(f)
    print(f"  Loaded {len(graph_cache)} graphs", flush=True)
else:
    graph_cache = {}

from rdkit import Chem
from torch_geometric.data import Data
ATOM_TYPES = [5,6,7,8,9,15,16,17,35,53]; BOND_TYPES = [1,2,3,12]

def mol_to_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    atom_types = [a.GetAtomicNum() for a in mol.GetAtoms()]
    x = torch.zeros(len(atom_types), len(ATOM_TYPES)+3)
    for i, at in enumerate(atom_types):
        if at in ATOM_TYPES: x[i, ATOM_TYPES.index(at)] = 1
        else: x[i,-3]=1
        x[i,-2]=mol.GetAtomWithIdx(i).GetDegree()/4.0
        x[i,-1]=mol.GetAtomWithIdx(i).GetTotalNumHs()/3.0
    ei, ea = [], []
    for b in mol.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); ei+=[[i,j],[j,i]]
        bt=b.GetBondTypeAsDouble(); feat=[1.0 if bt==b else 0.0 for b in BOND_TYPES]; ea+=[feat,feat]
    if not ei: ei=[[0,0]]; ea=[[1.0,0,0,0]]
    return Data(x=x, edge_index=torch.tensor(ei,dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(ea,dtype=torch.float))

def get_graphs(indices):
    graphs, valid = [], []
    for i, idx in enumerate(indices):
        smi=str(smiles_list[idx])
        g=graph_cache.get(smi) or mol_to_graph(smi)
        if g is not None:
            graph_cache[smi]=g; graphs.append(g); valid.append(idx)
        if (i+1)%20000==0: print(f"  {i+1}/{len(indices)}", flush=True)
    return graphs, np.array(valid,dtype=int)

train_graphs,train_v=get_graphs(train_idx)
val_graphs,val_v=get_graphs(val_idx)
test_graphs,test_v=get_graphs(test_idx)
print(f"Graphs: {len(train_graphs)}/{len(val_graphs)}/{len(test_graphs)}", flush=True)

if not os.path.exists(CACHE_PATH):
    with open(CACHE_PATH,"wb") as f: pickle.dump(graph_cache,f)
    print(f"Cache saved ({len(graph_cache)} graphs)", flush=True)

# Align
y_train_g = torch.tensor(y[train_v],dtype=torch.float)
y_val_g = torch.tensor(y[val_v],dtype=torch.float)
y_test_g = torch.tensor(y[test_v],dtype=torch.float)
ps_train_g = torch.tensor(ps[train_v],dtype=torch.long)
ps_test_g = ps[test_v]
seq_train = torch.tensor(protein_tokens[train_v])
seq_val = torch.tensor(protein_tokens[val_v])
seq_test = torch.tensor(protein_tokens[test_v])
for g,lbl in zip(train_graphs,y_train_g): g.y=lbl
for g,lbl in zip(val_graphs,y_val_g): g.y=lbl
for g,lbl in zip(test_graphs,y_test_g): g.y=lbl

# ====== Dataset ======
class DTADataset(Dataset):
    def __init__(self, graphs, seqs, labels, ps=None):
        self.graphs, self.seqs, self.labels, self.ps = graphs, seqs, labels, ps
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx):
        if self.ps is not None: return self.graphs[idx],self.seqs[idx],self.labels[idx],self.ps[idx]
        return self.graphs[idx],self.seqs[idx],self.labels[idx]

def collate(batch):
    from torch_geometric.data import Batch
    has_ps = len(batch[0])==4
    g, s, l = zip(*[(b[0],b[1],b[2]) for b in batch])
    bg, st, lt = Batch.from_data_list(list(g)), torch.stack(list(s)), torch.stack(list(l))
    if has_ps: return bg,st,lt,torch.tensor([b[3] for b in batch],dtype=torch.long)
    return bg,st,lt

tr_ds = DTADataset(train_graphs, seq_train, y_train_g, ps_train_g)
va_ds = DTADataset(val_graphs, seq_val, y_val_g, torch.tensor(ps[val_v],dtype=torch.long))
te_ds = DTADataset(test_graphs, seq_test, y_test_g)
tr_loader = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, collate_fn=collate)
va_loader = DataLoader(va_ds, BATCH_SIZE, collate_fn=collate)
te_loader = DataLoader(te_ds, BATCH_SIZE, collate_fn=collate)

# ====== Gradient Reversal Layer ======
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)

# ====== Models ======
class ProteinCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(25,32,padding_idx=0)
        self.c1=nn.Conv1d(32,128,5,padding=2); self.b1=nn.BatchNorm1d(128)
        self.c2=nn.Conv1d(128,128,5,padding=2); self.b2=nn.BatchNorm1d(128)
        self.c3=nn.Conv1d(128,128,5,padding=2); self.b3=nn.BatchNorm1d(128)
        self.pool=nn.AdaptiveMaxPool1d(1); self.drop=nn.Dropout(0.2)
    def forward(self, seq):
        x=self.emb(seq).permute(0,2,1)
        x=self.drop(F.relu(self.b1(self.c1(x))))
        x=self.drop(F.relu(self.b2(self.c2(x))))
        x=F.relu(self.b3(self.c3(x)))
        return self.pool(x).squeeze(-1)

class GCNDrug(nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
        self.c1=GCNConv(len(ATOM_TYPES)+3,128); self.c2=GCNConv(128,128); self.c3=GCNConv(128,64)
        self.proj=nn.Linear(64,128); self.pool=global_mean_pool
    def forward(self, data):
        x=F.relu(self.c1(data.x,data.edge_index))
        x=F.relu(self.c2(x,data.edge_index))
        x=F.relu(self.c3(x,data.edge_index))
        return self.proj(self.pool(x,data.batch))

class AdvDTAModel(nn.Module):
    """DTA model with adversarial PAINS classifier on drug embedding."""
    def __init__(self, lambda_=0.5):
        super().__init__()
        self.drug = GCNDrug()
        self.protein = ProteinCNN()
        self.regressor = nn.Sequential(
            nn.Linear(256,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,1))
        # PAINS adversary on drug embedding
        self.adversary = nn.Sequential(
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,2))
        self.lambda_ = lambda_

    def forward(self, drug_input, seq, alpha=1.0):
        d = self.drug(drug_input)
        p = self.protein(seq)

        # PAINS adversary with gradient reversal
        d_rev = grad_reverse(d, self.lambda_ * alpha)
        pains_logits = self.adversary(d_rev)

        # Regression
        fused = torch.cat([d, p], 1)
        pred = self.regressor(fused).squeeze(-1)
        return pred, pains_logits

# ====== Training ======
def train_adv(model, loader, val_loader, lambda_=0.5, epochs=EPOCHS):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val_loss = float("inf"); best_state = None

    for epoch in range(epochs):
        # Progressively increase GRL strength (warmup)
        p = epoch / epochs  # 0→1
        alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0  # sigmoid ramp 0→1

        model.train(); tr_loss, n = 0.0, 0
        for batch in tr_loader:
            bg, seq, yb, psb = [x.to(device) for x in batch]
            opt.zero_grad()
            pred, logits = model(bg, seq, alpha)

            # Regression loss (MSE)
            reg_loss = F.mse_loss(pred, yb)

            # Adversarial loss (cross-entropy: predict PAINS status)
            adv_loss = F.cross_entropy(logits, psb)

            loss = reg_loss + 0.5 * adv_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += reg_loss.item()*len(yb); n += len(yb)
        sched.step()

        model.eval(); val_loss, nv = 0.0, 0
        with torch.no_grad():
            for batch in va_loader:
                bg, seq, yb, _ = [x.to(device) for x in batch]
                pred, _ = model(bg, seq, 0.0)  # eval: no adversary
                loss = F.mse_loss(pred, yb)
                val_loss += loss.item()*len(yb); nv += len(yb)
        val_loss /= nv

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        if (epoch+1)%10==0:
            print(f"  Ep {epoch+1:2d} tr={tr_loss/n:.4f} val={val_loss:.4f}", flush=True)

    model.load_state_dict(best_state)
    return model

@torch.no_grad()
def predict(model, loader):
    model.eval(); preds = []
    for batch in loader:
        bg, seq = batch[0].to(device), batch[1].to(device)
        pred, _ = model(bg, seq, 0.0)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds)

# ====== Run ======
from src.evaluation import evaluate_pains_aware
results = []

for lambda_ in [0.0, 0.1, 0.5, 1.0]:
    label = f"GCN-DTA_adv{lambda_:.1f}" if lambda_ > 0 else "GCN-DTA_baseline_adv"
    print(f"\n{'='*50}\n{label} (λ={lambda_})\n{'='*50}", flush=True)
    gc.collect(); torch.cuda.empty_cache()
    t0 = time.time()
    model = AdvDTAModel(lambda_=lambda_)
    model = train_adv(model, tr_loader, va_loader, lambda_=lambda_)
    y_pred = predict(model, te_loader)
    elapsed = time.time() - t0
    eval_d = evaluate_pains_aware(y_test_g.numpy(), y_pred, ps_test_g)
    delta = eval_d['delta_rmse']
    print(f"  RMSE={eval_d['overall_RMSE']:.4f} P+={eval_d['pains_pos_RMSE']:.4f} P-={eval_d['pains_neg_RMSE']:.4f} ΔRMSE={delta:.4f} FP={eval_d['fp_ratio']:.4f} [{elapsed:.0f}s]", flush=True)
    results.append({"model":"GCN-DTA","strategy":f"adversarial_lambda{lambda_}","overall_RMSE":eval_d["overall_RMSE"],"pains_pos_RMSE":eval_d["pains_pos_RMSE"],"pains_neg_RMSE":eval_d["pains_neg_RMSE"],"delta_rmse":delta,"fp_ratio":eval_d["fp_ratio"],"train_time_s":elapsed})
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"mitigation_adv_results.csv"),index=False)
    np.save(os.path.join(RESULTS_DIR,f"preds_adv_lambda{lambda_:.0f}.npy"),y_pred)

print("\nDone.", flush=True)
