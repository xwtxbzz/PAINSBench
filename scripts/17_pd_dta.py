"""
PD-DTA: PAINS-Debiased Drug-Target Affinity Prediction.
Adversarial debiasing via Gradient Reversal Layer on drug embeddings.
Core idea: force drug encoder to learn PAINS-agnostic representations.
"""
import os, sys, time, gc, warnings, pickle, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED
warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

BATCH_SIZE = 128; EPOCHS = 30; LR = 1e-3; WEIGHT_DECAY = 1e-5
PROTEIN_MAX_LEN = 1200; DRUG_DIM = 128; PROT_DIM = 128
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"; AA_TO_IDX = {aa:i+1 for i,aa in enumerate(AA_ORDER)}

def tokenize(s, ml=PROTEIN_MAX_LEN):
    t=[AA_TO_IDX.get(a,len(AA_ORDER)+1) for a in s[:ml]]
    t+=[0]*(ml-len(t)); return np.array(t,dtype=np.int64)

# ============== DATA ==============
print("Loading...", flush=True)
b=pd.read_csv(os.path.join(PROCESSED_DIR,"benchmark_dta_full.csv"))
bf=pd.read_csv(os.path.join(PROCESSED_DIR,"benchmark_full.csv"))
bf["_oi"]=np.arange(len(bf))
sp=b.merge(bf[["molregno","target_chembl_id","_oi"]],on=["molregno","target_chembl_id"],how="left")["_oi"].values
d=np.load(os.path.join(PROCESSED_DIR,"features_full.npz"),allow_pickle=True)
y=d["y"][sp].astype(np.float32); ps=d["pains_status"][sp]
sl=b["canonical_smiles"].values; seqs=b["sequence"].values
prot_tok=np.array([tokenize(s) for s in seqs],dtype=np.int64)

from sklearn.model_selection import train_test_split
ai=np.arange(len(y)); ti,te=train_test_split(ai,test_size=0.2,random_state=RANDOM_SEED,stratify=ps)
ti,vi=train_test_split(ti,test_size=0.125,random_state=RANDOM_SEED,stratify=ps[ti])
print(f"Train {len(ti)} Val {len(vi)} Test {len(te)}",flush=True)

# Graphs
CACHE_PATH=os.path.join(PROCESSED_DIR,"graph_cache.pkl")
print("Graphs...",flush=True)
gcache={}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH,"rb") as f: gcache=pickle.load(f)
    print(f"  Cached {len(gcache)}",flush=True)

from rdkit import Chem
from torch_geometric.data import Data
AT=([5,6,7,8,9,15,16,17,35,53]); BT=[1,2,3,12]

def m2g(smi):
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

def gg(idx):
    gs,va=[],[]
    for i,ix in enumerate(idx):
        sm=str(sl[ix])
        g=gcache.get(sm) or m2g(sm)
        if g is not None: gcache[sm]=g; gs.append(g); va.append(ix)
        if(i+1)%20000==0: print(f"  {i+1}/{len(idx)}",flush=True)
    return gs,np.array(va,dtype=int)

tg,tv=gg(ti); vg,vv=gg(vi); sg,sv=gg(te)
print(f"Graphs: {len(tg)}/{len(vg)}/{len(sg)}",flush=True)
if not os.path.exists(CACHE_PATH):
    with open(CACHE_PATH,"wb") as f: pickle.dump(gcache,f)

# Align
yt=torch.tensor(y[tv],dtype=torch.float); yv_=torch.tensor(y[vv],dtype=torch.float); ye=torch.tensor(y[sv],dtype=torch.float)
pt_=torch.tensor(ps[tv],dtype=torch.float); pv_=torch.tensor(ps[vv],dtype=torch.float); pse=ps[sv]
st_=torch.tensor(prot_tok[tv]); sv_=torch.tensor(prot_tok[vv]); se_=torch.tensor(prot_tok[sv])
for g,l in zip(tg,yt): g.y=l
for g,l in zip(vg,yv_): g.y=l
for g,l in zip(sg,ye): g.y=l

# Dataset
class DS(Dataset):
    def __init__(self,g,s,y,ps=None): self.g=g;self.s=s;self.y=y;self.ps=ps
    def __len__(self): return len(self.g)
    def __getitem__(self,i):
        if self.ps is not None: return self.g[i],self.s[i],self.y[i],self.ps[i]
        return self.g[i],self.s[i],self.y[i]
def cl(b):
    from torch_geometric.data import Batch
    hp=len(b[0])==4
    g,s,l=zip(*[(x[0],x[1],x[2]) for x in b])
    bg=Batch.from_data_list(list(g)); st=torch.stack(list(s)); lt=torch.stack(list(l))
    if hp: return bg,st,lt,torch.tensor([x[3] for x in b],dtype=torch.float)
    return bg,st,lt

tr=DataLoader(DS(tg,st_,yt,pt_),BATCH_SIZE,shuffle=True,collate_fn=cl)
va=DataLoader(DS(vg,sv_,yv_,pv_),BATCH_SIZE,collate_fn=cl)
te=DataLoader(DS(sg,se_,ye),BATCH_SIZE,collate_fn=cl)

# ============== MODELS ==============
# Protein CNN (3-layer, paper baseline)
class ProteinCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(25,32,padding_idx=0)
        self.c1=nn.Conv1d(32,64,5,padding=2); self.b1=nn.BatchNorm1d(64)
        self.c2=nn.Conv1d(64,128,5,padding=2); self.b2=nn.BatchNorm1d(128)
        self.c3=nn.Conv1d(128,PROT_DIM,5,padding=2); self.b3=nn.BatchNorm1d(PROT_DIM)
        self.pool=nn.AdaptiveMaxPool1d(1); self.do=nn.Dropout(0.2)
    def forward(self,s):
        x=self.emb(s).permute(0,2,1)
        x=self.do(F.relu(self.b1(self.c1(x))))
        x=self.do(F.relu(self.b2(self.c2(x))))
        x=F.relu(self.b3(self.c3(x)))
        return self.pool(x).squeeze(-1)

# Drug GCN encoder (output dim=DRUG_DIM, proven efficient from paper baselines)
class GCNDrug(nn.Module):
    """3-layer GCN producing DRUG_DIM-dim output."""
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
        self.c1=GCNConv(len(AT)+3,128); self.c2=GCNConv(128,128); self.c3=GCNConv(128,DRUG_DIM)
        self.pool=global_mean_pool
    def forward(self,data):
        x=F.relu(self.c1(data.x,data.edge_index))
        x=F.relu(self.c2(x,data.edge_index))
        x=F.relu(self.c3(x,data.edge_index))
        return self.pool(x,data.batch)

# Gradient Reversal Layer
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,alpha): ctx.alpha=alpha; return x
    @staticmethod
    def backward(ctx,go): return -ctx.alpha*go, None
def gr(x,alpha): return GradReverse.apply(x,alpha)

# PD-DTA: PAINS-Debiased DTA
class PDDTA(nn.Module):
    """PAINS-Debiased DTA with adversarial training.
    - Drug encoder → h_drug (256)
    - Protein encoder → h_prot (256)
    - Fusion: concat(h_drug, h_prot) → affinity prediction
    - Adversarial branch: h_drug → GRL → PAINS classifier
    """
    def __init__(self, beta=0.5):
        super().__init__()
        self.drug_enc = GCNDrug()
        self.prot_enc = ProteinCNN()
        self.fusion = nn.Sequential(
            nn.Linear(DRUG_DIM+PROT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))
        self.pains_cls = nn.Sequential(
            nn.Linear(DRUG_DIM, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid())
        self.beta = beta  # adversarial loss weight
        self.alpha = 0.0  # GRL strength (ramped up during training)

    def forward(self, drug_graph, prot_seq):
        h_drug = self.drug_enc(drug_graph)     # (B, 256)
        h_prot = self.prot_enc(prot_seq)        # (B, 256)
        h_fusion = torch.cat([h_drug, h_prot], dim=1)
        y_pred = self.fusion(h_fusion).squeeze(-1)

        # Adversarial branch with GRL
        h_rev = gr(h_drug, self.alpha)
        p_pains = self.pains_cls(h_rev).squeeze(-1)  # (B,)

        return y_pred, p_pains

# ============== TRAINING ==============
from src.evaluation import evaluate_pains_aware

def train_pddta(model, loader, val_loader, beta=0.5, epochs=EPOCHS):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val = float("inf"); best_state = None
    warmup_epochs = epochs // 3  # warmup over first third of training

    for ep in range(epochs):
        # Ramp alpha from 0→1
        model.alpha = min(1.0, ep / max(warmup_epochs, 1))

        model.train(); tl, n = 0.0, 0
        for batch in loader:
            bg, seq, yb, psb = [x.to(device) for x in batch]
            opt.zero_grad()
            y_pred, p_pains = model(bg, seq)
            loss_mse = F.mse_loss(y_pred, yb)
            loss_adv = F.binary_cross_entropy(p_pains, psb)
            loss = loss_mse + beta * loss_adv
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss_mse.item()*len(yb); n += len(yb)
        sch.step()

        model.eval(); vl, nv = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                bg, seq, yb, _ = [x.to(device) for x in batch]
                y_pred, _ = model(bg, seq)
                vl += F.mse_loss(y_pred, yb).item()*len(yb); nv += len(yb)
        vl /= nv
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep+1) % 10 == 0:
            print(f"  Ep {ep+1:2d} α={model.alpha:.2f} tr={tl/n:.4f} val={vl:.4f}", flush=True)

    model.load_state_dict(best_state)
    return model

@torch.no_grad()
def predict(model, loader):
    model.eval(); pr = []
    for batch in loader:
        bg, seq = batch[0].to(device), batch[1].to(device)
        y_pred, _ = model(bg, seq)
        pr.append(y_pred.cpu().numpy())
    return np.concatenate(pr)

# ============== RUN ==============
results = []
from src.evaluation import evaluate_pains_aware

# Baseline: GCN-DTA without adversarial
print("\n=== GCN-DTA Baseline ===", flush=True)
gc.collect(); torch.cuda.empty_cache(); t0 = time.time()
m_base = PDDTA(beta=0.0)  # beta=0 disables adversarial branch
m_base = train_pddta(m_base, tr, va, beta=0.0)
yp = predict(m_base, te); et = time.time()-t0
ed = evaluate_pains_aware(ye.numpy(), yp, pse)
print(f"  RMSE={ed['overall_RMSE']:.4f} ΔRMSE={ed['delta_rmse']:.4f} FP={ed['fp_ratio']:.4f} [{et:.0f}s]", flush=True)
results.append({"model":"GCN-DTA","variant":"baseline","overall_RMSE":ed["overall_RMSE"],"delta_rmse":ed["delta_rmse"],"fp_ratio":ed["fp_ratio"]})
np.save(os.path.join(RESULTS_DIR,"pd-dta_baseline.npy"),yp)
pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"pd-dta_results.csv"),index=False)

# PD-DTA with different beta values
for beta in [0.1, 0.3, 0.5, 1.0]:
    print(f"\n=== PD-DTA (β={beta}) ===", flush=True)
    gc.collect(); torch.cuda.empty_cache(); t0 = time.time()
    m = PDDTA(beta=beta)
    m = train_pddta(m, tr, va, beta=beta)
    yp = predict(m, te); et = time.time()-t0
    ed = evaluate_pains_aware(ye.numpy(), yp, pse)
    print(f"  RMSE={ed['overall_RMSE']:.4f} ΔRMSE={ed['delta_rmse']:.4f} FP={ed['fp_ratio']:.4f} [{et:.0f}s]", flush=True)
    results.append({"model":"GCN-DTA","variant":f"PD-DTA_β{beta}","overall_RMSE":ed["overall_RMSE"],"delta_rmse":ed["delta_rmse"],"fp_ratio":ed["fp_ratio"]})
    np.save(os.path.join(RESULTS_DIR,f"pd-dta_beta{beta:.0f}.npy"),yp)
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"pd-dta_results.csv"),index=False)

# Summary
rf = pd.DataFrame(results)
print(f"\n{'='*60}")
print("PD-DTA Results")
print(f"{'='*60}")
print(f"{'Variant':20s} {'RMSE':>8s} {'ΔRMSE':>8s} {'FP':>8s} {'Improve?':>10s}")
print("-"*60)
bl_d = rf.loc[0, 'delta_rmse']
for _, r in rf.iterrows():
    better = abs(r['delta_rmse']) < abs(bl_d)
    imp = "✓" if better else "✗"
    print(f"{r['variant']:20s} {r['overall_RMSE']:8.4f} {r['delta_rmse']:8.4f} {r['fp_ratio']:8.4f} {imp:>10s}")
print("\nDone.", flush=True)
