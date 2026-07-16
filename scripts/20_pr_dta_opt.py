"""
PR-DTA: Hyperparameter optimization + ablation study.
Runs: baseline, PR-DTA (filtered), PR-DTA (unfiltered), PR-DTA (gate_alpha=0.5)
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED
from src.evaluation import evaluate_pains_aware

warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

BATCH_SIZE = 128; EPOCHS = 30; LR = 1e-3; WEIGHT_DECAY = 1e-5
PROTEIN_MAX_LEN = 1200; DRUG_DIM = 128; PROT_DIM = 128
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"; AA_TO_IDX = {aa:i+1 for i,aa in enumerate(AA_ORDER)}
AT = [5,6,7,8,9,15,16,17,35,53]; BT = [1,2,3,12]

def tokenize(s, ml=PROTEIN_MAX_LEN):
    t=np.zeros(ml,dtype=np.int32); n=min(len(s),ml)
    for i in range(n): t[i]=AA_TO_IDX.get(s[i],len(AA_ORDER)+1)
    return t

print("="*60+"\n[1] Loading data\n"+"="*60, flush=True)
b=pd.read_csv(os.path.join(PROCESSED_DIR,"benchmark_dta_full.csv"))
bf=pd.read_csv(os.path.join(PROCESSED_DIR,"benchmark_full.csv"))
bf["_oi"]=np.arange(len(bf))
sp=b.merge(bf[["molregno","target_chembl_id","_oi"]],on=["molregno","target_chembl_id"],how="left")["_oi"].values
d=np.load(os.path.join(PROCESSED_DIR,"features_full.npz"),allow_pickle=True)
y_label=d["y"][sp].astype(np.float32); ps=d["pains_status"][sp]
sl=b["canonical_smiles"].values
seqs=b["sequence"].values
prot_tok=np.array([tokenize(s) for s in seqs],dtype=np.int32)
del seqs
print(f"  Total: {len(y_label)}, PAINS+: {ps.mean():.3f} ({int(ps.sum())})",flush=True)

ai=np.arange(len(y_label)); ti,te=train_test_split(ai,test_size=0.2,random_state=RANDOM_SEED,stratify=ps)
ti,vi=train_test_split(ti,test_size=0.125,random_state=RANDOM_SEED,stratify=ps[ti])
print(f"  Train: {len(ti)}, Val: {len(vi)}, Test: {len(te)}",flush=True)

print("Loading caches...",flush=True)
pm=np.load(os.path.join(PROCESSED_DIR,"pains_masks_puda.npz"))
pains_masks_all=pm["masks"]

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool

CACHE_PATH=os.path.join(PROCESSED_DIR,"graph_cache.pkl")
with open(CACHE_PATH,"rb") as f: gcache=pickle.load(f)
print(f"  Graphs cached: {len(gcache)}",flush=True)

def gg(idx):
    gs,va=[],[]
    for i,ix in enumerate(tqdm(idx,desc="Graphs",ncols=80)):
        sm=str(sl[ix]); g=gcache.get(sm)
        if g is not None: gs.append(g); va.append(ix)
    return gs,np.array(va,dtype=int)

tg,tv=gg(ti); vg,vv=gg(vi); sg,sv=gg(te)
print(f"  {len(tg)}/{len(vg)}/{len(sg)}",flush=True)
del gcache; gc.collect()

yt=torch.tensor(y_label[tv],dtype=torch.float)
yv_=torch.tensor(y_label[vv],dtype=torch.float)
ye=torch.tensor(y_label[sv],dtype=torch.float); pse=ps[sv].copy()
st_=torch.tensor(prot_tok[tv]); sv_=torch.tensor(prot_tok[vv]); se_=torch.tensor(prot_tok[sv])
pm_train=torch.tensor(pains_masks_all[tv],dtype=torch.float)
pm_val=torch.tensor(pains_masks_all[vv],dtype=torch.float); pm_test=torch.tensor(pains_masks_all[sv],dtype=torch.float)
ratio_train=pm_train.mean(dim=1); ratio_val=pm_val.mean(dim=1); ratio_test=pm_test.mean(dim=1)
del prot_tok, pains_masks_all, y_label, sl
gc.collect()

for g,l in zip(tg,yt): g.y=l
for g,l in zip(vg,yv_): g.y=l
for g,l in zip(sg,ye): g.y=l

# Build datasets with/without PAINS filtering
class DS(Dataset):
    def __init__(self,g,s,y,pm,pr):
        self.g=g;self.s=s;self.y=y;self.pm=pm;self.pr=pr
    def __len__(self):return len(self.g)
    def __getitem__(self,i):return self.g[i],self.s[i],self.y[i],self.pm[i],self.pr[i]

def cl(b):
    g,s,y,pm,pr=zip(*b)
    return Batch.from_data_list(list(g)),torch.stack(list(s)),torch.stack(list(y)),torch.stack(list(pm)),torch.stack(list(pr))

# Filtered training set (no PAINS+)
tm=ps[tv]==0; ti_f=tv[tm]
# Re-map for filtered: restrict tg, yt, etc.
tf_mask=tm
tf_tg=[tg[i] for i in range(len(tg)) if tf_mask[i]]
tf_yt=yt[tf_mask]
tf_st=st_[tf_mask]
tf_pm=pm_train[tf_mask]
tf_pr=ratio_train[tf_mask]
for g,l in zip(tf_tg,tf_yt): g.y=l

tr_filt=DataLoader(DS(tf_tg,tf_st,tf_yt,tf_pm,tf_pr),BATCH_SIZE,shuffle=True,collate_fn=cl)
tr_full=DataLoader(DS(tg,st_,yt,pm_train,ratio_train),BATCH_SIZE,shuffle=True,collate_fn=cl)
va=DataLoader(DS(vg,sv_,yv_,pm_val,ratio_val),BATCH_SIZE,collate_fn=cl)
te=DataLoader(DS(sg,se_,ye,pm_test,ratio_test),BATCH_SIZE,collate_fn=cl)
print(f"  Filtered train: {len(tf_tg)}, Full train: {len(tg)}",flush=True)

# ===== MODELS =====
class GINEncoder(nn.Module):
    def __init__(self,in_dim=len(AT)+3,hid=128,out=DRUG_DIM):
        super().__init__()
        n1=nn.Sequential(nn.Linear(in_dim,hid),nn.BatchNorm1d(hid),nn.ReLU(),nn.Linear(hid,hid))
        self.c1=GINConv(n1)
        n2=nn.Sequential(nn.Linear(hid,hid),nn.BatchNorm1d(hid),nn.ReLU(),nn.Linear(hid,hid))
        self.c2=GINConv(n2)
        n3=nn.Sequential(nn.Linear(hid,out),nn.BatchNorm1d(out),nn.ReLU(),nn.Linear(out,out))
        self.c3=GINConv(n3)
        self.pool=global_mean_pool
    def forward(self,data):
        x=F.relu(self.c1(data.x,data.edge_index))
        x=F.relu(self.c2(x,data.edge_index))
        x=F.relu(self.c3(x,data.edge_index))
        return self.pool(x,data.batch)

class PAINSGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(129,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1),nn.Sigmoid())
    def forward(self,mask,ratio):
        return self.net(torch.cat([mask.float(),ratio.unsqueeze(-1).float()],dim=1)).squeeze(-1)

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

class GCNDTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=GCNConv(len(AT)+3,128);self.c2=GCNConv(128,128);self.c3=GCNConv(128,DRUG_DIM)
        self.pool=global_mean_pool; self.pe=ProteinCNN()
        self.head=nn.Sequential(nn.Linear(DRUG_DIM+PROT_DIM,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,bg,seq):
        x=F.relu(self.c1(bg.x,bg.edge_index));x=F.relu(self.c2(x,bg.edge_index));x=F.relu(self.c3(x,bg.edge_index))
        return self.head(torch.cat([self.pool(x,bg.batch),self.pe(seq)],1)).squeeze(-1)

class PRDTA(nn.Module):
    def __init__(self, alpha=1.0):
        """alpha controls gate strength: h_drug = h_main * (1 - alpha * s_pains)"""
        super().__init__()
        self.alpha = alpha
        self.gin=GINEncoder()
        self.pains_gate=PAINSGate()
        self.prot_cnn=ProteinCNN()
        self.head=nn.Sequential(nn.Linear(DRUG_DIM+PROT_DIM,128),nn.ReLU(),nn.Dropout(0.2),
                                nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.1),nn.Linear(64,1))
    def forward(self,bg,seq,mask,ratio):
        h_main=self.gin(bg)
        s_pains=self.pains_gate(mask,ratio)
        h_drug=h_main*(1-self.alpha*s_pains.unsqueeze(-1))
        h_prot=self.prot_cnn(seq)
        return self.head(torch.cat([h_drug,h_prot],dim=1)).squeeze(-1)

# ===== TRAINING =====
def run_epoch(model,loader,opt=None):
    is_train=opt is not None; model.train() if is_train else model.eval()
    tl,n=0.0,0; desc="  train" if is_train else "  val"
    pb=tqdm(loader,desc=desc,ncols=80,leave=False)
    with torch.set_grad_enabled(is_train):
        for batch in pb:
            bg,seq,yb,mask,ratio=[x.to(device) for x in batch[:5]]
            if is_train: opt.zero_grad()
            pred=model(bg,seq,mask,ratio)
            loss=F.mse_loss(pred,yb)
            if is_train:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            bs=len(yb); tl+=loss.item()*bs; n+=bs
            pb.set_postfix({"loss":f"{loss.item():.4f}"})
    return tl/n

def run_base(model,loader,opt=None):
    is_train=opt is not None; model.train() if is_train else model.eval()
    tl,n=0.0,0
    pb=tqdm(loader,desc="  train" if is_train else "  val",ncols=80,leave=False)
    with torch.set_grad_enabled(is_train):
        for batch in pb:
            bg,seq,yb=[x.to(device) for x in batch[:3]]
            if is_train: opt.zero_grad()
            pred=model(bg,seq); loss=F.mse_loss(pred,yb)
            if is_train:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            bs=len(yb); tl+=loss.item()*bs; n+=bs
            pb.set_postfix({"loss":f"{loss.item():.4f}"})
    return tl/n

def train_base(model,tr,vl_loader):
    model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    bv=float("inf"); bs=None
    ep_pb=tqdm(range(1,EPOCHS+1),desc="GCN-DTA",ncols=95)
    for ep in ep_pb:
        run_base(model,tr,opt); sched.step()
        va_l=run_base(model,vl_loader)
        if va_l<bv: bv=va_l; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}
        ep_pb.set_postfix({"val":f"{va_l:.4f}"})
    model.load_state_dict(bs); return model

def train_prdta(model,tr,vl):
    model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    bv=float("inf"); bs=None
    ep_pb=tqdm(range(1,EPOCHS+1),desc="PR-DTA",ncols=95)
    for ep in ep_pb:
        tr_l=run_epoch(model,tr,opt); sched.step()
        va_l=run_epoch(model,vl)
        if va_l<bv: bv=va_l; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}
        ep_pb.set_postfix({"tr":f"{tr_l:.4f}","val":f"{va_l:.4f}"})
    model.load_state_dict(bs); return model

@torch.no_grad()
def predict(model,loader):
    model.eval(); pr=[]
    for batch in tqdm(loader,desc="  predict",ncols=80,leave=False):
        bg,seq,_,mask,ratio=[x.to(device) for x in batch[:5]]
        pr.append(model(bg,seq,mask,ratio).cpu().numpy())
    return np.concatenate(pr)

@torch.no_grad()
def predict_base(model,loader):
    model.eval(); pr=[]
    for batch in tqdm(loader,desc="  predict",ncols=80,leave=False):
        bg,seq=[x.to(device) for x in batch[:2]]
        pr.append(model(bg,seq).cpu().numpy())
    return np.concatenate(pr)

# ===== RUN ALL VARIANTS =====
results=[]

def run_and_record(name, model, train_fn, loader_tr, loader_va, loader_te):
    gc.collect(); torch.cuda.empty_cache()
    t0=time.time()
    m=train_fn(model,loader_tr,loader_va)
    yp=predict(m,loader_te) if hasattr(m,'pains_gate') else predict_base(m,loader_te)
    et=time.time()-t0
    ed=evaluate_pains_aware(ye.numpy(),yp,pse)
    print(f"  >> RMSE={ed['overall_RMSE']:.4f} P+={ed['pains_pos_RMSE']:.4f} P-={ed['pains_neg_RMSE']:.4f} DR={ed['delta_rmse']:.4f} FP={ed['fp_ratio']:.4f} [{et:.0f}s]",flush=True)
    results.append({"v":name,"RMSE":ed["overall_RMSE"],"P+":ed["pains_pos_RMSE"],"P-":ed["pains_neg_RMSE"],"Δ":ed["delta_rmse"],"FP":ed["fp_ratio"],"t":et})
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"prdta_opt_results.csv"),index=False)
    # Aggressive cleanup
    del m, yp; gc.collect(); torch.cuda.empty_cache()

# 1. GCN-DTA Baseline (full training, no gate)
print("\n--- 1. GCN-DTA Baseline ---",flush=True)
run_and_record("GCN-DTA_baseline",GCNDTA(),train_base,tr_full,va,te)

# 2. PR-DTA with filtered training + gate α=1.0
print("\n--- 2. PR-DTA (filtered, α=1.0) ---",flush=True)
run_and_record("PR-DTA_filt_a1.0",PRDTA(alpha=1.0),train_prdta,tr_filt,va,te)

# 3. PR-DTA with full training + gate α=1.0
print("\n--- 3. PR-DTA (unfiltered, α=1.0) ---",flush=True)
run_and_record("PR-DTA_full_a1.0",PRDTA(alpha=1.0),train_prdta,tr_full,va,te)

# 4. PR-DTA with full training + gate α=0.5 (weaker gate)
print("\n--- 4. PR-DTA (unfiltered, α=0.5) ---",flush=True)
run_and_record("PR-DTA_full_a0.5",PRDTA(alpha=0.5),train_prdta,tr_full,va,te)

# 5. PR-DTA with full training + gate α=2.0 (stronger gate)
print("\n--- 5. PR-DTA (unfiltered, α=2.0) ---",flush=True)
run_and_record("PR-DTA_full_a2.0",PRDTA(alpha=2.0),train_prdta,tr_full,va,te)

# Summary
rf=pd.DataFrame(results)
print(f"\n{'='*75}")
print(f"{'Variant':25s} {'RMSE':>7s} {'P+':>7s} {'P-':>7s} {'ΔRMSE':>7s} {'FP':>7s} {'Time':>6s}")
print("-"*75)
bl_rmse=rf.loc[0,"RMSE"]
for _,r in rf.iterrows():
    imp=(bl_rmse-r["RMSE"])/bl_rmse*100
    print(f"{r['v']:25s} {r['RMSE']:7.4f} {r['P+']:7.4f} {r['P-']:7.4f} {r['Δ']:7.4f} {r['FP']:7.4f} {r['t']:5.0f}s  {imp:+5.1f}%")
print(f"\nBest variant: {rf.loc[rf['RMSE'].idxmin(),'v']} (RMSE={rf['RMSE'].min():.4f})")
print("Done.",flush=True)
