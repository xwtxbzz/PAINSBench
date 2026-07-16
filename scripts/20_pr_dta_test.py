"""
PR-DTA: quick test on 600-sample subset.
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
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

BATCH_SIZE = 32; EPOCHS = 5; LR = 1e-3; WEIGHT_DECAY = 1e-5
PROTEIN_MAX_LEN = 1200; DRUG_DIM = 128; PROT_DIM = 128
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"; AA_TO_IDX = {aa:i+1 for i,aa in enumerate(AA_ORDER)}
AT = [5,6,7,8,9,15,16,17,35,53]; BT = [1,2,3,12]

def tokenize(s, ml=PROTEIN_MAX_LEN):
    t=np.zeros(ml,dtype=np.int32); n=min(len(s),ml)
    for i in range(n): t[i]=AA_TO_IDX.get(s[i],len(AA_ORDER)+1)
    return t

# ===== LOAD SUBSET =====
print("Loading subset...", flush=True)
b=pd.read_csv(os.path.join(PROCESSED_DIR,"benchmark_dta_subset.csv"))
d=np.load(os.path.join(PROCESSED_DIR,"features_subset.npz"),allow_pickle=True)
y_label=d["y"].astype(np.float32); ps=d["pains_status"]
sl=b["canonical_smiles"].values

# Protein tokens
seqs=b["sequence"].values
prot_tok=np.array([tokenize(s) for s in seqs],dtype=np.int32)

ai=np.arange(len(y_label))
ti,te=train_test_split(ai,test_size=0.2,random_state=RANDOM_SEED,stratify=ps)
ti,vi=train_test_split(ti,test_size=0.125,random_state=RANDOM_SEED,stratify=ps[ti])
tm=ps[ti]==0; ti=ti[tm]
print(f"Train: {len(ti)} Val: {len(vi)} Test: {len(te)}",flush=True)

# ===== CACHES =====
print("Loading caches...",flush=True)
pm=np.load(os.path.join(PROCESSED_DIR,"pains_masks_subset.npz"))
pains_masks_all=pm["masks"]
print(f"  PAINS masks: {pains_masks_all.shape}",flush=True)

CACHE_PATH=os.path.join(PROCESSED_DIR,"graph_cache_subset.pkl")
with open(CACHE_PATH,"rb") as f: gcache=pickle.load(f)
print(f"  Graphs: {len(gcache)}",flush=True)

def gg(idx):
    gs,va=[],[]
    for i,ix in enumerate(idx):
        sm=str(sl[ix]); g=gcache.get(sm)
        if g is not None: gs.append(g); va.append(ix)
    return gs,np.array(va,dtype=int)

tg,tv=gg(ti); vg,vv=gg(vi); sg,sv=gg(te)
print(f"  Graphs: {len(tg)}/{len(vg)}/{len(sg)}",flush=True)

y_label_t=torch.tensor(y_label,dtype=torch.float)
yt=y_label_t[tv]; yv_=y_label_t[vv]; ye=y_label_t[sv]; pse=ps[sv].copy()
st_=torch.tensor(prot_tok[tv]); sv_=torch.tensor(prot_tok[vv]); se_=torch.tensor(prot_tok[sv])
pm_train=torch.tensor(pains_masks_all[tv]); pm_val=torch.tensor(pains_masks_all[vv]); pm_test=torch.tensor(pains_masks_all[sv])
ratio_train=pm_train.mean(dim=1); ratio_val=pm_val.mean(dim=1); ratio_test=pm_test.mean(dim=1)
for g,l in zip(tg,yt): g.y=l
for g,l in zip(vg,yv_): g.y=l
for g,l in zip(sg,ye): g.y=l

class DS(Dataset):
    def __init__(self,g,s,y,pm,pr,ps_=None):
        self.g=g;self.s=s;self.y=y;self.pm=pm;self.pr=pr;self.ps=ps_
    def __len__(self):return len(self.g)
    def __getitem__(self,i):
        r=(self.g[i],self.s[i],self.y[i],self.pm[i],self.pr[i])
        if self.ps is not None: r+=(self.ps[i],)
        return r
def cl(b):
    hp=len(b[0])==6; g,s,y,pm,pr,*_=zip(*b)
    bg=Batch.from_data_list(list(g)); st=torch.stack(list(s)); yt_=torch.stack(list(y))
    pmt=torch.stack(list(pm)); prt=torch.stack(list(pr))
    if hp: return bg,st,yt_,pmt,prt,torch.tensor(_.pop(),dtype=torch.float)
    return bg,st,yt_,pmt,prt

tr_loader=DataLoader(DS(tg,st_,yt,pm_train,ratio_train),BATCH_SIZE,shuffle=True,collate_fn=cl)
va_loader=DataLoader(DS(vg,sv_,yv_,pm_val,ratio_val),BATCH_SIZE,collate_fn=cl)
te_loader=DataLoader(DS(sg,se_,ye,pm_test,ratio_test),BATCH_SIZE,collate_fn=cl)

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

class PRDTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.gin=GINEncoder()
        self.pains_gate=PAINSGate()
        self.prot_cnn=ProteinCNN()
        self.head=nn.Sequential(nn.Linear(DRUG_DIM+PROT_DIM,128),nn.ReLU(),nn.Dropout(0.2),
                                nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.1),nn.Linear(64,1))
    def forward(self,bg,seq,mask,ratio):
        h_main=self.gin(bg)
        s_pains=self.pains_gate(mask,ratio)
        h_drug=h_main*(1-s_pains.unsqueeze(-1))
        h_prot=self.prot_cnn(seq)
        return self.head(torch.cat([h_drug,h_prot],dim=1)).squeeze(-1)

class GCNDTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=GCNConv(len(AT)+3,128);self.c2=GCNConv(128,128);self.c3=GCNConv(128,DRUG_DIM)
        self.pool=global_mean_pool; self.pe=ProteinCNN()
        self.head=nn.Sequential(nn.Linear(DRUG_DIM+PROT_DIM,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,bg,seq):
        x=F.relu(self.c1(bg.x,bg.edge_index));x=F.relu(self.c2(x,bg.edge_index));x=F.relu(self.c3(x,bg.edge_index))
        return self.head(torch.cat([self.pool(x,bg.batch),self.pe(seq)],1)).squeeze(-1)

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
    tl,n=0.0,0; desc="  train" if is_train else "  val"
    pb=tqdm(loader,desc=desc,ncols=80,leave=False)
    with torch.set_grad_enabled(is_train):
        for batch in pb:
            bg,seq,yb=[x.to(device) for x in batch[:3]]
            if is_train: opt.zero_grad()
            pred=model(bg,seq)
            loss=F.mse_loss(pred,yb)
            if is_train:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            bs=len(yb); tl+=loss.item()*bs; n+=bs
            pb.set_postfix({"loss":f"{loss.item():.4f}"})
    return tl/n

def train_model(model,tr,vl,epochs=EPOCHS):
    model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    bv=float("inf"); bs=None
    ep_pb=tqdm(range(1,epochs+1),desc="Train",ncols=95)
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

# ===== RUN =====
results=[]

# Baseline
print("\n--- GCN-DTA Baseline ---",flush=True)
gc.collect();torch.cuda.empty_cache();t0=time.time()
bm=GCNDTA().to(device); bo=torch.optim.AdamW(bm.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
bs_=torch.optim.lr_scheduler.CosineAnnealingLR(bo,T_max=EPOCHS); bv=float("inf"); bss=None
for ep in tqdm(range(1,EPOCHS+1),desc="Baseline",ncols=95):
    run_base(bm,tr_loader,bo); bs_.step()
    vl=run_base(bm,va_loader)
    if vl<bv: bv=vl; bss={k:v.cpu().clone() for k,v in bm.state_dict().items()}
bm.load_state_dict(bss)
bp=predict_base(bm,te_loader); et=time.time()-t0
ed=evaluate_pains_aware(ye.numpy(),bp,pse)
print(f"  RMSE={ed['overall_RMSE']:.4f} P+={ed['pains_pos_RMSE']:.4f} P-={ed['pains_neg_RMSE']:.4f} ΔRMSE={ed['delta_rmse']:.4f} FP={ed['fp_ratio']:.4f} [{et:.0f}s]",flush=True)
results.append({"v":"GCN-DTA_baseline","RMSE":ed["overall_RMSE"],"P+":ed["pains_pos_RMSE"],"P-":ed["pains_neg_RMSE"],"Δ":ed["delta_rmse"],"FP":ed["fp_ratio"]})

# PR-DTA
print("\n--- PR-DTA ---",flush=True)
gc.collect();torch.cuda.empty_cache();t0=time.time()
m=PRDTA(); m=train_model(m,tr_loader,va_loader)
yp=predict(m,te_loader); et=time.time()-t0
ed=evaluate_pains_aware(ye.numpy(),yp,pse)
print(f"  RMSE={ed['overall_RMSE']:.4f} P+={ed['pains_pos_RMSE']:.4f} P-={ed['pains_neg_RMSE']:.4f} ΔRMSE={ed['delta_rmse']:.4f} FP={ed['fp_ratio']:.4f} [{et:.0f}s]",flush=True)
results.append({"v":"PR-DTA_full","RMSE":ed["overall_RMSE"],"P+":ed["pains_pos_RMSE"],"P-":ed["pains_neg_RMSE"],"Δ":ed["delta_rmse"],"FP":ed["fp_ratio"]})

# Summary
rf=pd.DataFrame(results)
print(f"\n{'='*55}")
print(f"{'Variant':22s} {'RMSE':>7s} {'P+':>7s} {'P-':>7s} {'ΔRMSE':>7s} {'FP':>7s}")
print("-"*55)
for _,r in rf.iterrows():
    print(f"{r['v']:22s} {r['RMSE']:7.4f} {r['P+']:7.4f} {r['P-']:7.4f} {r['Δ']:7.4f} {r['FP']:7.4f}")
print("Done.",flush=True)
