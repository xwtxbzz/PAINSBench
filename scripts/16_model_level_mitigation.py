"""
Step 16: Model-Level PAINS Mitigation.
S1: Adversarial GRL | S2: Dual-path Uncertainty | S3: Ranking Margin Loss
"""
import os, sys, time, gc, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED
from src.evaluation import evaluate_pains_aware
warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

BATCH_SIZE = 256; EPOCHS = 30; LR = 1e-3; WEIGHT_DECAY = 1e-5
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

def aac(s):
    t=len(s); return np.zeros(20,dtype=np.float32) if t==0 else np.array([s.count(a)/t for a in AA_ORDER],dtype=np.float32)
def dpc(s):
    t=max(len(s)-1,1); d=np.zeros(400,dtype=np.float32)
    for i in range(len(s)-1):
        k=s[i:i+2]
        if k[0] in AA_ORDER and k[1] in AA_ORDER: d[AA_ORDER.index(k[0])*20+AA_ORDER.index(k[1])]+=1.0
    return d/t

print("Loading...", flush=True)
b=pd.read_csv(os.path.join(PROCESSED_DIR,"benchmark_dta_full.csv"))
bf=pd.read_csv(os.path.join(PROCESSED_DIR,"benchmark_full.csv"))
bf["_oi"]=np.arange(len(bf))
sp=b.merge(bf[["molregno","target_chembl_id","_oi"]],on=["molregno","target_chembl_id"],how="left")["_oi"].values
d=np.load(os.path.join(PROCESSED_DIR,"features_full.npz"),allow_pickle=True)
X=d["X"][sp,:2048]; y=d["y"][sp].astype(np.float32); ps=d["pains_status"][sp]; seqs=b["sequence"].values

X=np.concatenate([X,np.array([aac(s) for s in seqs],dtype=np.float32),np.array([dpc(s) for s in seqs],dtype=np.float32)],axis=1)

from sklearn.model_selection import train_test_split
ai=np.arange(len(y))
ti,te=train_test_split(ai,test_size=0.2,random_state=RANDOM_SEED,stratify=ps)
ti,vi=train_test_split(ti,test_size=0.125,random_state=RANDOM_SEED,stratify=ps[ti])

Xt=torch.tensor(X[ti],dtype=torch.float); yt=torch.tensor(y[ti],dtype=torch.float); pt=torch.tensor(ps[ti],dtype=torch.long)
Xv=torch.tensor(X[vi],dtype=torch.float); yv=torch.tensor(y[vi],dtype=torch.float); pv=torch.tensor(ps[vi],dtype=torch.long)
Xe=torch.tensor(X[te],dtype=torch.float); ye=torch.tensor(y[te],dtype=torch.float); pse=ps[te]

# Pair dataset
class PairDS(torch.utils.data.Dataset):
    def __init__(s,x,y,ps):
        s.x=x;s.y=y;s.ps=ps;s.pi=torch.where(ps==1)[0];s.ni=torch.where(ps==0)[0]
    def __len__(s):return min(len(s.pi),len(s.ni))
    def __getitem__(s,i):
        return s.x[s.pi[i%len(s.pi)]],s.y[s.pi[i%len(s.pi)]],s.ps[s.pi[i%len(s.pi)]],s.x[s.ni[i%len(s.ni)]],s.y[s.ni[i%len(s.ni)]],s.ps[s.ni[i%len(s.ni)]]
def pc(b):
    xp,yp,pp,xn,yn,pn=zip(*b)
    return torch.stack(xp),torch.stack(yp),torch.stack(pp),torch.stack(xn),torch.stack(yn),torch.stack(pn)

tl=DataLoader(TensorDataset(Xt,yt,pt),BATCH_SIZE,shuffle=True)
vl=DataLoader(TensorDataset(Xv,yv,pv),BATCH_SIZE)
el=DataLoader(TensorDataset(Xe,ye),BATCH_SIZE)
pl=DataLoader(PairDS(Xt,yt,pt),BATCH_SIZE,shuffle=True,collate_fn=pc)

# Models
class MLP(nn.Module):
    def __init__(s,in_dim=2468):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(in_dim,1024),nn.BatchNorm1d(1024),nn.ReLU(),nn.Dropout(0.3),nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,1))
    def forward(s,x):return s.net(x).squeeze(-1)

# GRL
class GRF(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,l):ctx.l=l;return x.view_as(x)
    @staticmethod
    def backward(ctx,g):return -ctx.l*g,None
def gr(x,l):return GRF.apply(x,l)

class Adv(nn.Module):
    def __init__(s,in_dim=2468):
        super().__init__()
        s.enc=nn.Sequential(nn.Linear(in_dim,1024),nn.BatchNorm1d(1024),nn.ReLU(),nn.Dropout(0.3),nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,128),nn.ReLU())
        s.reg=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,1))
        s.adv=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,2))
    def forward(s,x,a=1.0,l=0.5):
        e=s.enc(x);p=s.reg(e).squeeze(-1);r=gr(e,l*a);lg=s.adv(r)
        return p,lg

class Dual(nn.Module):
    def __init__(s,in_dim=2468):
        super().__init__()
        s.sh=nn.Sequential(nn.Linear(in_dim,1024),nn.BatchNorm1d(1024),nn.ReLU(),nn.Dropout(0.3))
        s.pa=nn.Sequential(nn.Linear(1024,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1))
        s.pb=nn.Sequential(nn.Linear(1024,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1),nn.Sigmoid())
    def forward(s,x):
        h=s.sh(x);a=s.pa(h).squeeze(-1);b=s.pb(h).squeeze(-1)
        return a*(1-0.3*b),a,b

@torch.no_grad()
def pred(m,loader):
    m.eval();pr=[]
    for x,_ in loader:
        x=x.to(device)
        if isinstance(m,Adv):p,_=m(x,0,0)
        elif isinstance(m,Dual):p,_,_=m(x)
        else:p=m(x)
        pr.append(p.cpu().numpy())
    return np.concatenate(pr)

def train(m,loader,loss_fn,val_loader):
    m=m.to(device);o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sc=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=EPOCHS)
    bl=float("inf");bs=None
    for ep in range(EPOCHS):
        m.train();tl=0.0;n=0
        for batch in loader:
            o.zero_grad();l=loss_fn(m,batch)
            l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);o.step()
            tl+=l.item()*batch[0].size(0);n+=batch[0].size(0)
        sc.step()
        m.eval();vl=0.0;nv=0
        with torch.no_grad():
            for xb,yb,_ in val_loader:
                xb,yb=xb.to(device),yb.to(device)
                if isinstance(m,Adv):p,_=m(xb,0,0)
                elif isinstance(m,Dual):p,_,_=m(xb)
                else:p=m(xb)
                vl+=F.mse_loss(p,yb).item()*yb.size(0);nv+=yb.size(0)
        vl/=nv
        if vl<bl:bl=vl;bs={k:v.cpu().clone() for k,v in m.state_dict().items()}
        if(ep+1)%10==0:print(f"  Ep{ep+1:2d} tl={tl/n:.4f} vl={vl:.4f}",flush=True)
    m.load_state_dict(bs);return m

results=[]

# Baseline
print("=== Baseline ===",flush=True)
gc.collect();torch.cuda.empty_cache();t0=time.time()
m=train(MLP(),tl,lambda m,b:F.mse_loss(m(b[0].to(device)),b[1].to(device)),vl)
yp=pred(m,el);et=time.time()-t0
ed=evaluate_pains_aware(ye.numpy(),yp,pse)
print(f"RMSE={ed['overall_RMSE']:.4f} Δ={ed['delta_rmse']:.4f} FP={ed['fp_ratio']:.4f} [{et:.0f}s]",flush=True)
results.append({"s":"baseline","RMSE":ed["overall_RMSE"],"D":ed["delta_rmse"],"FP":ed["fp_ratio"]})
pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"mitigation_model_level.csv"),index=False)

# S1: Adversarial
for lam in [0.3,0.5,1.0]:
    print(f"=== Adv λ={lam} ===",flush=True)
    gc.collect();torch.cuda.empty_cache();t0=time.time()
    def mk(l_):
        def fn(m,b):
            x,y,ps=b;x=x.to(device);y=y.to(device);ps=ps.to(device)
            p,lg=m.forward(x,1.0,l_)
            return F.mse_loss(p,y)+0.5*F.cross_entropy(lg,ps)
        return fn
    m_=train(Adv(),tl,mk(lam),vl)
    yp_=pred(m_,el);et_=time.time()-t0
    ed_=evaluate_pains_aware(ye.numpy(),yp_,pse)
    print(f"RMSE={ed_['overall_RMSE']:.4f} Δ={ed_['delta_rmse']:.4f} FP={ed_['fp_ratio']:.4f} [{et_:.0f}s]",flush=True)
    results.append({"s":f"adv_{lam}","RMSE":ed_["overall_RMSE"],"D":ed_["delta_rmse"],"FP":ed_["fp_ratio"]})
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"mitigation_model_level.csv"),index=False)

# S2: Dual
print("=== Dual ===",flush=True)
gc.collect();torch.cuda.empty_cache();t0=time.time()
m_=train(Dual(),tl,lambda m,b:(lambda x,y: (lambda p,pa,pb: F.mse_loss(p,y)+0.1*F.mse_loss(pa,y))(*m(x)))(b[0].to(device),b[1].to(device)),vl)
yp_=pred(m_,el);et_=time.time()-t0
ed_=evaluate_pains_aware(ye.numpy(),yp_,pse)
print(f"RMSE={ed_['overall_RMSE']:.4f} Δ={ed_['delta_rmse']:.4f} FP={ed_['fp_ratio']:.4f} [{et_:.0f}s]",flush=True)
results.append({"s":"dual","RMSE":ed_["overall_RMSE"],"D":ed_["delta_rmse"],"FP":ed_["fp_ratio"]})
pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"mitigation_model_level.csv"),index=False)

# S3: Ranking
for gamma in [0.05,0.1]:
    for lr_ in [0.3,1.0]:
        print(f"=== Rank γ={gamma} λ={lr_} ===",flush=True)
        gc.collect();torch.cuda.empty_cache();t0=time.time()
        def mk2(g_,la_):
            def fn(m,b):
                xp,yp,_,xn,yn,_=[x.to(device) for x in b]
                pp,pn=m(xp),m(xn)
                mse=F.mse_loss(pp,yp)+F.mse_loss(pn,yn)
                ep=torch.abs(pp-yp);en=torch.abs(pn-yn)
                rk=torch.mean(F.relu(en-ep+g_))
                return mse+la_*rk
            return fn
        m_=train(MLP(),pl,mk2(gamma,lr_),vl)
        yp_=pred(m_,el);et_=time.time()-t0
        ed_=evaluate_pains_aware(ye.numpy(),yp_,pse)
        print(f"RMSE={ed_['overall_RMSE']:.4f} Δ={ed_['delta_rmse']:.4f} FP={ed_['fp_ratio']:.4f} [{et_:.0f}s]",flush=True)
        results.append({"s":f"rank_g{gamma}_l{lr_}","RMSE":ed_["overall_RMSE"],"D":ed_["delta_rmse"],"FP":ed_["fp_ratio"]})
        pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"mitigation_model_level.csv"),index=False)

# Summary
rf=pd.DataFrame(results)
print(f"\n{'='*60}")
print("Model-Level Mitigation Summary")
print(f"{'='*60}")
print(f"{'Strategy':25s} {'RMSE':>8s} {'ΔRMSE':>8s} {'FP':>8s}")
print("-"*60)
for _,r in rf.iterrows():
    ok="✓" if abs(r['D'])<abs(rf.loc[0,'D']) else "✗"
    print(f"{r['s']:25s} {r['RMSE']:8.4f} {r['D']:8.4f} {r['FP']:8.4f} {ok}")
print("Done.",flush=True)
