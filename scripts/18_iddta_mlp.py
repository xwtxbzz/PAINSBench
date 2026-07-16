"""
ID-DTA (MLP): Invariant Debiasing DTA.
Environment-invariant learning + MI bottleneck + counterfactual + bias net.
"""
import os, sys, time, gc, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED
warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

BATCH_SIZE = 256; EPOCHS = 30; LR = 1e-3; WEIGHT_DECAY = 1e-5; FD = 128; NE = 5
AA = "ACDEFGHIKLMNPQRSTVWY"

def aac(s):
    t=len(s); return np.zeros(20,dtype=np.float32) if t==0 else np.array([s.count(a)/t for a in AA],dtype=np.float32)
def dpc(s):
    t=max(len(s)-1,1); d=np.zeros(400,dtype=np.float32)
    for i in range(len(s)-1):
        k=s[i:i+2]
        if k[0] in AA and k[1] in AA: d[AA.index(k[0])*20+AA.index(k[1])]+=1.0
    return d/t

print("Loading data...", flush=True)
b=pd.read_csv(os.path.join(PROCESSED_DIR,"benchmark_dta_full.csv"))
bf=pd.read_csv(os.path.join(PROCESSED_DIR,"benchmark_full.csv"))
bf["_oi"]=np.arange(len(bf))
sp=b.merge(bf[["molregno","target_chembl_id","_oi"]],on=["molregno","target_chembl_id"],how="left")["_oi"].values
dd=np.load(os.path.join(PROCESSED_DIR,"features_full.npz"),allow_pickle=True)
X=dd["X"][sp,:2048]; y=dd["y"][sp].astype(np.float32); ps=dd["pains_status"][sp]; seqs=b["sequence"].values
X=np.concatenate([X,np.array([aac(s) for s in seqs]),np.array([dpc(s) for s in seqs])],axis=1)
print(f"X: {X.shape} y: {y.shape}", flush=True)

from sklearn.model_selection import train_test_split
ai=np.arange(len(y)); ti,te=train_test_split(ai,test_size=0.2,random_state=RANDOM_SEED,stratify=ps)
ti,vi=train_test_split(ti,test_size=0.125,random_state=RANDOM_SEED,stratify=ps[ti])

print("Clustering environments...", flush=True)
km=KMeans(n_clusters=NE,random_state=RANDOM_SEED,n_init=5); el_all=km.fit_predict(X[:,:2048])
print(f"  Envs: {np.bincount(el_all)}", flush=True)

pff=np.column_stack([ps,X[:,:10].mean(axis=1)])
Xcf=X.copy(); pi=np.where(ps==1)[0]; rs=np.random.RandomState(RANDOM_SEED)
for i in pi: Xcf[i]=X[i]+rs.normal(0,0.1,X.shape[1]).astype(np.float32)*0.5

Xt=torch.tensor(X[ti],dtype=torch.float); yt=torch.tensor(y[ti],dtype=torch.float)
Xv=torch.tensor(X[vi],dtype=torch.float); yv=torch.tensor(y[vi],dtype=torch.float)
Xe=torch.tensor(X[te],dtype=torch.float); ye=torch.tensor(y[te],dtype=torch.float); pse=ps[te]
Xcft=torch.tensor(Xcf[ti],dtype=torch.float)
et=torch.tensor(el_all[ti],dtype=torch.long); ev=torch.tensor(el_all[vi],dtype=torch.long)
pft=torch.tensor(pff[ti],dtype=torch.float); pfv=torch.tensor(pff[vi],dtype=torch.float)

class IDDS(torch.utils.data.Dataset):
    def __init__(self,x,xc,y,e,pf):
        self.x=x;self.xc=xc;self.y=y;self.e=e;self.pf=pf
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return self.x[i],self.xc[i],self.y[i],self.e[i],self.pf[i]

tr_ds=IDDS(Xt,Xcft,yt,et,pft); va_ds=IDDS(Xv,Xv,yv,ev,pfv)
tr_loader=DataLoader(tr_ds,BATCH_SIZE,shuffle=True)
va_loader=DataLoader(va_ds,BATCH_SIZE)
te_loader=DataLoader(TensorDataset(Xe,ye),BATCH_SIZE)

class GRF(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,a): ctx.a=a; return x
    @staticmethod
    def backward(ctx,g): return -ctx.a*g,None
def gr(x,a): return GRF.apply(x,a)

class IDDTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Linear(X.shape[1],1024),nn.BatchNorm1d(1024),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(512,FD),nn.ReLU())
        self.pred=nn.Sequential(nn.Linear(FD,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,1))
        self.bias=nn.Sequential(nn.Linear(pff.shape[1],16),nn.ReLU(),nn.Linear(16,1))
        self.ecls=nn.Linear(FD,NE)
        self.ls=nn.Parameter(torch.tensor(-2.0))
    def forward(self,x,xc,pf,al=0.0,ra=False):
        ph=self.enc(x); phc=self.enc(xc)
        pm=self.pred(phc).squeeze(-1); bi=self.bias(pf).squeeze(-1); pt=pm+bi
        pn=ph+torch.randn_like(ph)*torch.exp(self.ls); el=self.ecls(gr(pn,float(al)))
        if ra: return pt,pm,bi,ph,el
        return pt

def train_m(m,lo,va,ep=EPOCHS,le=0.5,lm=0.1,lc=1.0,lb=0.01,nm="ID-DTA"):
    m=m.to(device); o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sc=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=ep); bl=float("inf"); bs=None; wu=ep//3
    pb=tqdm(range(1,ep+1),desc=nm,ncols=90)
    for e in pb:
        al=min(1.0,e/max(wu,1)); m.train(); ml,el_,mil,cl,tl=0.0,0.0,0.0,0.0,0.0; n=0
        for bx,bxc,by,be,bpf in lo:
            bx,bxc,by,be,bpf=[t.to(device) for t in (bx,bxc,by,be,bpf)]
            o.zero_grad()
            pt,pm,bi,ph,_=m.forward(bx,bxc,bpf,al,True)
            lmse=F.mse_loss(pt,by)
            els=[]
            for ee in range(NE):
                mk=be==ee
                if mk.sum()>1: els.append(F.mse_loss(pt[mk],by[mk]).unsqueeze(0))
            l_env=torch.cat(els).var() if len(els)>1 else torch.tensor(0.0,device=by.device)
            l_mi=F.cross_entropy(m.ecls(gr(m.enc(bx),float(al))),be)
            l_cf=F.mse_loss(pm,by)+(1-F.cosine_similarity(m.enc(bx),m.enc(bxc),dim=1).mean())
            l_br=bi.pow(2).mean()
            loss=lmse+le*l_env+lm*l_mi+lc*l_cf+lb*l_br
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); o.step()
            bs_=by.size(0); ml+=lmse.item()*bs_; el_+=(l_env.item() if isinstance(l_env,torch.Tensor) else 0)*bs_
            mil+=l_mi.item()*bs_; cl+=l_cf.item()*bs_; tl+=loss.item()*bs_; n+=bs_
        sc.step()
        m.eval(); vl,nv=0.0,0
        with torch.no_grad():
            for bx,bxc,by,be,bpf in va:
                bx,bxc,by=[t.to(device) for t in (bx,bxc,by)]
                bpf=torch.zeros(bx.size(0),pff.shape[1],device=device)
                pt,_,_,_,_=m.forward(bx,bxc,bpf,0.0,True)
                vl+=F.mse_loss(pt,by).item()*by.size(0); nv+=by.size(0)
        vl/=nv
        if vl<bl: bl=vl; bs={k:v.cpu().clone() for k,v in m.state_dict().items()}
        pb.set_postfix({"L":f"{ml/n:.4f}","E":f"{el_/n:.4f}","M":f"{mil/n:.4f}","V":f"{vl:.4f}","a":f"{al:.2f}"})
    m.load_state_dict(bs); return m

@torch.no_grad()
def pred(m,lo):
    m.eval(); pr=[]
    for x,_ in lo:
        x=x.to(device)
        pt,_,_,_,_=m.forward(x,x,torch.zeros(x.size(0),pff.shape[1],device=device),0.0,True)
        pr.append(pt.cpu().numpy())
    return np.concatenate(pr)

from src.evaluation import evaluate_pains_aware
results=[]

print("\\n--- Baseline ---",flush=True)
gc.collect(); torch.cuda.empty_cache(); t0=time.time()
m=train_m(IDDTA(),tr_loader,va_loader,le=0,lm=0,lc=0,lb=0,nm="Baseline")
yp=pred(m,te_loader); et=time.time()-t0
ed=evaluate_pains_aware(ye.numpy(),yp,pse)
print(f"  RMSE={ed['overall_RMSE']:.4f} D={ed['delta_rmse']:.4f} FP={ed['fp_ratio']:.4f} [{et:.0f}s]",flush=True)
results.append({"v":"baseline","RMSE":ed["overall_RMSE"],"D":ed["delta_rmse"],"FP":ed["fp_ratio"]})
pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"iddta_results.csv"),index=False)

print("\\n--- ID-DTA Full ---",flush=True)
gc.collect(); torch.cuda.empty_cache(); t0=time.time()
m=train_m(IDDTA(),tr_loader,va_loader,le=0.5,lm=0.1,lc=1.0,lb=0.01,nm="ID-DTA")
yp=pred(m,te_loader); et=time.time()-t0
ed=evaluate_pains_aware(ye.numpy(),yp,pse)
print(f"  RMSE={ed['overall_RMSE']:.4f} D={ed['delta_rmse']:.4f} FP={ed['fp_ratio']:.4f} [{et:.0f}s]",flush=True)
results.append({"v":"ID-DTA_full","RMSE":ed["overall_RMSE"],"D":ed["delta_rmse"],"FP":ed["fp_ratio"]})
pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"iddta_results.csv"),index=False)

for le_,lm_,lc_,lb_,nm in [(0.5,0,1,0.01,"no_MI"),(0,0.1,1,0.01,"no_Env"),(0.5,0.1,0,0.01,"no_CF"),(0.5,0.1,1,0,"no_bias")]:
    print(f"\\n--- ID-DTA {nm} ---",flush=True)
    gc.collect(); torch.cuda.empty_cache(); t0=time.time()
    m=train_m(IDDTA(),tr_loader,va_loader,le=le_,lm=lm_,lc=lc_,lb=lb_,nm=nm)
    yp=pred(m,te_loader); et=time.time()-t0
    ed=evaluate_pains_aware(ye.numpy(),yp,pse)
    print(f"  RMSE={ed['overall_RMSE']:.4f} D={ed['delta_rmse']:.4f} FP={ed['fp_ratio']:.4f} [{et:.0f}s]",flush=True)
    results.append({"v":f"ID-DTA_{nm}","RMSE":ed["overall_RMSE"],"D":ed["delta_rmse"],"FP":ed["fp_ratio"]})
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"iddta_results.csv"),index=False)

print(f"\\n{'='*55}")
print(f"{'Variant':22s} {'RMSE':>8s} {'DRMSE':>8s} {'FP':>8s} {'Red%':>8s}")
print('-'*55)
bd=results[0]["D"]
for r in results:
    pc=(1-abs(r["D"])/abs(bd))*100 if bd!=0 else 0
    print(f"{r['v']:22s} {r['RMSE']:8.4f} {r['D']:8.4f} {r['FP']:8.4f} {pc:+7.1f}%")
print("Done.",flush=True)
