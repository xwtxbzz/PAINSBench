"""
PUDA: PAINS-Unbiased Decoupled Architecture.
Optimized version: GCN backbone, in-place masking, proper caching, faster training.
"""
import os, sys, time, gc, warnings, pickle, math
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from torch_geometric.data import Data, Batch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED
from src.evaluation import evaluate_pains_aware

warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# Hyperparams
BATCH_SIZE = 256; EPOCHS = 20; LR = 1e-3; WEIGHT_DECAY = 1e-5
PROTEIN_MAX_LEN = 1000; BIO_DIM = 128; PAINS_DIM = 32; PROT_DIM = 128
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"; AA_TO_IDX = {aa:i+1 for i,aa in enumerate(AA_ORDER)}
AT = [5,6,7,8,9,15,16,17,35,53]; BT = [1,2,3,12]

def tokenize(s, ml=PROTEIN_MAX_LEN):
    t=[AA_TO_IDX.get(a,len(AA_ORDER)+1) for a in s[:ml]]
    t+=[0]*(ml-len(t)); return np.array(t,dtype=np.int64)

def mol2graph(smi):
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

# ========== DATA ==========
print("[1/5] Loading data...", flush=True)
b=pd.read_csv(os.path.join(PROCESSED_DIR,"benchmark_dta_full.csv"))
bf=pd.read_csv(os.path.join(PROCESSED_DIR,"benchmark_full.csv"))
bf["_oi"]=np.arange(len(bf))
sp=b.merge(bf[["molregno","target_chembl_id","_oi"]],on=["molregno","target_chembl_id"],how="left")["_oi"].values
dd=np.load(os.path.join(PROCESSED_DIR,"features_full.npz"),allow_pickle=True)
y_label=dd["y"][sp].astype(np.float32); ps=dd["pains_status"][sp]
sl=b["canonical_smiles"].values; seqs=b["sequence"].values
prot_tok=np.array([tokenize(s) for s in seqs],dtype=np.int64)

ai=np.arange(len(y_label)); ti,te=train_test_split(ai,test_size=0.2,random_state=RANDOM_SEED,stratify=ps)
ti,vi=train_test_split(ti,test_size=0.125,random_state=RANDOM_SEED,stratify=ps[ti])
print(f"  Train {len(ti)} Val {len(vi)} Test {len(te)}", flush=True)

# ========== PAINS FEATURES (cached) ==========
print("[2/5] PAINS features...", flush=True)
CACHE_FILE=os.path.join(PROCESSED_DIR,"pains_masks_puda.npz")
if os.path.exists(CACHE_FILE):
    pm=np.load(CACHE_FILE); pains_masks=pm["masks"]; pains_fp=pm["fp"]
    print(f"  Cached: masks {pains_masks.shape}, fp {pains_fp.shape}",flush=True)
else:
    params=FilterCatalogParams(); params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    catalog=FilterCatalog(params)
    pains_masks=[]; pains_fp=[]
    for i,smi in enumerate(tqdm(sl,desc="PAINS",ncols=80)):
        mol=Chem.MolFromSmiles(smi)
        if mol is None: pains_masks.append(np.zeros(128)); pains_fp.append(np.zeros(7)); continue
        mask=np.zeros(128); pf=np.zeros(7)
        for entry in catalog.GetMatches(mol):
            if not(isinstance(entry,tuple) and len(entry)>1): continue
            mt=entry[-1]
            if isinstance(mt,tuple):
                for idx in mt:
                    if idx<128: mask[idx]=1.0
            if hasattr(entry[0],'GetDescription'):
                d=entry[0].GetDescription()
                if 'PAINS_A' in d: pf[0]=1.0
                elif 'PAINS_B' in d: pf[1]=1.0
                elif 'PAINS_C' in d: pf[2]=1.0
        pf[3]=float(ps[i]); pf[4]=mask.mean(); pains_masks.append(mask); pains_fp.append(pf)
    pains_masks=np.array(pains_masks); pains_fp=np.array(pains_fp)
    np.savez(CACHE_FILE,masks=pains_masks,fp=pains_fp)
    print(f"  Computed: masks {pains_masks.shape}, avg PAINS atoms: {pains_masks.mean():.3f}",flush=True)

# ========== GRAPHS with PAINS masking ==========
print("[3/5] Graphs (cached)...", flush=True)
CACHE_PATH=os.path.join(PROCESSED_DIR,"graph_cache.pkl")
gcache={}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH,"rb") as f: gcache=pickle.load(f)
    print(f"  Loaded {len(gcache)}",flush=True)

def get_graphs(idx):
    gs,va=[],[]
    for i,ix in enumerate(idx):
        sm=str(sl[ix]); g=gcache.get(sm) or mol2graph(sm)
        if g is not None: gcache[sm]=g; gs.append(g); va.append(ix)
        if(i+1)%30000==0: print(f"  {i+1}/{len(idx)}",flush=True)
    return gs,np.array(va,dtype=int)

tg,tv=get_graphs(ti); vg,vv=get_graphs(vi); sg,sv=get_graphs(te)
print(f"  {len(tg)}/{len(vg)}/{len(sg)}",flush=True)

print("  Masking graphs...",flush=True)
def make_masked(g,mask_src,strength=0.95):
    g=deepcopy(g); n=g.x.size(0)
    if len(mask_src)>=n: m=torch.tensor(mask_src[:n],dtype=torch.float).unsqueeze(-1)
    else: m=torch.tensor(np.pad(mask_src,(0,n-len(mask_src)),'constant')[:n],dtype=torch.float).unsqueeze(-1)
    g.x=g.x*(1-m*strength); return g

tm=[make_masked(tg[i],pains_masks[tv[i]]) for i in tqdm(range(len(tg)),desc="Mask",ncols=80)]
vm=[make_masked(vg[i],pains_masks[vv[i]]) for i in range(len(vg))]

# Align
yt=torch.tensor(y_label[tv],dtype=torch.float); yv_=torch.tensor(y_label[vv],dtype=torch.float)
ye=torch.tensor(y_label[sv],dtype=torch.float); pse=ps[sv]
st_=torch.tensor(prot_tok[tv]); sv_=torch.tensor(prot_tok[vv]); se_=torch.tensor(prot_tok[sv])
pft=torch.tensor(pains_fp[tv],dtype=torch.float); pfv=torch.tensor(pains_fp[vv],dtype=torch.float); pfe_=torch.tensor(pains_fp[sv],dtype=torch.float)
for g,l in zip(tg,yt): g.y=l
for g,l in zip(vg,yv_): g.y=l
for g,l in zip(sg,ye): g.y=l
for g,l in zip(tm,yt): g.y=l
for g,l in zip(vm,yv_): g.y=l

# ========== DATASET ==========
class DS(Dataset):
    def __init__(self,g,gm,s,y,pf,ps_=None):
        self.g=g;self.gm=gm;self.s=s;self.y=y;self.pf=pf;self.ps=ps_
    def __len__(self):return len(self.g)
    def __getitem__(self,i):
        if self.ps is not None: return self.g[i],self.gm[i],self.s[i],self.y[i],self.pf[i],self.ps[i]
        return self.g[i],self.gm[i],self.s[i],self.y[i],self.pf[i]

def collate(b):
    hp=len(b[0])==6
    g,gm,s,y,pf=zip(*[(x[0],x[1],x[2],x[3],x[4]) for x in b])
    bg=Batch.from_data_list(list(g)); bgm=Batch.from_data_list(list(gm))
    st=torch.stack(list(s)); yt=torch.stack(list(y)); pft=torch.stack(list(pf))
    if hp: return bg,bgm,st,yt,pft,torch.tensor([x[5] for x in b],dtype=torch.float)
    return bg,bgm,st,yt,pft

tr_ds=DS(tg,tm,st_,yt,pft,torch.tensor(ps[tv],dtype=torch.float))
va_ds=DS(vg,vm,sv_,yv_,pfv)
te_ds=DS(sg,sg,se_,ye,pfe_)
tr_loader=DataLoader(tr_ds,BATCH_SIZE,shuffle=True,collate_fn=collate)
va_loader=DataLoader(va_ds,BATCH_SIZE,collate_fn=collate)
te_loader=DataLoader(te_ds,BATCH_SIZE,collate_fn=collate)
print(f"  Data ready: {len(tr_ds)}/{len(va_ds)}/{len(te_ds)}",flush=True)

# ========== MODEL ==========
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,a): ctx.a=a; return x
    @staticmethod
    def backward(ctx,g): return -ctx.a*g,None
def gr(x,a): return GradReverse.apply(x,a)

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

class GCNDrug(nn.Module):
    def __init__(self,out_dim=BIO_DIM):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
        self.c1=GCNConv(len(AT)+3,128); self.c2=GCNConv(128,128); self.c3=GCNConv(128,out_dim)
        self.pool=global_mean_pool
    def forward(self,data):
        x=F.relu(self.c1(data.x,data.edge_index))
        x=F.relu(self.c2(x,data.edge_index))
        x=F.relu(self.c3(x,data.edge_index))
        return self.pool(x,data.batch)

class PAINSEncoder(nn.Module):
    def __init__(self,in_dim=7,hidden=32,out_dim=PAINS_DIM):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,out_dim))
    def forward(self,x): return self.net(x)

class CrossAttn(nn.Module):
    def __init__(self,d_model=128,nhead=4):
        super().__init__()
        self.wq=nn.Linear(PROT_DIM,d_model); self.wk=nn.Linear(BIO_DIM,d_model); self.wv=nn.Linear(BIO_DIM,d_model)
        self.attn=nn.MultiheadAttention(d_model,nhead,batch_first=True)
        self.proj=nn.Linear(d_model,64)
    def forward(self,zp,zb):
        Q=self.wq(zp).unsqueeze(1); K=self.wk(zb).unsqueeze(1); V=self.wv(zb).unsqueeze(1)
        out,_=self.attn(Q,K,V); return self.proj(out.squeeze(1))

class PUDA(nn.Module):
    def __init__(self):
        super().__init__()
        self.bio_enc=GCNDrug(BIO_DIM)
        self.pains_enc=PAINSEncoder(in_dim=7,out_dim=PAINS_DIM)
        self.prot_enc=ProteinCNN()
        self.cross_attn=CrossAttn()
        self.head=nn.Sequential(nn.Linear(64+PAINS_DIM,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,1))
        self.disc=nn.Sequential(nn.Linear(BIO_DIM,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,1),nn.Sigmoid())
        self.pcls=nn.Sequential(nn.Linear(PAINS_DIM,16),nn.ReLU(),nn.Linear(16,1),nn.Sigmoid())
    def forward(self,dg,dgm,ps,pf,al=0.0,um=True,ra=False):
        inp=dgm if um else dg
        zb=self.bio_enc(inp); zp=self.pains_enc(pf); zpt=self.prot_enc(ps)
        fus=self.cross_attn(zpt,zb)
        pred=self.head(torch.cat([fus,zp],1)).squeeze(-1)
        dpr=self.disc(gr(zb,float(al))).squeeze(-1)
        ppr=self.pcls(zp).squeeze(-1)
        if ra: return pred,zb,zp,dpr,ppr
        return pred

# ========== TRAINING ==========
def train_one(model, loader, val_loader, le=0.5, lm=0.1, lc=1.0, lb=0.01, name="PUDA"):
    model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    bl=float("inf"); bs=None; wu=EPOCHS//3
    pb=tqdm(range(1,EPOCHS+1),desc=name,ncols=95)
    for ep in pb:
        al=min(1.0,ep/max(wu,1)); model.train(); ll={"m":0,"o":0,"a":0,"p":0,"t":0}; n=0
        for batch in loader:
            bg,bgm,s,yb,eb,pb_=[x.to(device) for x in batch]
            opt.zero_grad()
            um_=bool(torch.randint(0,2,(1,)).item())
            pred,zb,zp,dpr,ppr=model.forward(bg,bgm if um_ else bg,s,eb,al,um_,True)
            lm_=F.mse_loss(pred,yb)
            cs=F.cosine_similarity(zb,F.pad(zp,(0,BIO_DIM-PAINS_DIM)),dim=1).pow(2).mean()
            al_=min(0.1*ep/wu,0.1); am_=min(0.5*ep/wu,0.5)
            lo=lm_+al_*cs+am_*F.binary_cross_entropy(dpr,torch.full_like(dpr,0.5))+F.binary_cross_entropy(ppr,pb_)
            lo.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            bs_=yb.size(0); ll["m"]+=lm_.item()*bs_; ll["o"]+=cs.item()*bs_; ll["a"]+=dpr.mean().item()*bs_; ll["t"]+=lo.item()*bs_; n+=bs_
        sch.step()
        model.eval(); vl,nv=0.0,0
        with torch.no_grad():
            for bg,bgm,s,yb,eb in val_loader:
                bg,bgm,s,yb=[x.to(device) for x in (bg,bgm,s,yb)]
                bs_=yb.size(0)
                eb=torch.zeros(bs_,7,device=device)
                pred,_,_,_,_=model.forward(bg,bgm,s,eb,0.0,True,True)
                vl+=F.mse_loss(pred,yb).item()*yb.size(0); nv+=yb.size(0)
        vl/=nv
        if vl<bl: bl=vl; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}
        pb.set_postfix({"M":f"{ll['m']/n:.4f}","O":f"{ll['o']/n:.4f}","V":f"{vl:.4f}","a":f"{al:.2f}"})
    model.load_state_dict(bs); return model

@torch.no_grad()
def predict(model,loader):
    model.eval(); pr=[]
    for bg,_,s,_,pf_ in loader:
        bg,s,pf_=[x.to(device) for x in (bg,s,pf_)]
        pred=model.forward(bg,bg,s,pf_,0.0,False)
        pr.append(pred.cpu().numpy())
    return np.concatenate(pr)

# ========== RUN ==========
print("[4/5] Training...", flush=True)
results=[]

# Quick validation: check if PAINS features are correct
print(f"  PAINS+ ratio in train: {ps[tv].mean():.3f}, val: {ps[vv].mean():.3f}", flush=True)

print("\n--- GCN-DTA Baseline ---",flush=True)
gc.collect(); torch.cuda.empty_cache(); t0=time.time()
mb=PUDA()
mb=mb.to(device); ob=torch.optim.AdamW(mb.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
sb_=torch.optim.lr_scheduler.CosineAnnealingLR(ob,T_max=EPOCHS)
bb=float("inf"); bsb=None
for ep in tqdm(range(1,EPOCHS+1),desc="Baseline",ncols=80):
    mb.train()
    for bg,bgm,s,yb,eb,pb_ in tr_loader:
        bg,s,yb=[x.to(device) for x in (bg,s,yb)]
        bs_=yb.size(0)
        eb=torch.zeros(bs_,7,device=device)
        ob.zero_grad(); pred=mb.forward(bg,bg,s,eb,0.0,False); lo=F.mse_loss(pred,yb)
        lo.backward(); torch.nn.utils.clip_grad_norm_(mb.parameters(),1.0); ob.step()
    sb_.step()
    mb.eval(); vl,nv=0.0,0
    with torch.no_grad():
        for bg,bgm,s,yb,eb in va_loader:
            bg,s,yb=[x.to(device) for x in (bg,s,yb)]
            bs_=yb.size(0)
            eb=torch.zeros(bs_,7,device=device)
            vl+=F.mse_loss(mb.forward(bg,bg,s,eb,0.0,False),yb).item()*yb.size(0); nv+=yb.size(0)
    vl/=nv
    if vl<bb: bb=vl; bsb={k:v.cpu().clone() for k,v in mb.state_dict().items()}
mb.load_state_dict(bsb)
yp=predict(mb,te_loader); et=time.time()-t0
ed=evaluate_pains_aware(ye.numpy(),yp,pse)
print(f"  RMSE={ed['overall_RMSE']:.4f} DR={ed['delta_rmse']:.4f} FP={ed['fp_ratio']:.4f} [{et:.0f}s]",flush=True)
results.append({"v":"GCN_baseline","RMSE":ed["overall_RMSE"],"D":ed["delta_rmse"],"FP":ed["fp_ratio"]})
pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"puda_results.csv"),index=False)

print("\n--- PUDA Full ---",flush=True)
gc.collect(); torch.cuda.empty_cache(); t0=time.time()
mp=PUDA(); mp=train_one(mp,tr_loader,va_loader,name="PUDA")
yp=predict(mp,te_loader); et=time.time()-t0
ed=evaluate_pains_aware(ye.numpy(),yp,pse)
print(f"  RMSE={ed['overall_RMSE']:.4f} DR={ed['delta_rmse']:.4f} FP={ed['fp_ratio']:.4f} [{et:.0f}s]",flush=True)
results.append({"v":"PUDA_full","RMSE":ed["overall_RMSE"],"D":ed["delta_rmse"],"FP":ed["fp_ratio"]})
pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"puda_results.csv"),index=False)

# Ablations
for le_,lm_,lc_,nm in [(0,0.1,1.0,"no_env"),(0.5,0,1.0,"no_MI"),(0.5,0.1,0,"no_CF")]:
    print(f"\n--- PUDA {nm} ---",flush=True)
    gc.collect(); torch.cuda.empty_cache(); t0=time.time()
    ma=PUDA(); ma=train_one(ma,tr_loader,va_loader,le=le_,lm=lm_,lc=lc_,name=nm)
    yp=predict(ma,te_loader); et=time.time()-t0
    ed=evaluate_pains_aware(ye.numpy(),yp,pse)
    print(f"  RMSE={ed['overall_RMSE']:.4f} DR={ed['delta_rmse']:.4f} FP={ed['fp_ratio']:.4f} [{et:.0f}s]",flush=True)
    results.append({"v":f"PUDA_{nm}","RMSE":ed["overall_RMSE"],"D":ed["delta_rmse"],"FP":ed["fp_ratio"]})
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"puda_results.csv"),index=False)

# ========== SUMMARY ==========
print("\n"+f"{'='*55}")
print(f"{'Variant':22s} {'RMSE':>8s} {'DRMSE':>8s} {'FP':>8s} {'BiasRed':>10s}")
print('-'*55)
bd=results[0]["D"]
for r in results:
    pc=(1-abs(r["D"])/abs(bd))*100 if bd!=0 else 0
    chk="OK" if pc>0 else "NO"
    print(f"{r['v']:22s} {r['RMSE']:8.4f} {r['D']:8.4f} {r['FP']:8.4f} {chk:>4s} {pc:+6.1f}%")
print("[5/5] Done.",flush=True)
