"""
Step 14b: Alternative PAINS Mitigation Strategies.
1) Inverse weighting: weight PAINS+ < 1.0 (dilute PAINS shortcut signal)
2) ΔRMSE regularization: penalize mismatch between PAINS+ and PAINS- errors
"""
import os, sys, time, gc, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED

warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

BATCH_SIZE = 256; EPOCHS = 30; LR = 1e-3; WEIGHT_DECAY = 1e-5
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

def compute_aac(seq):
    total = len(seq)
    if total == 0: return np.zeros(20, dtype=np.float32)
    return np.array([seq.count(aa)/total for aa in AA_ORDER], dtype=np.float32)

def compute_dpc(seq):
    total = max(len(seq)-1, 1); dpc = np.zeros(400, dtype=np.float32)
    for i in range(len(seq)-1):
        k = seq[i:i+2]
        if k[0] in AA_ORDER and k[1] in AA_ORDER:
            dpc[AA_ORDER.index(k[0])*20 + AA_ORDER.index(k[1])] += 1.0
    return dpc / total

print("Loading benchmark...", flush=True)
bench = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_dta_full.csv"))
bench_full = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_full.csv"))
bench_full["_orig_idx"] = np.arange(len(bench_full))
sp = bench.merge(bench_full[["molregno","target_chembl_id","_orig_idx"]],
                 on=["molregno","target_chembl_id"],how="left")["_orig_idx"].values
data = np.load(os.path.join(PROCESSED_DIR, "features_full.npz"), allow_pickle=True)
X_fp = data["X"][sp, :2048]; y = data["y"][sp].astype(np.float32); ps = data["pains_status"][sp]
sequences = bench["sequence"].values

paac = np.array([compute_aac(s) for s in sequences], dtype=np.float32)
pdpc = np.array([compute_dpc(s) for s in sequences], dtype=np.float32)
X_all = np.concatenate([X_fp, paac, pdpc], axis=1)

from sklearn.model_selection import train_test_split
all_idx = np.arange(len(y))
train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=RANDOM_SEED, stratify=ps)
train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=RANDOM_SEED, stratify=ps[train_idx])

X_train = torch.tensor(X_all[train_idx], dtype=torch.float); y_train = torch.tensor(y[train_idx], dtype=torch.float)
ps_train = torch.tensor(ps[train_idx], dtype=torch.long)
X_val = torch.tensor(X_all[val_idx], dtype=torch.float); y_val = torch.tensor(y[val_idx], dtype=torch.float)
ps_val = torch.tensor(ps[val_idx], dtype=torch.long)
X_test = torch.tensor(X_all[test_idx], dtype=torch.float); y_test = torch.tensor(y[test_idx], dtype=torch.float)
ps_test = ps[test_idx]

train_loader = DataLoader(TensorDataset(X_train, y_train, ps_train), BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val, ps_val), BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test, y_test), BATCH_SIZE)

class MLPNet(nn.Module):
    def __init__(self, in_dim=2048+420, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden//2, hidden//4), nn.ReLU(),
            nn.Linear(hidden//4, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

from src.evaluation import evaluate_pains_aware
results = []

# Strategy 1: Inverse weighting (PAINS+ weight < 1.0)
def train_inverse(alpha_pos):
    """Inverse weighting: scale PAINS+ loss by alpha_pos (< 1.0 dilutes shortcut)."""
    model = MLPNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_loss = float("inf"); best_state = None
    for epoch in range(EPOCHS):
        model.train(); tr_loss, n = 0.0, 0
        for xb, yb, psb in train_loader:
            xb, yb, psb = xb.to(device), yb.to(device), psb.to(device)
            opt.zero_grad(); pred = model(xb)
            w = torch.where(psb == 1, alpha_pos, 1.0).float()
            loss = (w * (pred - yb) ** 2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tr_loss += loss.item()*len(yb); n += len(yb)
        sched.step()
        model.eval(); val_loss, nv = 0.0, 0
        with torch.no_grad():
            for xb, yb, psb in val_loader:
                xb, yb, psb = xb.to(device), yb.to(device), psb.to(device)
                w = torch.where(psb == 1, alpha_pos, 1.0).float()
                loss = (w * (model(xb) - yb) ** 2).mean()
                val_loss += loss.item()*len(yb); nv += len(yb)
        val_loss /= nv
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        if (epoch+1) % 10 == 0:
            print(f"  Ep {epoch+1:2d} tr={tr_loss/n:.4f} val={val_loss:.4f}", flush=True)
    model.load_state_dict(best_state); return model

# Strategy 2: ΔRMSE regularization
def train_drmse_reg(lmbda=0.5):
    """Add penalty for ΔRMSE (MSE_pos - MSE_neg)^2 to reduce bias."""
    model = MLPNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_loss = float("inf"); best_state = None
    for epoch in range(EPOCHS):
        model.train(); tr_loss, n = 0.0, 0
        for xb, yb, psb in train_loader:
            xb, yb, psb = xb.to(device), yb.to(device), psb.to(device)
            opt.zero_grad(); pred = model(xb)
            mse = (pred - yb) ** 2
            mse_pos = mse[psb == 1].mean() if (psb == 1).any() else 0
            mse_neg = mse[psb == 0].mean() if (psb == 0).any() else 0
            base_loss = mse.mean()
            reg_loss = lmbda * (mse_pos - mse_neg) ** 2
            loss = base_loss + reg_loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tr_loss += base_loss.item()*len(yb); n += len(yb)
        sched.step()
        model.eval(); val_loss, nv = 0.0, 0
        with torch.no_grad():
            for xb, yb, psb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = F.mse_loss(model(xb), yb)
                val_loss += loss.item()*len(yb); nv += len(yb)
        val_loss /= nv
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        if (epoch+1) % 10 == 0:
            print(f"  Ep {epoch+1:2d} tr={tr_loss/n:.4f} val={val_loss:.4f}", flush=True)
    model.load_state_dict(best_state); return model

@torch.no_grad()
def predict(model, loader):
    model.eval(); preds = []
    for xb, *_ in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(preds)

# ====== Inverse Weighting Experiments ======
# If PAINS+ weight < 1, model pays LESS attention to PAINS shortcuts
for alpha_pos in [0.75, 0.5, 0.25]:
    label = f"MLP-DTA_invw{alpha_pos:.2f}"
    print(f"\n{'='*50}\n{label} (PAINS+ weight = {alpha_pos})\n{'='*50}", flush=True)
    gc.collect(); torch.cuda.empty_cache()
    t0 = time.time()
    model = train_inverse(alpha_pos)
    y_pred = predict(model, test_loader)
    elapsed = time.time() - t0
    eval_d = evaluate_pains_aware(y_test.numpy(), y_pred, ps_test)
    delta = eval_d['delta_rmse']
    print(f"  RMSE={eval_d['overall_RMSE']:.4f} P+={eval_d['pains_pos_RMSE']:.4f} P-={eval_d['pains_neg_RMSE']:.4f} ΔRMSE={delta:.4f} FP={eval_d['fp_ratio']:.4f} [{elapsed:.0f}s]", flush=True)
    results.append({"model":"MLP-DTA","strategy":f"inverse_w{alpha_pos}","overall_RMSE":eval_d["overall_RMSE"],"pains_pos_RMSE":eval_d["pains_pos_RMSE"],"pains_neg_RMSE":eval_d["pains_neg_RMSE"],"delta_rmse":delta,"fp_ratio":eval_d["fp_ratio"],"train_time_s":elapsed})
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"mitigation_v2_results.csv"),index=False)

# ====== ΔRMSE Regularization ======
for lmbda in [0.1, 0.5, 1.0]:
    label = f"MLP-DTA_drmse{lmbda:.1f}"
    print(f"\n{'='*50}\n{label} (λ={lmbda})\n{'='*50}", flush=True)
    gc.collect(); torch.cuda.empty_cache()
    t0 = time.time()
    model = train_drmse_reg(lmbda=lmbda)
    y_pred = predict(model, test_loader)
    elapsed = time.time() - t0
    eval_d = evaluate_pains_aware(y_test.numpy(), y_pred, ps_test)
    delta = eval_d['delta_rmse']
    print(f"  RMSE={eval_d['overall_RMSE']:.4f} P+={eval_d['pains_pos_RMSE']:.4f} P-={eval_d['pains_neg_RMSE']:.4f} ΔRMSE={delta:.4f} FP={eval_d['fp_ratio']:.4f} [{elapsed:.0f}s]", flush=True)
    results.append({"model":"MLP-DTA","strategy":f"drmse_reg{lmbda}","overall_RMSE":eval_d["overall_RMSE"],"pains_pos_RMSE":eval_d["pains_pos_RMSE"],"pains_neg_RMSE":eval_d["pains_neg_RMSE"],"delta_rmse":delta,"fp_ratio":eval_d["fp_ratio"],"train_time_s":elapsed})
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"mitigation_v2_results.csv"),index=False)

# ====== Summary ======
res_df = pd.DataFrame(results)
print(f"\n{'='*80}")
print("Mitigation Results Summary")
print(f"{'='*80}")
print(f"{'Strategy':25s} {'RMSE':>8s} {'P+':>8s} {'P-':>8s} {'ΔRMSE':>8s} {'FP':>8s} {'Time':>7s}")
print("-"*80)
for _, r in res_df.iterrows():
    print(f"{r['strategy']:25s} {r['overall_RMSE']:8.4f} {r['pains_pos_RMSE']:8.4f} {r['pains_neg_RMSE']:8.4f} {r['delta_rmse']:8.4f} {r['fp_ratio']:8.4f} {r['train_time_s']:7.0f}s")

# Also load v1 results if they exist
v1_path = os.path.join(RESULTS_DIR, "mitigation_results.csv")
if os.path.exists(v1_path):
    v1 = pd.read_csv(v1_path)
    print(f"\n{'='*80}")
    print("V1 Results (Upweighting PAINS+)")
    print(f"{'='*80}")
    for _, r in v1.iterrows():
        print(f"  α={r['alpha']:4.1f}  RMSE={r['overall_RMSE']:.4f}  ΔRMSE={r['delta_rmse']:.4f}  FP={r['fp_ratio']:.4f}")

print("\nDone.", flush=True)
