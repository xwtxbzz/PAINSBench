"""
Step 6: Deep learning model comparison (MLP, GCN, GAT, AttentiveFP).
PAINS-aware evaluation on full 229K dataset with GPU training.
"""
import os, sys, time, gc, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, RANDOM_SEED

warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5

# ========== data loading ==========
print("\nLoading data...")
benchmark = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_full.csv"))
data = np.load(os.path.join(PROCESSED_DIR, "features_full.npz"), allow_pickle=True)
X_fp = data["X"][:, :2048]  # Morgan only
y = data["y"].astype(np.float32)
ps = data["pains_status"]

n_pos = int(ps.sum()); n_neg = len(ps) - n_pos
print(f"Full dataset: {len(y):,} (PAINS+ {n_pos:,} / PAINS- {n_neg:,})")

# Train/val/test split
from sklearn.model_selection import train_test_split
train_idx, test_idx, _, _, ps_train, ps_test = train_test_split(
    np.arange(len(y)), y, ps, test_size=0.2, random_state=RANDOM_SEED, stratify=ps)
train_idx, val_idx, y_train, y_val = train_test_split(
    train_idx, y[train_idx], test_size=0.125, random_state=RANDOM_SEED,
    stratify=ps[train_idx])

print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

# ========== Graph construction ==========
print("\nBuilding molecular graphs...")
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, DataLoader

ATOM_TYPES = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # B, C, N, O, F, P, S, Cl, Br, I
BOND_TYPES = [1, 2, 3, 12]  # single, double, triple, aromatic

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Node features
    atom_types = [a.GetAtomicNum() for a in mol.GetAtoms()]
    x = torch.zeros(len(atom_types), len(ATOM_TYPES) + 3)
    for i, at in enumerate(atom_types):
        if at in ATOM_TYPES: x[i, ATOM_TYPES.index(at)] = 1
        else: x[i, -3] = 1  # other
        x[i, -2] = mol.GetAtomWithIdx(i).GetDegree() / 4.0
        x[i, -1] = mol.GetAtomWithIdx(i).GetTotalNumHs() / 3.0

    # Edge features + adjacency
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        bt = bond.GetBondTypeAsDouble()
        feat = [1.0 if bt == b else 0.0 for b in BOND_TYPES]
        edge_attr += [feat, feat]

    if not edge_index:
        # Isolated atom: add self-loop
        edge_index = [[0, 0]]
        edge_attr = [[1.0, 0, 0, 0]]

    return Data(x=x, edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float) if edge_attr else None)

# Convert SMILES to graphs
graph_cache = {}
def get_graphs(indices):
    graphs, valid_idx = [], []
    for i, (_, row) in enumerate(benchmark.iloc[indices].iterrows()):
        smi = str(row.get("canonical_smiles", ""))
        if not smi or smi == "nan":
            continue
        if smi in graph_cache and graph_cache[smi] is not None:
            g = graph_cache[smi]
        else:
            g = mol_to_graph(smi)
            if g is not None:
                graph_cache[smi] = g
        if g is not None:
            graphs.append(g)
            valid_idx.append(indices[i])
    return graphs, np.array(valid_idx, dtype=int)

train_graphs, train_v = get_graphs(train_idx)
val_graphs, val_v = get_graphs(val_idx)
test_graphs, test_v = get_graphs(test_idx)

# Map labels
y_train_g = torch.tensor(y[train_v], dtype=torch.float)
y_val_g = torch.tensor(y[val_v], dtype=torch.float)
y_test_g = torch.tensor(y[test_v], dtype=torch.float)
ps_test_g = ps[test_v]

# Set labels on graph objects
for g, lbl in zip(train_graphs, y_train_g):
    g.y = lbl
for g, lbl in zip(val_graphs, y_val_g):
    g.y = lbl
for g, lbl in zip(test_graphs, y_test_g):
    g.y = lbl

train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)
print(f"Graphs: train {len(train_graphs)}, val {len(val_graphs)}, test {len(test_graphs)}")

# FP-based data for MLP (use same valid indices)
X_fp_t = torch.tensor(X_fp, dtype=torch.float)
fp_train = torch.utils.data.TensorDataset(X_fp_t[train_v], y_train_g)
fp_val = torch.utils.data.TensorDataset(X_fp_t[val_v], y_val_g)
fp_test = torch.utils.data.TensorDataset(X_fp_t[test_v], y_test_g)
fp_train_loader = DataLoader(fp_train, batch_size=BATCH_SIZE, shuffle=True)
fp_val_loader = DataLoader(fp_val, batch_size=BATCH_SIZE)
fp_test_loader = DataLoader(fp_test, batch_size=BATCH_SIZE)

# ========== Models ==========
class MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

class GCNNet(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden // 2)
        self.lin = nn.Sequential(
            nn.Linear(hidden // 2, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1))
        self.pool = global_mean_pool
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.pool(x, batch)
        return self.lin(x).squeeze(-1)

class GATNet(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, heads=4):
        super().__init__()
        from torch_geometric.nn import GATConv, global_mean_pool
        self.conv1 = GATConv(in_dim, hidden // heads, heads=heads)
        self.conv2 = GATConv(hidden, hidden // heads, heads=heads)
        self.conv3 = GATConv(hidden, hidden // 2, heads=1)
        self.lin = nn.Sequential(nn.Linear(hidden // 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.pool = global_mean_pool
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = self.pool(x, batch)
        return self.lin(x).squeeze(-1)

class AttentiveFPNet(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128):
        super().__init__()
        from torch_geometric.nn import AttentiveFP
        self.encoder = nn.Linear(in_dim, hidden)
        self.attentive_fp = AttentiveFP(
            in_channels=hidden, hidden_channels=hidden, out_channels=1,
            edge_dim=len(BOND_TYPES), num_layers=2, num_timesteps=2, dropout=0.2)
    def forward(self, data):
        x = self.encoder(data.x)
        return self.attentive_fp(x, data.edge_index, data.edge_attr, data.batch).squeeze(-1)

# ========== Training ==========
def train_model(model, loader, val_loader, epochs=EPOCHS):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y_b = batch
                x, y_b = x.to(device), y_b.to(device)
                is_graph = False
            else:
                x, y_b = batch.to(device), batch.y.to(device)
                is_graph = True
            opt.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y_b)
            loss.backward(); opt.step()
            train_loss += loss.item() * len(y_b)
            n_train += len(y_b)
        scheduler.step()
        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x, y_b = batch; x, y_b = x.to(device), y_b.to(device)
                else:
                    x, y_b = batch.to(device), batch.y.to(device)
                pred = model(x)
                val_loss += F.mse_loss(pred, y_b).item() * len(y_b)
                n_val += len(y_b)
        val_loss /= n_val if n_val else 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 20 == 0:
            train_loss /= n_train if n_train else 1
            print(f"  Epoch {epoch+1:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
    model.load_state_dict(best_state)
    return model

def predict(model, loader):
    model.eval()
    all_pred = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            all_pred.append(model(x).cpu().numpy())
    return np.concatenate(all_pred)

# ========== Train & evaluate ==========
from src.evaluation import evaluate_pains_aware
from src.visualization import plot_scatter, plot_residual_distribution

models = {
    "MLP": lambda: MLP(in_dim=2048),
    "GCN": lambda: GCNNet(),
    "GAT": lambda: GATNet(),
    "AttentiveFP": lambda: AttentiveFPNet(),
}

results_rows = []

for name, build_fn in models.items():
    print(f"\n{'=' * 50}")
    print(f"Training {name}...")
    t0 = time.time()

    loader = train_loader if name != "MLP" else fp_train_loader
    val_loader_ = val_loader if name != "MLP" else fp_val_loader
    test_loader_ = test_loader if name != "MLP" else fp_test_loader

    model = train_model(build_fn(), loader, val_loader_)
    y_pred = predict(model, test_loader_)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Align predictions with test indices
    eval_dict = evaluate_pains_aware(y_test_g.numpy(), y_pred, ps_test_g)
    eval_dict["model"] = name
    eval_dict["train_time_s"] = elapsed
    results_rows.append(eval_dict)

    print(f"  Overall RMSE:  {eval_dict['overall_RMSE']:.4f}")
    print(f"  PAINS+ RMSE:   {eval_dict['pains_pos_RMSE']:.4f}")
    print(f"  PAINS- RMSE:   {eval_dict['pains_neg_RMSE']:.4f}")
    print(f"  ΔRMSE:         {eval_dict['delta_rmse']:.4f}")
    print(f"  FP Ratio:      {eval_dict['fp_ratio']:.4f}")

    plot_scatter(y_test_g.numpy(), y_pred, ps_test_g, f"DL_{name}",
                 filename=f"scatter_DL_{name}.png")
    res_pos = np.abs(y_test_g.numpy()[ps_test_g == 1] - y_pred[ps_test_g == 1])
    res_neg = np.abs(y_test_g.numpy()[ps_test_g == 0] - y_pred[ps_test_g == 0])
    plot_residual_distribution(res_pos, res_neg, f"DL_{name}",
                               filename=f"residuals_DL_{name}.png")

# ========== Results ==========
results_df = pd.DataFrame(results_rows)
cols = ["model"] + [c for c in results_df.columns if c != "model"]
results_df = results_df[cols]
results_df.to_csv(os.path.join(RESULTS_DIR, "dl_comparison_results.csv"), index=False)

print(f"\n{'=' * 50}")
print("DL Model Comparison Results:")
print(f"{'Model':15s} {'RMSE':>8s} {'PAINS+':>8s} {'PAINS-':>8s} {'ΔRMSE':>8s} {'FP_Ratio':>9s} {'Time':>8s}")
print("-" * 60)
for _, r in results_df.iterrows():
    print(f"{r['model']:15s} {r['overall_RMSE']:8.4f} {r['pains_pos_RMSE']:8.4f} "
          f"{r['pains_neg_RMSE']:8.4f} {r['delta_rmse']:8.4f} {r['fp_ratio']:9.4f} "
          f"{r['train_time_s']:7.0f}s")

# Plot comparison
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# RMSE comparison
x = np.arange(len(results_df))
w = 0.25
ax = axes[0]
ax.bar(x - w, results_df["pains_pos_RMSE"], w, label="PAINS+", color="#e74c3c")
ax.bar(x, results_df["pains_neg_RMSE"], w, label="PAINS-", color="#3498db")
ax.bar(x + w, results_df["overall_RMSE"], w, label="Overall", color="#2ecc71")
ax.set_xticks(x); ax.set_xticklabels(results_df["model"])
ax.set_ylabel("RMSE"); ax.set_title("RMSE by Model & PAINS Status"); ax.legend()

ax = axes[1]
colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in results_df["delta_rmse"]]
ax.bar(x, results_df["delta_rmse"], color=colors)
ax.axhline(0, color="gray", lw=1)
ax.set_xticks(x); ax.set_xticklabels(results_df["model"])
ax.set_ylabel("ΔRMSE"); ax.set_title("ΔRMSE (PAINS+ − PAINS-)")

ax = axes[2]
ax.bar(x, results_df["fp_ratio"], color="#e67e22")
ax.axhline(1, color="gray", ls="--", lw=1)
ax.set_xticks(x); ax.set_xticklabels(results_df["model"])
ax.set_ylabel("FP Ratio"); ax.set_title("FP Ratio (Residual Ratio)")

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "dl_comparison_summary.png"), dpi=150)
print(f"\nFigure saved: dl_comparison_summary.png")
print(f"Results saved: dl_comparison_results.csv")
print("Done.")
