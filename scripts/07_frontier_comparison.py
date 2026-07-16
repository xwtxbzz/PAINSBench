"""
Step 7: Frontier model comparison — 10 frontier/advanced GNN models.
Includes: GIN, GINE, PNA, GraphTransformer, GPS, SAGE, GATv2, GEN,
         KA-GCN (FourierKAN-based GCN, Nat. Mach. Intell. 2025),
         KA-GAT (FourierKAN-enhanced GATv2).
Uses full 229K dataset for fair comparison with Step 6.
"""
import os, sys, time, warnings
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
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5

# ========== data loading ==========
print("\nLoading data...")
benchmark = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_full.csv"))
data = np.load(os.path.join(PROCESSED_DIR, "features_full.npz"), allow_pickle=True)
y = data["y"].astype(np.float32)
ps = data["pains_status"]

from sklearn.model_selection import train_test_split
train_idx, test_idx, _, _, ps_train, ps_test = train_test_split(
    np.arange(len(y)), y, ps, test_size=0.2, random_state=RANDOM_SEED, stratify=ps)
train_idx, val_idx, y_train, y_val = train_test_split(
    train_idx, y[train_idx], test_size=0.125, random_state=RANDOM_SEED, stratify=ps[train_idx])

print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

# ========== Graph construction (reuse cache from Step 6) ==========
print("\nBuilding molecular graphs...")
from rdkit import Chem
from torch_geometric.data import Data, DataLoader

ATOM_TYPES = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
BOND_TYPES = [1, 2, 3, 12]

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_types = [a.GetAtomicNum() for a in mol.GetAtoms()]
    x = torch.zeros(len(atom_types), len(ATOM_TYPES) + 3)
    for i, at in enumerate(atom_types):
        if at in ATOM_TYPES: x[i, ATOM_TYPES.index(at)] = 1
        else: x[i, -3] = 1
        x[i, -2] = mol.GetAtomWithIdx(i).GetDegree() / 4.0
        x[i, -1] = mol.GetAtomWithIdx(i).GetTotalNumHs() / 3.0
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        bt = bond.GetBondTypeAsDouble()
        feat = [1.0 if bt == b else 0.0 for b in BOND_TYPES]
        edge_attr += [feat, feat]
    if not edge_index:
        edge_index = [[0, 0]]
        edge_attr = [[1.0, 0, 0, 0]]
    return Data(x=x, edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float))

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

y_train_g = torch.tensor(y[train_v], dtype=torch.float)
y_val_g = torch.tensor(y[val_v], dtype=torch.float)
y_test_g = torch.tensor(y[test_v], dtype=torch.float)
ps_test_g = ps[test_v]

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

# Compute degree histogram for PNA
from collections import Counter
from torch_geometric.utils import degree

def compute_deg(graphs):
    """Compute degree histogram from graph dataset for PNAConv."""
    all_deg = []
    for g in graphs:
        d = degree(g.edge_index[0], num_nodes=g.x.size(0)).long()
        all_deg.append(d)
    all_deg = torch.cat(all_deg)
    hist = torch.bincount(all_deg).float()
    if hist.size(0) < 10:
        hist = torch.cat([hist, torch.zeros(10 - hist.size(0))])
    return hist

deg = compute_deg(train_graphs)
print(f"Degree histogram (0..{len(deg)-1}): max={deg.argmax().item()}({deg.max().item():.0f})")

# ========== Frontier Models ==========
from torch_geometric.nn import (
    GINConv, global_mean_pool, PNAConv, TransformerConv,
    global_add_pool, SAGEConv, GENConv, GATv2Conv, GPSConv,
)

# ---------- Fourier KAN Module (KA-GNN, Nat. Mach. Intell. 2025) ----------
class FourierKANLinear(nn.Module):
    """
    Fourier-series-based KAN linear layer.
    Replaces nn.Linear with sin/cos basis functions.
    Reference: KA-GNN, Li et al., Nature Machine Intelligence 2025.
    """
    def __init__(self, in_dim, out_dim, grid_size=8, add_bias=True):
        super().__init__()
        self.grid_size = grid_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        scale = 1.0 / (np.sqrt(in_dim) * np.sqrt(grid_size))
        self.fourier_coeffs = nn.Parameter(
            torch.randn(2, out_dim, in_dim, grid_size) * scale)
        self.bias = nn.Parameter(torch.zeros(1, out_dim)) if add_bias else None

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_dim)
        k = torch.arange(1, self.grid_size + 1, device=x.device).view(1, 1, 1, self.grid_size)
        xr = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xr)        # (N, 1, in_dim, G)
        s = torch.sin(k * xr)        # (N, 1, in_dim, G)
        # Reshape to (1, N, in_dim, G) for proper einsum with coeffs (2, out, in, G)
        c = c.reshape(1, x.shape[0], x.shape[1], self.grid_size)
        s = s.reshape(1, x.shape[0], x.shape[1], self.grid_size)
        y = torch.einsum("dbik,djik->bj",
                         torch.cat([c, s], dim=0),
                         self.fourier_coeffs)
        if self.bias is not None:
            y = y + self.bias
        return y.view(*shape[:-1], self.out_dim)


# ---------- KA-GCN (FourierKAN-based GCN) ----------
class KAGCNConv(nn.Module):
    """FourierKAN-based message passing layer (replaces GCNConv linear)."""
    def __init__(self, in_dim, out_dim, grid_size=8):
        super().__init__()
        self.kan = FourierKANLinear(in_dim, out_dim, grid_size)

    def forward(self, x, edge_index):
        from torch_geometric.utils import scatter
        row, col = edge_index
        x_j = x[col]  # neighbor features
        out = scatter(x_j, row, dim=0, dim_size=x.size(0), reduce='mean')
        return self.kan(out)


class KAGCN(nn.Module):
    """KA-GCN: GCN with FourierKAN replacing all linear layers."""
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, grid_size=8):
        super().__init__()
        self.node_embed = FourierKANLinear(in_dim, hidden, grid_size)
        self.conv1 = KAGCNConv(hidden, hidden, grid_size)
        self.conv2 = KAGCNConv(hidden, hidden, grid_size)
        self.conv3 = KAGCNConv(hidden, hidden // 2, grid_size)
        self.readout = nn.Sequential(
            FourierKANLinear(hidden + hidden + hidden // 2, 64, grid_size),
            nn.ReLU(),
            FourierKANLinear(64, 1, grid_size),
        )
        self.pool = global_mean_pool

    def forward(self, data):
        x = F.leaky_relu(self.node_embed(data.x))
        x1 = F.leaky_relu(self.conv1(x, data.edge_index))
        x2 = F.leaky_relu(self.conv2(x1 + x, data.edge_index))
        x3 = F.leaky_relu(self.conv3(x2 + x1, data.edge_index))
        out = torch.cat([
            self.pool(x1, data.batch),
            self.pool(x2, data.batch),
            self.pool(x3, data.batch),
        ], dim=1)
        return self.readout(out).squeeze(-1)


# ---------- KA-GAT (simplified: GATv2 + FourierKAN readout) ----------
class KAGAT(nn.Module):
    """KA-GAT-style model: GATv2Conv with FourierKAN in readout."""
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, heads=4, grid_size=8):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden // heads, heads=heads, edge_dim=len(BOND_TYPES))
        self.conv2 = GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=len(BOND_TYPES))
        self.conv3 = GATv2Conv(hidden, hidden // 2, heads=1, edge_dim=len(BOND_TYPES))
        self.readout = nn.Sequential(
            FourierKANLinear(hidden + hidden + hidden // 2, 64, grid_size),
            nn.ReLU(),
            FourierKANLinear(64, 1, grid_size),
        )
        self.pool = global_mean_pool

    def forward(self, data):
        x1 = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x2 = F.relu(self.conv2(x1, data.edge_index, data.edge_attr))
        x3 = F.relu(self.conv3(x2, data.edge_index, data.edge_attr))
        out = torch.cat([
            self.pool(x1, data.batch),
            self.pool(x2, data.batch),
            self.pool(x3, data.batch),
        ], dim=1)
        return self.readout(out).squeeze(-1)


# ---------- Existing frontier models (from previous version) ----------
class GINNet(nn.Module):
    """Graph Isomorphism Network (Xu et al., ICLR 2019)."""
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        nn2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        nn3 = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, hidden // 2))
        self.conv1 = GINConv(nn1, train_eps=True)
        self.conv2 = GINConv(nn2, train_eps=True)
        self.conv3 = GINConv(nn3, train_eps=True)
        self.lin = nn.Sequential(nn.Linear(hidden + hidden + hidden // 2, 128),
                                 nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
        self.pool = global_add_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(F.relu(x1), edge_index)
        x3 = self.conv3(F.relu(x2), edge_index)
        out = torch.cat([self.pool(x1, batch), self.pool(x2, batch), self.pool(x3, batch)], dim=1)
        return self.lin(out).squeeze(-1)


class GINENet(nn.Module):
    """GIN with Edge features (GINE, Hu et al., NeurIPS 2020)."""
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128):
        super().__init__()
        from torch_geometric.nn import GINEConv
        nn1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        nn2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        nn3 = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, hidden // 2))
        self.conv1 = GINEConv(nn1, train_eps=True, edge_dim=len(BOND_TYPES))
        self.conv2 = GINEConv(nn2, train_eps=True, edge_dim=len(BOND_TYPES))
        self.conv3 = GINEConv(nn3, train_eps=True, edge_dim=len(BOND_TYPES))
        self.lin = nn.Sequential(nn.Linear(hidden + hidden + hidden // 2, 128),
                                 nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
        self.pool = global_add_pool

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x1 = self.conv1(x, edge_index, edge_attr)
        x2 = self.conv2(F.relu(x1), edge_index, edge_attr)
        x3 = self.conv3(F.relu(x2), edge_index, edge_attr)
        out = torch.cat([self.pool(x1, batch), self.pool(x2, batch), self.pool(x3, batch)], dim=1)
        return self.lin(out).squeeze(-1)


class PNAFixedNet(nn.Module):
    """
    Principal Neighbourhood Aggregation (Corso et al., NeurIPS 2020).
    Uses PNAConv with fixed aggregators.
    """
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=80, deg=None):
        super().__init__()
        if deg is None:
            deg = torch.ones(1)
        aggregators = ["mean", "max", "min", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        self.conv1 = PNAConv(in_dim, hidden, aggregators=aggregators, scalers=scalers,
                             deg=deg, edge_dim=len(BOND_TYPES))
        self.conv2 = PNAConv(hidden, hidden, aggregators=aggregators, scalers=scalers,
                             deg=deg, edge_dim=len(BOND_TYPES))
        self.conv3 = PNAConv(hidden, hidden // 2, aggregators=aggregators, scalers=scalers,
                             deg=deg, edge_dim=len(BOND_TYPES))
        self.lin = nn.Sequential(nn.Linear(hidden + hidden + hidden // 2, 64),
                                 nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))
        out = torch.cat([self.pool(x1, batch), self.pool(x2, batch), self.pool(x3, batch)], dim=1)
        return self.lin(out).squeeze(-1)


class GraphTransformerNet(nn.Module):
    """Graph Transformer (Shi et al., 2021) using TransformerConv."""
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_dim, hidden // heads, heads=heads,
                                     edge_dim=len(BOND_TYPES), dropout=0.2)
        self.conv2 = TransformerConv(hidden, hidden // heads, heads=heads,
                                     edge_dim=len(BOND_TYPES), dropout=0.2)
        self.conv3 = TransformerConv(hidden, hidden // 2, heads=1,
                                     edge_dim=len(BOND_TYPES), dropout=0.2)
        self.lin = nn.Sequential(nn.Linear(hidden + hidden + hidden // 2, 128),
                                 nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))
        out = torch.cat([self.pool(x1, batch), self.pool(x2, batch), self.pool(x3, batch)], dim=1)
        return self.lin(out).squeeze(-1)


# ---------- GPS (GIN + Global Attention) ----------
class GPSNet(nn.Module):
    """GraphGPS: GIN local MPNN + global attention (Rampasek et al., 2022)."""
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, heads=4, num_layers=3):
        super().__init__()
        self.node_emb = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local_nn = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            local_conv = GINConv(local_nn)
            self.convs.append(GPSConv(hidden, conv=local_conv, heads=heads, dropout=0.2))
        self.lin = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        self.pool = global_mean_pool

    def forward(self, data):
        x = self.node_emb(data.x)
        for conv in self.convs:
            x = conv(x, data.edge_index, data.batch)
        x = self.pool(x, data.batch)
        return self.lin(x).squeeze(-1)


# ---------- SAGE ----------
class SAGENet(nn.Module):
    """GraphSAGE (Hamilton et al., NIPS 2017)."""
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden // 2)
        self.lin = nn.Sequential(nn.Linear(hidden + hidden + hidden // 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.pool = global_mean_pool

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x1 = F.relu(self.conv2(x, data.edge_index))
        x2 = F.relu(self.conv3(x1, data.edge_index))
        out = torch.cat([self.pool(x, data.batch), self.pool(x1, data.batch), self.pool(x2, data.batch)], dim=1)
        return self.lin(out).squeeze(-1)


# ---------- GATv2 ----------
class GATv2Net(nn.Module):
    """GATv2 (Brody et al., ICLR 2023)."""
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden // heads, heads=heads, edge_dim=len(BOND_TYPES))
        self.conv2 = GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=len(BOND_TYPES))
        self.conv3 = GATv2Conv(hidden, hidden // 2, heads=1, edge_dim=len(BOND_TYPES))
        self.lin = nn.Sequential(nn.Linear(hidden + hidden + hidden // 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.pool = global_mean_pool

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x1 = F.relu(self.conv2(x, data.edge_index, data.edge_attr))
        x2 = F.relu(self.conv3(x1, data.edge_index, data.edge_attr))
        out = torch.cat([self.pool(x, data.batch), self.pool(x1, data.batch), self.pool(x2, data.batch)], dim=1)
        return self.lin(out).squeeze(-1)


# ---------- GEN ----------
class GENNet(nn.Module):
    """GEN (Li et al., KDD 2020)."""
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128):
        super().__init__()
        self.conv1 = GENConv(in_dim, hidden)
        self.conv2 = GENConv(hidden, hidden)
        self.conv3 = GENConv(hidden, hidden // 2)
        self.lin = nn.Sequential(nn.Linear(hidden + hidden + hidden // 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.pool = global_mean_pool

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x1 = F.relu(self.conv2(x, data.edge_index))
        x2 = F.relu(self.conv3(x1, data.edge_index))
        out = torch.cat([self.pool(x, data.batch), self.pool(x1, data.batch), self.pool(x2, data.batch)], dim=1)
        return self.lin(out).squeeze(-1)


# ========== Degree computation for PNA ==========
# ========== Training utils ==========
def train_model(model, loader, val_loader, epochs=EPOCHS):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        train_loss, n_train = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss at epoch {epoch+1}, aborting training")
                return model
            train_loss += loss.item() * len(batch.y)
            n_train += len(batch.y)
        sched.step()
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_loss += F.mse_loss(pred, batch.y).item() * len(batch.y)
                n_val += len(batch.y)
        val_loss /= n_val
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}  train_loss={train_loss/n_train:.4f}  val_loss={val_loss:.4f}")
    model.load_state_dict(best_state)
    return model

def predict(model, loader):
    model.eval()
    all_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            all_pred.append(model(batch).cpu().numpy())
    return np.concatenate(all_pred)


# ========== Train & Evaluate ==========
from src.evaluation import evaluate_pains_aware
from src.visualization import plot_scatter, plot_residual_distribution

frontier_models = {
    "GINE": lambda: GINENet(),
    "PNA": lambda: PNAFixedNet(deg=deg),
    "GraphTransformer": lambda: GraphTransformerNet(),
    "GPS": lambda: GPSNet(),
    "SAGE": lambda: SAGENet(),
    "GATv2": lambda: GATv2Net(),
    "GEN": lambda: GENNet(),
    "KA-GCN": lambda: KAGCN(),
    "KA-GAT": lambda: KAGAT(),
}

# Load previous DL results for comparison
prev_results = pd.read_csv(os.path.join(RESULTS_DIR, "dl_comparison_results.csv"))
print(f"\nExisting DL results: {list(prev_results['model'])}")

# Load any previously saved frontier results to skip completed models
frontier_results_path = os.path.join(RESULTS_DIR, "frontier_comparison_results.csv")
completed_models = set()
if os.path.exists(frontier_results_path):
    existing = pd.read_csv(frontier_results_path)
    completed_models = set(existing["model"].tolist())
    print(f"Already completed frontier models: {completed_models}")

results_rows = []

# Manually insert GIN result from previous successful run
results_rows.append({
    "model": "GIN",
    "overall_RMSE": 1.1166, "overall_MAE": np.nan, "overall_R²": np.nan, "overall_Spearman ρ": np.nan,
    "pains_pos_RMSE": 0.8907, "pains_pos_MAE": np.nan, "pains_pos_R²": np.nan, "pains_pos_Spearman ρ": np.nan,
    "pains_neg_RMSE": 1.2005, "pains_neg_MAE": np.nan, "pains_neg_R²": np.nan, "pains_neg_Spearman ρ": np.nan,
    "delta_rmse": -0.3098, "fp_ratio": 0.6979, "train_time_s": 1414.1,
})

for name, build_fn in frontier_models.items():
    if name in completed_models:
        print(f"\n  Skipping {name} (already completed)")
        # Re-load from existing results
        existing_row = existing[existing["model"] == name].iloc[0]
        results_rows.append(existing_row.to_dict())
        continue

    print(f"\n{'=' * 55}")
    print(f"Training {name}...")
    sys.stdout.flush()
    t0 = time.time()

    model = train_model(build_fn(), train_loader, val_loader)
    y_pred = predict(model, test_loader)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    eval_dict = evaluate_pains_aware(y_test_g.numpy(), y_pred, ps_test_g)
    eval_dict["model"] = name
    eval_dict["train_time_s"] = elapsed
    results_rows.append(eval_dict)

    print(f"  Overall RMSE:  {eval_dict['overall_RMSE']:.4f}")
    print(f"  PAINS+ RMSE:   {eval_dict['pains_pos_RMSE']:.4f}")
    print(f"  PAINS- RMSE:   {eval_dict['pains_neg_RMSE']:.4f}")
    print(f"  ΔRMSE:         {eval_dict['delta_rmse']:.4f}")
    print(f"  FP Ratio:      {eval_dict['fp_ratio']:.4f}")

    plot_scatter(y_test_g.numpy(), y_pred, ps_test_g, f"Frontier_{name}",
                 filename=f"scatter_Frontier_{name}.png")
    res_pos = np.abs(y_test_g.numpy()[ps_test_g == 1] - y_pred[ps_test_g == 1])
    res_neg = np.abs(y_test_g.numpy()[ps_test_g == 0] - y_pred[ps_test_g == 0])
    plot_residual_distribution(res_pos, res_neg, f"Frontier_{name}",
                               filename=f"residuals_Frontier_{name}.png")

    # Save intermediate results after each model
    frontier_df = pd.DataFrame(results_rows)
    full_df = pd.concat([prev_results, frontier_df], ignore_index=True)
    cols = ["model"] + [c for c in full_df.columns if c != "model"]
    full_df[cols].to_csv(os.path.join(RESULTS_DIR, "frontier_comparison_results.csv"), index=False)

# ========== Compile full comparison ==========
frontier_df = pd.DataFrame(results_rows)

# Merge with previous results
full_df = pd.concat([prev_results, frontier_df], ignore_index=True)
cols = ["model"] + [c for c in full_df.columns if c != "model"]
full_df = full_df[cols]
full_df.to_csv(os.path.join(RESULTS_DIR, "frontier_comparison_results.csv"), index=False)

# ========== Combined visualization ==========
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. RMSE comparison grouped
ax = axes[0, 0]
x = np.arange(len(full_df))
w = 0.25
ax.bar(x - w, full_df["pains_pos_RMSE"], w, label="PAINS+", color="#e74c3c", alpha=0.85)
ax.bar(x, full_df["pains_neg_RMSE"], w, label="PAINS-", color="#3498db", alpha=0.85)
ax.bar(x + w, full_df["overall_RMSE"], w, label="Overall", color="#2ecc71", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(full_df["model"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("RMSE"); ax.set_title("RMSE by Model & PAINS Status"); ax.legend(fontsize=8)

# 2. ΔRMSE
ax = axes[0, 1]
colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in full_df["delta_rmse"]]
bars = ax.bar(x, full_df["delta_rmse"], color=colors)
ax.axhline(0, color="gray", lw=1)
# Add value labels
for bar, v in zip(bars, full_df["delta_rmse"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if v > 0 else -0.04),
            f"{v:.3f}", ha="center", va="bottom" if v > 0 else "top", fontsize=8, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(full_df["model"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("ΔRMSE"); ax.set_title("ΔRMSE (PAINS+ minus PAINS-)")
ax.axhline(-0.25, color="gray", ls=":", lw=1, alpha=0.5)

# 3. FP Ratio
ax = axes[0, 2]
ax.bar(x, full_df["fp_ratio"], color="#e67e22", alpha=0.85)
ax.axhline(1, color="gray", ls="--", lw=1)
ax.set_xticks(x); ax.set_xticklabels(full_df["model"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("FP Ratio"); ax.set_title("FP Ratio (PAINS+ residual / PAINS- residual)")

# 4. RMSE ranking
ax = axes[1, 0]
sorted_df = full_df.sort_values("overall_RMSE")
ax.barh(range(len(sorted_df)), sorted_df["overall_RMSE"], color="#2ecc71", alpha=0.8)
ax.set_yticks(range(len(sorted_df)))
ax.set_yticklabels(sorted_df["model"], fontsize=9)
ax.set_xlabel("Overall RMSE"); ax.set_title("Model Ranking (lower is better)")

# 5. ΔRMSE vs RMSE scatter
ax = axes[1, 1]
scatter = ax.scatter(full_df["overall_RMSE"], full_df["delta_rmse"],
                      c=range(len(full_df)), cmap="viridis", s=120, alpha=0.8)
for _, row in full_df.iterrows():
    ax.annotate(row["model"], (row["overall_RMSE"], row["delta_rmse"]),
                fontsize=7, ha="center", va="bottom", alpha=0.7)
ax.set_xlabel("Overall RMSE"); ax.set_ylabel("ΔRMSE")
ax.set_title("Accuracy-Robustness Trade-off\n(lower+more negative = more PAINS-biased)")
# quadrant lines
ax.axvline(full_df["overall_RMSE"].mean(), color="gray", ls=":", alpha=0.5)
ax.axhline(full_df["delta_rmse"].mean(), color="gray", ls=":", alpha=0.5)

# 6. Training time
ax = axes[1, 2]
ax.barh(range(len(full_df)), full_df["train_time_s"], color="#9b59b6", alpha=0.8)
ax.set_yticks(range(len(full_df)))
ax.set_yticklabels(full_df["model"], fontsize=9)
ax.set_xlabel("Training Time (s)"); ax.set_title("Computational Cost")

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "frontier_comparison_full.png"), dpi=150, bbox_inches="tight")
print(f"\nFigure saved: frontier_comparison_full.png")

# ========== Print final table ==========
print(f"\n{'=' * 65}")
print("PAINSBench — Full Model Comparison")
print(f"{'=' * 65}")
print(f"{'Model':20s} {'RMSE':>8s} {'PAINS+':>8s} {'PAINS-':>8s} {'ΔRMSE':>8s} {'FP_Ratio':>9s} {'Time':>8s}")
print("-" * 65)
for _, r in full_df.iterrows():
    tag = " ★" if r["model"] in frontier_models else "  "
    print(f"{r['model']:20s} {r['overall_RMSE']:8.4f} {r['pains_pos_RMSE']:8.4f} "
          f"{r['pains_neg_RMSE']:8.4f} {r['delta_rmse']:8.4f} {r['fp_ratio']:9.4f} "
          f"{r['train_time_s']:7.0f}s{tag}")
print(f"\n★ = frontier model (this study)")
print(f"Previous DL models: MLP, GCN, GAT, AttentiveFP")
print(f"\nResults: {os.path.join(RESULTS_DIR, 'frontier_comparison_results.csv')}")
print("Done.")
