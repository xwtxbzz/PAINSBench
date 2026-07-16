"""
Step 12: DTA model comparison by assay type (Ki vs IC50).
Runs all 16 DTA models on a specified assay subset.
Usage:
    python scripts/12_assay_comparison.py --assay ki
    python scripts/12_assay_comparison.py --assay ic50
"""
import os, sys, time, gc, warnings, math, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, RANDOM_SEED

warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

BATCH_SIZE = 128
TRANSFORMER_BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
PROTEIN_MAX_LEN = 1200
DRUG_EMBED_DIM = 128
PROTEIN_EMBED_DIM = 128
SMILES_MAX_LEN = 200

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i+1 for i, aa in enumerate(AA_ORDER)}

def tokenize_sequence(seq, max_len=PROTEIN_MAX_LEN):
    tokens = [AA_TO_IDX.get(aa, len(AA_ORDER)+1) for aa in seq[:max_len]]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    return np.array(tokens, dtype=np.int64)

def compute_aac(seq):
    aa_list = AA_ORDER; total = len(seq)
    if total == 0: return np.zeros(20, dtype=np.float32)
    return np.array([seq.count(aa) / total for aa in aa_list], dtype=np.float32)

def compute_dpc(seq):
    aa_list = AA_ORDER; total = max(len(seq) - 1, 1)
    dpc_vec = np.zeros(400, dtype=np.float32)
    for i in range(len(seq) - 1):
        key = seq[i:i+2]
        if key[0] in aa_list and key[1] in aa_list:
            idx = aa_list.index(key[0]) * 20 + aa_list.index(key[1])
            dpc_vec[idx] += 1.0
    return dpc_vec / total

def build_smiles_vocab(smiles_list):
    chars = set()
    for smi in smiles_list: chars.update(str(smi))
    sorted_chars = sorted(chars)
    return {c: i+2 for i, c in enumerate(sorted_chars)}, len(sorted_chars) + 2

def tokenize_smiles(smiles, vocab, max_len=SMILES_MAX_LEN):
    tokens = [vocab.get(c, 1) for c in str(smiles)[:max_len]]
    if len(tokens) < max_len: tokens += [0] * (max_len - len(tokens))
    return np.array(tokens, dtype=np.int64)

parser = argparse.ArgumentParser()
parser.add_argument("--assay", type=str, required=True, choices=["ki", "ic50"])
args = parser.parse_args()
ASSAY = args.assay.upper()
print(f"\n{'='*60}")
print(f"Assay Type: {ASSAY}")
print(f"{'='*60}")

# ========== Data loading ==========
csv_path = os.path.join(PROCESSED_DIR, f"benchmark_dta_{args.assay}.csv")
print(f"\nLoading {csv_path}...")
bench = pd.read_csv(csv_path)
print(f"  Total: {len(bench):,} (PAINS+ {bench['PAINS_status'].sum():,} / PAINS- {(bench['PAINS_status']==0).sum():,})")

bench_full = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_full.csv"))
bench_full["_orig_idx"] = np.arange(len(bench_full))
sp_indices = bench.merge(
    bench_full[["molregno", "target_chembl_id", "_orig_idx"]],
    on=["molregno", "target_chembl_id"], how="left"
)["_orig_idx"].values

full_data = np.load(os.path.join(PROCESSED_DIR, "features_full.npz"), allow_pickle=True)
X_fp = full_data["X"][sp_indices, :2048]
y = full_data["y"][sp_indices].astype(np.float32)
ps = full_data["pains_status"][sp_indices]
smiles_list = bench["canonical_smiles"].values
sequences = bench["sequence"].values
print(f"  Data shape: {X_fp.shape}, y range: [{y.min():.2f}, {y.max():.2f}]")

print("Computing protein features (AAC+DPC)...")
t0 = time.time()
protein_aac = np.array([compute_aac(s) for s in sequences], dtype=np.float32)
protein_dpc = np.array([compute_dpc(s) for s in sequences], dtype=np.float32)
protein_static = np.concatenate([protein_aac, protein_dpc], axis=1)
print(f"  Done in {time.time()-t0:.1f}s")

print("Tokenizing sequences for CNN...")
protein_tokens = np.array([tokenize_sequence(s) for s in sequences], dtype=np.int64)
print(f"  Token shape: {protein_tokens.shape}")

print("Building SMILES vocabulary...")
smiles_vocab, smiles_vocab_size = build_smiles_vocab(smiles_list)
print(f"  SMILES vocab size: {smiles_vocab_size}")
smiles_tokens = np.array([tokenize_smiles(s, smiles_vocab) for s in smiles_list], dtype=np.int64)

# ========== Train/val/test split ==========
from sklearn.model_selection import train_test_split
N = len(y)
all_idx = np.arange(N)
train_idx, test_idx, _, _, ps_train, ps_test = train_test_split(
    all_idx, y, ps, test_size=0.2, random_state=RANDOM_SEED, stratify=ps)
train_idx, val_idx, y_train, y_val = train_test_split(
    train_idx, y[train_idx], test_size=0.125, random_state=RANDOM_SEED, stratify=ps[train_idx])
print(f"Split: train {len(train_idx):,}  val {len(val_idx):,}  test {len(test_idx):,}")

# ========== Graph construction ==========
print("Building molecular graphs...")
from rdkit import Chem
from torch_geometric.data import Data

ATOM_TYPES = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
BOND_TYPES = [1, 2, 3, 12]

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
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
        edge_index = [[0, 0]]; edge_attr = [[1.0, 0, 0, 0]]
    return Data(x=x, edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float))

graph_cache = {}
def get_graphs(indices, verbose=False):
    graphs, valid_idx = [], []
    for i, idx in enumerate(indices):
        smi = str(smiles_list[idx])
        if smi in graph_cache and graph_cache[smi] is not None:
            g = graph_cache[smi]
        else:
            g = mol_to_graph(smi)
            if g is not None: graph_cache[smi] = g
        if g is not None:
            graphs.append(g); valid_idx.append(idx)
        if verbose and (i+1) % 10000 == 0:
            print(f"    {i+1}/{len(indices)} graphs built...")
    return graphs, np.array(valid_idx, dtype=int)

train_graphs, train_v = get_graphs(train_idx, verbose=True)
val_graphs, val_v = get_graphs(val_idx)
test_graphs, test_v = get_graphs(test_idx)
print(f"  Train: {len(train_graphs)}  Val: {len(val_graphs)}  Test: {len(test_graphs)}")

# Align labels and features
y_train_g = torch.tensor(y[train_v], dtype=torch.float)
y_val_g = torch.tensor(y[val_v], dtype=torch.float)
y_test_g = torch.tensor(y[test_v], dtype=torch.float)
ps_test_g = ps[test_v]
seq_train = protein_tokens[train_v]; seq_val = protein_tokens[val_v]; seq_test = protein_tokens[test_v]
static_train = torch.tensor(protein_static[train_v], dtype=torch.float)
static_val = torch.tensor(protein_static[val_v], dtype=torch.float)
static_test = torch.tensor(protein_static[test_v], dtype=torch.float)
smi_train = smiles_tokens[train_v]; smi_val = smiles_tokens[val_v]; smi_test = smiles_tokens[test_v]

for g, lbl in zip(train_graphs, y_train_g): g.y = lbl
for g, lbl in zip(val_graphs, y_val_g): g.y = lbl
for g, lbl in zip(test_graphs, y_test_g): g.y = lbl

# Degree for PNA
from torch_geometric.utils import degree
def compute_deg(graphs):
    all_deg = torch.cat([degree(g.edge_index[0], num_nodes=g.x.size(0)).long() for g in graphs])
    hist = torch.bincount(all_deg).float()
    if hist.size(0) < 10: hist = torch.cat([hist, torch.zeros(10 - hist.size(0))])
    return hist
dta_deg = compute_deg(train_graphs)

# ==================== Datasets ====================
class DTAGraphDataset(Dataset):
    def __init__(self, graphs, seqs, labels):
        self.graphs = graphs; self.seqs = seqs; self.labels = labels
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx], self.seqs[idx], self.labels[idx]

def dta_collate(batch):
    from torch_geometric.data import Batch
    graphs, seqs, labels = zip(*batch)
    return Batch.from_data_list(list(graphs)), torch.stack(list(seqs)), torch.stack(list(labels))

class SmilesDTADataset(Dataset):
    def __init__(self, smiles_tokens, seqs, labels):
        self.smiles = smiles_tokens; self.seqs = seqs; self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.smiles[idx], self.seqs[idx], self.labels[idx]

def smiles_dta_collate(batch):
    smiles, seqs, labels = zip(*batch)
    return torch.stack(list(smiles)), torch.stack(list(seqs)), torch.stack(list(labels))

# ==================== DataLoaders ====================
fp_train_t = torch.tensor(X_fp[train_v], dtype=torch.float)
fp_val_t = torch.tensor(X_fp[val_v], dtype=torch.float)
fp_test_t = torch.tensor(X_fp[test_v], dtype=torch.float)
mlp_train_input = torch.cat([fp_train_t, static_train], dim=1)
mlp_val_input = torch.cat([fp_val_t, static_val], dim=1)
mlp_test_input = torch.cat([fp_test_t, static_test], dim=1)

mlp_train_ds = torch.utils.data.TensorDataset(mlp_train_input, y_train_g)
mlp_val_ds = torch.utils.data.TensorDataset(mlp_val_input, y_val_g)
mlp_test_ds = torch.utils.data.TensorDataset(mlp_test_input, y_test_g)
mlp_train_loader = DataLoader(mlp_train_ds, batch_size=BATCH_SIZE*2, shuffle=True)
mlp_val_loader = DataLoader(mlp_val_ds, batch_size=BATCH_SIZE*2)
mlp_test_loader = DataLoader(mlp_test_ds, batch_size=BATCH_SIZE*2)

gnn_train_dataset = DTAGraphDataset(train_graphs, torch.tensor(protein_tokens[train_v]), y_train_g)
gnn_val_dataset = DTAGraphDataset(val_graphs, torch.tensor(protein_tokens[val_v]), y_val_g)
gnn_test_dataset = DTAGraphDataset(test_graphs, torch.tensor(protein_tokens[test_v]), y_test_g)
gnn_train_loader = DataLoader(gnn_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dta_collate)
gnn_val_loader = DataLoader(gnn_val_dataset, batch_size=BATCH_SIZE, collate_fn=dta_collate)
gnn_test_loader = DataLoader(gnn_test_dataset, batch_size=BATCH_SIZE, collate_fn=dta_collate)

gnn_train_loader_small = DataLoader(gnn_train_dataset, batch_size=TRANSFORMER_BATCH_SIZE,
                                    shuffle=True, collate_fn=dta_collate)
gnn_val_loader_small = DataLoader(gnn_val_dataset, batch_size=TRANSFORMER_BATCH_SIZE, collate_fn=dta_collate)
gnn_test_loader_small = DataLoader(gnn_test_dataset, batch_size=TRANSFORMER_BATCH_SIZE, collate_fn=dta_collate)

# ==================== Model Components ====================
class ProteinCNN(nn.Module):
    def __init__(self, vocab_size=25, embed_dim=32, hidden=PROTEIN_EMBED_DIM, kernel_size=5, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, hidden, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, seq):
        x = self.embedding(seq).permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x))); x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        return self.pool(x).squeeze(-1)

# Drug encoders
class MLPNet(nn.Module):
    def __init__(self, in_dim=2048+420, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden//2, hidden//4), nn.ReLU(), nn.Linear(hidden//4, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class MLPDrug(nn.Module):
    def __init__(self, in_dim=2048, hidden=1024, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden//2, hidden//4), nn.ReLU(), nn.Linear(hidden//4, out_dim))
    def forward(self, x): return self.net(x)

class GCNDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
        self.conv1 = GCNConv(in_dim, hidden); self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden//2); self.proj = nn.Linear(hidden//2, out_dim)
        self.pool = global_mean_pool
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
        return self.proj(self.pool(x, data.batch))

class GATDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, heads=4, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import GATConv, global_mean_pool
        self.conv1 = GATConv(in_dim, hidden//heads, heads=heads)
        self.conv2 = GATConv(hidden, hidden//heads, heads=heads)
        self.conv3 = GATConv(hidden, hidden//2, heads=1)
        self.proj = nn.Linear(hidden//2, out_dim); self.pool = global_mean_pool
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
        return self.proj(self.pool(x, data.batch))

class AttentiveFPDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import AttentiveFP
        self.encoder = nn.Linear(in_dim, hidden)
        self.attentive_fp = AttentiveFP(hidden, hidden, out_dim, edge_dim=len(BOND_TYPES), num_layers=2, num_timesteps=2, dropout=0.2)
    def forward(self, data):
        return self.attentive_fp(self.encoder(data.x), data.edge_index, data.edge_attr, data.batch)

class GINDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import GINConv, global_add_pool
        self.conv1 = GINConv(nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)), train_eps=True)
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)), train_eps=True)
        self.conv3 = GINConv(nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, hidden//2)), train_eps=True)
        self.proj = nn.Linear(hidden+hidden+hidden//2, out_dim); self.pool = global_add_pool
    def forward(self, data):
        x1 = self.conv1(data.x, data.edge_index); x2 = self.conv2(F.relu(x1), data.edge_index)
        x3 = self.conv3(F.relu(x2), data.edge_index)
        return self.proj(torch.cat([self.pool(x1, data.batch), self.pool(x2, data.batch), self.pool(x3, data.batch)], dim=1))

class GINEDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import GINEConv, global_add_pool
        self.conv1 = GINEConv(nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)), train_eps=True, edge_dim=len(BOND_TYPES))
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)), train_eps=True, edge_dim=len(BOND_TYPES))
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, hidden//2)), train_eps=True, edge_dim=len(BOND_TYPES))
        self.proj = nn.Linear(hidden+hidden+hidden//2, out_dim); self.pool = global_add_pool
    def forward(self, data):
        x1 = self.conv1(data.x, data.edge_index, data.edge_attr)
        x2 = self.conv2(F.relu(x1), data.edge_index, data.edge_attr)
        x3 = self.conv3(F.relu(x2), data.edge_index, data.edge_attr)
        return self.proj(torch.cat([self.pool(x1, data.batch), self.pool(x2, data.batch), self.pool(x3, data.batch)], dim=1))

class PNADrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=80, out_dim=DRUG_EMBED_DIM, deg=None):
        super().__init__()
        from torch_geometric.nn import PNAConv, global_mean_pool
        if deg is None: deg = torch.ones(1)
        aggr = ["mean","max","min","std"]; scalers = ["identity","amplification","attenuation"]
        self.conv1 = PNAConv(in_dim, hidden, aggregators=aggr, scalers=scalers, deg=deg, edge_dim=len(BOND_TYPES))
        self.conv2 = PNAConv(hidden, hidden, aggregators=aggr, scalers=scalers, deg=deg, edge_dim=len(BOND_TYPES))
        self.conv3 = PNAConv(hidden, hidden//2, aggregators=aggr, scalers=scalers, deg=deg, edge_dim=len(BOND_TYPES))
        self.proj = nn.Linear(hidden+hidden+hidden//2, out_dim); self.pool = global_mean_pool
    def forward(self, data):
        x1 = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x2 = F.relu(self.conv2(x1, data.edge_index, data.edge_attr))
        x3 = F.relu(self.conv3(x2, data.edge_index, data.edge_attr))
        return self.proj(torch.cat([self.pool(x1, data.batch), self.pool(x2, data.batch), self.pool(x3, data.batch)], dim=1))

class GTDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, heads=4, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import TransformerConv, global_mean_pool
        self.conv1 = TransformerConv(in_dim, hidden//heads, heads=heads, edge_dim=len(BOND_TYPES), dropout=0.2)
        self.conv2 = TransformerConv(hidden, hidden//heads, heads=heads, edge_dim=len(BOND_TYPES), dropout=0.2)
        self.conv3 = TransformerConv(hidden, hidden//2, heads=1, edge_dim=len(BOND_TYPES), dropout=0.2)
        self.proj = nn.Linear(hidden+hidden+hidden//2, out_dim); self.pool = global_mean_pool
    def forward(self, data):
        x1 = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x2 = F.relu(self.conv2(x1, data.edge_index, data.edge_attr))
        x3 = F.relu(self.conv3(x2, data.edge_index, data.edge_attr))
        return self.proj(torch.cat([self.pool(x1, data.batch), self.pool(x2, data.batch), self.pool(x3, data.batch)], dim=1))

class GPSDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, heads=4, out_dim=DRUG_EMBED_DIM, num_layers=3):
        super().__init__()
        from torch_geometric.nn import GPSConv, GINConv, global_mean_pool
        self.node_emb = nn.Linear(in_dim, hidden); self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GPSConv(hidden, conv=GINConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))), heads=heads, dropout=0.2))
        self.proj = nn.Linear(hidden, out_dim); self.pool = global_mean_pool
    def forward(self, data):
        x = self.node_emb(data.x)
        for conv in self.convs: x = conv(x, data.edge_index, data.batch)
        return self.proj(self.pool(x, data.batch))

class SAGEDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import SAGEConv, global_mean_pool
        self.conv1 = SAGEConv(in_dim, hidden); self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden//2); self.proj = nn.Linear(hidden//2, out_dim)
        self.pool = global_mean_pool
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
        return self.proj(self.pool(x, data.batch))

class GATv2Drug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, heads=4, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import GATv2Conv, global_mean_pool
        self.conv1 = GATv2Conv(in_dim, hidden//heads, heads=heads, edge_dim=len(BOND_TYPES))
        self.conv2 = GATv2Conv(hidden, hidden//heads, heads=heads, edge_dim=len(BOND_TYPES))
        self.conv3 = GATv2Conv(hidden, hidden//2, heads=1, edge_dim=len(BOND_TYPES))
        self.proj = nn.Linear(hidden//2, out_dim); self.pool = global_mean_pool
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.relu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.relu(self.conv3(x, data.edge_index, data.edge_attr))
        return self.proj(self.pool(x, data.batch))

class GENDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import GENConv, global_mean_pool
        self.conv1 = GENConv(in_dim, hidden); self.conv2 = GENConv(hidden, hidden)
        self.conv3 = GENConv(hidden, hidden//2); self.proj = nn.Linear(hidden//2, out_dim)
        self.pool = global_mean_pool
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
        return self.proj(self.pool(x, data.batch))

class DTAModel(nn.Module):
    def __init__(self, drug_encoder, protein_encoder, fusion_in=DRUG_EMBED_DIM+PROTEIN_EMBED_DIM):
        super().__init__()
        self.drug_encoder = drug_encoder; self.protein_encoder = protein_encoder
        self.fc = nn.Sequential(nn.Linear(fusion_in, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, drug_input, seq):
        return self.fc(torch.cat([self.drug_encoder(drug_input), self.protein_encoder(seq)], dim=1)).squeeze(-1)

# ==================== Frontier Model Components ====================
# GS-DTA
class GSDTA_DrugEncoder(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool
        self.gat1 = GATv2Conv(in_dim, hidden//4, heads=4, edge_dim=len(BOND_TYPES))
        self.gcn1 = GCNConv(in_dim, hidden)
        self.gat2 = GATv2Conv(hidden, hidden//4, heads=4, edge_dim=len(BOND_TYPES))
        self.gcn2 = GCNConv(hidden, hidden)
        self.gat3 = GATv2Conv(hidden, hidden, heads=1, edge_dim=len(BOND_TYPES))
        self.gcn3 = GCNConv(hidden, hidden)
        self.pool = global_mean_pool; self.proj = nn.Linear(hidden*3, out_dim)
    def forward(self, data):
        attn1 = F.relu(self.gat1(data.x, data.edge_index, data.edge_attr))
        x1 = F.relu(self.gcn1(data.x, data.edge_index)) + attn1
        attn2 = F.relu(self.gat2(x1, data.edge_index, data.edge_attr))
        x2 = F.relu(self.gcn2(x1, data.edge_index)) + attn2
        attn3 = F.relu(self.gat3(x2, data.edge_index, data.edge_attr))
        x3 = F.relu(self.gcn3(x2, data.edge_index)) + attn3
        return self.proj(torch.cat([self.pool(x1, data.batch), self.pool(x2, data.batch), self.pool(x3, data.batch)], dim=1))

class GSDTA_ProteinEncoder(nn.Module):
    def __init__(self, vocab_size=25, embed_dim=32, hidden=64, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, hidden, 5, stride=2, padding=2); self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, 5, stride=2, padding=2); self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = nn.Conv1d(hidden, hidden, 5, padding=2); self.bn3 = nn.BatchNorm1d(hidden)
        trans_layer = nn.TransformerEncoderLayer(hidden, nhead=2, dim_feedforward=hidden*2, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(trans_layer, num_layers=2)
        self.pool = nn.AdaptiveMaxPool1d(1); self.proj = nn.Linear(hidden, PROTEIN_EMBED_DIM)
        self.dropout = nn.Dropout(dropout)
    def forward(self, seq):
        x = self.embedding(seq).permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x))); x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x))); x = self.dropout(x)
        x = self.transformer(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.proj(self.pool(x).squeeze(-1))

class GSDTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_encoder = GSDTA_DrugEncoder(); self.protein_encoder = GSDTA_ProteinEncoder()
        self.fc = nn.Sequential(nn.Linear(DRUG_EMBED_DIM+PROTEIN_EMBED_DIM, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, drug_input, seq):
        return self.fc(torch.cat([self.drug_encoder(drug_input), self.protein_encoder(seq)], dim=1)).squeeze(-1)

# Mamba-DTA
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner*2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner, bias=False)
        self.channel_mix = nn.Sequential(nn.Linear(self.d_inner, self.d_inner), nn.GELU())
        self.out_proj = nn.Linear(self.d_inner, d_model)
    def forward(self, x):
        xz = self.in_proj(x); x_half, z = xz.chunk(2, dim=-1)
        x_conv = F.silu(self.conv1d(x_half.transpose(1, 2))[..., :x.size(1)]).transpose(1, 2)
        y = self.out_proj(self.channel_mix(x_conv) * F.silu(z))
        return y

class MambaProtein(nn.Module):
    def __init__(self, vocab_size=25, embed_dim=32, hidden=PROTEIN_EMBED_DIM, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, hidden, 5, padding=2); self.bn1 = nn.BatchNorm1d(hidden)
        self.mamba = MambaBlock(d_model=hidden, d_conv=4, expand=2)
        self.pool = nn.AdaptiveMaxPool1d(1); self.dropout = nn.Dropout(dropout)
    def forward(self, seq):
        x = self.embedding(seq).permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x))); x = self.dropout(x)
        return self.pool(self.mamba(x.permute(0, 2, 1)).permute(0, 2, 1)).squeeze(-1)

class MambaDTA(nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import TransformerConv, global_mean_pool
        drug_in = len(ATOM_TYPES)+3
        self.gt1 = TransformerConv(drug_in, 128//4, heads=4, edge_dim=len(BOND_TYPES), dropout=0.2)
        self.gt2 = TransformerConv(128, 128//4, heads=4, edge_dim=len(BOND_TYPES), dropout=0.2)
        self.gt3 = TransformerConv(128, 64, heads=1, edge_dim=len(BOND_TYPES), dropout=0.2)
        self.drug_proj = nn.Linear(128+128+64, DRUG_EMBED_DIM); self.pool = global_mean_pool
        self.protein_encoder = MambaProtein()
        self.fc = nn.Sequential(nn.Linear(DRUG_EMBED_DIM+PROTEIN_EMBED_DIM, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, drug_input, seq):
        x1 = F.relu(self.gt1(drug_input.x, drug_input.edge_index, drug_input.edge_attr))
        x2 = F.relu(self.gt2(x1, drug_input.edge_index, drug_input.edge_attr))
        x3 = F.relu(self.gt3(x2, drug_input.edge_index, drug_input.edge_attr))
        drug_embed = self.drug_proj(torch.cat([self.pool(x1, drug_input.batch), self.pool(x2, drug_input.batch), self.pool(x3, drug_input.batch)], dim=1))
        return self.fc(torch.cat([drug_embed, self.protein_encoder(seq)], dim=1)).squeeze(-1)

# CrossAttn-DTA
class CrossAttnDTAModel(nn.Module):
    def __init__(self, drug_encoder, protein_encoder, d_model=DRUG_EMBED_DIM, nhead=2):
        super().__init__()
        self.drug_encoder = drug_encoder; self.protein_encoder = protein_encoder
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(nn.Linear(d_model*2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, drug_input, seq):
        drug_embed = self.drug_encoder(drug_input); protein_embed = self.protein_encoder(seq)
        attended, _ = self.cross_attn(drug_embed.unsqueeze(1), protein_embed.unsqueeze(1), protein_embed.unsqueeze(1))
        attended = self.norm(attended.squeeze(1) + drug_embed)
        return self.fc(torch.cat([attended, protein_embed], dim=1)).squeeze(-1)

# TransformerProt-DTA
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model)*0.02)
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class TransformerProtein(nn.Module):
    def __init__(self, vocab_size=25, embed_dim=32, hidden=PROTEIN_EMBED_DIM, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 64, 5, stride=2, padding=2); self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, hidden, 5, stride=2, padding=2); self.bn2 = nn.BatchNorm1d(hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.pos_enc = LearnedPositionalEncoding(hidden, max_len=300)
        trans_layer = nn.TransformerEncoderLayer(hidden, nhead=2, dim_feedforward=hidden*2, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(trans_layer, num_layers=2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, seq):
        x = self.embedding(seq).permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x))); x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.dropout(x)
        return self.transformer(self.pos_enc(self.proj(x.permute(0, 2, 1)))).mean(dim=1)

# ==================== Training ====================
def train_dta(model, train_loader, val_loader, epochs=EPOCHS, is_mlp=False, is_text=False):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train(); train_loss, n_train = 0.0, 0
        for batch in train_loader:
            opt.zero_grad()
            if is_mlp: x, y_b = batch; x, y_b = x.to(device), y_b.to(device); pred = model(x)
            elif is_text: drug_tokens, seq, y_b = batch; drug_tokens, seq, y_b = drug_tokens.to(device), seq.to(device), y_b.to(device); pred = model(drug_tokens, seq)
            else: batch_graph, seq, y_b = batch; batch_graph, seq, y_b = batch_graph.to(device), seq.to(device), y_b.to(device); pred = model(batch_graph, seq)
            loss = F.mse_loss(pred, y_b); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); opt.step()
            if torch.isnan(loss) or torch.isinf(loss): print(f"  WARNING: NaN/Inf at epoch {epoch+1}"); return model
            train_loss += loss.item()*len(y_b); n_train += len(y_b)
        sched.step()
        model.eval(); val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if is_mlp: x, y_b = batch; pred = model(x.to(device))
                elif is_text: drug_tokens, seq, y_b = batch; pred = model(drug_tokens.to(device), seq.to(device))
                else: batch_graph, seq, y_b = batch; pred = model(batch_graph.to(device), seq.to(device))
                val_loss += F.mse_loss(pred, y_b.to(device)).item()*len(y_b); n_val += len(y_b)
        val_loss /= n_val
        if val_loss < best_val_loss: best_val_loss = val_loss; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch+1)%10==0: print(f"  Epoch {epoch+1:3d}  train_loss={train_loss/n_train:.4f}  val_loss={val_loss:.4f}")
    model.load_state_dict(best_state); return model

def predict_dta(model, loader, is_mlp=False, is_text=False):
    model.eval(); all_pred = []
    with torch.no_grad():
        for batch in loader:
            if is_mlp: x, _ = batch; pred = model(x.to(device))
            elif is_text: drug_tokens, seq, _ = batch; pred = model(drug_tokens.to(device), seq.to(device))
            else: batch_graph, seq, _ = batch; pred = model(batch_graph.to(device), seq.to(device))
            all_pred.append(pred.cpu().numpy())
    return np.concatenate(all_pred)

# ==================== Model Registry ====================
from src.evaluation import evaluate_pains_aware
from src.visualization import plot_scatter, plot_residual_distribution

def build_dta(drug_cls, **kwargs):
    return DTAModel(drug_cls(in_dim=len(ATOM_TYPES)+3, **kwargs), ProteinCNN())

results_rows = []
out_path = os.path.join(RESULTS_DIR, f"dta_{args.assay}_results.csv")
completed_models = set()
if os.path.exists(out_path):
    existing = pd.read_csv(out_path); completed_models = set(existing["model"].tolist())
    print(f"Already completed: {completed_models}")

# ==================== 1. MLP-DTA ====================
name = "MLP-DTA"
if name in completed_models: print(f"  Skipping {name}")
else:
    print(f"\n{'='*55}\nTraining {name}..."); sys.stdout.flush(); gc.collect(); torch.cuda.empty_cache()
    t0 = time.time()
    model = train_dta(MLPNet(in_dim=2048+420), mlp_train_loader, mlp_val_loader, is_mlp=True)
    y_pred = predict_dta(model, mlp_test_loader, is_mlp=True)
    elapsed = time.time()-t0; print(f"  Done in {elapsed:.1f}s")
    eval_dict = evaluate_pains_aware(y_test_g.numpy(), y_pred, ps_test_g)
    eval_dict.update({"model": name, "train_time_s": elapsed})
    results_rows.append(eval_dict); print(f"  RMSE: {eval_dict['overall_RMSE']:.4f}  ΔRMSE: {eval_dict['delta_rmse']:.4f}")
    pd.DataFrame(results_rows).to_csv(out_path, index=False)

# ==================== 2-12. GNN DTA models ====================
gnn_models = {"GCN-DTA": GCNDrug, "GAT-DTA": GATDrug, "AttFP-DTA": AttentiveFPDrug,
    "GIN-DTA": GINDrug, "GINE-DTA": GINEDrug, "PNA-DTA": PNADrug, "GT-DTA": GTDrug,
    "GPS-DTA": GPSDrug, "SAGE-DTA": SAGEDrug, "GATv2-DTA": GATv2Drug, "GEN-DTA": GENDrug}

for name, drug_cls in gnn_models.items():
    if name in completed_models: print(f"  Skipping {name}"); continue
    print(f"\n{'='*55}\nTraining {name}..."); sys.stdout.flush(); gc.collect(); torch.cuda.empty_cache()
    t0 = time.time()
    kwargs = {"deg": dta_deg} if name == "PNA-DTA" else {}
    model = train_dta(build_dta(drug_cls, **kwargs), gnn_train_loader, gnn_val_loader)
    y_pred = predict_dta(model, gnn_test_loader)
    elapsed = time.time()-t0; print(f"  Done in {elapsed:.1f}s")
    eval_dict = evaluate_pains_aware(y_test_g.numpy(), y_pred, ps_test_g)
    eval_dict.update({"model": name, "train_time_s": elapsed})
    results_rows.append(eval_dict)
    print(f"  RMSE: {eval_dict['overall_RMSE']:.4f}  ΔRMSE: {eval_dict['delta_rmse']:.4f}")
    plot_scatter(y_test_g.numpy(), y_pred, ps_test_g, f"DTA_{name}_{ASSAY}", filename=f"scatter_DTA_{name}_{ASSAY}.png")
    pd.DataFrame(results_rows).to_csv(out_path, index=False)

# ==================== 13. GS-DTA ====================
for name, cls, bs in [("GS-DTA", GSDTA, BATCH_SIZE), ("Mamba-DTA", MambaDTA, BATCH_SIZE),
                       ("CrossAttn-DTA", lambda: CrossAttnDTAModel(GATv2Drug(in_dim=len(ATOM_TYPES)+3), ProteinCNN()), BATCH_SIZE),
                       ("TransformerProt-DTA", lambda: DTAModel(GCNDrug(in_dim=len(ATOM_TYPES)+3), TransformerProtein()), TRANSFORMER_BATCH_SIZE)]:
    if name in completed_models: print(f"  Skipping {name}"); continue
    print(f"\n{'='*55}\nTraining {name} ({ASSAY})..."); sys.stdout.flush(); gc.collect(); torch.cuda.empty_cache()
    t0 = time.time()
    train_loader = gnn_train_loader_small if bs == TRANSFORMER_BATCH_SIZE else gnn_train_loader
    val_loader = gnn_val_loader_small if bs == TRANSFORMER_BATCH_SIZE else gnn_val_loader
    test_loader = gnn_test_loader_small if bs == TRANSFORMER_BATCH_SIZE else gnn_test_loader
    model = train_dta(cls(), train_loader, val_loader)
    y_pred = predict_dta(model, test_loader)
    elapsed = time.time()-t0; print(f"  Done in {elapsed:.1f}s")
    eval_dict = evaluate_pains_aware(y_test_g.numpy(), y_pred, ps_test_g)
    eval_dict.update({"model": name, "train_time_s": elapsed})
    results_rows.append(eval_dict)
    print(f"  RMSE: {eval_dict['overall_RMSE']:.4f}  ΔRMSE: {eval_dict['delta_rmse']:.4f}")
    plot_scatter(y_test_g.numpy(), y_pred, ps_test_g, f"DTA_{name}_{ASSAY}", filename=f"scatter_DTA_{name}_{ASSAY}.png")
    pd.DataFrame(results_rows).to_csv(out_path, index=False)

# ==================== Final Summary ====================
results_df = pd.DataFrame(results_rows)
results_df.to_csv(out_path, index=False)
print(f"\n{'='*70}")
print(f"PAINSBench-DTA: {ASSAY} Results")
print(f"{'='*70}")
print(f"{'Model':20s} {'RMSE':>8s} {'PAINS+':>8s} {'PAINS-':>8s} {'ΔRMSE':>8s} {'FP_Ratio':>9s} {'Time':>8s}")
print("-"*70)
for _, r in results_df.iterrows():
    print(f"{r['model']:20s} {r['overall_RMSE']:8.4f} {r['pains_pos_RMSE']:8.4f} {r['pains_neg_RMSE']:8.4f} {r['delta_rmse']:8.4f} {r['fp_ratio']:9.4f} {r['train_time_s']:7.0f}s")
print(f"\nResults saved: {out_path}")
print(f"Done. ({ASSAY})")
