"""
Step 9: DTA (Drug-Target Affinity) full model comparison.
Uses single-protein benchmark (112K) with protein sequences.
12 dual-branch DTA models: drug encoder + protein encoder → fusion.
"""
import os, sys, time, gc, warnings, math
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
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
PROTEIN_MAX_LEN = 1200
DRUG_EMBED_DIM = 128
PROTEIN_EMBED_DIM = 128

# ========== Amino acid vocabulary ==========
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i+1 for i, aa in enumerate(AA_ORDER)}  # 0 = padding

def tokenize_sequence(seq, max_len=PROTEIN_MAX_LEN):
    """Convert AA sequence to integer tokens, pad/truncate."""
    tokens = [AA_TO_IDX.get(aa, len(AA_ORDER)+1) for aa in seq[:max_len]]  # unknowns → 22
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    return np.array(tokens, dtype=np.int64)

def compute_aac(seq):
    """20-dim amino acid composition."""
    aa_list = AA_ORDER
    total = len(seq)
    if total == 0:
        return np.zeros(20, dtype=np.float32)
    return np.array([seq.count(aa) / total for aa in aa_list], dtype=np.float32)

def compute_dpc(seq):
    """400-dim dipeptide composition."""
    aa_list = AA_ORDER
    total = max(len(seq) - 1, 1)
    dpc_vec = np.zeros(400, dtype=np.float32)
    for i in range(len(seq) - 1):
        key = seq[i:i+2]
        if key[0] in aa_list and key[1] in aa_list:
            idx = aa_list.index(key[0]) * 20 + aa_list.index(key[1])
            dpc_vec[idx] += 1.0
    return dpc_vec / total

# ========== Data loading ==========
print("\nLoading single-protein benchmark...")
bench = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_dta_full.csv"))
print(f"  Total: {len(bench):,} (PAINS+ {bench['PAINS_status'].sum():,} / PAINS- {(bench['PAINS_status']==0).sum():,})")

# Align with features_full.npz via molregno+target merge
bench_full = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_full.csv"))
bench_full["_orig_idx"] = np.arange(len(bench_full))
sp_indices = bench.merge(
    bench_full[["molregno", "target_chembl_id", "_orig_idx"]],
    on=["molregno", "target_chembl_id"], how="left"
)["_orig_idx"].values

# Load features_full.npz (rows aligned with benchmark_full.csv)
full_data = np.load(os.path.join(PROCESSED_DIR, "features_full.npz"), allow_pickle=True)
X_fp = full_data["X"][sp_indices, :2048]  # Morgan FP only
y = full_data["y"][sp_indices].astype(np.float32)
ps = full_data["pains_status"][sp_indices]
smiles_list = bench["canonical_smiles"].values
sequences = bench["sequence"].values
target_ids = bench["target_chembl_id"].values

print(f"  Data shape: {X_fp.shape}, y range: [{y.min():.2f}, {y.max():.2f}]")

# Precompute protein static features (AAC+DPC)
print("\nComputing protein features (AAC+DPC)...")
t0 = time.time()
protein_aac = np.array([compute_aac(s) for s in sequences], dtype=np.float32)
protein_dpc = np.array([compute_dpc(s) for s in sequences], dtype=np.float32)
protein_static = np.concatenate([protein_aac, protein_dpc], axis=1)  # (N, 420)
print(f"  Done in {time.time()-t0:.1f}s, shape={protein_static.shape}")

# Tokenize sequences for CNN
print("Tokenizing sequences for CNN...")
protein_tokens = np.array([tokenize_sequence(s) for s in sequences], dtype=np.int64)
print(f"  Token shape: {protein_tokens.shape}")

# ========== Train/val/test split ==========
from sklearn.model_selection import train_test_split

N = len(y)
all_idx = np.arange(N)
train_idx, test_idx, _, _, ps_train, ps_test = train_test_split(
    all_idx, y, ps, test_size=0.2, random_state=RANDOM_SEED, stratify=ps)
train_idx, val_idx, y_train, y_val = train_test_split(
    train_idx, y[train_idx], test_size=0.125, random_state=RANDOM_SEED, stratify=ps[train_idx])

print(f"\nSplit: train {len(train_idx):,}  val {len(val_idx):,}  test {len(test_idx):,}")
print(f"  Train PAINS+: {ps_train.sum():,}  PAINS-: {(ps_train==0).sum():,}")

# ========== Graph construction ==========
print("\nBuilding molecular graphs...")
from rdkit import Chem
from torch_geometric.data import Data, DataLoader as PyGDataLoader

ATOM_TYPES = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
BOND_TYPES = [1, 2, 3, 12]

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_types = [a.GetAtomicNum() for a in mol.GetAtoms()]
    x = torch.zeros(len(atom_types), len(ATOM_TYPES) + 3)
    for i, at in enumerate(atom_types):
        if at in ATOM_TYPES:
            x[i, ATOM_TYPES.index(at)] = 1
        else:
            x[i, -3] = 1
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
    return Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )


graph_cache = {}

def get_graphs(indices, verbose=False):
    graphs, valid_idx = [], []
    for i, idx in enumerate(indices):
        smi = str(smiles_list[idx])
        if smi in graph_cache and graph_cache[smi] is not None:
            g = graph_cache[smi]
        else:
            g = mol_to_graph(smi)
            if g is not None:
                graph_cache[smi] = g
        if g is not None:
            graphs.append(g)
            valid_idx.append(idx)
        if verbose and (i + 1) % 10000 == 0:
            print(f"    {i+1}/{len(indices)} graphs built...")
    return graphs, np.array(valid_idx, dtype=int)


train_graphs, train_v = get_graphs(train_idx, verbose=True)
val_graphs, val_v = get_graphs(val_idx)
test_graphs, test_v = get_graphs(test_idx)
print(f"  Train: {len(train_graphs)}  Val: {len(val_graphs)}  Test: {len(test_graphs)}")

# Compute degree histogram for PNA
from torch_geometric.utils import degree

def compute_deg(graphs):
    all_deg = []
    for g in graphs:
        d = degree(g.edge_index[0], num_nodes=g.x.size(0)).long()
        all_deg.append(d)
    all_deg = torch.cat(all_deg)
    hist = torch.bincount(all_deg).float()
    if hist.size(0) < 10:
        hist = torch.cat([hist, torch.zeros(10 - hist.size(0))])
    return hist

dta_deg = compute_deg(train_graphs)
print(f"  DTA degree histogram (0..{len(dta_deg)-1}): max={dta_deg.argmax().item()}({dta_deg.max().item():.0f})")

# Align labels and protein features
y_train_g = torch.tensor(y[train_v], dtype=torch.float)
y_val_g = torch.tensor(y[val_v], dtype=torch.float)
y_test_g = torch.tensor(y[test_v], dtype=torch.float)
ps_test_g = ps[test_v]

seq_train = protein_tokens[train_v]
seq_val = protein_tokens[val_v]
seq_test = protein_tokens[test_v]

static_train = torch.tensor(protein_static[train_v], dtype=torch.float)
static_val = torch.tensor(protein_static[val_v], dtype=torch.float)
static_test = torch.tensor(protein_static[test_v], dtype=torch.float)

# FP data for MLP (pre-concatenated with AAC+DPC)
fp_train = torch.tensor(X_fp[train_v], dtype=torch.float)
fp_val = torch.tensor(X_fp[val_v], dtype=torch.float)
fp_test = torch.tensor(X_fp[test_v], dtype=torch.float)

# MLP-DTA dataset: FP + AAC+DPC pre-concatenated
mlp_train_input = torch.cat([fp_train, static_train], dim=1)
mlp_val_input = torch.cat([fp_val, static_val], dim=1)
mlp_test_input = torch.cat([fp_test, static_test], dim=1)

# Set graph labels
for g, lbl in zip(train_graphs, y_train_g):
    g.y = lbl
for g, lbl in zip(val_graphs, y_val_g):
    g.y = lbl
for g, lbl in zip(test_graphs, y_test_g):
    g.y = lbl

# ==================== DTA Dataset ====================
class DTAGraphDataset(Dataset):
    """For GNN-DTA models: graph + seq tokens."""
    def __init__(self, graphs, seqs, labels):
        self.graphs = graphs
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.seqs[idx], self.labels[idx]


def dta_collate(batch):
    """Collate for DTAGraphDataset: batch graphs, pad sequences."""
    graphs, seqs, labels = zip(*batch)
    # PyG handles graph batching via its own Batch class
    from torch_geometric.data import Batch
    batch_graph = Batch.from_data_list(list(graphs))
    seq_tensor = torch.stack(list(seqs))
    label_tensor = torch.stack(list(labels))
    return batch_graph, seq_tensor, label_tensor


# ==================== Protein Encoder ====================
class ProteinCNN(nn.Module):
    """1D CNN for protein sequence encoding."""
    def __init__(self, vocab_size=25, embed_dim=32, hidden=PROTEIN_EMBED_DIM,
                 kernel_size=5, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, hidden, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        x = self.embedding(seq)  # (B, L, embed_dim)
        x = x.permute(0, 2, 1)  # (B, embed_dim, L)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # (B, hidden)
        return x


# ==================== Drug Encoders (return embeddings) ====================
class MLPNet(nn.Module):
    """End-to-end MLP for fingerprint+protein features (no DTA wrapper needed)."""
    def __init__(self, in_dim=2048 + 420, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPDrug(nn.Module):
    def __init__(self, in_dim=2048, hidden=1024, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GCNDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden // 2)
        self.proj = nn.Linear(hidden // 2, out_dim)
        self.pool = global_mean_pool

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
        x = self.pool(x, data.batch)
        return self.proj(x)


class GATDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, heads=4, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import GATConv, global_mean_pool
        self.conv1 = GATConv(in_dim, hidden // heads, heads=heads)
        self.conv2 = GATConv(hidden, hidden // heads, heads=heads)
        self.conv3 = GATConv(hidden, hidden // 2, heads=1)
        self.proj = nn.Linear(hidden // 2, out_dim)
        self.pool = global_mean_pool

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
        x = self.pool(x, data.batch)
        return self.proj(x)


class AttentiveFPDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import AttentiveFP
        self.encoder = nn.Linear(in_dim, hidden)
        self.attentive_fp = AttentiveFP(
            hidden, hidden, out_dim,  # use out_dim as final channels
            edge_dim=len(BOND_TYPES), num_layers=2, num_timesteps=2, dropout=0.2
        )

    def forward(self, data):
        x = self.encoder(data.x)
        return self.attentive_fp(x, data.edge_index, data.edge_attr, data.batch)


class GINDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import GINConv, global_add_pool
        nn1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        nn2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        nn3 = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                           nn.Linear(hidden // 2, hidden // 2))
        self.conv1 = GINConv(nn1, train_eps=True)
        self.conv2 = GINConv(nn2, train_eps=True)
        self.conv3 = GINConv(nn3, train_eps=True)
        self.proj = nn.Linear(hidden + hidden + hidden // 2, out_dim)
        self.pool = global_add_pool

    def forward(self, data):
        x = data.x
        x1 = self.conv1(x, data.edge_index)
        x2 = self.conv2(F.relu(x1), data.edge_index)
        x3 = self.conv3(F.relu(x2), data.edge_index)
        out = torch.cat([self.pool(x1, data.batch), self.pool(x2, data.batch),
                        self.pool(x3, data.batch)], dim=1)
        return self.proj(out)


class GINEDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import GINEConv, global_add_pool
        nn1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        nn2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        nn3 = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                           nn.Linear(hidden // 2, hidden // 2))
        self.conv1 = GINEConv(nn1, train_eps=True, edge_dim=len(BOND_TYPES))
        self.conv2 = GINEConv(nn2, train_eps=True, edge_dim=len(BOND_TYPES))
        self.conv3 = GINEConv(nn3, train_eps=True, edge_dim=len(BOND_TYPES))
        self.proj = nn.Linear(hidden + hidden + hidden // 2, out_dim)
        self.pool = global_add_pool

    def forward(self, data):
        x1 = self.conv1(data.x, data.edge_index, data.edge_attr)
        x2 = self.conv2(F.relu(x1), data.edge_index, data.edge_attr)
        x3 = self.conv3(F.relu(x2), data.edge_index, data.edge_attr)
        out = torch.cat([self.pool(x1, data.batch), self.pool(x2, data.batch),
                        self.pool(x3, data.batch)], dim=1)
        return self.proj(out)


class PNADrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=80, out_dim=DRUG_EMBED_DIM, deg=None):
        super().__init__()
        from torch_geometric.nn import PNAConv, global_mean_pool
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
        self.proj = nn.Linear(hidden + hidden + hidden // 2, out_dim)
        self.pool = global_mean_pool

    def forward(self, data):
        x1 = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x2 = F.relu(self.conv2(x1, data.edge_index, data.edge_attr))
        x3 = F.relu(self.conv3(x2, data.edge_index, data.edge_attr))
        out = torch.cat([self.pool(x1, data.batch), self.pool(x2, data.batch),
                        self.pool(x3, data.batch)], dim=1)
        return self.proj(out)


class GTDrug(nn.Module):
    """Graph Transformer (TransformerConv)."""
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, heads=4, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import TransformerConv, global_mean_pool
        self.conv1 = TransformerConv(in_dim, hidden // heads, heads=heads,
                                     edge_dim=len(BOND_TYPES), dropout=0.2)
        self.conv2 = TransformerConv(hidden, hidden // heads, heads=heads,
                                     edge_dim=len(BOND_TYPES), dropout=0.2)
        self.conv3 = TransformerConv(hidden, hidden // 2, heads=1,
                                     edge_dim=len(BOND_TYPES), dropout=0.2)
        self.proj = nn.Linear(hidden + hidden + hidden // 2, out_dim)
        self.pool = global_mean_pool

    def forward(self, data):
        x1 = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x2 = F.relu(self.conv2(x1, data.edge_index, data.edge_attr))
        x3 = F.relu(self.conv3(x2, data.edge_index, data.edge_attr))
        out = torch.cat([self.pool(x1, data.batch), self.pool(x2, data.batch),
                        self.pool(x3, data.batch)], dim=1)
        return self.proj(out)


class GPSDrug(nn.Module):
    """GraphGPS: GIN local MPNN + global attention."""
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, heads=4, out_dim=DRUG_EMBED_DIM,
                 num_layers=3):
        super().__init__()
        from torch_geometric.nn import GPSConv, GINConv, global_mean_pool
        self.node_emb = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local_nn = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            local_conv = GINConv(local_nn)
            self.convs.append(GPSConv(hidden, conv=local_conv, heads=heads, dropout=0.2))
        self.proj = nn.Linear(hidden, out_dim)
        self.pool = global_mean_pool

    def forward(self, data):
        x = self.node_emb(data.x)
        for conv in self.convs:
            x = conv(x, data.edge_index, data.batch)
        x = self.pool(x, data.batch)
        return self.proj(x)


class SAGEDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import SAGEConv, global_mean_pool
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden // 2)
        self.proj = nn.Linear(hidden // 2, out_dim)
        self.pool = global_mean_pool

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
        x = self.pool(x, data.batch)
        return self.proj(x)


class GATv2Drug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, heads=4, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import GATv2Conv, global_mean_pool
        self.conv1 = GATv2Conv(in_dim, hidden // heads, heads=heads, edge_dim=len(BOND_TYPES))
        self.conv2 = GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=len(BOND_TYPES))
        self.conv3 = GATv2Conv(hidden, hidden // 2, heads=1, edge_dim=len(BOND_TYPES))
        self.proj = nn.Linear(hidden // 2, out_dim)
        self.pool = global_mean_pool

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.relu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.relu(self.conv3(x, data.edge_index, data.edge_attr))
        x = self.pool(x, data.batch)
        return self.proj(x)


class GENDrug(nn.Module):
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import GENConv, global_mean_pool
        self.conv1 = GENConv(in_dim, hidden)
        self.conv2 = GENConv(hidden, hidden)
        self.conv3 = GENConv(hidden, hidden // 2)
        self.proj = nn.Linear(hidden // 2, out_dim)
        self.pool = global_mean_pool

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
        x = self.pool(x, data.batch)
        return self.proj(x)


# ==================== DTA Model ====================
class DTAModel(nn.Module):
    """Unified DTA: drug_encoder + protein_encoder + fusion."""
    def __init__(self, drug_encoder, protein_encoder,
                 fusion_in=DRUG_EMBED_DIM + PROTEIN_EMBED_DIM):
        super().__init__()
        self.drug_encoder = drug_encoder
        self.protein_encoder = protein_encoder
        self.fc = nn.Sequential(
            nn.Linear(fusion_in, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, drug_input, seq):
        drug_embed = self.drug_encoder(drug_input)
        protein_embed = self.protein_encoder(seq)
        out = torch.cat([drug_embed, protein_embed], dim=1)
        return self.fc(out).squeeze(-1)


# ==================== Training ====================
def train_dta(model, train_loader, val_loader, epochs=EPOCHS, is_mlp=False):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss, n_train = 0.0, 0
        for batch in train_loader:
            opt.zero_grad()
            if is_mlp:
                x, y_b = batch
                x, y_b = x.to(device), y_b.to(device)
                pred = model(x)
            else:
                batch_graph, seq, y_b = batch
                batch_graph = batch_graph.to(device)
                seq, y_b = seq.to(device), y_b.to(device)
                pred = model(batch_graph, seq)
            loss = F.mse_loss(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss at epoch {epoch+1}, aborting training")
                return model
            train_loss += loss.item() * len(y_b)
            n_train += len(y_b)
        sched.step()

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if is_mlp:
                    x, y_b = batch
                    x, y_b = x.to(device), y_b.to(device)
                    pred = model(x)
                else:
                    batch_graph, seq, y_b = batch
                    batch_graph = batch_graph.to(device)
                    seq, y_b = seq.to(device), y_b.to(device)
                    pred = model(batch_graph, seq)
                val_loss += F.mse_loss(pred, y_b).item() * len(y_b)
                n_val += len(y_b)

        val_loss /= n_val
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}  train_loss={train_loss/n_train:.4f}  val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


def predict_dta(model, loader, is_mlp=False):
    model.eval()
    all_pred = []
    with torch.no_grad():
        for batch in loader:
            if is_mlp:
                x, _ = batch
                x = x.to(device)
                pred = model(x)
            else:
                batch_graph, seq, _ = batch
                batch_graph = batch_graph.to(device)
                seq = seq.to(device)
                pred = model(batch_graph, seq)
            all_pred.append(pred.cpu().numpy())
    return np.concatenate(all_pred)


# ==================== Build DataLoaders ====================
# MLP data (pre-concatenated FP + AAC+DPC)
mlp_train = torch.utils.data.TensorDataset(mlp_train_input, y_train_g)
mlp_val = torch.utils.data.TensorDataset(mlp_val_input, y_val_g)
mlp_test = torch.utils.data.TensorDataset(mlp_test_input, y_test_g)
mlp_train_loader = DataLoader(mlp_train, batch_size=BATCH_SIZE * 2, shuffle=True)
mlp_val_loader = DataLoader(mlp_val, batch_size=BATCH_SIZE * 2)
mlp_test_loader = DataLoader(mlp_test, batch_size=BATCH_SIZE * 2)

# GNN data
gnn_train_dataset = DTAGraphDataset(train_graphs, torch.tensor(protein_tokens[train_v]), y_train_g)
gnn_val_dataset = DTAGraphDataset(val_graphs, torch.tensor(protein_tokens[val_v]), y_val_g)
gnn_test_dataset = DTAGraphDataset(test_graphs, torch.tensor(protein_tokens[test_v]), y_test_g)
gnn_train_loader = DataLoader(gnn_train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=dta_collate)
gnn_val_loader = DataLoader(gnn_val_dataset, batch_size=BATCH_SIZE,
                            collate_fn=dta_collate)
gnn_test_loader = DataLoader(gnn_test_dataset, batch_size=BATCH_SIZE,
                             collate_fn=dta_collate)


# ==================== Model Registry ====================
from src.evaluation import evaluate_pains_aware
from src.visualization import plot_scatter, plot_residual_distribution

def build_dta(drug_cls, **kwargs):
    prot_encoder = ProteinCNN()
    drug_encoder = drug_cls(in_dim=len(ATOM_TYPES) + 3, **kwargs)
    return DTAModel(drug_encoder, prot_encoder)

results_rows = []

# Load any previously saved DTA results to skip completed models
dta_results_path = os.path.join(RESULTS_DIR, "dta_comparison_results.csv")
completed_models = set()
if os.path.exists(dta_results_path):
    existing = pd.read_csv(dta_results_path)
    completed_models = set(existing["model"].tolist())
    print(f"Already completed DTA models: {completed_models}")

# ==================== 1. MLP-DTA (separate path — FP + AAC+DPC concat) ====================
if "MLP-DTA" in completed_models:
    print(f"\n  Skipping MLP-DTA (already completed)")
    existing_row = existing[existing["model"] == "MLP-DTA"].iloc[0]
    results_rows.append(existing_row.to_dict())
else:
    print(f"\n{'=' * 55}")
    print("Training MLP-DTA (FP + AAC+DPC → MLP)...")
    sys.stdout.flush()
    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.time()
    mlp_model = MLPNet(in_dim=2048 + 420)
    mlp_model = train_dta(mlp_model, mlp_train_loader, mlp_val_loader, is_mlp=True)
    y_pred_mlp = predict_dta(mlp_model, mlp_test_loader, is_mlp=True)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    eval_dict = evaluate_pains_aware(y_test_g.numpy(), y_pred_mlp, ps_test_g)
    eval_dict["model"] = "MLP-DTA"
    eval_dict["train_time_s"] = elapsed
    results_rows.append(eval_dict)
    print(f"  Overall RMSE: {eval_dict['overall_RMSE']:.4f}  ΔRMSE: {eval_dict['delta_rmse']:.4f}")
    plot_scatter(y_test_g.numpy(), y_pred_mlp, ps_test_g, "DTA_MLP-DTA", filename="scatter_DTA_MLP-DTA.png")
    res_pos = np.abs(y_test_g.numpy()[ps_test_g == 1] - y_pred_mlp[ps_test_g == 1])
    res_neg = np.abs(y_test_g.numpy()[ps_test_g == 0] - y_pred_mlp[ps_test_g == 0])
    plot_residual_distribution(res_pos, res_neg, "DTA_MLP-DTA", filename="residuals_DTA_MLP-DTA.png")
    pd.DataFrame(results_rows).to_csv(dta_results_path, index=False)

# ==================== 2-12. GNN-DTA models ====================
gnn_dta_models = {
    "GCN-DTA": GCNDrug,
    "GAT-DTA": GATDrug,
    "AttFP-DTA": AttentiveFPDrug,
    "GIN-DTA": GINDrug,
    "GINE-DTA": GINEDrug,
    "PNA-DTA": PNADrug,
    "GT-DTA": GTDrug,
    "GPS-DTA": GPSDrug,
    "SAGE-DTA": SAGEDrug,
    "GATv2-DTA": GATv2Drug,
    "GEN-DTA": GENDrug,
}

for name, drug_cls in gnn_dta_models.items():
    if name in completed_models:
        print(f"\n  Skipping {name} (already completed)")
        existing_row = existing[existing["model"] == name].iloc[0]
        results_rows.append(existing_row.to_dict())
        continue

    print(f"\n{'=' * 55}")
    print(f"Training {name}...")
    sys.stdout.flush()
    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.time()

    extra_kwargs = {"deg": dta_deg} if name == "PNA-DTA" else {}
    model = build_dta(drug_cls, **extra_kwargs)
    model = train_dta(model, gnn_train_loader, gnn_val_loader)
    y_pred = predict_dta(model, gnn_test_loader)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Align: test_v has the indices into the full dataset
    eval_dict = evaluate_pains_aware(y_test_g.numpy(), y_pred, ps_test_g)
    eval_dict["model"] = name
    eval_dict["train_time_s"] = elapsed
    eval_dict["data_size"] = len(train_idx) + len(val_idx)
    results_rows.append(eval_dict)

    print(f"  Overall RMSE:  {eval_dict['overall_RMSE']:.4f}")
    print(f"  PAINS+ RMSE:   {eval_dict['pains_pos_RMSE']:.4f}")
    print(f"  PAINS- RMSE:   {eval_dict['pains_neg_RMSE']:.4f}")
    print(f"  ΔRMSE:         {eval_dict['delta_rmse']:.4f}")
    print(f"  FP Ratio:      {eval_dict['fp_ratio']:.4f}")

    plot_scatter(y_test_g.numpy(), y_pred, ps_test_g, f"DTA_{name}",
                 filename=f"scatter_DTA_{name}.png")
    res_pos = np.abs(y_test_g.numpy()[ps_test_g == 1] - y_pred[ps_test_g == 1])
    res_neg = np.abs(y_test_g.numpy()[ps_test_g == 0] - y_pred[ps_test_g == 0])
    plot_residual_distribution(res_pos, res_neg, f"DTA_{name}",
                               filename=f"residuals_DTA_{name}.png")

    # Save intermediate results after each model
    intermed_df = pd.DataFrame(results_rows)
    intermed_df.to_csv(os.path.join(RESULTS_DIR, "dta_comparison_results.csv"), index=False)

# ========== Results ==========
results_df = pd.DataFrame(results_rows)
cols = ["model"] + [c for c in results_df.columns if c != "model"]
results_df = results_df[cols]
results_df.to_csv(os.path.join(RESULTS_DIR, "dta_comparison_results.csv"), index=False)

print(f"\n{'=' * 70}")
print("PAINSBench-DTA: Full Model Comparison Results")
print(f"{'=' * 70}")
print(f"{'Model':15s} {'RMSE':>8s} {'PAINS+':>8s} {'PAINS-':>8s} {'ΔRMSE':>8s} "
      f"{'FP_Ratio':>9s} {'Time':>8s}")
print("-" * 70)
for _, r in results_df.iterrows():
    print(f"{r['model']:15s} {r['overall_RMSE']:8.4f} {r['pains_pos_RMSE']:8.4f} "
          f"{r['pains_neg_RMSE']:8.4f} {r['delta_rmse']:8.4f} {r['fp_ratio']:9.4f} "
          f"{r['train_time_s']:7.0f}s")

# ========== Visualization ==========
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(20, 11))

x = np.arange(len(results_df))
w = 0.25

# 1. RMSE grouped
ax = axes[0, 0]
ax.bar(x - w, results_df["pains_pos_RMSE"], w, label="PAINS+", color="#e74c3c", alpha=0.85)
ax.bar(x, results_df["pains_neg_RMSE"], w, label="PAINS-", color="#3498db", alpha=0.85)
ax.bar(x + w, results_df["overall_RMSE"], w, label="Overall", color="#2ecc71", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(results_df["model"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("RMSE")
ax.set_title("DTA: RMSE by Model & PAINS Status")
ax.legend(fontsize=8)

# 2. ΔRMSE
ax = axes[0, 1]
colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in results_df["delta_rmse"]]
bars = ax.bar(x, results_df["delta_rmse"], color=colors)
ax.axhline(0, color="gray", lw=1)
for bar, v in zip(bars, results_df["delta_rmse"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.005 if v > 0 else -0.025),
            f"{v:.3f}", ha="center", va="bottom" if v > 0 else "top", fontsize=7, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels(results_df["model"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("ΔRMSE")
ax.set_title("ΔRMSE (PAINS+ minus PAINS-)")

# 3. FP Ratio
ax = axes[0, 2]
ax.bar(x, results_df["fp_ratio"], color="#e67e22", alpha=0.85)
ax.axhline(1, color="gray", ls="--", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(results_df["model"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("FP Ratio")
ax.set_title("FP Ratio (PAINS+ residual / PAINS- residual)")

# 4. RMSE ranking
ax = axes[1, 0]
sorted_df = results_df.sort_values("overall_RMSE")
ax.barh(range(len(sorted_df)), sorted_df["overall_RMSE"], color="#2ecc71", alpha=0.8)
ax.set_yticks(range(len(sorted_df)))
ax.set_yticklabels(sorted_df["model"], fontsize=8)
ax.set_xlabel("Overall RMSE")
ax.set_title("DTA: Model Ranking (lower is better)")

# 5. ΔRMSE vs RMSE scatter
ax = axes[1, 1]
sc = ax.scatter(results_df["overall_RMSE"], results_df["delta_rmse"],
                c=range(len(results_df)), cmap="viridis", s=150, alpha=0.8)
for _, row in results_df.iterrows():
    ax.annotate(row["model"], (row["overall_RMSE"], row["delta_rmse"]),
                fontsize=7, ha="center", va="bottom", alpha=0.7)
ax.set_xlabel("Overall RMSE")
ax.set_ylabel("ΔRMSE")
ax.set_title("Accuracy-Robustness Trade-off")
ax.axvline(results_df["overall_RMSE"].mean(), color="gray", ls=":", alpha=0.4)
ax.axhline(results_df["delta_rmse"].mean(), color="gray", ls=":", alpha=0.4)

# 6. Training time
ax = axes[1, 2]
ax.barh(range(len(results_df)), results_df["train_time_s"], color="#9b59b6", alpha=0.8)
ax.set_yticks(range(len(results_df)))
ax.set_yticklabels(results_df["model"], fontsize=8)
ax.set_xlabel("Training Time (s)")
ax.set_title("Computational Cost")

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "dta_comparison_full.png"), dpi=150, bbox_inches="tight")
print(f"\nFigure saved: dta_comparison_full.png")
print(f"Results saved: dta_comparison_results.csv")
print("Done.")
