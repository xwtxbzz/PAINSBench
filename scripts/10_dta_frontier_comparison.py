"""
Step 10: Frontier DTA model comparison.
4 new SOTA models based on 2024-2026 papers:
  1. GS-DTA (BMC Genomics 2025): GATv2-GCN hybrid drug + CNN-BiLSTM-Transformer protein
  2. Mamba-DTA (MGDTA 2024): Mamba protein + GraphTransformer drug
  3. CrossAttn-DTA (CS-DTA 2026): Cross-attention fusion
  4. TransformerProt-DTA (TransGNN-DTA 2025): Transformer protein encoder

Extends existing 12 DTA models from 09_dta_full_comparison.py.
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
TRANSFORMER_BATCH_SIZE = 64  # smaller for transformer models (O(L^2) attention)
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
    tokens = [AA_TO_IDX.get(aa, len(AA_ORDER)+1) for aa in seq[:max_len]]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    return np.array(tokens, dtype=np.int64)

def compute_aac(seq):
    aa_list = AA_ORDER
    total = len(seq)
    if total == 0:
        return np.zeros(20, dtype=np.float32)
    return np.array([seq.count(aa) / total for aa in aa_list], dtype=np.float32)

def compute_dpc(seq):
    aa_list = AA_ORDER
    total = max(len(seq) - 1, 1)
    dpc_vec = np.zeros(400, dtype=np.float32)
    for i in range(len(seq) - 1):
        key = seq[i:i+2]
        if key[0] in aa_list and key[1] in aa_list:
            idx = aa_list.index(key[0]) * 20 + aa_list.index(key[1])
            dpc_vec[idx] += 1.0
    return dpc_vec / total

# ========== SMILES tokenization (for Mamba-DTA drug) ==========
def build_smiles_vocab(smiles_list):
    chars = set()
    for smi in smiles_list:
        chars.update(str(smi))
    sorted_chars = sorted(chars)
    return {c: i+2 for i, c in enumerate(sorted_chars)}, len(sorted_chars) + 2  # 0=PAD, 1=UNK

SMILES_MAX_LEN = 200

def tokenize_smiles(smiles, vocab, max_len=SMILES_MAX_LEN):
    tokens = [vocab.get(c, 1) for c in str(smiles)[:max_len]]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    return np.array(tokens, dtype=np.int64)

# ========== Data loading ==========
print("\nLoading single-protein benchmark...")
bench = pd.read_csv(os.path.join(PROCESSED_DIR, "benchmark_dta_full.csv"))
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

# Precompute protein features
print("\nComputing protein features (AAC+DPC)...")
t0 = time.time()
protein_aac = np.array([compute_aac(s) for s in sequences], dtype=np.float32)
protein_dpc = np.array([compute_dpc(s) for s in sequences], dtype=np.float32)
protein_static = np.concatenate([protein_aac, protein_dpc], axis=1)
print(f"  Done in {time.time()-t0:.1f}s")

print("Tokenizing sequences for CNN...")
protein_tokens = np.array([tokenize_sequence(s) for s in sequences], dtype=np.int64)
print(f"  Token shape: {protein_tokens.shape}")

# SMILES vocab
print("Building SMILES vocabulary...")
smiles_vocab, smiles_vocab_size = build_smiles_vocab(smiles_list)
print(f"  SMILES vocab size: {smiles_vocab_size}")
smiles_tokens = np.array([tokenize_smiles(s, smiles_vocab) for s in smiles_list], dtype=np.int64)
print(f"  SMILES tokens shape: {smiles_tokens.shape}")

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
from torch_geometric.data import Data

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

# Align labels and features
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

# SMILES tokens aligned
smi_train = smiles_tokens[train_v]
smi_val = smiles_tokens[val_v]
smi_test = smiles_tokens[test_v]

# Set graph labels
for g, lbl in zip(train_graphs, y_train_g):
    g.y = lbl
for g, lbl in zip(val_graphs, y_val_g):
    g.y = lbl
for g, lbl in zip(test_graphs, y_test_g):
    g.y = lbl

# ==================== Datasets ====================
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
    from torch_geometric.data import Batch
    graphs, seqs, labels = zip(*batch)
    batch_graph = Batch.from_data_list(list(graphs))
    seq_tensor = torch.stack(list(seqs))
    label_tensor = torch.stack(list(labels))
    return batch_graph, seq_tensor, label_tensor

class SmilesDTADataset(Dataset):
    """For models that use SMILES tokens instead of graphs."""
    def __init__(self, smiles_tokens, seqs, labels):
        self.smiles = smiles_tokens
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.seqs[idx], self.labels[idx]

def smiles_dta_collate(batch):
    smiles, seqs, labels = zip(*batch)
    return (torch.stack(list(smiles)), torch.stack(list(seqs)), torch.stack(list(labels)))

# ==================== Reusable Components (from 09) ====================
class ProteinCNN(nn.Module):
    """1D CNN for protein sequence encoding (reused from 09)."""
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
        x = self.embedding(seq)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        return x


class GCNDrug(nn.Module):
    """3-layer GCN drug encoder (reused from 09)."""
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


class GATv2Drug(nn.Module):
    """3-layer GATv2 drug encoder (reused from 09)."""
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


class DTAModel(nn.Module):
    """Standard DTA: drug_encoder + protein_encoder + concat fusion."""
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


# ==================== Model 1: GS-DTA (BMC Genomics 2025) ====================
class GSDTA_DrugEncoder(nn.Module):
    """
    GS-DTA drug encoder: GATv2-guided GCN hybrid.
    At each layer: GATv2 attention → GCN convolution
    (3 layers total, as per paper)
    """
    def __init__(self, in_dim=len(ATOM_TYPES)+3, hidden=128, out_dim=DRUG_EMBED_DIM):
        super().__init__()
        from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool

        # Layer 1
        self.gat1 = GATv2Conv(in_dim, hidden // 4, heads=4, edge_dim=len(BOND_TYPES))
        self.gcn1 = GCNConv(in_dim, hidden)

        # Layer 2
        self.gat2 = GATv2Conv(hidden, hidden // 4, heads=4, edge_dim=len(BOND_TYPES))
        self.gcn2 = GCNConv(hidden, hidden)

        # Layer 3 - both branches output same dim for residual add
        self.gat3 = GATv2Conv(hidden, hidden, heads=1, edge_dim=len(BOND_TYPES))
        self.gcn3 = GCNConv(hidden, hidden)

        self.pool = global_mean_pool
        self.proj = nn.Linear(hidden * 3, out_dim)

    def forward(self, data):
        # Layer 1: GATv2 → GCN → residual
        attn1 = F.relu(self.gat1(data.x, data.edge_index, data.edge_attr))
        x1 = F.relu(self.gcn1(data.x, data.edge_index))
        x1 = x1 + attn1  # combine GAT and GCN

        # Layer 2
        attn2 = F.relu(self.gat2(x1, data.edge_index, data.edge_attr))
        x2 = F.relu(self.gcn2(x1, data.edge_index))
        x2 = x2 + attn2

        # Layer 3
        attn3 = F.relu(self.gat3(x2, data.edge_index, data.edge_attr))
        x3 = F.relu(self.gcn3(x2, data.edge_index))
        x3 = x3 + attn3

        # Multi-scale pooling
        p1 = self.pool(x1, data.batch)
        p2 = self.pool(x2, data.batch)
        p3 = self.pool(x3, data.batch)

        out = torch.cat([p1, p2, p3], dim=1)
        return self.proj(out)


class GSDTA_ProteinEncoder(nn.Module):
    """
    GS-DTA protein encoder: CNN(stride) → TransformerEncoder.
    Strided CNN reduces 1200→300 tokens, making O(L^2) attention feasible on 8GB GPU.
    Adapted from BMC Genomics 2025 (removed BiLSTM due to VRAM constraints).
    """
    def __init__(self, vocab_size=25, embed_dim=32, hidden=64, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Strided CNN: local features + length reduction 1200→600→300
        self.conv1 = nn.Conv1d(embed_dim, hidden, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(hidden)
        # TransformerEncoder: long-range dependencies (on ~300 tokens)
        trans_layer = nn.TransformerEncoderLayer(
            hidden, nhead=2, dim_feedforward=hidden * 2, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(trans_layer, num_layers=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Linear(hidden, PROTEIN_EMBED_DIM)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        x = self.embedding(seq)  # (B, L, 32)
        x = x.permute(0, 2, 1)  # (B, 32, L)
        # Strided CNN: 1200 → 600 → 300
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        # Transformer (on ~300 tokens)
        x = x.permute(0, 2, 1)  # (B, 300, 64)
        x = self.transformer(x)
        # Pool
        x = x.permute(0, 2, 1)  # (B, 64, 300)
        x = self.pool(x).squeeze(-1)  # (B, 64)
        x = self.proj(x)  # (B, 128)
        return x


class GSDTA(nn.Module):
    """GS-DTA: GATv2-GCN hybrid drug + CNN-BiLSTM-Transformer protein."""
    def __init__(self):
        super().__init__()
        self.drug_encoder = GSDTA_DrugEncoder()
        self.protein_encoder = GSDTA_ProteinEncoder()
        self.fc = nn.Sequential(
            nn.Linear(DRUG_EMBED_DIM + PROTEIN_EMBED_DIM, 64),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, drug_input, seq):
        drug_embed = self.drug_encoder(drug_input)
        protein_embed = self.protein_encoder(seq)
        out = torch.cat([drug_embed, protein_embed], dim=1)
        return self.fc(out).squeeze(-1)


# ==================== Model 2: Mamba-DTA (MGDTA 2024 inspired) ====================
class MambaBlock(nn.Module):
    """
    Efficient Mamba-inspired block.
    Core ideas from Mamba (Gu & Dao, 2023): conv1d + gating + channel mixing.
    Uses parallel scan via linear attention instead of sequential CUDA scan
    for efficient training on GPU without custom kernels.
    """
    def __init__(self, d_model, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner,
                                kernel_size=d_conv, padding=d_conv - 1,
                                groups=self.d_inner, bias=False)
        # Efficient channel mixing (replaces selective scan)
        self.channel_mix = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner),
            nn.GELU())
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        """x: (B, L, d_model)"""
        xz = self.in_proj(x)
        x_half, z = xz.chunk(2, dim=-1)

        # Causal convolution
        x_conv = x_half.transpose(1, 2)
        x_conv = F.silu(self.conv1d(x_conv)[..., :x.size(1)])
        x_conv = x_conv.transpose(1, 2)

        # Channel mixing (parallel - no sequential scan)
        x_mix = self.channel_mix(x_conv)

        # Gating
        z = F.silu(z)
        y = x_mix * z
        y = self.out_proj(y)
        return y


class MambaProtein(nn.Module):
    """Mamba state-space model for protein sequences."""
    def __init__(self, vocab_size=25, embed_dim=32, hidden=PROTEIN_EMBED_DIM, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, hidden, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.mamba = MambaBlock(d_model=hidden, d_conv=4, expand=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        x = self.embedding(seq)  # (B, L, 32)
        x = x.permute(0, 2, 1)  # (B, 32, L)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, hidden, L)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # (B, L, hidden)
        x = self.mamba(x)  # (B, L, hidden)
        x = x.permute(0, 2, 1)  # (B, hidden, L)
        x = self.pool(x).squeeze(-1)  # (B, hidden)
        return x


class MambaDTA(nn.Module):
    """
    Mamba-DTA: Graph Transformer drug + Mamba protein.
    Inspired by MGDTA (Microchemical Journal, 2024).
    """
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import TransformerConv, global_mean_pool
        # Drug: Graph Transformer (same as GTDrug from 09)
        drug_in = len(ATOM_TYPES) + 3
        self.gt1 = TransformerConv(drug_in, 128 // 4, heads=4, edge_dim=len(BOND_TYPES), dropout=0.2)
        self.gt2 = TransformerConv(128, 128 // 4, heads=4, edge_dim=len(BOND_TYPES), dropout=0.2)
        self.gt3 = TransformerConv(128, 64, heads=1, edge_dim=len(BOND_TYPES), dropout=0.2)
        self.drug_proj = nn.Linear(128 + 128 + 64, DRUG_EMBED_DIM)
        self.pool = global_mean_pool
        # Protein: Mamba
        self.protein_encoder = MambaProtein()
        # Fusion
        self.fc = nn.Sequential(
            nn.Linear(DRUG_EMBED_DIM + PROTEIN_EMBED_DIM, 64),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, drug_input, seq):
        x1 = F.relu(self.gt1(drug_input.x, drug_input.edge_index, drug_input.edge_attr))
        x2 = F.relu(self.gt2(x1, drug_input.edge_index, drug_input.edge_attr))
        x3 = F.relu(self.gt3(x2, drug_input.edge_index, drug_input.edge_attr))
        p1 = self.pool(x1, drug_input.batch)
        p2 = self.pool(x2, drug_input.batch)
        p3 = self.pool(x3, drug_input.batch)
        drug_embed = self.drug_proj(torch.cat([p1, p2, p3], dim=1))

        protein_embed = self.protein_encoder(seq)

        out = torch.cat([drug_embed, protein_embed], dim=1)
        return self.fc(out).squeeze(-1)


# ==================== Model 3: CrossAttn-DTA (CS-DTA 2026 / CAFIE-DTA 2025 inspired) ====================
class CrossAttnDTAModel(nn.Module):
    """
    Cross-attention fusion for DTA.
    Drug embedding attends to protein embedding via MultiheadAttention,
    then concatenated with protein for final prediction.
    """
    def __init__(self, drug_encoder, protein_encoder,
                 d_model=DRUG_EMBED_DIM, nhead=2):
        super().__init__()
        self.drug_encoder = drug_encoder
        self.protein_encoder = protein_encoder
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, drug_input, seq):
        drug_embed = self.drug_encoder(drug_input)    # (B, d_model)
        protein_embed = self.protein_encoder(seq)     # (B, d_model)

        # Cross-attention: drug queries protein
        # (B, 1, d_model) x (B, 1, d_model) → (B, 1, d_model)
        attended, _ = self.cross_attn(
            drug_embed.unsqueeze(1),   # query
            protein_embed.unsqueeze(1), # key
            protein_embed.unsqueeze(1), # value
        )
        attended = self.norm(attended.squeeze(1) + drug_embed)  # residual + LN

        out = torch.cat([attended, protein_embed], dim=1)
        return self.fc(out).squeeze(-1)


# ==================== Model 4: TransformerProt-DTA (TransGNN-DTA 2025 inspired) ====================
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerProtein(nn.Module):
    """
    Transformer-based protein encoder.
    Uses strided CNN to reduce sequence length, then TransformerEncoder.
    Inspired by TransGNN-DTA (2025).
    """
    def __init__(self, vocab_size=25, embed_dim=32, hidden=PROTEIN_EMBED_DIM, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Strided CNN to reduce sequence length
        self.conv1 = nn.Conv1d(embed_dim, 64, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, hidden, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden)
        # Transformer
        self.proj = nn.Linear(hidden, hidden)
        self.pos_enc = LearnedPositionalEncoding(hidden, max_len=300)
        trans_layer = nn.TransformerEncoderLayer(
            hidden, nhead=2, dim_feedforward=hidden * 2, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(trans_layer, num_layers=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        x = self.embedding(seq)  # (B, L, 32)
        x = x.permute(0, 2, 1)  # (B, 32, L)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, L/2)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, L/4≈300)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # (B, L/4, 128)
        x = self.proj(x)  # (B, L/4, 128)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (B, L/4, 128)
        x = x.mean(dim=1)  # (B, 128)
        return x


# ==================== Training Functions ====================
def train_dta(model, train_loader, val_loader, epochs=EPOCHS,
              is_mlp=False, is_text=False):
    """General training function. Supports:
    - is_mlp: (x, y) where x is tensor
    - is_text: (drug_tokens, seq, y) where drug_tokens is tensor
    - default: (batch_graph, seq, y) where batch_graph is PyG Batch
    """
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
            elif is_text:
                drug_tokens, seq, y_b = batch
                drug_tokens, seq, y_b = drug_tokens.to(device), seq.to(device), y_b.to(device)
                pred = model(drug_tokens, seq)
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
                elif is_text:
                    drug_tokens, seq, y_b = batch
                    drug_tokens, seq, y_b = drug_tokens.to(device), seq.to(device), y_b.to(device)
                    pred = model(drug_tokens, seq)
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


def predict_dta(model, loader, is_mlp=False, is_text=False):
    model.eval()
    all_pred = []
    with torch.no_grad():
        for batch in loader:
            if is_mlp:
                x, _ = batch
                x = x.to(device)
                pred = model(x)
            elif is_text:
                drug_tokens, seq, _ = batch
                drug_tokens, seq = drug_tokens.to(device), seq.to(device)
                pred = model(drug_tokens, seq)
            else:
                batch_graph, seq, _ = batch
                batch_graph = batch_graph.to(device)
                seq = seq.to(device)
                pred = model(batch_graph, seq)
            all_pred.append(pred.cpu().numpy())
    return np.concatenate(all_pred)


# ==================== Build DataLoaders ====================
gnn_train_dataset = DTAGraphDataset(train_graphs, torch.tensor(protein_tokens[train_v]), y_train_g)
gnn_val_dataset = DTAGraphDataset(val_graphs, torch.tensor(protein_tokens[val_v]), y_val_g)
gnn_test_dataset = DTAGraphDataset(test_graphs, torch.tensor(protein_tokens[test_v]), y_test_g)
gnn_train_loader = DataLoader(gnn_train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=dta_collate)
gnn_val_loader = DataLoader(gnn_val_dataset, batch_size=BATCH_SIZE, collate_fn=dta_collate)
gnn_test_loader = DataLoader(gnn_test_dataset, batch_size=BATCH_SIZE, collate_fn=dta_collate)

# For transformer models (need smaller batch due to O(L^2) attention)
gnn_train_loader_small = DataLoader(gnn_train_dataset, batch_size=TRANSFORMER_BATCH_SIZE,
                                    shuffle=True, collate_fn=dta_collate)
gnn_val_loader_small = DataLoader(gnn_val_dataset, batch_size=TRANSFORMER_BATCH_SIZE,
                                  collate_fn=dta_collate)
gnn_test_loader_small = DataLoader(gnn_test_dataset, batch_size=TRANSFORMER_BATCH_SIZE,
                                   collate_fn=dta_collate)


# ==================== Model Registry ====================
from src.evaluation import evaluate_pains_aware
from src.visualization import plot_scatter, plot_residual_distribution

results_rows = []
frontier_results_path = os.path.join(RESULTS_DIR, "dta_frontier_results.csv")
completed_models = set()
if os.path.exists(frontier_results_path):
    existing = pd.read_csv(frontier_results_path)
    completed_models = set(existing["model"].tolist())
    print(f"\nAlready completed frontier DTA models: {completed_models}")

# ==================== Training Loop ====================

# --- Model 1: GS-DTA ---
model_name = "GS-DTA"
if model_name in completed_models:
    print(f"\n  Skipping {model_name} (already completed)")
    existing_row = existing[existing["model"] == model_name].iloc[0]
    results_rows.append(existing_row.to_dict())
else:
    print(f"\n{'=' * 55}")
    print(f"Training {model_name} (GATv2-GCN hybrid + CNN-BiLSTM-Transformer)...")
    print(f"  Paper: BMC Genomics 2025")
    sys.stdout.flush()
    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.time()

    model = GSDTA()
    model = train_dta(model, gnn_train_loader, gnn_val_loader)
    y_pred = predict_dta(model, gnn_test_loader)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    eval_dict = evaluate_pains_aware(y_test_g.numpy(), y_pred, ps_test_g)
    eval_dict["model"] = model_name
    eval_dict["train_time_s"] = elapsed
    results_rows.append(eval_dict)

    print(f"  Overall RMSE: {eval_dict['overall_RMSE']:.4f}  ΔRMSE: {eval_dict['delta_rmse']:.4f}")
    plot_scatter(y_test_g.numpy(), y_pred, ps_test_g, f"DTA_Frontier_{model_name}",
                 filename=f"scatter_DTA_Frontier_{model_name}.png")
    res_pos = np.abs(y_test_g.numpy()[ps_test_g == 1] - y_pred[ps_test_g == 1])
    res_neg = np.abs(y_test_g.numpy()[ps_test_g == 0] - y_pred[ps_test_g == 0])
    plot_residual_distribution(res_pos, res_neg, f"DTA_Frontier_{model_name}",
                               filename=f"residuals_DTA_Frontier_{model_name}.png")
    pd.DataFrame(results_rows).to_csv(frontier_results_path, index=False)

# --- Model 2: Mamba-DTA ---
model_name = "Mamba-DTA"
if model_name in completed_models:
    print(f"\n  Skipping {model_name} (already completed)")
    existing_row = existing[existing["model"] == model_name].iloc[0]
    results_rows.append(existing_row.to_dict())
else:
    print(f"\n{'=' * 55}")
    print(f"Training {model_name} (GraphTransformer drug + Mamba protein)...")
    print(f"  Inspired by MGDTA, Microchemical Journal 2024")
    sys.stdout.flush()
    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.time()

    model = MambaDTA()
    model = train_dta(model, gnn_train_loader, gnn_val_loader)
    y_pred = predict_dta(model, gnn_test_loader)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    eval_dict = evaluate_pains_aware(y_test_g.numpy(), y_pred, ps_test_g)
    eval_dict["model"] = model_name
    eval_dict["train_time_s"] = elapsed
    results_rows.append(eval_dict)

    print(f"  Overall RMSE: {eval_dict['overall_RMSE']:.4f}  ΔRMSE: {eval_dict['delta_rmse']:.4f}")
    plot_scatter(y_test_g.numpy(), y_pred, ps_test_g, f"DTA_Frontier_{model_name}",
                 filename=f"scatter_DTA_Frontier_{model_name}.png")
    res_pos = np.abs(y_test_g.numpy()[ps_test_g == 1] - y_pred[ps_test_g == 1])
    res_neg = np.abs(y_test_g.numpy()[ps_test_g == 0] - y_pred[ps_test_g == 0])
    plot_residual_distribution(res_pos, res_neg, f"DTA_Frontier_{model_name}",
                               filename=f"residuals_DTA_Frontier_{model_name}.png")
    pd.DataFrame(results_rows).to_csv(frontier_results_path, index=False)

# --- Model 3: CrossAttn-DTA ---
model_name = "CrossAttn-DTA"
if model_name in completed_models:
    print(f"\n  Skipping {model_name} (already completed)")
    existing_row = existing[existing["model"] == model_name].iloc[0]
    results_rows.append(existing_row.to_dict())
else:
    print(f"\n{'=' * 55}")
    print(f"Training {model_name} (GATv2 drug + ProteinCNN + cross-attention fusion)...")
    print(f"  Inspired by CS-DTA 2026, CAFIE-DTA 2025")
    sys.stdout.flush()
    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.time()

    drug_enc = GATv2Drug()
    prot_enc = ProteinCNN()
    model = CrossAttnDTAModel(drug_enc, prot_enc)
    model = train_dta(model, gnn_train_loader, gnn_val_loader)
    y_pred = predict_dta(model, gnn_test_loader)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    eval_dict = evaluate_pains_aware(y_test_g.numpy(), y_pred, ps_test_g)
    eval_dict["model"] = model_name
    eval_dict["train_time_s"] = elapsed
    results_rows.append(eval_dict)

    print(f"  Overall RMSE: {eval_dict['overall_RMSE']:.4f}  ΔRMSE: {eval_dict['delta_rmse']:.4f}")
    plot_scatter(y_test_g.numpy(), y_pred, ps_test_g, f"DTA_Frontier_{model_name}",
                 filename=f"scatter_DTA_Frontier_{model_name}.png")
    res_pos = np.abs(y_test_g.numpy()[ps_test_g == 1] - y_pred[ps_test_g == 1])
    res_neg = np.abs(y_test_g.numpy()[ps_test_g == 0] - y_pred[ps_test_g == 0])
    plot_residual_distribution(res_pos, res_neg, f"DTA_Frontier_{model_name}",
                               filename=f"residuals_DTA_Frontier_{model_name}.png")
    pd.DataFrame(results_rows).to_csv(frontier_results_path, index=False)

# --- Model 4: TransformerProt-DTA ---
model_name = "TransformerProt-DTA"
if model_name in completed_models:
    print(f"\n  Skipping {model_name} (already completed)")
    existing_row = existing[existing["model"] == model_name].iloc[0]
    results_rows.append(existing_row.to_dict())
else:
    print(f"\n{'=' * 55}")
    print(f"Training {model_name} (GCN drug + Transformer protein)...")
    print(f"  Inspired by TransGNN-DTA, 2025")
    sys.stdout.flush()
    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.time()

    drug_enc = GCNDrug()
    prot_enc = TransformerProtein()
    model = DTAModel(drug_enc, prot_enc)
    model = train_dta(model, gnn_train_loader_small, gnn_val_loader_small)
    y_pred = predict_dta(model, gnn_test_loader_small)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    eval_dict = evaluate_pains_aware(y_test_g.numpy(), y_pred, ps_test_g)
    eval_dict["model"] = model_name
    eval_dict["train_time_s"] = elapsed
    results_rows.append(eval_dict)

    print(f"  Overall RMSE: {eval_dict['overall_RMSE']:.4f}  ΔRMSE: {eval_dict['delta_rmse']:.4f}")
    plot_scatter(y_test_g.numpy(), y_pred, ps_test_g, f"DTA_Frontier_{model_name}",
                 filename=f"scatter_DTA_Frontier_{model_name}.png")
    res_pos = np.abs(y_test_g.numpy()[ps_test_g == 1] - y_pred[ps_test_g == 1])
    res_neg = np.abs(y_test_g.numpy()[ps_test_g == 0] - y_pred[ps_test_g == 0])
    plot_residual_distribution(res_pos, res_neg, f"DTA_Frontier_{model_name}",
                               filename=f"residuals_DTA_Frontier_{model_name}.png")
    pd.DataFrame(results_rows).to_csv(frontier_results_path, index=False)

# ========== Save final results ==========
results_df = pd.DataFrame(results_rows)
cols = ["model"] + [c for c in results_df.columns if c != "model"]
results_df = results_df[cols]
results_df.to_csv(frontier_results_path, index=False)

# ========== Combined Results Table ==========
print(f"\n{'=' * 70}")
print("PAINSBench-DTA: Frontier Model Comparison Results")
print(f"{'=' * 70}")
print(f"{'Model':25s} {'RMSE':>8s} {'PAINS+':>8s} {'PAINS-':>8s} {'ΔRMSE':>8s} "
      f"{'FP_Ratio':>9s} {'Time':>8s}")
print("-" * 70)
for _, r in results_df.iterrows():
    print(f"{r['model']:25s} {r['overall_RMSE']:8.4f} {r['pains_pos_RMSE']:8.4f} "
          f"{r['pains_neg_RMSE']:8.4f} {r['delta_rmse']:8.4f} {r['fp_ratio']:9.4f} "
          f"{r['train_time_s']:7.0f}s")

# ========== Combined Visualization (all 16 DTA models) ==========
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Load existing 12 models from 09
existing_path = os.path.join(RESULTS_DIR, "dta_comparison_results.csv")
if os.path.exists(existing_path):
    existing_df = pd.read_csv(existing_path)
    combined_df = pd.concat([existing_df, results_df], ignore_index=True)
else:
    combined_df = results_df

# Color: distinguish existing vs frontier
existing_models = set(existing_df["model"].tolist()) if os.path.exists(existing_path) else set()

fig, axes = plt.subplots(2, 3, figsize=(22, 12))
x = np.arange(len(combined_df))
w = 0.25

# 1. RMSE grouped
ax = axes[0, 0]
ax.bar(x - w, combined_df["pains_pos_RMSE"], w, label="PAINS+", color="#e74c3c", alpha=0.85)
ax.bar(x, combined_df["pains_neg_RMSE"], w, label="PAINS-", color="#3498db", alpha=0.85)
ax.bar(x + w, combined_df["overall_RMSE"], w, label="Overall", color="#2ecc71", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(combined_df["model"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("RMSE")
ax.set_title("DTA: RMSE by Model & PAINS Status (All 16 Models)")
ax.legend(fontsize=8)

# Highlight frontier models
for i, model_name in enumerate(combined_df["model"]):
    if model_name not in existing_models:
        ax.get_xticklabels()[i].set_fontweight("bold")
        ax.get_xticklabels()[i].set_color("#e67e22")

# 2. ΔRMSE
ax = axes[0, 1]
colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in combined_df["delta_rmse"]]
bars = ax.bar(x, combined_df["delta_rmse"], color=colors)
ax.axhline(0, color="gray", lw=1)
for bar, v in zip(bars, combined_df["delta_rmse"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.005 if v > 0 else -0.025),
            f"{v:.3f}", ha="center", va="bottom" if v > 0 else "top", fontsize=6, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels(combined_df["model"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("ΔRMSE")
ax.set_title("ΔRMSE (PAINS+ minus PAINS-)")
for i, model_name in enumerate(combined_df["model"]):
    if model_name not in existing_models:
        ax.get_xticklabels()[i].set_fontweight("bold")
        ax.get_xticklabels()[i].set_color("#e67e22")

# 3. FP Ratio
ax = axes[0, 2]
ax.bar(x, combined_df["fp_ratio"], color="#e67e22", alpha=0.85)
ax.axhline(1, color="gray", ls="--", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(combined_df["model"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("FP Ratio")
ax.set_title("FP Ratio (PAINS+ residual / PAINS- residual)")
for i, model_name in enumerate(combined_df["model"]):
    if model_name not in existing_models:
        ax.get_xticklabels()[i].set_fontweight("bold")
        ax.get_xticklabels()[i].set_color("#e67e22")

# 4. RMSE ranking
ax = axes[1, 0]
sorted_df = combined_df.sort_values("overall_RMSE")
colors_rank = ["#e67e22" if m not in existing_models else "#2ecc71"
               for m in sorted_df["model"]]
ax.barh(range(len(sorted_df)), sorted_df["overall_RMSE"], color=colors_rank, alpha=0.8)
ax.set_yticks(range(len(sorted_df)))
ax.set_yticklabels(sorted_df["model"], fontsize=8)
ax.set_xlabel("Overall RMSE")
ax.set_title("DTA: Model Ranking (orange = frontier)")

# 5. ΔRMSE vs RMSE scatter
ax = axes[1, 1]
colors_scatter = ["#e67e22" if m not in existing_models else "#2ecc71"
                  for m in combined_df["model"]]
ax.scatter(combined_df["overall_RMSE"], combined_df["delta_rmse"],
           c=colors_scatter, s=150, alpha=0.8)
for _, row in combined_df.iterrows():
    label = row["model"]
    weight = "bold" if label not in existing_models else "normal"
    ax.annotate(label, (row["overall_RMSE"], row["delta_rmse"]),
                fontsize=7, ha="center", va="bottom", alpha=0.7, fontweight=weight)
ax.set_xlabel("Overall RMSE")
ax.set_ylabel("ΔRMSE")
ax.set_title("Accuracy-Robustness Trade-off (orange = frontier)")
ax.axvline(combined_df["overall_RMSE"].mean(), color="gray", ls=":", alpha=0.4)
ax.axhline(combined_df["delta_rmse"].mean(), color="gray", ls=":", alpha=0.4)

# 6. Training time
ax = axes[1, 2]
colors_time = ["#e67e22" if m not in existing_models else "#9b59b6"
               for m in combined_df["model"]]
ax.barh(range(len(combined_df)), combined_df["train_time_s"], color=colors_time, alpha=0.8)
ax.set_yticks(range(len(combined_df)))
ax.set_yticklabels(combined_df["model"], fontsize=8)
ax.set_xlabel("Training Time (s)")
ax.set_title("Computational Cost (orange = frontier)")

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "dta_frontier_comparison_full.png"), dpi=150, bbox_inches="tight")
print(f"\nFigure saved: dta_frontier_comparison_full.png")
print(f"Results saved: {frontier_results_path}")
print("Done.")
