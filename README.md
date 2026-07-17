# PAINSBench: Evaluating the Robustness of Drug-Target Affinity Prediction Models to PAINS Interference

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **PAINSBench** is the first comprehensive benchmark for evaluating Drug-Target Affinity (DTA) prediction models against Pan-Assay Interference Compounds (PAINS) — compounds that produce false-positive signals through non-specific mechanisms. Built from ChEMBL 36, PAINSBench comprises **126,452 compound-target pairs** (18.4% PAINS-positive) and evaluates **15 DTA models** across 3 data configurations.

---

## 🔬 Key Finding: The PAINS Paradox

**All 33 evaluated models** exhibit **negative ΔRMSE** (range: -0.17 to -0.33), meaning PAINS-positive compounds are **predicted more accurately** than regular compounds — a systematic shortcut learning phenomenon where models exploit statistical regularities in PAINS substructures rather than learning genuine biophysical interactions.

- **False-Positive Ratio**: 0.68–0.90 — PAINS compounds consistently yield **10–32% smaller residuals**
- **Cross-Assay Consistency**: Near-perfect correlation (Pearson \( r = 0.89 \)) between \(\mathbf{K}_i\) and \(\mathrm{IC}_{50}\) ΔRMSE, confirming PAINS bias is independent of measurement protocol
- **Loss-Level Mitigation Fails**: Upweighting, inverse weighting, and ΔRMSE regularization all fail to substantially reduce bias — PAINS shortcut learning is deeply embedded and requires architectural or data-level solutions

---

## 📊 Evaluated Models

| Category | Models |
|----------|--------|
| **12 Baselines** | MLP-DTA, GCN-DTA, GAT-DTA, GATv2-DTA, AttFP-DTA, GIN-DTA, GINE-DTA, PNA-DTA, GT-DTA, GPS-DTA, SAGE-DTA, GEN-DTA |
| **3 Frontier** | GS-DTA (GATv2-GCN + Transformer), Mamba-DTA (GraphTransformer + Mamba), TranGNN-DTA (Transformer protein encoder) |
| **Mitigation** | PR-DTA (Pains-Resistant DTA) — dual-branch GNN with differentiable gated fusion |

---
The dataset is too large to be hosted on GitHub. Researchers interested in obtaining the dataset are welcome to request access via email at wzs13141@gmail.com.
