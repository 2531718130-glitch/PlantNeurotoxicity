"""
17_dtrl_representation_analysis.py
Extract and evaluate DTRL disentangled representations.

Metrics reported:
  (1) PK reconstruction R^2
  (2) Cosine similarity between subspaces (orthogonality check)
  (3) Discriminative power of each subspace (t-test between toxic and non-toxic)

Input:  results/dtrl/dtrl_gnn_5fold.pt
Output: results/dtrl/dtrl_representations.npz
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Reuse feature functions and model definition from 08_dtrl_gnn_5fold
# (Copy only the functions/classes needed for inference.)


def get_atom_features(atom):
    hybrid = atom.GetHybridization()
    features = [
        atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
        atom.GetTotalNumHs(), int(atom.GetIsAromatic()), int(atom.IsInRing()),
        int(atom.IsInRingSize(5)), int(atom.IsInRingSize(6)),
        atom.GetExplicitValence(), atom.GetMass() / 100.0,
        int(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
        int(atom.GetNumRadicalElectrons() > 0),
    ]
    features += [
        hybrid == Chem.rdchem.HybridizationType.SP,
        hybrid == Chem.rdchem.HybridizationType.SP2,
        hybrid == Chem.rdchem.HybridizationType.SP3,
        hybrid == Chem.rdchem.HybridizationType.SP3D,
        hybrid == Chem.rdchem.HybridizationType.SP3D2,
    ]
    return features


def get_bond_features(bond):
    bt = bond.GetBondType()
    return [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        int(bond.GetIsConjugated()), int(bond.IsInRing()),
    ]


def get_global_features(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, config.MORGAN_RADIUS,
                                               nBits=config.MORGAN_NBITS)
    fp_arr = np.array(fp, dtype=np.float32)
    desc = np.array([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Descriptors.HeavyAtomCount(mol), Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAliphaticRings(mol), Descriptors.NumAromaticRings(mol),
        rdMolDescriptors.CalcNumHBA(mol), rdMolDescriptors.CalcNumHBD(mol),
        Descriptors.FractionCSP3(mol),
    ], dtype=np.float32)
    return np.concatenate([desc, fp_arr])


def mol_to_graph(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    x = torch.tensor([get_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        feat = get_bond_features(bond)
        edge_attr += [feat, feat]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    global_feat = get_global_features(mol)
    y = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                global_feat=global_feat, y=y), global_feat


class DisentangledHead(nn.Module):
    def __init__(self, input_dim, tox_dim=64, pk_dim=64, scaf_dim=64, dropout=0.2):
        super().__init__()
        self.tox_proj = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, tox_dim)
        )
        self.pk_proj = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, pk_dim)
        )
        self.scaf_proj = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, scaf_dim)
        )
        self.pk_recon = nn.Linear(pk_dim, 3)
        self.classifier = nn.Linear(tox_dim + pk_dim + scaf_dim, 1)

    def forward(self, h_graph):
        h_tox = self.tox_proj(h_graph)
        h_pk = self.pk_proj(h_graph)
        h_scaf = self.scaf_proj(h_graph)
        h_cat = torch.cat([h_tox, h_pk, h_scaf], dim=-1)
        logit = self.classifier(h_cat)
        pk_pred = self.pk_recon(h_pk)
        return logit, h_tox, h_pk, h_scaf, pk_pred


class DTRL_GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim=128, heads=8, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv(node_dim, hidden_dim, heads=heads, concat=False,
                             edge_dim=edge_dim, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False,
                             edge_dim=edge_dim, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False,
                             edge_dim=edge_dim, dropout=dropout)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.conv4 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False,
                             edge_dim=edge_dim, dropout=dropout)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        self.res_proj = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else None
        self.jk_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout)
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        graph_dim = hidden_dim * 4
        combined_dim = graph_dim + 128
        self.dtr_head = DisentangledHead(
            combined_dim, tox_dim=64, pk_dim=64, scaf_dim=64, dropout=dropout
        )

    def forward(self, x, edge_index, edge_attr, batch, global_feat):
        batch_size = batch.max().item() + 1
        global_feat = global_feat.view(batch_size, -1)

        x0 = self.res_proj(x) if self.res_proj is not None else x
        x1 = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x1 = x1 + x0 if x1.shape == x0.shape else x1
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index, edge_attr)))
        x2 = x2 + x1
        x3 = F.relu(self.bn3(self.conv3(x2, edge_index, edge_attr)))
        x3 = x3 + x2
        x4 = F.relu(self.bn4(self.conv4(x3, edge_index, edge_attr)))

        x_jk = self.jk_proj(torch.cat([x1, x2, x3, x4], dim=-1))
        x_mean = global_mean_pool(x_jk, batch)
        x_max = global_max_pool(x_jk, batch)
        graph_feat = torch.cat([x_mean, x_max], dim=-1)
        global_feat = self.global_proj(global_feat)
        combined = torch.cat([graph_feat, global_feat], dim=-1)
        logit, h_tox, h_pk, h_scaf, pk_pred = self.dtr_head(combined)
        return logit, h_tox, h_pk, h_scaf, pk_pred


if __name__ == "__main__":
    device = torch.device('cpu')
    OUTPUT_DIR = os.path.join(config.RESULTS_DIR, 'dtrl')

    checkpoint_path = os.path.join(OUTPUT_DIR, 'dtrl_gnn_5fold.pt')
    print("Loading DTRL checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    fold_states = checkpoint['fold_states']
    scaler = checkpoint['scaler']
    dims = checkpoint['dims']
    fold_aucs = checkpoint['fold_aucs']
    best_fold = int(np.argmax(fold_aucs))
    print(f"Using fold {best_fold + 1} (AUC: {fold_aucs[best_fold]:.4f})")

    model = DTRL_GNN(dims['node'], dims['edge'], dims['global'],
                     hidden_dim=128, heads=8, dropout=0).to(device)
    model.load_state_dict(fold_states[best_fold])
    model.eval()

    df = pd.read_csv(config.CSV_PATH)
    print(f"Data: {len(df)} compounds, running per-graph inference...")

    all_pk_true, all_pk_pred = [], []
    all_h_tox, all_h_pk, all_h_scaf = [], [], []
    all_labels, all_probs = [], []

    for idx, (smiles, label) in enumerate(zip(df['smiles'], df['label'])):
        g, gf_raw = mol_to_graph(smiles, label)
        if g is None:
            continue
        gf = gf_raw.copy()
        gf[:10] = scaler.transform(gf[:10].reshape(1, -1))[0]
        g.global_feat = torch.tensor(gf, dtype=torch.float).unsqueeze(0)
        g.batch = torch.zeros(g.x.size(0), dtype=torch.long)

        with torch.no_grad():
            logit, h_tox, h_pk, h_scaf, pk_pred = model(
                g.x.to(device), g.edge_index.to(device), g.edge_attr.to(device),
                g.batch.to(device), g.global_feat.to(device)
            )

        all_pk_true.append(gf[:3])
        all_pk_pred.append(pk_pred.squeeze(0).cpu().numpy())
        all_h_tox.append(h_tox.squeeze(0).cpu().numpy())
        all_h_pk.append(h_pk.squeeze(0).cpu().numpy())
        all_h_scaf.append(h_scaf.squeeze(0).cpu().numpy())
        all_labels.append(label)
        all_probs.append(torch.sigmoid(logit).item())

        if (idx + 1) % 3000 == 0:
            print(f"  {idx + 1}/{len(df)}")

    pk_true = np.array(all_pk_true)
    pk_pred = np.array(all_pk_pred)
    h_tox = np.array(all_h_tox)
    h_pk = np.array(all_h_pk)
    h_scaf = np.array(all_h_scaf)
    labels = np.array(all_labels)
    probs = np.array(all_probs)

    print(f"\npk_true: {pk_true.shape}, pk_pred: {pk_pred.shape}")
    print(f"h_tox: {h_tox.shape}")

    # (1) PK reconstruction R^2
    pk_r2 = r2_score(pk_true, pk_pred)
    print(f"\n[1] PK reconstruction R^2: {pk_r2:.4f}")
    for i, name in enumerate(['LogP', 'TPSA', 'MolWt']):
        r2_i = r2_score(pk_true[:, i], pk_pred[:, i])
        print(f"    {name}: R^2 = {r2_i:.4f}")

    # (2) Subspace cosine similarity
    def cos_sim(a, b):
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.mean(np.sum(a_n * b_n, axis=1))

    sim_tp = cos_sim(h_tox, h_pk)
    sim_ts = cos_sim(h_tox, h_scaf)
    sim_ps = cos_sim(h_pk, h_scaf)
    print(f"\n[2] Subspace cosine similarity:")
    print(f"    tox-pk:   {sim_tp:.4f}")
    print(f"    tox-scaf: {sim_ts:.4f}")
    print(f"    pk-scaf:  {sim_ps:.4f}")

    # (3) Discriminative power (t-test)
    tox_norm = np.linalg.norm(h_tox, axis=1)
    pk_norm = np.linalg.norm(h_pk, axis=1)
    scaf_norm = np.linalg.norm(h_scaf, axis=1)

    t_tox, p_tox = stats.ttest_ind(tox_norm[labels == 1], tox_norm[labels == 0])
    t_pk, p_pk = stats.ttest_ind(pk_norm[labels == 1], pk_norm[labels == 0])
    t_scaf, p_scaf = stats.ttest_ind(scaf_norm[labels == 1], scaf_norm[labels == 0])

    print(f"\n[3] Subspace discriminative power (t-test):")
    print(f"    h_tox : t = {t_tox:.2f}, p = {p_tox:.2e}")
    print(f"    h_pk  : t = {t_pk:.2f}, p = {p_pk:.2e}")
    print(f"    h_scaf: t = {t_scaf:.2f}, p = {p_scaf:.2e}")

    np.savez(os.path.join(OUTPUT_DIR, 'dtrl_representations.npz'),
             h_tox=h_tox, h_pk=h_pk, h_scaf=h_scaf,
             pk_pred=pk_pred, pk_true=pk_true,
             labels=labels, probs=probs)
    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'dtrl_representations.npz')}")