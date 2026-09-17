"""
18_toxproto.py
Toxicity Prototype Network (ToxProto): discover mechanism-specific toxicity
subtypes from the DTRL toxicity subspace via K-means clustering.

Input:  results/dtrl/dtrl_gnn_5fold.pt (trained DTRL-GNN)
Output: results/dtrl/toxproto_results.npz
        results/dtrl/toxproto_prototypes.csv
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
from sklearn.cluster import KMeans

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# (Import feature functions, DisentangledHead, and DTRL_GNN from 11_dtrl_gnn_5fold.py
#  by copying them here — same code as in 17_dtrl_representation_analysis.py.)

N_PROTOTYPES = 6


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
        self.dtr_head = DisentangledHead(combined_dim, dropout=dropout)

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

    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'dtrl_gnn_5fold.pt'),
                            map_location='cpu', weights_only=False)
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
    graphs = []
    for _, row in df.iterrows():
        g, _ = mol_to_graph(row['smiles'], row['label'])
        if g is not None:
            graphs.append(g)

    global_arr = np.stack([g.global_feat for g in graphs])
    global_arr[:, :10] = scaler.transform(global_arr[:, :10])
    for i, g in enumerate(graphs):
        g.global_feat = torch.tensor(global_arr[i], dtype=torch.float)

    labels = np.array([int(g.y.item()) for g in graphs])
    print(f"Data: {len(graphs)} compounds")

    print("\nExtracting h_tox representations...")
    h_tox_list = []
    with torch.no_grad():
        for g in graphs:
            x = g.x.to(device)
            edge_index = g.edge_index.to(device)
            edge_attr = g.edge_attr.to(device)
            batch = torch.zeros(g.x.size(0), dtype=torch.long).to(device)
            gf = g.global_feat.to(device)
            if gf.dim() == 1:
                gf = gf.unsqueeze(0)
            _, h_tox, _, _, _ = model(x, edge_index, edge_attr, batch, gf)
            h_tox_list.append(h_tox.squeeze(0).cpu().numpy())

    h_tox_all = np.array(h_tox_list)
    print(f"h_tox shape: {h_tox_all.shape}")

    print(f"\nInitializing {N_PROTOTYPES} prototypes with K-means...")
    kmeans = KMeans(n_clusters=N_PROTOTYPES, random_state=config.RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(h_tox_all)
    prototypes = kmeans.cluster_centers_

    print("\nPrototype annotation:")
    print("=" * 60)
    prototype_info = []
    for k in range(N_PROTOTYPES):
        mask = cluster_labels == k
        n_total = mask.sum()
        n_toxic = labels[mask].sum()
        tox_rate = n_toxic / n_total if n_total > 0 else 0
        proto_norm = np.linalg.norm(prototypes[k])
        prototype_info.append({
            'prototype': k,
            'n_molecules': int(n_total),
            'n_toxic': int(n_toxic),
            'toxicity_rate': tox_rate,
            'prototype_norm': proto_norm,
        })
        print(f"Prototype {k}: {n_total} molecules | toxicity rate: {tox_rate:.1%} | norm: {proto_norm:.2f}")

    proto_df = pd.DataFrame(prototype_info)
    proto_df.to_csv(os.path.join(OUTPUT_DIR, 'toxproto_prototypes.csv'), index=False)

    np.savez(os.path.join(OUTPUT_DIR, 'toxproto_results.npz'),
             h_tox=h_tox_all,
             cluster_labels=cluster_labels,
             prototypes=prototypes,
             labels=labels)

    print(f"\nSaved:")
    print(f"  {os.path.join(OUTPUT_DIR, 'toxproto_prototypes.csv')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'toxproto_results.npz')}")