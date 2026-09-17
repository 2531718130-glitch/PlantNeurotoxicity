"""
15_gnn_attention_visualization.py
GNN attention weight visualization: highlight key substructures for neurotoxicity.

Loads the final 5-fold GNN ensemble (best_gnn_5fold.pt) and extracts attention
weights from the best fold. Attention is aggregated across all 5 folds for
robustness.

Input:  models/gnn_5fold.pt
        data/step3_final_v4.csv
Output: results/visualization/  (individual molecular images + summary CSV)
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
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ==================== Output directory ====================
OUTPUT_DIR = os.path.join(config.RESULTS_DIR, 'visualization')
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cpu')


# ==================== Feature engineering (matching 10_gnn_5fold.py) ====================

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


# ==================== Model definition (matching 10_gnn_5fold.py) ====================

class ToxicityGNN(nn.Module):
    """4-layer GAT with residual connections and Jumping Knowledge."""

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
        self.classifier = nn.Sequential(
            nn.Linear(graph_dim + 128, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
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
        return self.classifier(combined)

    def extract_attention(self, x, edge_index, edge_attr, batch, global_feat):
        """Extract per-atom importance scores from the 4-layer GAT."""
        batch_size = batch.max().item() + 1
        global_feat = global_feat.view(batch_size, -1)

        x0 = self.res_proj(x) if self.res_proj is not None else x
        h1 = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        h1 = h1 + x0 if h1.shape == x0.shape else h1
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index, edge_attr)))
        h2 = h2 + h1
        h3 = F.relu(self.bn3(self.conv3(h2, edge_index, edge_attr)))
        h3 = h3 + h2
        h4 = F.relu(self.bn4(self.conv4(h3, edge_index, edge_attr)))

        # Weighted aggregation: deeper layers contribute more
        importance = (0.1 * h1.norm(dim=1) + 0.2 * h2.norm(dim=1) +
                      0.3 * h3.norm(dim=1) + 0.4 * h4.norm(dim=1)).cpu().numpy()
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

        x_jk = self.jk_proj(torch.cat([h1, h2, h3, h4], dim=-1))
        x_mean = global_mean_pool(x_jk, batch)
        x_max = global_max_pool(x_jk, batch)
        graph_feat = torch.cat([x_mean, x_max], dim=-1)
        global_feat = self.global_proj(global_feat)
        combined = torch.cat([graph_feat, global_feat], dim=-1)
        logit = self.classifier(combined).item()

        return importance, logit


# ==================== Load 5-fold ensemble ====================

def load_ensemble():
    checkpoint = torch.load(os.path.join(config.MODEL_DIR, 'gnn_5fold.pt'),
                            map_location=device, weights_only=False)
    return (checkpoint['fold_models'], checkpoint['fold_temps'],
            checkpoint['scaler'], checkpoint['config'], checkpoint['dims'])


def extract_attention_ensemble(smiles, fold_states, scaler, dims):
    """Aggregate attention importance across all 5 folds."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    x = torch.tensor([get_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        edge_attr += [get_bond_features(bond), get_bond_features(bond)]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    gf = get_global_features(mol)
    gf[:10] = scaler.transform(gf[:10].reshape(1, -1))[0]
    global_feat = torch.tensor(gf, dtype=torch.float).unsqueeze(0)
    batch = torch.zeros(x.size(0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                global_feat=global_feat, batch=batch).to(device)

    all_importance = []
    all_logits = []
    for state in fold_states:
        model = ToxicityGNN(dims['node'], dims['edge'], dims['global'],
                            hidden_dim=128, heads=8, dropout=0).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            importance, logit = model.extract_attention(
                data.x, data.edge_index, data.edge_attr, data.batch, data.global_feat
            )
            all_importance.append(importance)
            all_logits.append(logit)

    mean_importance = np.mean(all_importance, axis=0)
    mean_importance = (mean_importance - mean_importance.min()) / (mean_importance.max() - mean_importance.min() + 1e-8)
    mean_logit = np.mean(all_logits)

    return mean_importance, mean_logit, mol


# ==================== Visualization ====================

def visualize_molecule(smiles, fold_states, scaler, dims, save_path):
    """Highlight atoms by attention weight: blue (low) -> white (mid) -> red (high)."""
    importance, logit, mol = extract_attention_ensemble(smiles, fold_states, scaler, dims)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None

    prob = torch.sigmoid(torch.tensor(logit)).item()
    n_atoms = mol.GetNumAtoms()

    colors = {}
    for i in range(n_atoms):
        imp = importance[i]
        if imp > 0.5:
            r = 1.0
            g = b = 1.0 - (imp - 0.5) * 2
        else:
            r = g = imp * 2
            b = 1.0
        colors[i] = (float(r), float(g), float(b))

    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette = False
    opts.highlightRadius = 0.35
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, highlightAtoms=list(range(n_atoms)),
        highlightAtomColors=colors,
    )
    drawer.FinishDrawing()
    with open(save_path, 'wb') as f:
        f.write(drawer.GetDrawingText())

    top3 = np.argsort(importance)[-3:][::-1]
    print(f"Saved: {save_path} | p = {prob:.3f} | Top-3 atoms: {list(top3)}")
    return prob, importance, top3


def batch_visualize(fold_states, scaler, dims, n_pos=5, n_neg=5):
    """Select high-confidence positive and negative samples for visualization."""
    df = pd.read_csv(config.CSV_PATH)

    all_probs = []
    for smiles in df['smiles']:
        _, logit, _ = extract_attention_ensemble(smiles, fold_states, scaler, dims)
        all_probs.append(torch.sigmoid(torch.tensor(logit)).item() if logit is not None else 0.5)

    df['pred_prob'] = all_probs

    pos_samples = df[df['label'] == 1].nlargest(n_pos, 'pred_prob')
    neg_samples = df[df['label'] == 0].nsmallest(n_neg, 'pred_prob')

    print(f"\nVisualizing {n_pos} high-confidence toxic + {n_neg} high-confidence non-toxic molecules...\n")

    results = []
    for _, row in pos_samples.iterrows():
        safe_name = row['smiles'][:30].replace('/', '_').replace('\\', '_')
        path = os.path.join(OUTPUT_DIR, f"POS_p{row['pred_prob']:.3f}_{safe_name}.png")
        out = visualize_molecule(row['smiles'], fold_states, scaler, dims, path)
        if out:
            prob, imp, top3 = out
            results.append({'smiles': row['smiles'], 'label': 1, 'prob': prob,
                            'top3_atoms': list(top3), 'path': path})

    for _, row in neg_samples.iterrows():
        safe_name = row['smiles'][:30].replace('/', '_').replace('\\', '_')
        path = os.path.join(OUTPUT_DIR, f"NEG_p{row['pred_prob']:.3f}_{safe_name}.png")
        out = visualize_molecule(row['smiles'], fold_states, scaler, dims, path)
        if out:
            prob, imp, top3 = out
            results.append({'smiles': row['smiles'], 'label': 0, 'prob': prob,
                            'top3_atoms': list(top3), 'path': path})

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'visualization_results.csv'), index=False)
    print(f"\nSummary saved: {os.path.join(OUTPUT_DIR, 'visualization_results.csv')}")
    return results


# ==================== Main ====================

if __name__ == "__main__":
    print("Loading 5-fold GNN ensemble...")
    fold_states, fold_temps, scaler, gnn_config, dims = load_ensemble()
    print(f"Model loaded | 5 folds | Temperatures: {[f'{t:.3f}' for t in fold_temps]}")

    print("\n=== Batch visualization of high-confidence samples ===")
    results = batch_visualize(fold_states, scaler, dims, n_pos=5, n_neg=5)

    print(f"\nAll images saved to: {OUTPUT_DIR}")
    print("\nInterpretation:")
    print("- Red highlight = atoms/substructures with highest attention (contribute most to toxicity)")
    print("- Comparison of toxic (POS) vs non-toxic (NEG) molecules reveals toxicity-associated patterns")