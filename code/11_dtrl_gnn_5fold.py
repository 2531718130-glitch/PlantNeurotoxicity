"""
11_dtrl_gnn_5fold.py
Train a Disentangled Toxicity Representation Learning (DTRL) GNN with 5-fold CV.

The DTRL head decomposes the graph-level representation into three orthogonal
subspaces (toxicity, pharmacokinetic, scaffold) and enforces orthogonality.

Input:  data/step3_final_v4.csv
Output: Per-fold model weights and OOF probabilities, saved to
        results/dtrl/dtrl_gnn_5fold.pt
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DTRL_CONFIG = {
    'batch_size': 64,
    'hidden_dim': 128,
    'heads': 8,
    'dropout': 0.2,
    'lr': 5e-4,
    'weight_decay': 5e-4,
    'epochs': 150,
    'patience': 30,
    'device': 'cpu',
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'drop_edge_prob': 0.1,
    'label_smoothing': 0.1,
    'tox_dim': 64,
    'pk_dim': 64,
    'scaf_dim': 64,
    'lambda_orth': 0.01,
    'lambda_pk': 0.1,
}

torch.manual_seed(config.RANDOM_STATE)
np.random.seed(config.RANDOM_STATE)


# ==================== Feature engineering ====================

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
                global_feat=global_feat, y=y, smiles=smiles), global_feat


# ==================== DTRL head ====================

class DisentangledHead(nn.Module):
    """Decompose the graph representation into three orthogonal subspaces."""

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


def orthogonality_loss(h_tox, h_pk, h_scaf):
    """Penalize the correlation between the three subspaces."""
    loss = 0.0
    loss += torch.norm(h_tox.T @ h_pk, p='fro') ** 2
    loss += torch.norm(h_tox.T @ h_scaf, p='fro') ** 2
    loss += torch.norm(h_pk.T @ h_scaf, p='fro') ** 2
    return loss


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
            combined_dim,
            tox_dim=DTRL_CONFIG['tox_dim'],
            pk_dim=DTRL_CONFIG['pk_dim'],
            scaf_dim=DTRL_CONFIG['scaf_dim'],
            dropout=dropout
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


# ==================== Loss and training utilities ====================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - 2 * self.label_smoothing) + self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class EarlyStopping:
    def __init__(self, patience=30, delta=0.0005):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_auc, model):
        if self.best_score is None or val_auc > self.best_score + self.delta:
            self.best_score = val_auc
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_focal, total_orth, total_pk = 0, 0, 0, 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        logit, h_tox, h_pk, h_scaf, pk_pred = model(
            data.x, data.edge_index, data.edge_attr, data.batch, data.global_feat
        )

        n_pos = max((data.y == 1).sum().item(), 1)
        pos_weight = torch.tensor([(data.y.shape[0] - n_pos) / n_pos], device=device)
        focal_criterion = FocalLoss(
            alpha=DTRL_CONFIG['focal_alpha'],
            gamma=DTRL_CONFIG['focal_gamma'],
            pos_weight=pos_weight,
            label_smoothing=DTRL_CONFIG['label_smoothing']
        )
        loss_focal = focal_criterion(logit.view(-1), data.y.view(-1))
        loss_orth = orthogonality_loss(h_tox, h_pk, h_scaf)

        gf = data.global_feat
        if gf.dim() == 1:
            gf = gf.unsqueeze(0)
        pk_targets = gf[:, :3]
        loss_pk = F.mse_loss(pk_pred, pk_targets)

        loss = loss_focal + DTRL_CONFIG['lambda_orth'] * loss_orth + DTRL_CONFIG['lambda_pk'] * loss_pk
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_focal += loss_focal.item() * data.num_graphs
        total_orth += loss_orth.item() * data.num_graphs
        total_pk += loss_pk.item() * data.num_graphs

    n = len(loader.dataset)
    return total_loss / n, total_focal / n, total_orth / n, total_pk / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(device)
        logit, _, _, _, _ = model(
            data.x, data.edge_index, data.edge_attr, data.batch, data.global_feat
        )
        probs = torch.sigmoid(logit).view(-1).cpu().numpy()
        all_preds.extend(np.atleast_1d(probs))
        all_labels.extend(data.y.cpu().numpy())
    return roc_auc_score(all_labels, all_preds)


# ==================== Main ====================

if __name__ == "__main__":
    device = torch.device(DTRL_CONFIG['device'])
    OUTPUT_DIR = os.path.join(config.RESULTS_DIR, 'dtrl')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("DTRL-GNN 5-fold cross-validation")
    print("=" * 60)

    df = pd.read_csv(config.CSV_PATH)
    print(f"Dataset: {len(df)} compounds")

    graphs, global_feats = [], []
    for _, row in df.iterrows():
        g, gf = mol_to_graph(row['smiles'], row['label'])
        if g is not None:
            graphs.append(g)
            global_feats.append(gf)

    global_arr = np.stack(global_feats)
    scaler = StandardScaler()
    global_arr[:, :10] = scaler.fit_transform(global_arr[:, :10])
    for i, g in enumerate(graphs):
        g.global_feat = torch.tensor(global_arr[i], dtype=torch.float)

    labels = np.array([int(g.y.item()) for g in graphs])
    print(f"Valid graphs: {len(graphs)}")

    node_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    global_dim = graphs[0].global_feat.shape[0]
    dims = {'node': node_dim, 'edge': edge_dim, 'global': global_dim}
    print(f"Node: {node_dim}, Edge: {edge_dim}, Global: {global_dim}")

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    fold_aucs, fold_states = [], []
    oof_preds = np.zeros(len(graphs))
    oof_labels = np.zeros(len(graphs))

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*55}")
        print(f"Fold {fold + 1}/{config.N_FOLDS}")
        print(f"{'='*55}")

        train_g = [graphs[i] for i in train_idx]
        val_g = [graphs[i] for i in val_idx]

        train_loader = DataLoader(train_g, batch_size=DTRL_CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_g, batch_size=DTRL_CONFIG['batch_size'])

        model = DTRL_GNN(node_dim, edge_dim, global_dim,
                         hidden_dim=DTRL_CONFIG['hidden_dim'],
                         heads=DTRL_CONFIG['heads'],
                         dropout=DTRL_CONFIG['dropout']).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=DTRL_CONFIG['lr'],
                                      weight_decay=DTRL_CONFIG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        early_stopper = EarlyStopping(patience=DTRL_CONFIG['patience'])

        for epoch in range(1, DTRL_CONFIG['epochs'] + 1):
            loss, loss_f, loss_o, loss_pk = train_epoch(model, train_loader, optimizer, device)
            val_auc = evaluate(model, val_loader, device)
            scheduler.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Focal: {loss_f:.4f} | "
                      f"Orth: {loss_o:.4f} | PK: {loss_pk:.4f} | AUC: {val_auc:.4f}")

            early_stopper(val_auc, model)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch} (best AUC: {early_stopper.best_score:.4f})")
                break

        model.load_state_dict(early_stopper.best_state)
        fold_aucs.append(early_stopper.best_score)
        fold_states.append(early_stopper.best_state)

        model.eval()
        for i, g in zip(val_idx, val_g):
            single_loader = DataLoader([g], batch_size=1)
            for data in single_loader:
                data = data.to(device)
                with torch.no_grad():
                    logit, _, _, _, _ = model(data.x, data.edge_index, data.edge_attr,
                                              data.batch, data.global_feat)
                    prob = torch.sigmoid(logit).item()
                oof_preds[i] = prob
                oof_labels[i] = g.y.item()

        print(f"Fold {fold + 1} done | AUC: {early_stopper.best_score:.4f}")

    print(f"\n{'='*55}")
    print("DTRL-GNN 5-fold CV summary")
    print(f"{'='*55}")
    print(f"Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Mean +/- Std: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")

    torch.save({
        'fold_states': fold_states,
        'fold_aucs': fold_aucs,
        'scaler': scaler,
        'config': DTRL_CONFIG,
        'dims': dims,
    }, os.path.join(OUTPUT_DIR, 'dtrl_gnn_5fold.pt'))

    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'dtrl_gnn_5fold.pt')}")
    print(f"{'=' * 55}")