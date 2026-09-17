"""
10_gnn_5fold.py
Train an attention-based graph neural network (GNN) with 5-fold cross-validation.

Key features:
  - 4-layer Graph Attention Network (GAT) with residual connections and Jumping Knowledge.
  - Focal loss with label smoothing for class imbalance.
  - DropEdge augmentation and Exponential Moving Average (EMA) for regularization.
  - Temperature scaling fitted on out-of-fold logits for probability calibration.

Input:  data/step3_final_v4.csv
Output: Per-fold model weights, temperature parameters, calibrated OOF probabilities,
        and the final ensemble checkpoint (best_gnn_5fold.pt).
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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ==================== GNN configuration ====================
GNN_CONFIG = {
    'batch_size': 64,
    'hidden_dim': 128,
    'heads': 8,
    'dropout': 0.2,
    'lr': 5e-4,
    'weight_decay': 5e-4,
    'epochs': 300,
    'patience': 40,
    'device': 'cpu',
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'drop_edge_prob': 0.1,
    'use_ema': True,
    'ema_decay': 0.999,
    'label_smoothing': 0.1,
}

torch.manual_seed(config.RANDOM_STATE)
np.random.seed(config.RANDOM_STATE)


# ==================== Feature engineering ====================

def get_atom_features(atom):
    """Return a 17-dimensional atom feature vector."""
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
    """Return a 6-dimensional bond feature vector."""
    bt = bond.GetBondType()
    return [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        int(bond.GetIsConjugated()), int(bond.IsInRing()),
    ]


def get_global_features(mol):
    """Return Morgan fingerprint (2048 bits) + 10 physicochemical descriptors."""
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
    """Convert a SMILES string into a PyTorch Geometric Data object."""
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


# ==================== Data augmentation ====================

def augment_graph(data, drop_edge_prob=0.1):
    """Randomly drop edges (DropEdge) as data augmentation during training."""
    if drop_edge_prob > 0 and data.edge_index.size(1) > 0:
        mask = torch.rand(data.edge_index.size(1)) > drop_edge_prob
        data.edge_index = data.edge_index[:, mask]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[mask]
    return data


# ==================== GNN model ====================

class ToxicityGNN(nn.Module):
    """4-layer Graph Attention Network with residual connections and Jumping Knowledge."""

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
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
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


# ==================== Focal loss with label smoothing ====================

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
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ==================== EMA ====================

class EMA:
    """Exponential Moving Average of model weights for stable evaluation."""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

    def get_shadow_state(self):
        return {k: v.clone() for k, v in self.shadow.items()}


# ==================== Temperature scaling ====================

class TemperatureScaler:
    """Post-hoc probability calibration via a single temperature parameter."""

    def __init__(self):
        self.temperature = None

    def fit(self, logits, labels):
        self.temperature = nn.Parameter(torch.ones(1).to(logits.device))
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature.abs()
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.temperature.data = self.temperature.data.abs()
        print(f"[TemperatureScaler] Optimal temperature T = {self.temperature.item():.4f}")
        return self

    def scale(self, logits):
        return logits / self.temperature.abs()

    def predict_proba(self, logits):
        return torch.sigmoid(self.scale(logits))


# ==================== Early stopping ====================

class EarlyStopping:
    def __init__(self, patience=40, delta=0.0005):
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


# ==================== Training and evaluation ====================

def train_epoch(model, loader, optimizer, criterion, device, ema=None, drop_edge_prob=0.1):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        data = augment_graph(data, drop_edge_prob=drop_edge_prob)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.global_feat)
        loss = criterion(out.view(-1), data.y.view(-1))
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.global_feat)
        probs = torch.sigmoid(out).view(-1).cpu().numpy()
        all_preds.extend(np.atleast_1d(probs))
        all_labels.extend(data.y.cpu().numpy())
    auc = roc_auc_score(all_labels, all_preds)
    preds_bin = (np.array(all_preds) >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_bin)
    pre = precision_score(all_labels, preds_bin, zero_division=0)
    rec = recall_score(all_labels, preds_bin, zero_division=0)
    f1 = f1_score(all_labels, preds_bin, zero_division=0)
    return auc, acc, pre, rec, f1


# ==================== Dataset construction ====================

def build_dataset(df):
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
    return graphs, scaler


# ==================== Single fold training ====================

def train_fold(train_g, val_g, fold_idx, dims):
    device = torch.device(GNN_CONFIG['device'])

    train_loader = DataLoader(train_g, batch_size=GNN_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_g, batch_size=GNN_CONFIG['batch_size'])

    model = ToxicityGNN(dims['node'], dims['edge'], dims['global'],
                        hidden_dim=GNN_CONFIG['hidden_dim'],
                        heads=GNN_CONFIG['heads'],
                        dropout=GNN_CONFIG['dropout']).to(device)

    train_labels = [g.y.item() for g in train_g]
    pos_weight = (len(train_labels) - sum(train_labels)) / sum(train_labels) if sum(train_labels) > 0 else 1.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=GNN_CONFIG['lr'],
                                  weight_decay=GNN_CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    criterion = FocalLoss(
        alpha=GNN_CONFIG['focal_alpha'],
        gamma=GNN_CONFIG['focal_gamma'],
        pos_weight=torch.tensor([pos_weight], device=device),
        label_smoothing=GNN_CONFIG['label_smoothing']
    )
    ema = EMA(model, decay=GNN_CONFIG['ema_decay']) if GNN_CONFIG['use_ema'] else None
    early_stopper = EarlyStopping(patience=GNN_CONFIG['patience'])

    print(f"\n{'=' * 55}")
    print(f"Fold {fold_idx + 1}/{config.N_FOLDS}")
    print(f"{'=' * 55}")

    for epoch in range(1, GNN_CONFIG['epochs'] + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device,
                           ema=ema, drop_edge_prob=GNN_CONFIG['drop_edge_prob'])
        val_auc, val_acc, val_pre, val_rec, val_f1 = evaluate(model, val_loader, device)
        scheduler.step(epoch + val_auc)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | F1: {val_f1:.3f}")

        early_stopper(val_auc, model)
        if early_stopper.early_stop:
            print(f">>> Early stopping at epoch {epoch} (best Val AUC: {early_stopper.best_score:.4f})")
            break

    model.load_state_dict(early_stopper.best_state)
    best_auc = early_stopper.best_score
    best_state = early_stopper.best_state

    if ema is not None:
        ema.apply_shadow()
        val_auc_ema, _, _, _, _ = evaluate(model, val_loader, device)
        if val_auc_ema > best_auc:
            best_auc = val_auc_ema
            best_state = ema.get_shadow_state()
            print(f">>> Using EMA weights (Val AUC: {val_auc_ema:.4f})")
        else:
            model.load_state_dict(early_stopper.best_state)

    return best_state, best_auc


# ==================== 5-fold cross-validation ====================

def cross_validate(df):
    device = torch.device(GNN_CONFIG['device'])
    graphs, scaler = build_dataset(df)
    labels = np.array([int(g.y.item()) for g in graphs])
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    node_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    global_dim = graphs[0].global_feat.shape[0]
    dims = {'node': node_dim, 'edge': edge_dim, 'global': global_dim}

    fold_states = []
    fold_temps = []
    fold_aucs = []
    oof_preds = np.zeros(len(graphs))
    oof_labels = np.zeros(len(graphs))

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        train_g = [graphs[i] for i in train_idx]
        val_g = [graphs[i] for i in val_idx]

        state, best_auc = train_fold(train_g, val_g, fold, dims)
        fold_states.append(state)
        fold_aucs.append(best_auc)

        # Collect raw logits on the validation set for calibration
        model = ToxicityGNN(dims['node'], dims['edge'], dims['global'],
                            hidden_dim=GNN_CONFIG['hidden_dim'],
                            heads=GNN_CONFIG['heads'],
                            dropout=0).to(device)
        model.load_state_dict(state)
        model.eval()

        val_loader = DataLoader(val_g, batch_size=GNN_CONFIG['batch_size'])
        fold_logits, fold_labs = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.global_feat)
                fold_logits.extend(out.view(-1).cpu().numpy())
                fold_labs.extend(data.y.cpu().numpy())

        # Temperature scaling
        logits_tensor = torch.tensor(fold_logits, dtype=torch.float, device=device)
        labels_tensor = torch.tensor(fold_labs, dtype=torch.float, device=device)
        temp_scaler = TemperatureScaler()
        temp_scaler.fit(logits_tensor, labels_tensor)
        fold_temps.append(temp_scaler.temperature.item())

        calibrated_probs = temp_scaler.predict_proba(logits_tensor).detach().cpu().numpy()
        for i, idx in enumerate(val_idx):
            oof_preds[idx] = calibrated_probs[i]
            oof_labels[idx] = fold_labs[i]

        print(f"Fold {fold + 1} done | Val AUC: {best_auc:.4f} | T = {temp_scaler.temperature.item():.4f}")

    oof_auc = roc_auc_score(oof_labels, oof_preds)

    print(f"\n{'=' * 55}")
    print("5-fold cross-validation summary")
    print(f"Fold AUCs    : {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Mean +/- Std : {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    print(f"OOF AUC      : {oof_auc:.4f}")
    print(f"Temperatures : {[f'{t:.4f}' for t in fold_temps]}")
    print(f"{'=' * 55}")

    # Save checkpoint and calibrated OOF probabilities
    torch.save({
        'fold_models': fold_states,
        'fold_temps': fold_temps,
        'scaler': scaler,
        'fold_aucs': fold_aucs,
        'oof_auc': oof_auc,
        'config': GNN_CONFIG,
        'dims': dims,
    }, os.path.join(config.MODEL_DIR, 'gnn_5fold.pt'))
    print(f"\nModel saved: {os.path.join(config.MODEL_DIR, 'gnn_5fold.pt')}")

    np.save(os.path.join(config.RESULTS_DIR, 'prob_gnn_calibrated.npy'), oof_preds)
    np.save(os.path.join(config.RESULTS_DIR, 'label_gnn_calibrated.npy'), oof_labels)
    print("Calibrated OOF probabilities saved.")

    return fold_states, fold_temps, scaler, oof_preds, oof_labels


if __name__ == "__main__":
    df = pd.read_csv(config.CSV_PATH)
    cross_validate(df)